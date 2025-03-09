import numpy as np
import scipy
import tensorflow as tf


class Fitter:
    def __init__(self, indata, options):
        self.indata = indata
        self.binByBinStat = options.binByBinStat
        self.normalize = options.normalize
        self.systgroupsfull = self.indata.systgroups.tolist()
        self.systgroupsfull.append("stat")
        if self.binByBinStat:
            self.systgroupsfull.append("binByBinStat")

        if options.externalCovariance and not options.chisqFit:
            raise Exception(
                'option "--externalCovariance" only works with "--chisqFit"'
            )
        if options.externalCovariance and options.binByBinStat:
            raise Exception(
                'option "--binByBinStat" currently not supported for options "--externalCovariance"'
            )

        self.chisqFit = options.chisqFit
        self.externalCovariance = options.externalCovariance

        # if quantities are computed with profiling, usually set to true after the fit
        self.profile = False

        self.nsystgroupsfull = len(self.systgroupsfull)

        self.pois = []

        if options.POIMode == "mu":
            self.npoi = self.indata.nsignals
            poidefault = options.POIDefault * tf.ones(
                [self.npoi], dtype=self.indata.dtype
            )
            for signal in self.indata.signals:
                self.pois.append(signal)
        elif options.POIMode == "none":
            self.npoi = 0
            poidefault = tf.zeros([], dtype=self.indata.dtype)
        else:
            raise Exception("unsupported POIMode")

        self.parms = np.concatenate([self.pois, self.indata.systs])

        self.allowNegativePOI = options.allowNegativePOI

        if self.allowNegativePOI:
            self.xpoidefault = poidefault
        else:
            self.xpoidefault = tf.sqrt(poidefault)

        # tf variable containing all fit parameters
        thetadefault = tf.zeros([self.indata.nsyst], dtype=self.indata.dtype)
        if self.npoi > 0:
            xdefault = tf.concat([self.xpoidefault, thetadefault], axis=0)
        else:
            xdefault = thetadefault

        self.x = tf.Variable(xdefault, trainable=True, name="x")

        # observed number of events per bin
        self.nobs = tf.Variable(self.indata.data_obs, trainable=False, name="nobs")

        if self.chisqFit:
            if self.externalCovariance:
                if self.indata.data_cov_inv is None:
                    raise RuntimeError("No external covariance found in input data.")
                # provided covariance
                self.data_cov_inv = self.indata.data_cov_inv
            else:
                # covariance from data stat
                if tf.math.reduce_any(self.nobs <= 0).numpy():
                    raise RuntimeError(
                        "Bins in 'nobs <= 0' encountered, chi^2 fit can not be performed."
                    )
                self.data_cov_inv = tf.linalg.diag(tf.math.reciprocal(self.nobs))

        # constraint minima for nuisance parameters
        self.theta0 = tf.Variable(
            tf.zeros([self.indata.nsyst], dtype=self.indata.dtype),
            trainable=False,
            name="theta0",
        )

        # global observables for mc stat uncertainty
        self.beta0 = tf.Variable(
            tf.ones_like(self.indata.data_obs), trainable=False, name="beta0"
        )

        nexpfullcentral = self.expected_yield()
        self.nexpnom = tf.Variable(nexpfullcentral, trainable=False, name="nexpnom")

        # parameter covariance matrix
        self.cov = tf.Variable(self.prefit_covariance(), trainable=False, name="cov")

    def prefit_covariance(self, unconstrained_err=0.0):
        # free parameters are taken to have zero uncertainty for the purposes of prefit uncertainties
        var_poi = tf.zeros([self.npoi], dtype=self.indata.dtype)

        # nuisances have their uncertainty taken from the constraint term, but unconstrained nuisances
        # are set to a placeholder uncertainty (zero by default) for the purposes of prefit uncertainties
        var_theta = tf.where(
            self.indata.constraintweights == 0.0,
            unconstrained_err,
            tf.math.reciprocal(self.indata.constraintweights),
        )

        invhessianprefit = tf.linalg.diag(tf.concat([var_poi, var_theta], axis=0))
        return invhessianprefit

    @tf.function
    def val_jac(self, fun, *args, **kwargs):
        with tf.GradientTape() as t:
            val = fun(*args, **kwargs)
        jac = t.jacobian(val, self.x)

        return val, jac

    def theta0defaultassign(self):
        self.profile = False
        self.theta0.assign(tf.zeros([self.indata.nsyst], dtype=self.theta0.dtype))

    def xdefaultassign(self):
        if self.npoi == 0:
            self.x.assign(self.theta0)
        else:
            self.x.assign(tf.concat([self.xpoidefault, self.theta0], axis=0))

    def beta0defaultassign(self):
        self.beta0.assign(tf.ones_like(self.indata.data_obs, dtype=self.beta0.dtype))

    def defaultassign(self):
        self.cov.assign(self.prefit_covariance())
        self.theta0defaultassign()
        self.beta0defaultassign()
        self.xdefaultassign()

    def bayesassign(self):
        if self.npoi == 0:
            self.x.assign(
                self.theta0
                + tf.random.normal(shape=self.theta0.shape, dtype=self.theta0.dtype)
            )
        else:
            self.x.assign(
                tf.concat(
                    [
                        self.xpoidefault,
                        self.theta0
                        + tf.random.normal(
                            shape=self.theta0.shape, dtype=self.theta0.dtype
                        ),
                    ],
                    axis=0,
                )
            )

    def frequentistassign(self):
        self.theta0.assign(
            tf.random.normal(shape=self.theta0.shape, dtype=self.theta0.dtype)
        )

    def toyassign(self, bayesian=False, bootstrap_data=False):
        if bayesian:
            # randomize actual values
            self.bayesassign()
        else:
            # randomize nuisance constraint minima
            self.frequentistassign()

        if self.binByBinStat:
            # TODO properly implement randomization of constraint parameters associated with bin-by-bin stat nuisances for frequentist toys,
            # currently bin-by-bin stat fluctuations are always handled in a bayesian way in toys
            # this also means bin-by-bin stat fluctuations are not consistently propagated for bootstrap toys from data
            self.beta0.assign(
                tf.random.gamma(
                    shape=[],
                    alpha=self.indata.kstat + 1.0,
                    beta=self.indata.kstat,
                    dtype=self.beta0.dtype,
                )
            )

        if bootstrap_data:
            # randomize from observed data
            if self.binByBinStat and not bayesian:
                raise Exception(
                    "Since bin-by-bin statistical uncertainties are always propagated in a bayesian manner, \
                    they cannot currently be consistently propagated for bootstrap toys"
                )
            self.nobs.assign(
                tf.random.poisson(
                    lam=self.indata.data_obs, shape=[], dtype=self.nobs.dtype
                )
            )
        else:
            # randomize from expectation
            self.nobs.assign(
                tf.random.poisson(
                    lam=self.expected_yield(),
                    shape=[],
                    dtype=self.nobs.dtype,
                )
            )

        # assign start values for nuisance parameters to constraint minima
        self.xdefaultassign()
        # set likelihood offset
        self.nexpnom.assign(self.expected_yield())

    def _compute_impact_group(self, v, idxs):
        cov_reduced = tf.gather(self.cov[self.npoi :, self.npoi :], idxs, axis=0)
        cov_reduced = tf.gather(cov_reduced, idxs, axis=1)
        v_reduced = tf.gather(v, idxs, axis=1)
        invC_v = tf.linalg.solve(cov_reduced, tf.transpose(v_reduced))
        v_invC_v = tf.einsum("ij,ji->i", v_reduced, invC_v)
        return tf.sqrt(v_invC_v)

    @tf.function
    def impacts_parms(self, hess):
        # impact for poi at index i in covariance matrix from nuisance with index j is C_ij/sqrt(C_jj) = <deltax deltatheta>/sqrt(<deltatheta^2>)
        cov_poi = self.cov[: self.npoi]
        cov_noi = tf.gather(self.cov[self.npoi :], self.indata.noigroupidxs)
        v = tf.concat([cov_poi, cov_noi], axis=0)
        impacts = v / tf.reshape(tf.sqrt(tf.linalg.diag_part(self.cov)), [1, -1])

        nstat = self.npoi + self.indata.nsystnoconstraint
        hess_stat = hess[:nstat, :nstat]
        inv_hess_stat = tf.linalg.inv(hess_stat)

        if self.binByBinStat:
            # impact bin-by-bin stat
            val_no_bbb, grad_no_bbb, hess_no_bbb = self.loss_val_grad_hess(
                profile_grad=False
            )

            hess_stat_no_bbb = hess_no_bbb[:nstat, :nstat]
            inv_hess_stat_no_bbb = tf.linalg.inv(hess_stat_no_bbb)

            impacts_data_stat = tf.sqrt(tf.linalg.diag_part(inv_hess_stat_no_bbb))
            impacts_data_stat = tf.reshape(impacts_data_stat, (-1, 1))

            impacts_bbb_sq = tf.linalg.diag_part(inv_hess_stat - inv_hess_stat_no_bbb)
            impacts_bbb = tf.sqrt(tf.nn.relu(impacts_bbb_sq))  # max(0,x)
            impacts_bbb = tf.reshape(impacts_bbb, (-1, 1))
            impacts_grouped = tf.concat([impacts_data_stat, impacts_bbb], axis=1)
        else:
            impacts_data_stat = tf.sqrt(tf.linalg.diag_part(inv_hess_stat))
            impacts_data_stat = tf.reshape(impacts_data_stat, (-1, 1))
            impacts_grouped = impacts_data_stat

        if len(self.indata.systgroupidxs):
            impacts_grouped_syst = tf.map_fn(
                lambda idxs: self._compute_impact_group(v[:, self.npoi :], idxs),
                tf.ragged.constant(self.indata.systgroupidxs, dtype=tf.int32),
                fn_output_signature=tf.TensorSpec(
                    shape=(impacts.shape[0],), dtype=tf.float64
                ),
            )
            impacts_grouped_syst = tf.transpose(impacts_grouped_syst)
            impacts_grouped = tf.concat([impacts_grouped_syst, impacts_grouped], axis=1)

        return impacts, impacts_grouped

    def _compute_global_impact_group(self, d_squared, idxs):
        gathered = tf.gather(d_squared, idxs, axis=-1)
        d_squared_summed = tf.reduce_sum(gathered, axis=-1)
        return tf.sqrt(d_squared_summed)

    @tf.function
    def global_impacts_parms(self):
        # compute impacts for pois and nois
        dxdtheta0, dxdnobs, dxdbeta0 = self._compute_derivatives_x()

        dxdtheta0_poi = dxdtheta0[: self.npoi]
        dxdtheta0_noi = tf.gather(dxdtheta0[self.npoi :], self.indata.noigroupidxs)
        dxdtheta0 = tf.concat([dxdtheta0_poi, dxdtheta0_noi], axis=0)
        dxdtheta0_squared = tf.square(dxdtheta0)

        # global impact data stat
        dxdnobs_poi = dxdnobs[: self.npoi]
        dxdnobs_noi = tf.gather(dxdnobs[self.npoi :], self.indata.noigroupidxs)
        dxdnobs = tf.concat([dxdnobs_poi, dxdnobs_noi], axis=0)

        if self.externalCovariance:
            data_cov = tf.linalg.inv(self.data_cov_inv)
            # equivalent to tf.linalg.diag_part(dxdnobs @ data_cov @ tf.transpose(dxdnobs)) but avoiding computing full matrix
            data_stat = tf.einsum("ij,jk,ik->i", dxdnobs, data_cov, dxdnobs)
        else:
            data_stat = tf.reduce_sum(tf.square(dxdnobs) * self.nobs, axis=-1)

        data_stat = tf.sqrt(data_stat)
        impacts_data_stat = tf.reshape(data_stat, (-1, 1))

        if self.binByBinStat:
            # global impact bin-by-bin stat
            dxdbeta0_poi = dxdbeta0[: self.npoi]
            dxdbeta0_noi = tf.gather(dxdbeta0[self.npoi :], self.indata.noigroupidxs)
            dxdbeta0 = tf.concat([dxdbeta0_poi, dxdbeta0_noi], axis=0)

            impacts_bbb = tf.sqrt(
                tf.reduce_sum(
                    tf.square(dxdbeta0) * tf.math.reciprocal(self.indata.kstat), axis=-1
                )
            )
            impacts_bbb = tf.reshape(impacts_bbb, (-1, 1))
            impacts_grouped = tf.concat([impacts_data_stat, impacts_bbb], axis=1)
        else:
            impacts_grouped = impacts_data_stat

        if len(self.indata.systgroupidxs):
            impacts_grouped_syst = tf.map_fn(
                lambda idxs: self._compute_global_impact_group(dxdtheta0_squared, idxs),
                tf.ragged.constant(self.indata.systgroupidxs, dtype=tf.int32),
                fn_output_signature=tf.TensorSpec(
                    shape=(dxdtheta0_squared.shape[0],), dtype=tf.float64
                ),
            )
            impacts_grouped_syst = tf.transpose(impacts_grouped_syst)
            impacts_grouped = tf.concat([impacts_grouped_syst, impacts_grouped], axis=1)

        # global impacts of unconstrained parameters are always 0, only store impacts of constrained ones
        impacts = dxdtheta0[:, self.indata.nsystnoconstraint :]

        return impacts, impacts_grouped

    def _expvar_profiled(
        self, fun_exp, compute_cov=False, compute_global_impacts=False
    ):

        with tf.GradientTape() as t:
            t.watch([self.theta0, self.nobs, self.beta0])
            expected = fun_exp()
            expected_flat = tf.reshape(expected, (-1,))

        pdexpdx, pdexpdtheta0, pdexpdnobs, pdexpdbeta0 = t.jacobian(
            expected_flat,
            [self.x, self.theta0, self.nobs, self.beta0],
            unconnected_gradients="zero",
        )

        dxdtheta0, dxdnobs, dxdbeta0 = self._compute_derivatives_x()

        dexpdtheta0 = pdexpdtheta0 + pdexpdx @ dxdtheta0
        dexpdnobs = pdexpdnobs + pdexpdx @ dxdnobs
        dexpdbeta0 = pdexpdbeta0 + pdexpdx @ dxdbeta0

        # FIXME factorize this part better with the global impacts calculation

        var_theta0 = tf.where(
            self.indata.constraintweights == 0.0,
            tf.zeros_like(self.indata.constraintweights),
            tf.math.reciprocal(self.indata.constraintweights),
        )

        dtheta0 = tf.math.sqrt(var_theta0)
        dnobs = tf.math.sqrt(self.nobs)
        dbeta0 = tf.math.sqrt(tf.math.reciprocal(self.indata.kstat))

        dexpdtheta0 *= dtheta0[None, :]
        dexpdnobs *= dnobs[None, :]
        dexpdbeta0 *= dbeta0[None, :]

        if compute_cov:
            expcov_stat = dexpdnobs @ tf.transpose(dexpdnobs)
            expcov = dexpdtheta0 @ tf.transpose(dexpdtheta0) + expcov_stat
            if self.binByBinStat:
                expcov_binByBinStat = dexpdbeta0 @ tf.transpose(dexpdbeta0)
                expcov += expcov_binByBinStat

            expvar = tf.linalg.diag_part(expcov)
        else:
            expcov = None
            expvar_stat = tf.reduce_sum(tf.square(dexpdnobs), axis=-1)
            expvar = tf.reduce_sum(tf.square(dexpdtheta0), axis=-1) + expvar_stat
            if self.binByBinStat:
                expvar_binByBinStat = tf.reduce_sum(tf.square(dexpdbeta0), axis=-1)
                expvar += expvar_binByBinStat

        expvar = tf.reshape(expvar, expected.shape)

        if compute_global_impacts:
            # global impacts of unconstrained parameters are always 0, only store impacts of constrained ones
            impacts = dexpdtheta0[:, self.indata.nsystnoconstraint :]

            if compute_cov:
                expvar_stat = tf.linalg.diag_part(expcov_stat)
            impacts_stat = tf.sqrt(expvar_stat)
            impacts_stat = tf.reshape(impacts_stat, (-1, 1))

            if self.binByBinStat:
                if compute_cov:
                    expvar_binByBinStat = tf.linalg.diag_part(expcov_binByBinStat)
                impacts_binByBinStat = tf.sqrt(expvar_binByBinStat)
                impacts_binByBinStat = tf.reshape(impacts_binByBinStat, (-1, 1))
                impacts_grouped = tf.concat(
                    [impacts_stat, impacts_binByBinStat], axis=1
                )
            else:
                impacts_grouped = impacts_stat

            if len(self.indata.systgroupidxs):
                dexpdtheta0_squared = tf.square(dexpdtheta0)
                impacts_grouped_syst = tf.map_fn(
                    lambda idxs: self._compute_global_impact_group(
                        dexpdtheta0_squared, idxs
                    ),
                    tf.ragged.constant(self.indata.systgroupidxs, dtype=tf.int32),
                    fn_output_signature=tf.TensorSpec(
                        shape=(dexpdtheta0_squared.shape[0],), dtype=tf.float64
                    ),
                )
                impacts_grouped_syst = tf.transpose(impacts_grouped_syst)
                impacts_grouped = tf.concat(
                    [impacts_grouped_syst, impacts_grouped], axis=1
                )
        else:
            impacts = None
            impacts_grouped = None

        return expected, expvar, expcov, impacts, impacts_grouped

    def _expvar_optimized(self, fun_exp, skipBinByBinStat=False):
        # compute uncertainty on expectation propagating through uncertainty on fit parameters using full covariance matrix

        # FIXME this doesn't actually work for the positive semi-definite case
        invhesschol = tf.linalg.cholesky(self.cov)

        # since the full covariance matrix with respect to the bin counts is given by J^T R^T R J, then summing RJ element-wise squared over the parameter axis gives the diagonal elements

        expected = fun_exp()

        # dummy vector for implicit transposition
        u = tf.ones_like(expected)
        with tf.GradientTape(watch_accessed_variables=False) as t1:
            t1.watch(u)
            with tf.GradientTape() as t2:
                expected = fun_exp()
            # this returns dndx_j = sum_i u_i dn_i/dx_j
            Ju = t2.gradient(expected, self.x, output_gradients=u)
            Ju = tf.transpose(Ju)
            Ju = tf.reshape(Ju, [-1, 1])
            RJu = tf.matmul(tf.stop_gradient(invhesschol), Ju, transpose_a=True)
            RJu = tf.reshape(RJu, [-1])
        RJ = t1.jacobian(RJu, u)
        sRJ2 = tf.reduce_sum(RJ**2, axis=0)
        sRJ2 = tf.reshape(sRJ2, expected.shape)
        if self.binByBinStat and not skipBinByBinStat:
            # add MC stat uncertainty on variance
            sumw2 = tf.square(expected) / self.indata.kstat
            sRJ2 = sRJ2 + sumw2
        return expected, sRJ2

    def _chi2(self, res, rescov):
        resv = tf.reshape(res, (-1, 1))

        chi_square_value = tf.transpose(resv) @ tf.linalg.solve(rescov, resv)

        return chi_square_value[0, 0]

    def _expvar(self, fun_exp, compute_cov=False, compute_global_impacts=False):
        # compute uncertainty on expectation propagating through uncertainty on fit parameters using full covariance matrix
        # FIXME switch back to optimized version at some point?

        with tf.GradientTape() as t:
            t.watch([self.theta0, self.nobs, self.beta0])
            expected = fun_exp()
            expected_flat = tf.reshape(expected, (-1,))
        pdexpdx, pdexpdnobs, pdexpdbeta0 = t.jacobian(
            expected_flat,
            [self.x, self.nobs, self.beta0],
        )

        expcov = pdexpdx @ tf.matmul(self.cov, pdexpdx, transpose_b=True)

        if pdexpdnobs is not None:
            varnobs = self.nobs
            exp_cov_stat = pdexpdnobs @ (varnobs[:, None] * tf.transpose(pdexpdnobs))
            expcov += exp_cov_stat

        expcov_noBBB = expcov
        if self.binByBinStat:
            varbeta0 = tf.math.reciprocal(self.indata.kstat)
            exp_cov_BBB = pdexpdbeta0 @ (varbeta0[:, None] * tf.transpose(pdexpdbeta0))
            expcov += exp_cov_BBB

        if compute_global_impacts:
            raise NotImplementedError(
                "WARNING: Global impacts on observables without profiling is under development!"
            )
            # FIXME This is not correct

            dxdtheta0, dxdnobs, dxdbeta0 = self._compute_derivatives_x()

            # dexpdtheta0 = pdexpdtheta0 + pdexpdx @ dxdtheta0 # TODO: pdexpdtheta0 not available?
            dexpdtheta0 = pdexpdx @ dxdtheta0

            # TODO: including effect of beta0

            var_theta0 = tf.where(
                self.indata.constraintweights == 0.0,
                tf.zeros_like(self.indata.constraintweights),
                tf.math.reciprocal(self.indata.constraintweights),
            )
            dtheta0 = tf.math.sqrt(var_theta0)
            dexpdtheta0 *= dtheta0[None, :]

            dexpdtheta0_squared = tf.square(dexpdtheta0)

            # global impacts of unconstrained parameters are always 0, only store impacts of constrained ones
            impacts = dexpdtheta0[:, self.indata.nsystnoconstraint :]

            # stat global impact from all unconstrained parameters, not sure if this is correct TODO: check
            impacts_stat = tf.sqrt(
                tf.linalg.diag_part(expcov_noBBB)
                - tf.reduce_sum(dexpdtheta0_squared, axis=-1)
            )
            impacts_stat = tf.reshape(impacts_stat, (-1, 1))

            if self.binByBinStat:
                impacts_BBB_stat = tf.sqrt(tf.linalg.diag_part(exp_cov_BBB))
                impacts_BBB_stat = tf.reshape(impacts_BBB_stat, (-1, 1))
                impacts_grouped = tf.concat([impacts_stat, impacts_BBB_stat], axis=1)
            else:
                impacts_grouped = impacts_stat

            if len(self.indata.systgroupidxs):
                impacts_grouped_syst = tf.map_fn(
                    lambda idxs: self._compute_global_impact_group(
                        dexpdtheta0_squared, idxs
                    ),
                    tf.ragged.constant(self.indata.systgroupidxs, dtype=tf.int32),
                    fn_output_signature=tf.TensorSpec(
                        shape=(dexpdtheta0_squared.shape[0],), dtype=tf.float64
                    ),
                )
                impacts_grouped_syst = tf.transpose(impacts_grouped_syst)
                impacts_grouped = tf.concat(
                    [impacts_grouped_syst, impacts_grouped], axis=1
                )
        else:
            impacts = None
            impacts_grouped = None

        expvar = tf.linalg.diag_part(expcov)
        expvar = tf.reshape(expvar, expected.shape)

        return expected, expvar, expcov, impacts, impacts_grouped

    def _expvariations(self, fun_exp, correlations):
        with tf.GradientTape() as t:
            expected = fun_exp()
            expected_flat = tf.reshape(expected, (-1,))
        dexpdx = t.jacobian(expected_flat, self.x)

        if correlations:
            # construct the matrix such that the columns represent
            # the variations associated with profiling a given parameter
            # taking into account its correlations with the other parameters
            dx = self.cov / tf.math.sqrt(tf.linalg.diag_part(self.cov))[None, :]

            dexp = dexpdx @ dx
        else:
            dexp = dexpdx * tf.math.sqrt(tf.linalg.diag_part(self.cov))[None, :]

        dexp = tf.reshape(dexp, (*expected.shape, -1))

        down = expected[..., None] - dexp
        up = expected[..., None] + dexp

        expvars = tf.stack([down, up], axis=-1)

        return expvars

    def _compute_yields_noBBB(self, compute_normfull=False):
        xpoi = self.x[: self.npoi]
        theta = self.x[self.npoi :]

        if self.allowNegativePOI:
            poi = xpoi
        else:
            poi = tf.square(xpoi)

        rnorm = tf.concat(
            [poi, tf.ones([self.indata.nproc - poi.shape[0]], dtype=self.indata.dtype)],
            axis=0,
        )
        mrnorm = tf.expand_dims(rnorm, -1)
        ernorm = tf.reshape(rnorm, [1, -1])

        if self.indata.symmetric_tensor:
            mthetaalpha = tf.reshape(theta, [self.indata.nsyst, 1])
        else:
            # interpolation for asymmetric log-normal
            twox = 2.0 * theta
            twox2 = twox * twox
            alpha = 0.125 * twox * (twox2 * (3.0 * twox2 - 10.0) + 15.0)
            alpha = tf.clip_by_value(alpha, -1.0, 1.0)

            thetaalpha = theta * alpha

            mthetaalpha = tf.stack(
                [theta, thetaalpha], axis=0
            )  # now has shape [2,nsyst]
            mthetaalpha = tf.reshape(mthetaalpha, [2 * self.indata.nsyst, 1])

        if self.indata.sparse:
            logsnorm = tf.sparse.sparse_dense_matmul(
                self.indata.logk_sparse, mthetaalpha
            )
            logsnorm = tf.squeeze(logsnorm, -1)
            snorm = tf.exp(logsnorm)

            snormnorm_sparse = self.indata.norm_sparse.with_values(
                snorm * self.indata.norm_sparse.values
            )
            nexpfullcentral = tf.sparse.sparse_dense_matmul(snormnorm_sparse, mrnorm)
            nexpfullcentral = tf.squeeze(nexpfullcentral, -1)

            if compute_normfull:
                snormnorm = tf.sparse.to_dense(snormnorm_sparse)

        else:
            if self.indata.symmetric_tensor:
                mlogk = tf.reshape(
                    self.indata.logk,
                    [self.indata.nbins * self.indata.nproc, self.indata.nsyst],
                )
            else:
                mlogk = tf.reshape(
                    self.indata.logk,
                    [self.indata.nbins * self.indata.nproc, 2 * self.indata.nsyst],
                )

            logsnorm = tf.matmul(mlogk, mthetaalpha)
            logsnorm = tf.reshape(logsnorm, [self.indata.nbins, self.indata.nproc])

            snorm = tf.exp(logsnorm)

            snormnorm = snorm * self.indata.norm
            nexpfullcentral = tf.matmul(snormnorm, mrnorm)
            nexpfullcentral = tf.squeeze(nexpfullcentral, -1)

        # if options.saveHists:
        if compute_normfull:
            normfullcentral = ernorm * snormnorm
        else:
            normfullcentral = None

        if self.normalize:
            # FIXME this should be done per-channel ideally
            normscale = tf.reduce_sum(self.nobs) / tf.reduce_sum(nexpfullcentral)

            nexpfullcentral *= normscale
            if compute_normfull:
                normfullcentral *= normscale

        return nexpfullcentral, normfullcentral

    def _compute_yields_with_beta(self, profile_grad=True, compute_normfull=False):
        nexpfullcentral, normfullcentral = self._compute_yields_noBBB(compute_normfull)

        nexpfull = nexpfullcentral
        normfull = normfullcentral

        beta = None
        if self.binByBinStat:
            if self.profile:
                beta = (self.nobs + self.indata.kstat) / (
                    nexpfullcentral + self.indata.kstat
                )
                if not profile_grad:
                    beta = tf.stop_gradient(beta)
            else:
                beta = self.beta0
            nexpfull = beta * nexpfullcentral
            if compute_normfull:
                normfull = beta[..., None] * normfullcentral

            if self.normalize:
                # FIXME this is probably not fully consistent when combined with the binByBinStat
                normscale = tf.reduce_sum(self.nobs) / tf.reduce_sum(nexpfull)

                nexpfull *= normscale
                if compute_normfull:
                    normfull *= normscale

        if self.indata.exponential_transform:
            nexpfull = self.indata.exponential_transform_scale * tf.math.log(nexpfull)
            if compute_normfull:
                normfull = self.indata.exponential_transform_scale * tf.math.log(
                    normfull
                )

        return nexpfull, normfull, beta

    def _compute_yields(self, inclusive=True, profile_grad=True):
        nexpfullcentral, normfullcentral, beta = self._compute_yields_with_beta(
            profile_grad=profile_grad, compute_normfull=not inclusive
        )
        if inclusive:
            return nexpfullcentral
        else:
            return normfullcentral

    @tf.function
    def expected_with_variance(
        self, fun, compute_cov=False, compute_global_impacts=False
    ):
        if self.profile:
            return self._expvar_profiled(fun, compute_cov, compute_global_impacts)
        else:
            return self._expvar(fun, compute_cov, compute_global_impacts)

    @tf.function
    def expected_variations(self, fun, correlations=False):
        return self._expvariations(fun, correlations=correlations)

    def expected_events(
        self,
        inclusive=True,
        compute_variance=True,
        compute_cov=False,
        compute_global_impacts=False,
        compute_variations=False,
        correlated_variations=False,
        profile_grad=True,
        compute_chi2=False,
    ):

        def fun():
            return self._compute_yields(inclusive=inclusive, profile_grad=profile_grad)

        if compute_variations and (
            compute_variance or compute_cov or compute_global_impacts
        ):
            raise NotImplementedError()

        aux = [None] * 4
        if compute_cov or compute_variance or compute_global_impacts:
            # exp, var = self.expected_with_variance(fun, cov, profile=profile)
            exp, expvar, expcov, exp_impacts, exp_impacts_grouped = (
                self.expected_with_variance(
                    fun,
                    compute_cov=compute_cov,
                    compute_global_impacts=compute_global_impacts,
                )
            )
            aux = [expvar, expcov, exp_impacts, exp_impacts_grouped]
        elif compute_variations:
            exp = self.expected_variations(fun, correlations=correlated_variations)
        else:
            exp = tf.function(fun)()

        if compute_chi2:

            def fun_residual():
                return fun() - self.nobs

            if self.profile:
                res, resvar, rescov, _1, _2 = self._expvar_profiled(
                    fun_residual, compute_cov=True
                )
            else:
                res, resvar, rescov, _1, _2 = self._expvar(
                    fun_residual, compute_cov=True
                )

            chi2val = self.chi2(res, rescov).numpy()
            ndf = tf.size(exp).numpy() - self.normalize

            aux.append(self.chi2(res, rescov).numpy())  # chi2val
            aux.append(tf.size(exp).numpy() - self.normalize)  # ndf
        else:
            aux.append(None)
            aux.append(None)

        return exp, aux

    def expected_events_projection(
        self,
        channel,
        axes,
        inclusive=True,
        compute_variance=True,
        compute_cov=False,
        compute_global_impacts=False,
        compute_variations=False,
        correlated_variations=False,
        profile_grad=True,
        compute_chi2=False,
    ):

        def fun():
            return self._compute_yields(inclusive=inclusive, profile_grad=profile_grad)

        info = self.indata.channel_info[channel]
        start = info["start"]
        stop = info["stop"]

        channel_axes = info["axes"]

        exp_axes = channel_axes.copy()
        hist_axes = [axis for axis in channel_axes if axis.name in axes]

        if len(hist_axes) != len(axes):
            raise ValueError(
                f"Hist axes {[h.name for h in hist_axes]} != {axes} not found"
            )

        extra_axes = []
        if not inclusive:
            exp_axes.append(self.indata.axis_procs)
            hist_axes.append(self.indata.axis_procs)
            extra_axes.append(self.indata.axis_procs)

        exp_shape = tuple([len(a) for a in exp_axes])

        channel_axes_names = [axis.name for axis in channel_axes]
        extra_axes_names = [axis.name for axis in extra_axes]

        axis_idxs = [channel_axes_names.index(axis) for axis in axes]

        proj_idxs = [i for i in range(len(channel_axes)) if i not in axis_idxs]

        post_proj_axes_names = [
            axis for axis in channel_axes_names if axis in axes
        ] + extra_axes_names

        transpose_idxs = [post_proj_axes_names.index(axis) for axis in axes] + [
            post_proj_axes_names.index(axis) for axis in extra_axes_names
        ]

        def make_projection_fun(fun_flat):
            def proj_fun():
                exp = fun_flat()[start:stop]
                exp = tf.reshape(exp, exp_shape)
                if self.indata.exponential_transform:
                    exp = self.indata.exponential_transform_scale * tf.math.log(exp)
                exp = tf.reduce_sum(exp, axis=proj_idxs)
                exp = tf.transpose(exp, perm=transpose_idxs)

                return exp

            return proj_fun

        projection_fun = make_projection_fun(fun)

        if compute_variations and (
            compute_variance or compute_cov or compute_global_impacts
        ):
            raise NotImplementedError()

        aux = [None] * 4
        if compute_variance or compute_cov or compute_global_impacts:
            exp, expvar, expcov, exp_impacts, exp_impacts_grouped = (
                self.expected_with_variance(
                    projection_fun,
                    compute_cov=compute_cov,
                    compute_global_impacts=compute_global_impacts,
                )
            )
            aux = [expvar, expcov, exp_impacts, exp_impacts_grouped]
        elif compute_variations:
            exp = self.expected_variations(
                projection_fun, correlations=correlated_variations
            )
        else:
            exp = tf.function(projection_fun)()

        if compute_chi2:

            def fun_residual():
                return fun() - self.nobs

            projection_fun_residual = make_projection_fun(fun_residual)

            if self.profile:
                res, resvar, rescov, _1, _2 = self._expvar_profiled(
                    projection_fun_residual, compute_cov=True
                )
            else:
                res, resvar, rescov, _1, _2 = self._expvar(
                    projection_fun_residual, compute_cov=True
                )

            aux.append(self.chi2(res, rescov).numpy())  # chi2val
            aux.append(tf.size(exp).numpy() - self.normalize)  # ndf
        else:
            aux.append(None)
            aux.append(None)

        return exp, aux

    def _compute_derivatives_x(self):
        with tf.GradientTape() as t2:
            t2.watch([self.theta0, self.nobs, self.beta0])
            with tf.GradientTape() as t1:
                t1.watch([self.theta0, self.nobs, self.beta0])
                val = self._compute_loss()
            grad = t1.gradient(val, self.x, unconnected_gradients="zero")
        pd2ldxdtheta0, pd2ldxdnobs, pd2ldxdbeta0 = t2.jacobian(
            grad, [self.theta0, self.nobs, self.beta0], unconnected_gradients="zero"
        )

        dxdtheta0 = -self.cov @ pd2ldxdtheta0
        dxdnobs = -self.cov @ pd2ldxdnobs
        dxdbeta0 = -self.cov @ pd2ldxdbeta0

        return dxdtheta0, dxdnobs, dxdbeta0

    @tf.function
    def expected_yield(self):
        return self._compute_yields(inclusive=True)

    @tf.function
    def chi2(self, res, rescov):
        return self._chi2(res, rescov)

    @tf.function
    def saturated_nll(self):
        nobs = self.nobs

        nobsnull = tf.equal(nobs, tf.zeros_like(nobs))

        # saturated model
        nobssafe = tf.where(nobsnull, tf.ones_like(nobs), nobs)
        lognobs = tf.math.log(nobssafe)

        lsaturated = tf.reduce_sum(-nobs * lognobs + nobs, axis=-1)

        ndof = (
            tf.size(nobs) - self.npoi - self.indata.nsystnoconstraint - self.normalize
        )

        return lsaturated, ndof

    @tf.function
    def full_nll(self):
        l, lfull = self._compute_nll()
        return lfull

    @tf.function
    def reduced_nll(self):
        l, lfull = self._compute_nll()
        return l

    def _compute_nll(self, profile_grad=True):
        theta = self.x[self.npoi :]

        nexpfullcentral, _, beta = self._compute_yields_with_beta(
            profile_grad=profile_grad, compute_normfull=False
        )

        nexp = nexpfullcentral

        if self.chisqFit:
            residual = tf.reshape(self.nobs - nexp, [-1, 1])  # chi2 residual
            # Solve the system without inverting
            ln = lnfull = 0.5 * tf.reduce_sum(
                tf.matmul(
                    residual, tf.matmul(self.data_cov_inv, residual), transpose_a=True
                )
            )
        else:
            nobsnull = tf.equal(self.nobs, tf.zeros_like(self.nobs))

            nexpsafe = tf.where(nobsnull, tf.ones_like(self.nobs), nexp)
            lognexp = tf.math.log(nexpsafe)

            nexpnomsafe = tf.where(nobsnull, tf.ones_like(self.nobs), self.nexpnom)
            lognexpnom = tf.math.log(nexpnomsafe)

            # final likelihood computation

            # poisson term
            lnfull = tf.reduce_sum(-self.nobs * lognexp + nexp, axis=-1)

            # poisson term with offset to improve numerical precision
            ln = tf.reduce_sum(
                -self.nobs * (lognexp - lognexpnom) + nexp - self.nexpnom, axis=-1
            )

        # constraints
        lc = tf.reduce_sum(
            self.indata.constraintweights * 0.5 * tf.square(theta - self.theta0)
        )

        l = ln + lc
        lfull = lnfull + lc

        if self.binByBinStat:
            lbetavfull = (
                -self.indata.kstat * tf.math.log(beta / self.beta0)
                + self.indata.kstat * beta / self.beta0
            )

            lbetav = lbetavfull - self.indata.kstat
            lbeta = tf.reduce_sum(lbetav)

            l = l + lbeta
            lfull = lfull + lbeta

        return l, lfull

    def _compute_loss(self, profile_grad=True):
        l, lfull = self._compute_nll(profile_grad=profile_grad)
        return l

    @tf.function
    def loss_val(self):
        val = self._compute_loss()
        return val

    @tf.function
    def loss_val_grad(self):
        with tf.GradientTape() as t:
            val = self._compute_loss()
        grad = t.gradient(val, self.x)

        return val, grad

    @tf.function
    def loss_val_grad_hessp(self, p):
        with tf.autodiff.ForwardAccumulator(self.x, p) as acc:
            with tf.GradientTape() as grad_tape:
                val = self._compute_loss()
            grad = grad_tape.gradient(val, self.x)
        hessp = acc.jvp(grad)

        return val, grad, hessp

    @tf.function
    def loss_val_grad_hess(self, profile_grad=True):
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                val = self._compute_loss(profile_grad=profile_grad)
            grad = t1.gradient(val, self.x)
        hess = t2.jacobian(grad, self.x)

        return val, grad, hess

    def minimize(self):

        def scipy_loss(xval):
            self.x.assign(xval)
            val, grad = self.loss_val_grad()
            # print("scipy_loss", val)
            return val.numpy(), grad.numpy()

        def scipy_hessp(xval, pval):
            self.x.assign(xval)
            p = tf.convert_to_tensor(pval)
            val, grad, hessp = self.loss_val_grad_hessp(p)
            # print("scipy_hessp", val)
            return hessp.numpy()

        xval = self.x.numpy()

        res = scipy.optimize.minimize(
            scipy_loss, xval, method="trust-krylov", jac=True, hessp=scipy_hessp
        )

        xval = res["x"]

        self.x.assign(xval)

        print(res)

        return res

    def nll_scan(self, param, scan_range, scan_points, use_prefit=False):
        # make a likelihood scan for a single parameter
        # assuming the likelihood is minimized

        idx = np.where(self.parms.astype(str) == param)[0][0]

        # store current state of x temporarily
        xval = tf.identity(self.x)

        param_offsets = np.linspace(0, scan_range, scan_points // 2 + 1)
        if not use_prefit:
            param_offsets *= self.cov[idx, idx].numpy() ** 0.5

        nscans = 2 * len(param_offsets) - 1
        nlls = np.full(nscans, np.nan)
        scan_vals = np.zeros(nscans)

        # save delta nll w.r.t. global minimum
        reduced_nll = self.reduced_nll().numpy()
        # set central point
        nlls[nscans // 2] = 0
        scan_vals[nscans // 2] = xval[idx].numpy()
        # scan positive side and negative side independently to profit from previous step
        for sign in [-1, 1]:
            param_scan_values = xval[idx].numpy() + sign * param_offsets
            for i, ixval in enumerate(param_scan_values):
                if i == 0:
                    continue

                self.x.assign(tf.tensor_scatter_nd_update(self.x, [[idx]], [ixval]))

                def scipy_loss(xval):
                    self.x.assign(xval)
                    val, grad = self.loss_val_grad()
                    grad = grad.numpy()
                    grad[idx] = 0  # Zero out gradient for the frozen parameter
                    return val.numpy(), grad

                def scipy_hessp(xval, pval):
                    self.x.assign(xval)
                    pval[idx] = (
                        0  # Ensure the perturbation does not affect frozen parameter
                    )
                    p = tf.convert_to_tensor(pval)
                    val, grad, hessp = self.loss_val_grad_hessp(p)
                    hessp = hessp.numpy()
                    # TODO: worth testing modifying the loss/grad/hess functions to imply 1
                    # for the corresponding hessian element instead of 0,
                    # since this might allow the minimizer to converge more efficiently
                    hessp[idx] = (
                        0  # Zero out Hessian-vector product at the frozen index
                    )
                    return hessp

                res = scipy.optimize.minimize(
                    scipy_loss,
                    self.x,
                    method="trust-krylov",
                    jac=True,
                    hessp=scipy_hessp,
                )
                if res["success"]:
                    nlls[nscans // 2 + sign * i] = (
                        self.reduced_nll().numpy() - reduced_nll
                    )
                    scan_vals[nscans // 2 + sign * i] = ixval

            # reset x to original state
            self.x.assign(xval)

        return scan_vals, nlls

    def nll_scan2D(self, param_tuple, scan_range, scan_points, use_prefit=False):

        idx0 = np.where(self.parms.astype(str) == param_tuple[0])[0][0]
        idx1 = np.where(self.parms.astype(str) == param_tuple[1])[0][0]

        xval = tf.identity(self.x)

        dsigs = np.linspace(-scan_range, scan_range, scan_points)
        if not use_prefit:
            x_scans = xval[idx0] + dsigs * self.cov[idx0, idx0] ** 0.5
            y_scans = xval[idx1] + dsigs * self.cov[idx1, idx1] ** 0.5
        else:
            x_scans = dsigs
            y_scans = dsigs

        best_fit = (scan_points + 1) // 2 - 1
        nlls = np.full((len(x_scans), len(y_scans)), np.nan)
        nlls[best_fit, best_fit] = self.full_nll().numpy()
        # scan in a spiral around the best fit point
        dcol = -1
        drow = 0
        i = 0
        j = 0
        r = 1
        while r - 1 < best_fit:
            if i == r and drow == 1:
                drow = 0
                dcol = 1
            if j == r and dcol == 1:
                dcol = 0
                drow = -1
            elif i == -r and drow == -1:
                dcol = -1
                drow = 0
            elif j == -r and dcol == -1:
                drow = 1
                dcol = 0

            i += drow
            j += dcol

            if i == -r and j == -r:
                r += 1

            ix = best_fit - i
            iy = best_fit + j

            # print(f"i={i}, j={j}, r={r} drow={drow}, dcol={dcol} | ix={ix}, iy={iy}")

            self.x.assign(
                tf.tensor_scatter_nd_update(
                    self.x, [[idx0], [idx1]], [x_scans[ix], y_scans[iy]]
                )
            )

            def scipy_loss(xval):
                self.x.assign(xval)
                val, grad = self.loss_val_grad()
                grad = grad.numpy()
                grad[idx0] = 0
                grad[idx1] = 0
                return val.numpy(), grad

            def scipy_hessp(xval, pval):
                self.x.assign(xval)
                pval[idx0] = 0
                pval[idx1] = 0
                p = tf.convert_to_tensor(pval)
                val, grad, hessp = self.loss_val_grad_hessp(p)
                hessp = hessp.numpy()
                hessp[idx0] = 0
                hessp[idx1] = 0

                if np.allclose(hessp, 0, atol=1e-8):
                    return np.zeros_like(hessp)

                return hessp

            res = scipy.optimize.minimize(
                scipy_loss,
                self.x,
                method="trust-krylov",
                jac=True,
                hessp=scipy_hessp,
            )

            if res["success"]:
                nlls[ix, iy] = self.full_nll().numpy()

        self.x.assign(xval)
        return x_scans, y_scans, nlls

    def contour_scan(self, param, nll_min, cl=1):

        def scipy_grad(xval):
            self.x.assign(xval)
            val, grad = self.loss_val_grad()
            return grad.numpy()

        # def scipy_hessp(xval, pval):
        #     self.x.assign(xval)
        #     p = tf.convert_to_tensor(pval)
        #     val, grad, hessp = self.loss_val_grad_hessp(p)
        #     # print("scipy_hessp", val)
        #     return hessp.numpy()

        def scipy_loss(xval):
            self.x.assign(xval)
            val = self.loss_val()
            return val.numpy() - nll_min - 0.5 * cl**2

        nlc = scipy.optimize.NonlinearConstraint(
            fun=scipy_loss,
            lb=0,
            ub=0,
            jac=scipy_grad,
            hess=scipy.optimize.SR1(),  # TODO: use exact hessian or hessian vector product
        )

        # initial guess from covariance
        idx = np.where(self.parms.astype(str) == param)[0][0]
        xval = tf.identity(self.x)

        xup = xval[idx] + self.cov[idx, idx] ** 0.5
        xdn = xval[idx] - self.cov[idx, idx] ** 0.5

        xval_init = xval.numpy()

        intervals = np.full((2, len(self.parms)), np.nan)
        for i, sign in enumerate([-1.0, 1.0]):
            if sign == 1.0:
                xval_init[idx] = xdn
            else:
                xval_init[idx] = xup

            # Objective function and its derivatives
            def objective(params):
                return sign * params[idx]

            def objective_jac(params):
                jac = np.zeros_like(params)
                jac[idx] = sign
                return jac

            def objective_hessp(params, v):
                return np.zeros_like(v)

            res = scipy.optimize.minimize(
                objective,
                xval_init,
                method="trust-constr",
                jac=objective_jac,
                hessp=objective_hessp,
                constraints=[nlc],
                options={
                    "maxiter": 5000,
                    "xtol": 1e-10,
                    "gtol": 1e-10,
                    # "verbose": 3
                },
            )

            if res["success"]:
                intervals[i] = res["x"] - xval.numpy()

            self.x.assign(xval)

        return intervals

    def contour_scan2D(self, param_tuple, nll_min, cl=1, n_points=16):
        # Not yet working
        def scipy_loss(xval):
            self.x.assign(xval)
            val, grad = self.loss_val_grad()
            return val.numpy()

        def scipy_grad(xval):
            self.x.assign(xval)
            val, grad = self.loss_val_grad()
            return grad.numpy()

        xval = tf.identity(self.x)

        # Constraint function and its derivatives
        delta_nll = 0.5 * cl**2

        def constraint(params):
            return scipy_loss(params) - nll_min - delta_nll

        nlc = scipy.optimize.NonlinearConstraint(
            fun=constraint,
            lb=-np.inf,
            ub=0,
            jac=scipy_grad,
            hess=scipy.optimize.SR1(),
        )

        # initial guess from covariance
        xval_init = xval.numpy()
        idx0 = np.where(self.parms.astype(str) == param_tuple[0])[0][0]
        idx1 = np.where(self.parms.astype(str) == param_tuple[1])[0][0]

        intervals = np.full((2, n_points), np.nan)
        for i, t in enumerate(np.linspace(0, 2 * np.pi, n_points, endpoint=False)):
            print(f"Now at {i} with angle={t}")

            # Objective function and its derivatives
            def objective(params):
                # coordinate center (best fit)
                x = params[idx0] - xval[idx0]
                y = params[idx1] - xval[idx1]
                return -(x**2 + y**2)

            def objective_jac(params):
                x = params[idx0] - xval[idx0]
                y = params[idx1] - xval[idx1]
                jac = np.zeros_like(params)
                jac[idx0] = -2 * x
                jac[idx1] = -2 * y
                return jac

            def objective_hessp(params, v):
                hessp = np.zeros_like(v)
                hessp[idx0] = -2 * v[idx0]
                hessp[idx1] = -2 * v[idx1]
                return hessp

            def constraint_angle(params):
                # coordinate center (best fit)
                x = params[idx0] - xval[idx0]
                y = params[idx1] - xval[idx1]
                return x * np.sin(t) - y * np.cos(t)

            def constraint_angle_jac(params):
                jac = np.zeros_like(params)
                jac[idx0] = np.sin(t)
                jac[idx1] = -np.cos(t)
                return jac

            # constraint on angle
            tc = scipy.optimize.NonlinearConstraint(
                fun=constraint_angle,
                lb=0,
                ub=0,
                jac=constraint_angle_jac,
                hess=scipy.optimize.SR1(),
            )

            res = scipy.optimize.minimize(
                objective,
                xval_init,
                method="trust-constr",
                jac=objective_jac,
                hessp=objective_hessp,
                constraints=[nlc, tc],
                options={
                    "maxiter": 10000,
                    "xtol": 1e-14,
                    "gtol": 1e-14,
                    # "verbose": 3
                },
            )

            print(res)

            if res["success"]:
                intervals[0, i] = res["x"][idx0]
                intervals[1, i] = res["x"][idx1]

            self.x.assign(xval)

        return intervals
