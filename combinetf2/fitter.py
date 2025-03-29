import numpy as np
import scipy
import tensorflow as tf
import tensorflow_probability as tfp
from wums import logging

from combinetf2.tfhelpers import is_diag, simple_sparse_slice0end

logger = logging.child_logger(__name__)


class FitterCallback:
    def __init__(self, xv):
        self.iiter = 0
        self.xval = xv

    def __call__(self, intermediate_result):
        logger.debug(f"Iteration {self.iiter}: loss value {intermediate_result.fun}")
        self.xval = intermediate_result.x
        self.iiter += 1


class Fitter:
    def __init__(self, indata, options):
        self.indata = indata
        self.binByBinStat = options.binByBinStat
        self.binByBinStatType = options.binByBinStatType
        self.systgroupsfull = self.indata.systgroups.tolist()
        self.systgroupsfull.append("stat")
        if self.binByBinStat:
            self.systgroupsfull.append("binByBinStat")

        if options.externalCovariance and not options.chisqFit:
            raise Exception(
                'option "--externalCovariance" only works with "--chisqFit"'
            )
        if (
            options.externalCovariance
            and options.binByBinStat
            and options.binByBinStatType != "normal"
        ):
            raise Exception(
                'option "--binByBinStat" only for options "--externalCovariance" with "--binByBinStatType normal"'
            )

        if self.binByBinStatType not in ["gamma", "normal"]:
            raise RuntimeError(
                f"Invalid binByBinStatType {self.indata.binByBinStatType}, valid choices are 'gamma' or 'normal'"
            )

        if self.indata.systematic_type not in ["log_normal", "normal"]:
            raise RuntimeError(
                f"Invalid systematic_type {self.indata.systematic_type}, valid choices are 'log_normal' or 'normal'"
            )

        self.chisqFit = options.chisqFit
        self.externalCovariance = options.externalCovariance
        self.prefitUnconstrainedNuisanceUncertainty = (
            options.prefitUnconstrainedNuisanceUncertainty
        )

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
        self.data_cov_inv = None

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

        # constraint minima for nuisance parameters
        self.theta0 = tf.Variable(
            tf.zeros([self.indata.nsyst], dtype=self.indata.dtype),
            trainable=False,
            name="theta0",
        )

        # FIXME for now this is needed even if binByBinStat is off because of how it is used in the global impacts
        #  and uncertainty band computations (gradient is allowed to be zero or None and then propagated or skipped only later)

        # global observables for mc stat uncertainty
        self.beta0 = tf.Variable(self._default_beta0(), trainable=False, name="beta0")

        # nuisance parameters for mc stat uncertainty
        self.beta = tf.Variable(self.beta0, trainable=False, name="beta")

        # cache the constraint variance since it's used in several places
        # this is treated as a constant
        if self.binByBinStatType == "gamma":
            self.varbeta = tf.stop_gradient(tf.math.reciprocal(self.indata.kstat))
        elif self.binByBinStatType == "normal":
            n0 = self.expected_events_nominal()
            self.varbeta = tf.stop_gradient(n0**2 / self.indata.kstat)

            if self.binByBinStat and self.externalCovariance:
                # precompute decomposition of composite matrix to speed up
                # calculation of profiled beta values
                self.betaauxlu = tf.linalg.lu(
                    self.data_cov_inv + tf.diag(tf.reciprocal(self.varbeta))
                )

        self.nexpnom = tf.Variable(
            self.expected_yield(), trainable=False, name="nexpnom"
        )

        # parameter covariance matrix
        self.cov = tf.Variable(
            self.prefit_covariance(
                unconstrained_err=self.prefitUnconstrainedNuisanceUncertainty
            ),
            trainable=False,
            name="cov",
        )

        # determine if problem is linear (ie likelihood is purely quadratic)
        self.is_linear = (
            self.chisqFit
            and self.indata.symmetric_tensor
            and self.indata.systematic_type == "normal"
            and self.npoi == 0
            and ((not self.binByBinStat) or self.binByBinStatType == "normal")
        )

    def _default_beta0(self):
        if self.binByBinStatType == "gamma":
            return tf.ones_like(self.indata.kstat)
        elif self.binByBinStatType == "normal":
            return tf.zeros_like(self.indata.kstat)

    def prefit_covariance(self, unconstrained_err=0.0):
        # free parameters are taken to have zero uncertainty for the purposes of prefit uncertainties
        var_poi = tf.zeros([self.npoi], dtype=self.indata.dtype)

        # nuisances have their uncertainty taken from the constraint term, but unconstrained nuisances
        # are set to a placeholder uncertainty (zero by default) for the purposes of prefit uncertainties
        var_theta = tf.where(
            self.indata.constraintweights == 0.0,
            unconstrained_err**2,
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
        self.theta0.assign(tf.zeros([self.indata.nsyst], dtype=self.theta0.dtype))

    def xdefaultassign(self):
        if self.npoi == 0:
            self.x.assign(self.theta0)
        else:
            self.x.assign(tf.concat([self.xpoidefault, self.theta0], axis=0))

    def beta0defaultassign(self):
        self.beta0.assign(self._default_beta0())

    def betadefaultassign(self):
        self.beta.assign(self.beta0)

    def defaultassign(self):
        self.cov.assign(
            self.prefit_covariance(
                unconstrained_err=self.prefitUnconstrainedNuisanceUncertainty
            )
        )
        self.theta0defaultassign()
        if self.binByBinStat:
            self.beta0defaultassign()
            self.betadefaultassign()
        self.xdefaultassign()

    def bayesassign(self):
        # FIXME use theta0 as the mean and constraintweight to scale the width
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

        if self.binByBinStat:
            if self.binByBinStatType == "gamma":
                self.beta.assign(
                    tf.random.gamma(
                        shape=[],
                        alpha=self.indata.kstat * self.beta0 + 1.0,
                        beta=tf.ones_like(self.indata.kstat),
                        dtype=self.beta.dtype,
                    )
                    / self.indata.kstat
                )
            elif self.binByBinStatType == "normal":
                self.beta.assign(
                    tf.random.normal(
                        shape=[],
                        mean=self.beta0,
                        sigma=tf.sqrt(self.varbeta),
                        dtype=self.beta.dtype,
                    )
                )

    def frequentistassign(self):
        # FIXME use theta as the mean and constraintweight to scale the width
        self.theta0.assign(
            tf.random.normal(shape=self.theta0.shape, dtype=self.theta0.dtype)
        )
        if self.binByBinStat:
            if self.binByBinStatType == "gamma":
                self.beta0.assign(
                    tf.random.poisson(
                        shape=[],
                        lam=self.indata.kstat * self.beta,
                        dtype=self.beta.dtype,
                    )
                    / self.indata.kstat
                )
            elif self.binByBinStatType == "normal":
                self.beta0.assign(
                    tf.random.normal(
                        shape=[],
                        mean=self.beta,
                        sigma=tf.sqrt(self.varbeta),
                        dtype=self.beta.dtype,
                    )
                )

    def toyassign(
        self,
        syst_randomize="frequentist",
        data_randomize="poisson",
        data_mode="expected",
        randomize_parameters=False,
    ):
        if syst_randomize == "bayesian":
            # randomize actual values
            self.bayesassign()
        elif syst_randomize == "frequentist":
            # randomize nuisance constraint minima
            self.frequentistassign()

        if data_mode == "expected":
            data_nom = self.expected_yield()
        elif data_mode == "observed":
            data_nom = self.indata.data_obs

        if data_randomize == "poisson":
            if self.externalCovariance:
                raise RuntimeError(
                    "Toys with external covariance only possible with data_randomize=normal"
                )
            else:
                self.nobs.assign(
                    tf.random.poisson(lam=data_nom, shape=[], dtype=self.nobs.dtype)
                )
        elif data_randomize == "normal":
            if self.externalCovariance:
                pdata = tfp.distributions.MultivariateNormalTriL(
                    loc=data_nom,
                    scale_tril=tf.linalg.cholesky(tf.linalg.inv(self.data_cov_inv)),
                )
                self.nobs.assign(pdata.sample())
            else:
                self.nobs.assign(
                    tf.random.normal(
                        mean=data_nom,
                        stddev=tf.sqrt(data_nom),
                        shape=[],
                        dtype=self.nobs.dtype,
                    )
                )
        elif data_randomize == "none":
            self.nobs.assign(data_nom)

        # assign start values for nuisance parameters to constraint minima
        self.xdefaultassign()
        if self.binByBinStat:
            self.betadefaultassign()
        # set likelihood offset
        self.nexpnom.assign(self.expected_yield())

        if randomize_parameters:
            # the special handling of the diagonal case here speeds things up, but is also required
            # in case the prefit covariance has zero for some uncertainties (which is the default
            # for unconstrained nuisances for example) since the multivariate normal distribution
            # requires a positive-definite covariance matrix
            if is_diag(self.cov):
                self.x.assign(
                    tf.random.normal(
                        shape=[],
                        mean=self.x,
                        stddev=tf.sqrt(tf.linalg.diag_part(self.cov)),
                        dtype=self.x.dtype,
                    )
                )
            else:
                pparms = tfp.distributions.MultivariateNormalTriL(
                    loc=self.x, scale_tril=tf.linalg.cholesky(self.cov)
                )
                self.x.assign(pparms.sample())
            if self.binByBinStat:
                self.beta.assign(
                    tf.random.normal(
                        shape=[],
                        mean=self.beta0,
                        stddev=tf.sqrt(self.varbeta),
                        dtype=self.beta.dtype,
                    )
                )

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
                profile=False
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

            # FIXME consider implications of using kstat*beta as the variance
            impacts_bbb = tf.sqrt(
                tf.reduce_sum(tf.square(dxdbeta0) * self.varbeta, axis=-1)
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
        # FIXME consider implications of using kstat*beta as the variance
        dbeta0 = tf.math.sqrt(self.varbeta)

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

        expvar = tf.reshape(expvar, tf.shape(expected))

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
        sRJ2 = tf.reshape(sRJ2, tf.shape(expected))
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
            t.watch([self.theta0, self.nobs, self.beta])
            expected = fun_exp()
            expected_flat = tf.reshape(expected, (-1,))
        pdexpdx, pdexpdnobs, pdexpdbeta = t.jacobian(
            expected_flat,
            [self.x, self.nobs, self.beta],
        )

        expcov = pdexpdx @ tf.matmul(self.cov, pdexpdx, transpose_b=True)

        if pdexpdnobs is not None:
            varnobs = self.nobs
            exp_cov_stat = pdexpdnobs @ (varnobs[:, None] * tf.transpose(pdexpdnobs))
            expcov += exp_cov_stat

        expcov_noBBB = expcov
        if self.binByBinStat:
            varbeta = self.varbeta
            exp_cov_BBB = pdexpdbeta @ (varbeta[:, None] * tf.transpose(pdexpdbeta))
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
        expvar = tf.reshape(expvar, tf.shape(expected))

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

        new_shape = tf.concat([tf.shape(expected), [-1]], axis=0)
        dexp = tf.reshape(dexp, new_shape)

        down = expected[..., None] - dexp
        up = expected[..., None] + dexp

        expvars = tf.stack([down, up], axis=-1)

        return expvars

    def _compute_yields_noBBB(self, compute_norm=False, full=True):
        # compute_norm: compute yields for each process, otherwise inclusive
        # full: compute yields inclduing masked channels
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
            logsnorm = tf.sparse.sparse_dense_matmul(self.indata.logk, mthetaalpha)
            logsnorm = tf.squeeze(logsnorm, -1)

            if self.indata.systematic_type == "log_normal":
                snorm = tf.exp(logsnorm)
                snormnorm_sparse = self.indata.norm.with_values(
                    snorm * self.indata.norm.values
                )
            elif self.indata.systematic_type == "normal":
                snormnorm_sparse = self.indata.norm.with_values(
                    self.indata.norm.values + logsnorm
                )

            nexpfullcentral = tf.sparse.sparse_dense_matmul(snormnorm_sparse, mrnorm)
            nexpfullcentral = tf.squeeze(nexpfullcentral, -1)

            if not full and self.indata.nbinsmasked:
                snormnorm_sparse = simple_sparse_slice0end(
                    snormnorm_sparse, self.indata.nbins
                )

            nexpcentral = tf.sparse.sparse_dense_matmul(snormnorm_sparse, mrnorm)
            nexpcentral = tf.squeeze(nexpcentral, -1)

            if compute_norm:
                snormnorm = tf.sparse.to_dense(snormnorm_sparse)
        else:
            if full or self.indata.nbinsmasked == 0:
                nbins = self.indata.nbinsfull
                logk = self.indata.logk
                norm = self.indata.norm
            else:
                nbins = self.indata.nbins
                logk = self.indata.logk[:nbins]
                norm = self.indata.norm[:nbins]

            if self.indata.symmetric_tensor:
                mlogk = tf.reshape(
                    logk,
                    [nbins * self.indata.nproc, self.indata.nsyst],
                )
            else:
                mlogk = tf.reshape(
                    logk,
                    [nbins * self.indata.nproc, 2 * self.indata.nsyst],
                )

            logsnorm = tf.matmul(mlogk, mthetaalpha)
            logsnorm = tf.reshape(logsnorm, [nbins, self.indata.nproc])

            if self.indata.systematic_type == "log_normal":
                snorm = tf.exp(logsnorm)
                snormnorm = snorm * norm
            elif self.indata.systematic_type == "normal":
                snormnorm = norm + logsnorm

            nexpcentral = tf.matmul(snormnorm, mrnorm)
            nexpcentral = tf.squeeze(nexpcentral, -1)

        if compute_norm:
            normcentral = ernorm * snormnorm
        else:
            normcentral = None

        return nexpcentral, normcentral

    def _compute_yields_with_beta(self, profile=True, compute_norm=False, full=True):
        nexp, norm = self._compute_yields_noBBB(compute_norm, full=full)

        if self.binByBinStat:
            if profile:
                # analytic solution for profiled barlow-beeston lite parameters for each combination
                # of likelihood and uncertainty form

                nexp_profile = nexp[: self.indata.nbins]
                kstat = self.indata.kstat[: self.indata.nbins]
                beta0 = self.beta0[: self.indata.nbins]
                # denominator in Gaussian likelihood is treated as a constant when computing
                # global impacts for example
                nobs0 = tf.stop_gradient(self.nobs)

                if self.chisqFit:
                    if self.binByBinStatType == "gamma":
                        abeta = nexp_profile**2
                        bbeta = kstat * nobs0 - nexp_profile * self.nobs
                        cbeta = -kstat * nobs0 * self.beta0
                        beta = (
                            0.5
                            * (-bbeta + tf.sqrt(bbeta**2 - 4.0 * abeta * cbeta))
                            / abeta
                        )
                    elif self.binByBinStatType == "normal":
                        if self.externalCovariance:
                            beta = tf.linalg.lu_solve(
                                self.betaauxlu,
                                self.data_cov_inv
                                @ ((self.nobs - nexp_profile)[:, None])
                                + (beta0 / self.varbeta)[:, None],
                            )
                            beta = tf.squeeze(beta, axis=-1)
                        else:
                            beta = (
                                self.varbeta * (self.nobs - nexp_profile)
                                + nobs0 * beta0
                            ) / (nobs0 + self.varbeta)
                else:
                    if self.binByBinStatType == "gamma":
                        beta = (self.nobs + kstat * beta0) / (nexp_profile + kstat)
                    elif self.binByBinStatType == "normal":
                        bbeta = self.varbeta + nexp_profile - self.beta0
                        cbeta = (
                            self.varbeta * (nexp_profile - self.nobs)
                            - nexp_profile * self.beta0
                        )
                        beta = 0.5 * (-bbeta + tf.sqrt(bbeta**2 - 4.0 * cbeta))

                if full and self.indata.nbinsmasked:
                    beta = tf.concat([beta, self.beta[self.indata.nbins :]], axis=0)
            else:
                beta = self.beta
                if (not full) and self.indata.nbinsmasked:
                    beta = beta[: self.indata.nbins]

            if self.binByBinStatType == "gamma":
                nexp = nexp * beta
            elif self.binByBinStatType == "normal":
                nexp = nexp + beta

            if compute_norm:
                norm = beta[..., None] * norm
        else:
            beta = None

        return nexp, norm, beta

    @tf.function
    def _profile_beta(self):
        nexp, norm, beta = self._compute_yields_with_beta()
        self.beta.assign(beta)

    @tf.function
    def expected_events_nominal(self):
        rnorm = tf.ones(self.indata.nproc, dtype=self.indata.dtype)
        mrnorm = tf.expand_dims(rnorm, -1)

        if self.indata.sparse:
            nexpfullcentral = tf.sparse.sparse_dense_matmul(self.indata.norm, mrnorm)
            nexpfullcentral = tf.squeeze(nexpfullcentral, -1)
        else:
            nexpfullcentral = tf.matmul(self.indata.norm, mrnorm)
            nexpfullcentral = tf.squeeze(nexpfullcentral, -1)

        return nexpfullcentral

    def _compute_yields(self, inclusive=True, profile=True, full=True):
        nexpcentral, normcentral, beta = self._compute_yields_with_beta(
            profile=profile,
            compute_norm=not inclusive,
            full=full,
        )
        if inclusive:
            return nexpcentral
        else:
            return normcentral

    @tf.function
    def expected_with_variance(
        self, fun, profile=False, compute_cov=False, compute_global_impacts=False
    ):
        if profile:
            return self._expvar_profiled(fun, compute_cov, compute_global_impacts)
        else:
            return self._expvar(fun, compute_cov, compute_global_impacts)

    @tf.function
    def expected_variations(self, fun, correlations=False):
        return self._expvariations(fun, correlations=correlations)

    def expected_events(
        self,
        model,
        inclusive=True,
        compute_variance=True,
        compute_cov=False,
        compute_global_impacts=False,
        compute_variations=False,
        correlated_variations=False,
        profile=True,
        compute_chi2=False,
    ):

        def flat_fun():
            return self._compute_yields(
                inclusive=inclusive,
                profile=profile,
            )

        if compute_variations and (
            compute_variance or compute_cov or compute_global_impacts
        ):
            raise NotImplementedError()

        fun = model.make_fun(flat_fun, self.x, inclusive)

        aux = [None] * 4
        if compute_cov or compute_variance or compute_global_impacts:
            exp, exp_var, exp_cov, exp_impacts, exp_impacts_grouped = (
                self.expected_with_variance(
                    fun,
                    profile=profile,
                    compute_cov=compute_cov,
                    compute_global_impacts=compute_global_impacts,
                )
            )
            aux = [exp_var, exp_cov, exp_impacts, exp_impacts_grouped]
        elif compute_variations:
            exp = self.expected_variations(fun, correlations=correlated_variations)
        else:
            exp = tf.function(fun)()

        if compute_chi2:
            data, data_var, data_cov = model.get_data(self.nobs, self.data_cov_inv)

            # need to calculate prediction excluding masked channels
            def flat_fun():
                return self._compute_yields(
                    inclusive=inclusive, profile=profile, full=False
                )

            pred, pred_var, pred_cov, _1, _2 = self.expected_with_variance(
                model.make_fun(flat_fun, self.x, inclusive),
                profile=profile,
                compute_cov=True,
            )

            chi2val = self.chi2(pred - data, pred_cov + data_cov).numpy()
            ndf = tf.size(pred).numpy() - getattr(model, "normalize", False)

            aux.append(chi2val)
            aux.append(ndf)
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
    def expected_yield(self, profile=False, full=False):
        return self._compute_yields(inclusive=True, profile=profile, full=full)

    @tf.function
    def _expected_yield_noBBB(self, full=False):
        res, _ = self._compute_yields_noBBB(full=full)
        return res

    @tf.function
    def chi2(self, res, rescov):
        return self._chi2(res, rescov)

    @tf.function
    def saturated_nll(self):

        nobs = self.nobs

        if self.chisqFit:
            lsaturated = tf.zeros(shape=(), dtype=self.nobs.dtype)
        else:
            nobsnull = tf.equal(nobs, tf.zeros_like(nobs))

            # saturated model
            nobssafe = tf.where(nobsnull, tf.ones_like(nobs), nobs)
            lognobs = tf.math.log(nobssafe)

            lsaturated = tf.reduce_sum(-nobs * lognobs + nobs, axis=-1)

        if self.binByBinStat:
            if self.binByBinStatType == "gamma":
                kstat = self.indata.kstat[: self.indata.nbins]
                beta0 = self.beta0[: self.indata.nbins]
                lsaturated += tf.reduce_sum(
                    -kstat * beta0 * tf.math.log(beta0) + kstat * beta0
                )
            elif self.binByBinStatType == "normal":
                # mc stat contribution to the saturated likelihood is zero in this case
                pass

        ndof = tf.size(nobs) - self.npoi - self.indata.nsystnoconstraint

        return lsaturated, ndof

    @tf.function
    def full_nll(self):
        l, lfull = self._compute_nll()
        return lfull

    @tf.function
    def reduced_nll(self):
        l, lfull = self._compute_nll()
        return l

    def _compute_nll(self, profile=True):
        theta = self.x[self.npoi :]

        nexpfullcentral, _, beta = self._compute_yields_with_beta(
            profile=profile,
            compute_norm=False,
            full=False,
        )

        nexp = nexpfullcentral

        if self.chisqFit:
            if self.externalCovariance:
                # Solve the system without inverting
                residual = tf.reshape(self.nobs - nexp, [-1, 1])  # chi2 residual
                ln = lnfull = 0.5 * tf.reduce_sum(
                    tf.matmul(
                        residual,
                        tf.matmul(self.data_cov_inv, residual),
                        transpose_a=True,
                    )
                )
            else:
                # stop_gradient needed in denominator here because it should be considered
                # constant when evaluating global impacts from observed data
                ln = lnfull = 0.5 * tf.math.reduce_sum(
                    (nexp - self.nobs) ** 2 / tf.stop_gradient(self.nobs), axis=-1
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
            kstat = self.indata.kstat[: self.indata.nbins]
            beta0 = self.beta0[: self.indata.nbins]
            if self.binByBinStatType == "gamma":
                lbetavfull = -kstat * beta0 * tf.math.log(beta) + kstat * beta

                lbetav = -kstat * beta0 * tf.math.log(beta) + kstat * (beta - 1.0)

                lbetafull = tf.reduce_sum(lbetavfull)
                lbeta = tf.reduce_sum(lbetav)
            elif self.binByBinStatType == "normal":
                lbetavfull = 0.5 * (beta - beta0) ** 2 / self.varbeta

                lbetafull = tf.reduce_sum(lbetavfull)
                lbeta = lbetafull

            l = l + lbeta
            lfull = lfull + lbetafull

        return l, lfull

    def _compute_loss(self, profile=True):
        l, lfull = self._compute_nll(profile=profile)
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

    # FIXME in principle this version of the function is preferred
    # but seems to introduce some small numerical non-reproducibility
    @tf.function
    def loss_val_grad_hessp_fwdrev(self, p):
        p = tf.stop_gradient(p)
        with tf.autodiff.ForwardAccumulator(self.x, p) as acc:
            with tf.GradientTape() as grad_tape:
                val = self._compute_loss()
            grad = grad_tape.gradient(val, self.x)
        hessp = acc.jvp(grad)

        return val, grad, hessp

    @tf.function
    def loss_val_grad_hessp_revrev(self, p):
        p = tf.stop_gradient(p)
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                val = self._compute_loss()
            grad = t1.gradient(val, self.x)
        hessp = t2.gradient(grad, self.x, output_gradients=p)

        return val, grad, hessp

    loss_val_grad_hessp = loss_val_grad_hessp_revrev

    @tf.function
    def loss_val_grad_hess(self, profile=True):
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                val = self._compute_loss(profile=profile)
            grad = t1.gradient(val, self.x)
        hess = t2.jacobian(grad, self.x)

        return val, grad, hess

    @tf.function
    def loss_val_valfull_grad_hess(self, profile=True):
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                val, valfull = self._compute_nll(profile=profile)
            grad = t1.gradient(val, self.x)
        hess = t2.jacobian(grad, self.x)

        return val, valfull, grad, hess

    def minimize(self):

        if self.is_linear:
            logger.info(
                "Likelihood is purely quadratic, solving by Cholesky decomposition instead of iterative fit"
            )

            # no need to do a minimization, simple matrix solve is sufficient
            val, grad, hess = self.loss_val_grad_hess()

            # use a Cholesky decomposition to easily detect the non-positive-definite case
            chol = tf.linalg.cholesky(hess)

            # FIXME catch this exception to mark failed toys and continue
            if tf.reduce_any(tf.math.is_nan(chol)).numpy():
                raise ValueError(
                    "Cholesky decomposition failed, Hessian is not positive-definite"
                )

            del hess
            gradv = grad[..., None]
            dx = tf.linalg.cholesky_solve(chol, -gradv)[:, 0]
            del chol

            self.x.assign_add(dx)
        else:

            def scipy_loss(xval):
                self.x.assign(xval)
                val, grad = self.loss_val_grad()
                return val.__array__(), grad.__array__()

            def scipy_hessp(xval, pval):
                self.x.assign(xval)
                p = tf.convert_to_tensor(pval)
                val, grad, hessp = self.loss_val_grad_hessp(p)
                return hessp.__array__()

            xval = self.x.numpy()
            callback = FitterCallback(xval)

            try:
                res = scipy.optimize.minimize(
                    scipy_loss,
                    xval,
                    method="trust-krylov",
                    jac=True,
                    hessp=scipy_hessp,
                    tol=0.0,
                    callback=callback,
                )
            except Exception as ex:
                # minimizer could have called the loss or hessp functions with "random" values, so restore the
                # state from the end of the last iteration before the exception
                xval = callback.xval
                logger.debug(ex)
            else:
                xval = res["x"]
                logger.debug(res)

            self.x.assign(xval)

        # force profiling of beta with final parameter values
        # TODO avoid the extra calculation and jitting if possible since the relevant calculation
        # usually would have been done during the minimization
        if self.binByBinStat:
            self._profile_beta()

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
