import h5py
import hist
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio


def maketensor(h5dset):
    if "original_shape" in h5dset.attrs:
        shape = h5dset.attrs["original_shape"]
    else:
        shape = h5dset.shape

    if h5dset.size == 0:
        return tf.zeros(shape, h5dset.dtype)

    filename = h5dset.file.filename
    dsetpath = h5dset.name

    atensor = tfio.IOTensor.from_hdf5(filename)(dsetpath).to_tensor()

    atensor = tf.reshape(atensor, shape)
    return atensor


def makesparsetensor(h5group):
    indices = maketensor(h5group["indices"])
    values = maketensor(h5group["values"])
    dense_shape = h5group.attrs["dense_shape"]

    return tf.sparse.SparseTensor(indices, values, dense_shape)


class FitInputData:
    def __init__(self, filename, pseudodata=None, normalize=False):
        with h5py.File(filename, mode="r") as f:

            # load text arrays from file
            self.procs = f["hprocs"][...]
            self.signals = f["hsignals"][...]
            self.systs = f["hsysts"][...]
            self.systsnoprofile = f["hsystsnoprofile"][...]
            self.systsnoconstraint = f["hsystsnoconstraint"][...]
            self.systgroups = f["hsystgroups"][...]
            self.systgroupidxs = f["hsystgroupidxs"][...]

            self.noigroups = f["hnoigroups"][...]
            self.noigroupidxs = f["hnoigroupidxs"][...]
            if "hpseudodatanames" in f.keys():
                self.pseudodatanames = f["hpseudodatanames"][...].astype(str)
            else:
                self.pseudodatanames = []

            # load arrays from file
            hconstraintweights = f["hconstraintweights"]
            hdata_obs = f["hdata_obs"]

            if "hdata_cov_inv" in f.keys():
                hdata_cov_inv = f["hdata_cov_inv"]
                self.data_cov_inv = maketensor(hdata_cov_inv)
            else:
                self.data_cov_inv = None

            self.sparse = not "hnorm" in f

            if self.sparse:
                print(
                    "WARNING: The sparse tensor implementation is experimental and probably slower than with a dense tensor!"
                )
                hnorm_sparse = f["hnorm_sparse"]
                hlogk_sparse = f["hlogk_sparse"]
            else:
                hnorm = f["hnorm"]
                hlogk = f["hlogk"]

            # infer some metadata from loaded information
            self.dtype = hdata_obs.dtype
            self.nbins = hdata_obs.shape[-1]
            self.nproc = len(self.procs)
            self.nsyst = len(self.systs)
            self.nsystnoprofile = len(self.systsnoprofile)
            self.nsystnoconstraint = len(self.systsnoconstraint)
            self.nsignals = len(self.signals)
            self.nsystgroups = len(self.systgroups)
            self.nnoigroups = len(self.noigroups)

            # reference meta data if available
            self.metadata = {}
            if "meta" in f.keys():
                from narf.ioutils import pickle_load_h5py

                self.metadata = pickle_load_h5py(f["meta"])
                self.channel_info = self.metadata["channel_info"]

            else:
                self.channel_info = {
                    "ch0": {
                        "axes": [
                            hist.axis.Integer(
                                0,
                                self.nbins,
                                underflow=False,
                                overflow=False,
                                name="obs",
                            )
                        ]
                    }
                }

            self.symmetric_tensor = self.metadata.get("symmetric_tensor", False)
            self.exponential_transform = self.metadata.get(
                "exponential_transform", False
            )
            self.exponential_transform_scale = self.metadata.get(
                "exponential_transform_scale", 1000000
            )

            # compute indices for channels
            ibin = 0
            for channel, info in self.channel_info.items():
                axes = info["axes"]
                shape = tuple([len(a) for a in axes])
                size = np.prod(shape)

                start = ibin
                stop = start + size

                info["start"] = start
                info["stop"] = stop

                ibin = stop

            for channel, info in self.channel_info.items():
                print(channel, info)

            self.axis_procs = hist.axis.StrCategory(self.procs, name="processes")

            # build tensorflow graph for likelihood calculation

            # start by creating tensors which read in the hdf5 arrays (optimized for memory consumption)
            self.constraintweights = maketensor(hconstraintweights)

            # load data/pseudodata
            if pseudodata is not None:
                if pseudodata in self.pseudodatanames:
                    pseudodata_idx = np.where(self.pseudodatanames == pseudodata)[0][0]
                else:
                    raise Exception(
                        "Pseudodata %s not found, available pseudodata sets are %s"
                        % (pseudodata, self.pseudodatanames)
                    )
                print("Run pseudodata fit for index %i: " % (pseudodata_idx))
                print(self.pseudodatanames[pseudodata_idx])
                hdata_obs = f["hpseudodata"]

                data_obs = maketensor(hdata_obs)
                self.data_obs = data_obs[:, pseudodata_idx]
            else:
                self.data_obs = maketensor(hdata_obs)

            hkstat = f["hkstat"]
            self.kstat = maketensor(hkstat)

            if self.sparse:
                self.norm_sparse = makesparsetensor(hnorm_sparse)
                self.logk_sparse = makesparsetensor(hlogk_sparse)
            else:
                self.norm = maketensor(hnorm)
                self.logk = maketensor(hlogk)

            self.normalize = normalize
            if self.normalize:
                # normalize predictoin and each systematic to total event yield in data
                # FIXME this should be done per-channel ideally

                data_sum = tf.reduce_sum(self.data_obs)
                norm_sum = tf.reduce_sum(self.norm)
                lognorm_sum = tf.math.log(norm_sum)[None, None, ...]

                logkavg = self.logk[..., 0, :]
                logkhalfdiff = self.logk[..., 1, :]

                logkdown = logkavg - logkhalfdiff
                logdown_sum = tf.math.log(
                    tf.reduce_sum(tf.exp(-logkdown) * self.norm[..., None], axis=(0, 1))
                )[None, None, ...]
                logkdown = logkdown + logdown_sum - lognorm_sum

                logkup = logkavg + logkhalfdiff
                logup_sum = tf.math.log(
                    tf.reduce_sum(tf.exp(logkup) * self.norm[..., None], axis=(0, 1))
                )[None, None, ...]
                logkup = logkup - logup_sum + lognorm_sum

                # Compute new logkavg and logkhalfdiff
                logkavg = 0.5 * (logkup + logkdown)
                logkhalfdiff = 0.5 * (logkup - logkdown)

                # Stack logkavg and logkhalfdiff to form the new logk_array using tf.stack
                logk_array = tf.stack([logkavg, logkhalfdiff], axis=-2)

                # Finally, set self.logk to the new computed logk_array
                self.logk = logk_array
                self.norm = self.norm * (data_sum / norm_sum)[None, None, ...]

    def getImpactsAxes(self):
        impact_names = list(self.signals.astype(str)) + list(self.systs.astype(str))
        return hist.axis.StrCategory(impact_names, name="impacts")

    def getGlobalImpactsAxes(self):
        impact_names = list(self.systs.astype(str)[self.nsystnoconstraint :])
        return hist.axis.StrCategory(impact_names, name="impacts")

    def getImpactsAxesGrouped(self, bin_by_bin_stat=False):
        impact_names_grouped = list(self.systgroups.astype(str))
        # impact data stat
        impact_names_grouped.append("stat")
        if bin_by_bin_stat:
            # impact bin-by-bin stat
            impact_names_grouped.append("binByBinStat")
        return hist.axis.StrCategory(impact_names_grouped, name="impacts")
