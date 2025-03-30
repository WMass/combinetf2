import h5py
import hist
import numpy as np
import tensorflow as tf
from wums.ioutils import pickle_load_h5py

from combinetf2.h5pyutils import makesparsetensor, maketensor


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

            if "hdata_cov_inv" in f.keys():
                hdata_cov_inv = f["hdata_cov_inv"]
                self.data_cov_inv = maketensor(hdata_cov_inv)
            else:
                self.data_cov_inv = None

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
                self.data_obs = maketensor(f["hdata_obs"])

            hkstat = f["hkstat"]
            self.kstat = maketensor(hkstat)

            # start by creating tensors which read in the hdf5 arrays (optimized for memory consumption)
            self.constraintweights = maketensor(f["hconstraintweights"])

            self.sparse = not "hnorm" in f

            if self.sparse:
                print(
                    "WARNING: The sparse tensor implementation is experimental and probably slower than with a dense tensor!"
                )
                self.norm = makesparsetensor(f["hnorm_sparse"])
                self.logk = makesparsetensor(f["hlogk_sparse"])
            else:
                self.norm = maketensor(f["hnorm"])
                self.logk = maketensor(f["hlogk"])

            # reference meta data if available
            self.metadata = pickle_load_h5py(f["meta"])
            self.channel_info = self.metadata["channel_info"]

            self.symmetric_tensor = self.metadata.get("symmetric_tensor", False)

            if self.metadata.get("exponential_transform", False):
                raise NotImplementedError(
                    "exponential_transform functionality has been removed.   Please use systematic_type normal instead"
                )

            self.systematic_type = self.metadata.get("systematic_type", "log_normal")

            # infer some metadata from loaded information
            self.dtype = self.data_obs.dtype
            self.nbins = np.sum(
                [
                    np.prod([len(a) for a in info["axes"]])
                    for info in self.channel_info.values()
                    if not info["masked"]
                ],
                dtype=int,
            )
            self.nbinsmasked = np.sum(
                [
                    np.prod([len(a) for a in info["axes"]])
                    for info in self.channel_info.values()
                    if info["masked"]
                ],
                dtype=int,
            )
            self.nbinsfull = self.nbins + self.nbinsmasked
            self.nproc = len(self.procs)
            self.nsyst = len(self.systs)
            self.nsystnoprofile = len(self.systsnoprofile)
            self.nsystnoconstraint = len(self.systsnoconstraint)
            self.nsignals = len(self.signals)
            self.nsystgroups = len(self.systgroups)
            self.nnoigroups = len(self.noigroups)

            # compute indices for channels
            ibin = 0
            for channel, info in self.channel_info.items():
                axes = info["axes"]
                shape = tuple([len(a) for a in axes])
                size = int(np.prod(shape))

                start = ibin
                stop = start + size

                info["start"] = start
                info["stop"] = stop

                ibin = stop

            for channel, info in self.channel_info.items():
                print(channel, info)

            self.axis_procs = hist.axis.StrCategory(self.procs, name="processes")

            self.normalize = normalize
            if self.normalize:
                # normalize prediction and each systematic to total event yield in data
                # FIXME this should be done per-channel ideally

                data_sum = tf.reduce_sum(self.data_obs)
                norm_sum = tf.reduce_sum(self.norm)
                lognorm_sum = tf.math.log(norm_sum)[None, None, ...]

                if self.symmetric_tensor:
                    raise NotImplementedError(
                        "Normalized distributions with symmetric tensor is currently not supported. Need to validate implementation ..."
                    )
                    logkdown = self.logk[..., :]
                    logdown_sum = tf.math.log(
                        tf.reduce_sum(
                            tf.exp(-logkdown) * self.norm[..., None], axis=(0, 1)
                        )
                    )[None, None, ...]
                    logkdown = logkdown + logdown_sum - lognorm_sum

                    logkup = self.logk[..., :]
                    logup_sum = tf.math.log(
                        tf.reduce_sum(
                            tf.exp(logkup) * self.norm[..., None], axis=(0, 1)
                        )
                    )[None, None, ...]
                    logkup = logkup - logup_sum + lognorm_sum

                    # Compute new logkavg
                    logk_array = 0.5 * (logkup + logkdown)
                else:

                    logkavg = self.logk[..., 0, :]
                    logkhalfdiff = self.logk[..., 1, :]

                    logkdown = logkavg - logkhalfdiff
                    logdown_sum = tf.math.log(
                        tf.reduce_sum(
                            tf.exp(-logkdown) * self.norm[..., None], axis=(0, 1)
                        )
                    )[None, None, ...]
                    logkdown = logkdown + logdown_sum - lognorm_sum

                    logkup = logkavg + logkhalfdiff
                    logup_sum = tf.math.log(
                        tf.reduce_sum(
                            tf.exp(logkup) * self.norm[..., None], axis=(0, 1)
                        )
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
