import os

import h5py
import hist
import narf.ioutils
import numpy as np
import tensorflow as tf


def hist_boost(name, axes, values, variances=None, label=None):
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    storage_type = (
        hist.storage.Weight() if variances is not None else hist.storage.Double()
    )
    h = hist.Hist(*axes, storage=storage_type, name=name, label=label)
    h.values()[...] = memoryview(tf.reshape(values, h.shape))
    if variances is not None:
        h.variances()[...] = memoryview(tf.reshape(variances, h.shape))
    return h


def hist_H5PickleProxy(*args, **kwargs):
    return narf.ioutils.H5PickleProxy(hist_boost(*args, **kwargs))


def load_H5PickleProxy(proxy):
    return proxy.get()


class Workspace:
    def __init__(self, output_format="narf"):
        self.output_format = output_format

        if output_format == "narf":
            self.hist = hist_H5PickleProxy
            self.pack = narf.ioutils.H5PickleProxy
            self.unpack = load_H5PickleProxy
            self.dump = narf.ioutils.pickle_dump_h5py
            self.extension = "hdf5"
        elif output_format == "h5py":
            self.hist = hist_boost
            self.pack = lambda x: x
            self.unpack = lambda x: x
            self.dump = lambda name, obj, fout: fout.create_dataset(name, obj)
            self.extension = "h5"
            raise NotImplementedError(
                f"Writing to vanilla h5py is under development and not yet supported"
            )
        else:
            raise ValueError(f"Unknown output format {output_format}")

    def project(self, h, axes):
        h = self.unpack(h).project(*axes)
        return self.pack(h)

    def write(self, output, results, meta={}):
        outfolder = os.path.dirname(output)
        if outfolder:
            if not os.path.exists(outfolder):
                print(f"Creating output folder {outfolder}")
                os.makedirs(outfolder)

        if "." not in output and output.split(".")[-1]:
            output += f".{self.extension}"

        print(f"Write output file {output}")

        with h5py.File(output, "w") as fout:
            self.dump("results", results, fout)
            self.dump("meta", meta, fout)
