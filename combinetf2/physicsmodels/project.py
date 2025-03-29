import tensorflow as tf

from combinetf2.physicsmodels.basemodel import Basemodel


class Project(Basemodel):
    """
    A class to project a histogram to lower dimensions.
    Optionally the histogram can be normalized.
    The normalization is done to the integral of all processes or data.

    Parameters
    ----------
    channel_name : str
        Name of the channel. Required.
    axes_names : list of str, optional
        Names of the axes to keep. If empty, the histogram will be projected to a single bin.
    """

    name = "project"
    need_params = False

    def __init__(self, indata, channel, *axes_names):
        info = indata.channel_info[channel]

        self.start = info["start"]
        self.stop = info["stop"]

        channel_axes = info["axes"]

        hist_axes = [axis for axis in channel_axes if axis.name in axes_names]

        if len(hist_axes) != len(axes_names):
            raise ValueError(
                f"Hist axes {[h.name for h in hist_axes]} != {axes_names} not found"
            )

        exp_axes = channel_axes.copy()

        # for per process histograms
        exp_axes.append(indata.axis_procs)
        extra_axes = [indata.axis_procs]

        self.exp_shape = tuple([len(a) for a in exp_axes])

        channel_axes_names = [axis.name for axis in channel_axes]
        extra_axes_names = [axis.name for axis in extra_axes]

        axis_idxs = [channel_axes_names.index(axis) for axis in axes_names]

        self.proj_idxs = [i for i in range(len(channel_axes)) if i not in axis_idxs]

        post_proj_axes_names = [
            axis for axis in channel_axes_names if axis in axes_names
        ] + extra_axes_names

        self.transpose_idxs = [
            post_proj_axes_names.index(axis) for axis in axes_names
        ] + [post_proj_axes_names.index(axis) for axis in extra_axes_names]

        self.has_data = not info["masked"]

        self.channel_info = {
            channel: {
                "axes": hist_axes,
                "start": None,
                "stop": None,
            }
        }

        self.instance = channel
        if len(axes_names):
            self.instance += "_" + "_".join(axes_names)

    def project(self, values, out_shape, transpose_idxs):
        exp = values[self.start : self.stop]
        exp = tf.reshape(exp, out_shape)
        exp = tf.reduce_sum(exp, axis=self.proj_idxs)
        exp = tf.transpose(exp, perm=transpose_idxs)
        return exp

    def compute_per_process(self, observables):
        return self.project(observables, self.exp_shape, self.transpose_idxs)

    def compute(self, observables):
        out_shape = self.exp_shape[:-1]
        transpose_idxs = self.transpose_idxs[:-1]
        return self.project(observables, out_shape, transpose_idxs)
