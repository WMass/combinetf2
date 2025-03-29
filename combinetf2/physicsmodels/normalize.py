import tensorflow as tf

from combinetf2.physicsmodels.project import Project


class Normalize(Project):
    """
    Same as project but also normalize
    """

    name = "normalize"
    normalize = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def project(self, values, *args):
        norm = tf.reduce_sum(values[self.start : self.stop])
        out = values / norm
        return super().project(out, *args)
