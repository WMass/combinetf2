import tensorflow as tf


class Basemodel:
    """
    A class to output histograms without any transformation, can be used as base class to inherit custom physics models from.
    """

    name = "basemodel"
    need_observables = True  # if observables should be provided to the compute function
    need_params = (
        False  # if parameter values should be provided to the compute function
    )
    has_data = True  # if data histograms are stored or not, and if chi2 is calculated

    def __init__(self, indata):
        # The result of a model in the output dictionary is stored under 'result = fitresult[cls.name]'
        #   if the model can have different instances 'self.instance' must be set to a unique string and the result will be stored in 'result = result[self.instance]'
        #   each model can have different channels that are the same or different from channels from the input data. All channel specific results are stored under 'result["channels"]'
        self.instance = "distributions"
        self.channel_info = indata.channel_info

    # class function to parse strings as given by the argparse input e.g. -m PhysicsModel <arg1> <args2>
    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        return cls(indata, *args, **kwargs)

    # function to compute the transformation of the physics model, has to be differentiable.
    #    For custom physics models, this function should be overridden.
    #    observables are the provided histograms inclusive in processes: nbins
    #    params are the fit parameters
    def compute(self, observables, params=None):
        return observables

    # function to compute the transformation of the physics model, has to be differentiable.
    #    For custom physics models, this function can be overridden.
    #    observables are the provided histograms per process: nbins x nprocesses
    #    params are the fit parameters
    def compute_per_process(self, observables, params=None):
        return self.compute(observables)

    # generic version which should not need to be overridden
    def make_fun(self, fun_flat, params, inclusive=True):
        compute = self.compute if inclusive else self.compute_per_process

        def fun():
            if self.need_params and self.need_observables:
                exp = compute(fun_flat(), params)
            elif self.need_observables:
                exp = compute(fun_flat())
            elif self.need_params:
                exp = compute(params)
            else:
                exp = compute()
            return exp

        return fun

    # generic version which should not need to be overridden
    @tf.function
    def get_data(self, data, data_cov_inv=None):
        with tf.GradientTape() as t:
            t.watch(data)
            output = self.compute(data)

        jacobian = t.jacobian(output, data)

        # Ensure the Jacobian has at least 2 dimensions (expand in case output is a scalar)
        if len(jacobian.shape) == 1:
            jacobian = tf.expand_dims(jacobian, axis=0)

        if data_cov_inv is None:
            # Assume poisson uncertainties on data
            cov_output = (jacobian * data) @ tf.transpose(jacobian)
        else:
            # General case with full covariance matrix
            data_cov = tf.linalg.inv(data_cov_inv)
            cov_output = jacobian @ data_cov @ tf.transpose(jacobian)

        variances_output = tf.linalg.diag_part(cov_output)

        return output, variances_output, cov_output
