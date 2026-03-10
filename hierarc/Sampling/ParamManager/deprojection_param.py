import numpy as np


class DeprojectionParam(object):
    """Manager for the kinematics deprojection parameters (q_intrinsic)."""

    def __init__(
        self,
        deprojection_sampling=False,
        distribution_function="NONE",
        log_scatter=False,
        kwargs_fixed=None,
    ):
        """

        :param deprojection_sampling: bool, if True, makes use of this module, else ignores it's functionalities
        :param distribution_function: string, 'NONE', 'GAUSSIAN', 'GAUSSIAN_SCALED', description of the distribution
         function of the deprojection model parameters
        :param log_scatter: boolean, if True, samples the Gaussian scatter amplitude in log space (and thus flat prior in log)
        :param kwargs_fixed: keyword arguments of the fixed parameters
        """
        self._deprojection_sampling = deprojection_sampling
        self._distribution_function = distribution_function
        self._distribution_parameterization = "q_intrinsic"
        if kwargs_fixed is None:
            kwargs_fixed = {}
        self._kwargs_fixed = kwargs_fixed
        self._log_scatter = log_scatter

    def param_list(self, latex_style=False):
        """

        :param latex_style: bool, if True returns strings in latex symbols, else in the convention of the sampler
        :param i: int, index of the parameter to start with
        :return: list of the free parameters being sampled in the same order as the sampling
        """
        list = []
        if self._deprojection_sampling is True:
            if "q_intrinsic" not in self._kwargs_fixed:
                if latex_style is True:
                    list.append(r"$\langle q_{\rm int} \rangle$")
                else:
                    list.append("q_intrinsic")
            if self._distribution_function in ["GAUSSIAN", "GAUSSIAN_SCALED"]:
                if "q_intrinsic_sigma" not in self._kwargs_fixed:
                    if latex_style is True:
                        if self._log_scatter is True:
                            list.append(r"$\log_{10}\sigma(q_{\rm int})$")
                        else:
                            list.append(r"$\sigma(q_{\rm int})$")
                    else:
                        list.append("q_intrinsic_sigma")
        return list

    def args2kwargs(self, args, i=0):
        """

        :param args: sampling argument list
        :param i: integer, index to start reading out the argument list
        :return: keyword argument list with parameter names
        """
        kwargs = {}
        if self._deprojection_sampling is True:
            if "q_intrinsic" in self._kwargs_fixed:
                kwargs["q_intrinsic"] = self._kwargs_fixed["q_intrinsic"]
            else:
                kwargs["q_intrinsic"] = args[i]
                i += 1
            if self._distribution_function in ["GAUSSIAN", "GAUSSIAN_SCALED"]:
                if "q_intrinsic_sigma" in self._kwargs_fixed:
                    kwargs["q_intrinsic_sigma"] = self._kwargs_fixed[
                        "q_intrinsic_sigma"
                    ]
                else:
                    if self._log_scatter is True:
                        kwargs["q_intrinsic_sigma"] = 10 ** (args[i])
                    else:
                        kwargs["q_intrinsic_sigma"] = args[i]
                    i += 1
        return kwargs, i

    def kwargs2args(self, kwargs):
        """

        :param kwargs: keyword argument list of parameters
        :return: sampling argument list in specified order
        """
        args = []
        if self._deprojection_sampling is True:
            if "q_intrinsic" not in self._kwargs_fixed:
                args.append(kwargs["q_intrinsic"])
            if self._distribution_function in ["GAUSSIAN", "GAUSSIAN_SCALED"]:
                if "q_intrinsic_sigma" not in self._kwargs_fixed:
                    if self._log_scatter is True:
                        args.append(np.log10(kwargs["q_intrinsic_sigma"]))
                    else:
                        args.append(kwargs["q_intrinsic_sigma"])
        return args
