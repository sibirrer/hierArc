_LOS_DISTRIBUTIONS = ["GEV", "GAUSSIAN", "NONE"]


class LOSParam(object):
    """Manager for the source property parameters (currently particularly source
    magnitudes for SNe)"""

    def __init__(
        self,
        los_sampling=False,
        los_distributions=None,
        kwargs_fixed=None,
    ):
        """

        :param los_sampling: if sampling of the parameters should be done
        :type los_sampling: bool
        :param los_distributions: what distribution to be sampled
        :type los_distributions: list of str
        :param kwargs_fixed: fixed arguments in sampling
        :type kwargs_fixed: list of dictionaries or None
        """
        self._los_sampling = los_sampling
        if los_distributions is None:
            los_distributions = []
        for los_distribution in los_distributions:
            if los_distribution not in _LOS_DISTRIBUTIONS:
                raise ValueError(
                    "LOS distribution %s not supported. Please chose among %s."
                    % (los_distribution, _LOS_DISTRIBUTIONS)
                )
        self._los_distributions = los_distributions
        if kwargs_fixed is None:
            kwargs_fixed = [{} for _ in range(len(los_distributions))]
        self._kwargs_fixed = kwargs_fixed

    def param_list(self, latex_style=False):
        """

        :param latex_style: bool, if True returns strings in latex symbols, else in the convention of the sampler
        :return: list of the free parameters being sampled in the same order as the sampling
        """
        name_list = []
        if self._los_sampling is True:
            for i, los_distribution in enumerate(self._los_distributions):
                if los_distribution in ["GEV", "GAUSSIAN"]:
                    if "mean" not in self._kwargs_fixed[i]:
                        if latex_style is True:
                            name_list.append(r"$\mu_{\rm los %s}$" % i)
                        else:
                            name_list.append(str("mean_los_" + str(i)))
                    if "sigma" not in self._kwargs_fixed[i]:
                        if latex_style is True:
                            name_list.append(r"$\sigma_{\rm los %s}$" % i)
                        else:
                            name_list.append(str("sigma_los_" + str(i)))
                if los_distribution in ["GEV"]:
                    if "xi" not in self._kwargs_fixed[i]:
                        if latex_style is True:
                            name_list.append(r"$\xi_{\rm los} %s$" % i)
                        else:
                            name_list.append(str("xi_los_" + str(i)))
        # str(name + "_" + type + str(k))
        return name_list

    def args2kwargs(self, args, i=0):
        """

        :param args: sampling argument list
        :param i: index of argument list to start reading out
        :return: keyword argument list with parameter names
        """
        kwargs = [{} for _ in range(len(self._los_distributions))]
        if self._los_sampling is True:
            for k, los_distribution in enumerate(self._los_distributions):
                if los_distribution in ["GEV", "GAUSSIAN"]:
                    if "mean" in self._kwargs_fixed[k]:
                        kwargs[k]["mean"] = self._kwargs_fixed[k]["mean"]
                    else:
                        kwargs[k]["mean"] = args[i]
                        i += 1
                    if "sigma" in self._kwargs_fixed[k]:
                        kwargs[k]["sigma"] = self._kwargs_fixed[k]["sigma"]
                    else:
                        kwargs[k]["sigma"] = args[i]
                        i += 1
                if los_distribution in ["GEV"]:
                    if "xi" in self._kwargs_fixed[k]:
                        kwargs[k]["xi"] = self._kwargs_fixed[k]["xi"]
                    else:
                        kwargs[k]["xi"] = args[i]
                        i += 1
        return kwargs, i

    def kwargs2args(self, kwargs):
        """

        :param kwargs: keyword argument list of parameters
        :return: sampling argument list in specified order
        """
        args = []
        if self._los_sampling is True:
            for k, los_distribution in enumerate(self._los_distributions):
                if los_distribution in ["GEV", "GAUSSIAN"]:
                    if "mean" not in self._kwargs_fixed[k]:
                        args.append(kwargs[k]["mean"])
                    if "sigma" not in self._kwargs_fixed[k]:
                        args.append(kwargs[k]["sigma"])
                if los_distribution in ["GEV"]:
                    if "xi" not in self._kwargs_fixed[k]:
                        args.append(kwargs[k]["xi"])
        return args
