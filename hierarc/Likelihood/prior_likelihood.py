class PriorLikelihood(object):
    """Class to define priors for individual lenses, e.g. from lens models etc."""

    def __init__(self, prior_list=None):
        """

        :param prior_list: list of [[name, mean, sigma], [],...]
        """
        if prior_list is None:
            prior_list = []
        self._prior_list = prior_list
        self._param_name_list = []
        self._param_mean_list = []
        self._param_sigma_list = []
        for i, param in enumerate(prior_list):
            self._param_name_list.append(param[0])
            self._param_mean_list.append(param[1])
            self._param_sigma_list.append(param[2])

    def log_likelihood(self, kwargs):
        """

        :param kwargs:
        :type kwargs: dict
        :return: log likelihood
        """
        lnlikelihood = 0
        for i, param in enumerate(self._param_name_list):
            if param in kwargs:
                lnlikelihood -= (kwargs[param] - self._param_mean_list[i]) ** 2 / (
                    2 * self._param_sigma_list[i] ** 2
                )
        return lnlikelihood
