import numpy as np


class LensDistribution(object):
    """Class to draw lens parameters of individual lens from distributions."""

    def __init__(
        self,
        lambda_mst_sampling=False,
        lambda_mst_distribution="NONE",
        gamma_in_sampling=False,
        gamma_in_distribution="NONE",
        log_m2l_sampling=False,
        log_m2l_distribution="NONE",
        alpha_lambda_sampling=False,
        beta_lambda_sampling=False,
        alpha_gamma_in_sampling=False,
        alpha_log_m2l_sampling=False,
        log_scatter=False,
        mst_ifu=False,
        lambda_scaling_property=0,
        lambda_scaling_property_beta=0,
        kwargs_min=None,
        kwargs_max=None,
        gamma_pl_index=None,
        gamma_pl_global_sampling=False,
        gamma_pl_global_dist="NONE",
    ):
        """

        :param lambda_mst_sampling: bool, if True adds a global mass-sheet transform parameter in the sampling
        :param lambda_mst_distribution: string, distribution function of the MST transform
        :param gamma_in_sampling: bool, if True samples the inner slope of the GNFW profile
        :param gamma_in_distribution: string, distribution function of the inner
            slope of the GNFW profile
        :param log_m2l_sampling: bool, if True samples the mass to light ratio of
            the stars in logarithmic scale
        :param log_m2l_distribution: string, distribution function of the logarithm of mass to
            light ratio of the lens
        :param alpha_lambda_sampling: bool, if True samples a parameter alpha_lambda, which scales lambda_mst linearly
            according to a predefined quantity of the lens
        :param beta_lambda_sampling: bool, if True samples a parameter beta_lambda, which scales lambda_mst linearly
            according to a predefined quantity of the lens
        :param alpha_gamma_in_sampling: bool, if True samples a parameter alpha_gamma_in, which scales gamma_in linearly
        :param alpha_log_m2l_sampling: bool, if True samples a parameter alpha_log_m2l, which scales log_m2l linearly
        :param log_scatter: boolean, if True, samples the Gaussian scatter amplitude in log space
         (and thus flat prior in log)
        :param mst_ifu: bool, if True replaces the lambda_mst parameter by the lambda_ifu parameter (and distribution)
         in sampling this lens.
        :param lambda_scaling_property: float (optional), scaling of
         lambda_mst = lambda_mst_global + alpha * lambda_scaling_property
        :param lambda_scaling_property_beta: float (optional), scaling of
         lambda_mst = lambda_mst_global + beta * lambda_scaling_property_beta
        :param kwargs_min: minimum arguments of parameters supported by each lens
        :type kwargs_min: dict or None
        :param kwargs_max: maximum arguments of parameters supported by each lens
        :type kwargs_max: dict or None
        :param gamma_pl_index: index of gamma_pl parameter associated with this lens
        :type gamma_pl_index: int or None
        :param gamma_pl_global_sampling: if sampling a global power-law density slope distribution
        :type gamma_pl_global_sampling: bool
        :param gamma_pl_global_dist: distribution of global gamma_pl distribution ("GAUSSIAN" or "NONE")
        """
        self._lambda_mst_sampling = lambda_mst_sampling
        self._lambda_mst_distribution = lambda_mst_distribution
        self._gamma_in_sampling = gamma_in_sampling
        self._gamma_in_distribution = gamma_in_distribution
        self._log_m2l_sampling = log_m2l_sampling
        self._log_m2l_distribution = log_m2l_distribution
        self._alpha_lambda_sampling = alpha_lambda_sampling
        self._beta_lambda_sampling = beta_lambda_sampling
        self._alpha_gamma_in_sampling = alpha_gamma_in_sampling
        self._alpha_log_m2l_sampling = alpha_log_m2l_sampling
        self._mst_ifu = mst_ifu
        self._lambda_scaling_property = lambda_scaling_property
        self._lambda_scaling_property_beta = lambda_scaling_property_beta
        self._gamma_pl_global_sampling = gamma_pl_global_sampling
        self._gamma_pl_global_dist = gamma_pl_global_dist

        self._log_scatter = log_scatter
        if kwargs_max is None:
            kwargs_max = {}
        if kwargs_min is None:
            kwargs_min = {}
        self._gamma_in_min, self._gamma_in_max = kwargs_min.get(
            "gamma_in", -np.inf
        ), kwargs_max.get("gamma_in", np.inf)
        self._log_m2l_min, self._log_m2l_max = kwargs_min.get(
            "log_m2l", -np.inf
        ), kwargs_max.get("log_m2l", np.inf)
        if gamma_pl_index is not None:
            self._gamma_pl_model = True
            self._gamma_pl_index = gamma_pl_index
        else:
            self._gamma_pl_model = False
            self._gamma_pl_index = None

    def draw_lens(
        self,
        lambda_mst=1,
        lambda_mst_sigma=0,
        gamma_ppn=1,
        lambda_ifu=1,
        lambda_ifu_sigma=0,
        alpha_lambda=0,
        beta_lambda=0,
        gamma_in=1,
        gamma_in_sigma=0,
        alpha_gamma_in=0,
        log_m2l=1,
        log_m2l_sigma=0,
        alpha_log_m2l=0,
        gamma_pl_list=None,
        gamma_pl_mean=2,
        gamma_pl_sigma=0,
    ):
        """Draws a realization of a specific model from the hyperparameter distribution.

        :param lambda_mst: MST transform
        :param lambda_mst_sigma: spread in the distribution
        :param gamma_ppn: Post-Newtonian parameter
        :param lambda_ifu: secondary lambda_mst parameter for subset of lenses specified
            for
        :param lambda_ifu_sigma: secondary lambda_mst_sigma parameter for subset of
            lenses specified for
        :param alpha_lambda: float, linear slope of the lambda_int scaling relation with
            lens quantity self._lambda_scaling_property
        :param beta_lambda: float, a second linear slope of the lambda_int scaling
            relation with lens quantity self._lambda_scaling_property_beta
        :param gamma_in: inner slope of the NFW profile
        :param gamma_in_sigma: spread in the distribution
        :param alpha_gamma_in: float, linear slope of the gamma_in scaling relation with
            lens quantity self._lambda_scaling_property
        :param log_m2l: log(mass-to-light ratio)
        :param log_m2l_sigma: spread in the distribution
        :param alpha_log_m2l: float, linear slope of the log(m2l) scaling relation with
            lens quantity self._lambda_scaling_property
        :param gamma_pl_list: power-law density slopes as lists (for multiple lenses)
        :type gamma_pl_list: list or None
        :param gamma_pl_mean: mean of gamma_pl of the global distribution
        :param gamma_pl_sigma: sigma of the gamma_pl global distribution
        :return: draw from the distributions
        """
        kwargs_return = {}

        if self._mst_ifu is True:
            lambda_mst_mean_lens = lambda_ifu
        else:
            lambda_mst_mean_lens = lambda_mst

        lambda_lens = (
            lambda_mst_mean_lens
            + alpha_lambda * self._lambda_scaling_property
            + beta_lambda * self._lambda_scaling_property_beta
        )
        lambda_mst_draw = lambda_lens
        if self._lambda_mst_sampling:
            if self._lambda_mst_distribution in ["GAUSSIAN"]:
                lambda_mst_draw = np.random.normal(lambda_lens, lambda_ifu_sigma)

        kwargs_return["lambda_mst"] = lambda_mst_draw
        kwargs_return["gamma_ppn"] = gamma_ppn

        if self._gamma_in_sampling:
            if gamma_in < self._gamma_in_min or gamma_in > self._gamma_in_max:
                raise ValueError(
                    "gamma_in parameter is out of bounds of the interpolated range!"
                )
            if self._gamma_in_distribution in ["GAUSSIAN"]:
                gamma_in_lens = (
                    gamma_in + alpha_gamma_in * self._lambda_scaling_property
                )
            else:
                gamma_in_lens = gamma_in
            gamma_in_draw = np.random.normal(gamma_in_lens, gamma_in_sigma)
            if gamma_in_draw < self._gamma_in_min or gamma_in_draw > self._gamma_in_max:
                return self.draw_lens(
                    lambda_mst=lambda_mst,
                    lambda_mst_sigma=lambda_mst_sigma,
                    gamma_ppn=gamma_ppn,
                    lambda_ifu=lambda_ifu,
                    lambda_ifu_sigma=lambda_ifu_sigma,
                    alpha_lambda=alpha_lambda,
                    beta_lambda=beta_lambda,
                    gamma_in=gamma_in,
                    gamma_in_sigma=gamma_in_sigma,
                    alpha_gamma_in=alpha_gamma_in,
                    log_m2l=log_m2l,
                    log_m2l_sigma=log_m2l_sigma,
                    alpha_log_m2l=alpha_log_m2l,
                    gamma_pl_list=gamma_pl_list,
                    gamma_pl_mean=gamma_pl_mean,
                    gamma_pl_sigma=gamma_pl_sigma,
                )
            kwargs_return["gamma_in"] = gamma_in_draw
        if self._log_m2l_sampling:

            if log_m2l < self._log_m2l_min or log_m2l > self._log_m2l_max:
                raise ValueError(
                    "m2l parameter is out of bounds of the interpolated range!"
                )

            log_m2l_lens = log_m2l + alpha_log_m2l * self._lambda_scaling_property
            log_m2l_draw = np.random.normal(log_m2l_lens, log_m2l_sigma)

            if log_m2l_draw < self._log_m2l_min or log_m2l_draw > self._log_m2l_max:
                return self.draw_lens(
                    lambda_mst=lambda_mst,
                    lambda_mst_sigma=lambda_mst_sigma,
                    gamma_ppn=gamma_ppn,
                    lambda_ifu=lambda_ifu,
                    lambda_ifu_sigma=lambda_ifu_sigma,
                    alpha_lambda=alpha_lambda,
                    beta_lambda=beta_lambda,
                    gamma_in=gamma_in,
                    gamma_in_sigma=gamma_in_sigma,
                    alpha_gamma_in=alpha_gamma_in,
                    log_m2l=log_m2l,
                    log_m2l_sigma=log_m2l_sigma,
                    alpha_log_m2l=alpha_log_m2l,
                    gamma_pl_list=gamma_pl_list,
                    gamma_pl_mean=gamma_pl_mean,
                    gamma_pl_sigma=gamma_pl_sigma,
                )
            kwargs_return["log_m2l"] = log_m2l_draw
        if self._gamma_pl_model is True:
            kwargs_return["gamma_pl"] = gamma_pl_list[self._gamma_pl_index]
        elif self._gamma_pl_global_sampling is True:
            if self._gamma_pl_global_dist in ["GAUSSIAN"]:
                gamma_pl_draw = np.random.normal(gamma_pl_mean, gamma_pl_sigma)
            else:
                gamma_pl_draw = gamma_pl_mean
            kwargs_return["gamma_pl"] = gamma_pl_draw

        return kwargs_return
