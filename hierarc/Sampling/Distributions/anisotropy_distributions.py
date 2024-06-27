import numpy as np


class AnisotropyDistribution(object):
    """
    class to draw anisotropy parameters from hyperparameter distributions
    """
    def __init__(self, anisotropy_model, anisotropy_sampling,
                 distribution_function, kwargs_anisotropy_min, kwargs_anisotropy_max):
        """

        :param anisotropy_model: string, name of anisotropy model to consider
        :param anisotropy_sampling: bool, if True adds a global stellar anisotropy parameter that alters the single lens
         kinematic prediction
        :param distribution_function: string, 'NONE', 'GAUSSIAN', description of the distribution function of the
        anisotropy model parameters
        """

        self._anisotropy_model = anisotropy_model
        self._anisotropy_sampling = anisotropy_sampling
        self._distribution_function = distribution_function
        if kwargs_anisotropy_min is None:
            kwargs_anisotropy_min = {}
        if kwargs_anisotropy_max is None:
            kwargs_anisotropy_max = {}
        self._kwargs_min = kwargs_anisotropy_min
        self._kwargs_max = kwargs_anisotropy_max
        self._a_ani_min, self._a_ani_max = self._kwargs_min.get("a_ani", -np.inf), self._kwargs_max.get("a_ani", np.inf)
        self._beta_inf_min = self._kwargs_min.get("beta_inf", -np.inf)
        self._beta_inf_max = self._kwargs_max.get("beta_inf", np.inf)

    def draw_anisotropy(self, a_ani=None, a_ani_sigma=0, beta_inf=None, beta_inf_sigma=0):
        """Draw Gaussian distribution and re-sample if outside bounds.

        :param a_ani: mean of the distribution
        :param a_ani_sigma: std of the distribution
        :param beta_inf: anisotropy at infinity (relevant for GOM model)
        :param beta_inf_sigma: std of beta_inf distribution
        :return: random draw from the distribution
        """
        kwargs_return = {}
        if not self._anisotropy_sampling:
            if a_ani is not None:
                kwargs_return["a_ani"] = a_ani
            if beta_inf is not None:
                kwargs_return["beta_inf"] = beta_inf
            return kwargs_return
        if self._anisotropy_model in ["OM", "const", "GOM"]:
            if a_ani < self._a_ani_min or a_ani > self._a_ani_max:
                raise ValueError(
                    "anisotropy parameter is out of bounds of the interpolated range!"
                )
            # we draw a linear gaussian for 'const' anisotropy and a scaled proportional one for 'OM
            if self._distribution_function in ["GAUSSIAN"]:
                if self._anisotropy_model == "OM":
                    a_ani_draw = np.random.normal(a_ani, a_ani_sigma * a_ani)
                else:
                    a_ani_draw = np.random.normal(a_ani, a_ani_sigma)
                if a_ani_draw < self._a_ani_min or a_ani_draw > self._a_ani_max:
                    return self.draw_anisotropy(a_ani, a_ani_sigma)
                kwargs_return["a_ani"] = a_ani_draw
            else:
                kwargs_return["a_ani"] = a_ani
        if self._anisotropy_model in ["GOM"]:
            if beta_inf < self._beta_inf_min or beta_inf > self._beta_inf_max:
                raise ValueError(
                    "anisotropy parameter is out of bounds of the interpolated range!"
                )
            if self._distribution_function in ["GAUSSIAN"]:
                beta_inf_draw = np.random.normal(beta_inf, beta_inf_sigma)
            else:
                beta_inf_draw = beta_inf
            if beta_inf_draw < self._beta_inf_min or beta_inf_draw > self._beta_inf_max:
                return self.draw_anisotropy(
                    a_ani, a_ani_sigma, beta_inf, beta_inf_sigma
                )
            kwargs_return["beta_inf"] = beta_inf_draw
        return kwargs_return
