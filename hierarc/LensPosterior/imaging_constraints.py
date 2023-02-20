import numpy as np


class ImageModelPosterior(object):
    """
    class to manage lens and light model posteriors inferred from imaging data
    """
    def __init__(self, theta_E, theta_E_error, gamma, gamma_error, r_eff, r_eff_error):
        """

        :param theta_E: Einstein radius (in arc seconds)
        :param theta_E_error: 1-sigma error on Einstein radius
        :param gamma: power-law slope (2 = isothermal) estimated from imaging data
        :param gamma_error: 1-sigma uncertainty on power-law slope
        :param r_eff: half-light radius of the deflector (arc seconds)
        :param r_eff_error: uncertainty on half-light radius
        """
        self._theta_E, self._theta_E_error = theta_E, theta_E_error
        self._gamma, self._gamma_error = gamma, gamma_error
        self._r_eff, self._r_eff_error = r_eff, r_eff_error

    def draw_lens(self, no_error=False):
        """

        :param no_error: bool, if True, does not render from the uncertainty but uses the mean values instead
        :return: theta_E, gamma, r_eff, delta_r_eff
        """
        if no_error is True:
            return self._theta_E, self._gamma, self._r_eff, 1
        theta_E_draw = np.maximum(np.random.normal(loc=self._theta_E, scale=self._theta_E_error), 0)
        gamma_draw = np.random.normal(loc=self._gamma, scale=self._gamma_error)
        # distributions are drawn in the range [1, 3)
        # the power-law slope gamma=3 is divergent in mass in the center and values close close to =3 may be unstable
        # to compute the kinematics for.
        gamma_draw = np.maximum(gamma_draw, 1.)
        gamma_draw = np.minimum(gamma_draw, 2.999)
        # we make sure no negative r_eff are being sampled
        delta_r_eff = np.maximum(np.random.normal(loc=1, scale=self._r_eff_error/self._r_eff), 0.001)
        r_eff_draw = delta_r_eff * self._r_eff
        return theta_E_draw, gamma_draw, r_eff_draw, delta_r_eff
