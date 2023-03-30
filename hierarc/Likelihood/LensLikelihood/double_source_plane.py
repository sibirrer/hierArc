import numpy as np


class DSPLikelihood(object):
    """
    likelihood for Einstein ratios in double source plane lenses

    """

    def __init__(self, z_lens, z_source_1, z_source_2, beta_dspl, sigma_beta_dspl, normalized=False):
        """
        :param z_lens: lens redshift
        :param z_source_1: redshift of first source
        :param z_source_2: redshift of second source
        :param beta_dspl: measured ratio of Einstein rings theta_E_1 / theta_E_2
        :param sigma_beta_dspl:
        :param normalized: normalize the likelihood
        :type normalized: boolean
        """
        self._z_lens = z_lens
        self._z_source_1 = z_source_1
        self._z_source_2 = z_source_2
        self._beta_dspl = beta_dspl
        self._sigma_beta_dspl = sigma_beta_dspl
        self._normalized = normalized

    def lens_log_likelihood(self, cosmo, kwargs_lens=None, kwargs_kin=None, kwargs_source=None):
        """
        log likelihood of the data given a model

        :param cosmo: astropy.cosmology instance
        :param kwargs_lens: keyword arguments of lens
        :param kwargs_kin: keyword arguments of kinematics
        :param kwargs_source: keyword argument of source
        :return: log likelihood of data given model
        """
        beta_model = self._beta_model(cosmo)
        log_l = - 0.5 * ((beta_model - self._beta_dspl) / self._sigma_beta_dspl)**2
        if self._normalized:
            log_l -= 1 / 2. * np.log(2 * np.pi * self._sigma_beta_dspl**2)
        return log_l

    def num_data(self):
        """

        :return: number of data points
        """
        return 1

    def _beta_model(self, cosmo):
        """
        model prediction of ratio of Einstein radii theta_E_1 / theta_E_2 or scaled deflection angles

        :param cosmo: astropy.cosmology instance
        :return: beta
        """
        beta = beta_double_source_plane(self._z_lens, self._z_source_1, self._z_source_2, cosmo=cosmo)
        return beta


def beta_double_source_plane(z_lens, z_source_1, z_source_2, cosmo):
    """
    model prediction of ratio of Einstein radii theta_E_1 / theta_E_2 or scaled deflection angles

    :param z_lens: lens redshift
    :param z_source_1: source_1 redshift
    :param z_source_2: source_2 redshift
    :param cosmo: astropy.cosmology instance
    :return: beta
    """
    ds1 = cosmo.angular_diameter_distance(z=z_source_1).value
    dds1 = cosmo.angular_diameter_distance_z1z2(z1=z_lens, z2=z_source_1).value
    ds2 = cosmo.angular_diameter_distance(z=z_source_2).value
    dds2 = cosmo.angular_diameter_distance_z1z2(z1=z_lens, z2=z_source_2).value
    beta = dds1 / ds1 * ds2 / dds2
    return beta
