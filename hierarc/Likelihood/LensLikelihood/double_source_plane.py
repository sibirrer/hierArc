import numpy as np


class DSPLikelihood(object):
    """Likelihood for Einstein ratios in double source plane lenses."""

    def __init__(
        self,
        beta_dspl,
        sigma_beta_dspl,
        normalized=False,
    ):
        """
        :param beta_dspl: measured ratio of Einstein rings theta_E_1 / theta_E_2
        :param sigma_beta_dspl: 1-sigma uncertainty in the measurement of the Einstein radius ratio
        :param normalized: normalize the likelihood
        :type normalized: boolean
        """
        self._beta_dspl = beta_dspl
        self._sigma_beta_dspl = sigma_beta_dspl
        self._normalized = normalized

    def log_likelihood(
        self,
        beta_dsp,
        gamma_pl=2,
        lambda_mst=1,
    ):
        """Log likelihood of the data given a model.

        :param beta_dsp: scaled deflection angles alpha_1 / alpha_2 as ratio between
            z_source and z_source2 source planes
        :param gamma_pl: power-law density slope of main deflector (=2 being isothermal)
        :param lambda_mst: mass-sheet transform at the main deflector
        :return: log likelihood of data given model
        """
        theta_E_ratio = beta2theta_e_ratio(
            beta_dsp, gamma_pl=gamma_pl, lambda_mst=lambda_mst
        )
        log_l = -0.5 * ((theta_E_ratio - self._beta_dspl) / self._sigma_beta_dspl) ** 2
        if self._normalized:
            log_l -= 1 / 2.0 * np.log(2 * np.pi * self._sigma_beta_dspl**2)
        return log_l

    def num_data(self):
        """

        :return: number of data points
        """
        return 1


def beta_double_source_plane(z_lens, z_source_1, z_source_2, cosmo):
    """Model prediction of ratio of scaled
    deflection angles.

    :param z_lens: lens redshift
    :param z_source_1: source_1 redshift
    :param z_source_2: source_2 redshift
    :param cosmo: ~astropy.cosmology instance
    :return: beta
    """
    ds1 = cosmo.angular_diameter_distance(z=z_source_1).value
    dds1 = cosmo.angular_diameter_distance_z1z2(z1=z_lens, z2=z_source_1).value
    ds2 = cosmo.angular_diameter_distance(z=z_source_2).value
    dds2 = cosmo.angular_diameter_distance_z1z2(z1=z_lens, z2=z_source_2).value
    beta = dds1 / ds1 * ds2 / dds2
    return beta


def beta2theta_e_ratio(beta_dsp, gamma_pl=2, lambda_mst=1):
    """Calculates Einstein radii ratio for a power-law + MST profile with given
    parameters.

    :param beta_dsp: scaled deflection angles alpha_1 / alpha_2 as ratio between
        z_source and z_source2 source planes
    :param gamma_pl: power-law density slope of main deflector (=2 being isothermal)
    :param lambda_mst: mass-sheet transform at the main deflector
    :return: theta_E1 / theta_E2
    """
    return (beta_dsp - (1 - lambda_mst) * (1 - beta_dsp)) ** (1 / (gamma_pl - 1))
