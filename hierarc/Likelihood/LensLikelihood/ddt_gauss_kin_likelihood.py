from hierarc.Likelihood.LensLikelihood.kin_likelihood import KinLikelihood
from hierarc.Likelihood.LensLikelihood.lens_likelihood import TDLikelihoodGaussian


class DdtGaussKinLikelihood(object):

    """
    class for joint kinematics and time delay likelihood assuming that they are independent
    Uses KinLikelihood and DdtHistLikelihood combined
    """
    def __init__(self, z_lens, z_source, ddt_mean, ddt_sigma, sigma_v_measurement, j_model, error_cov_measurement,
                 error_cov_j_sqrt):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param ddt_mean: mean Ddt [Mpc] of time-delay and lens model likelihood
        :param ddt_sigma: 1-sigma uncertainty in Ddt
        :param sigma_v_measurement: numpy array, velocity dispersion measured
        :param j_model: numpy array of the predicted dimensionless dispersion on the IFU's
        :param error_cov_measurement: covariance matrix of the measured velocity dispersions in the IFU's
        :param error_cov_j_sqrt: covariance matrix of sqrt(J) of the model predicted dimensionless dispersion on the JFU's
        """
        self._tdLikelihood = TDLikelihoodGaussian(z_lens, z_source, ddt_mean=ddt_mean, ddt_sigma=ddt_sigma)
        self._kinlikelihood = KinLikelihood(z_lens, z_source, sigma_v_measurement, j_model, error_cov_measurement,
                                            error_cov_j_sqrt)
        self.num_data = self._tdLikelihood.num_data + self._kinlikelihood.num_data

    def log_likelihood(self, ddt, dd, aniso_scaling=None, sigma_v_sys_error=None):
        """

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :param aniso_scaling: numpy array of anisotropy scaling on prediction of Ds/Dds
        :return: log likelihood given the single lens analysis
        """
        lnlikelihood = self._tdLikelihood.log_likelihood(ddt) + self._kinlikelihood.log_likelihood(ddt, dd, aniso_scaling, sigma_v_sys_error=sigma_v_sys_error)
        return lnlikelihood
