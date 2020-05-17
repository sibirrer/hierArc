from hierarc.Likelihood.LensLikelihood.kin_likelihood import KinLikelihood
from hierarc.Likelihood.LensLikelihood.ddt_gauss_likelihood import DdtGaussianLikelihood


class DdtGaussKinLikelihood(object):

    """
    class for joint kinematics and time delay likelihood assuming that they are independent
    Uses KinLikelihood and DdtHistLikelihood combined
    """
    def __init__(self, z_lens, z_source, ddt_mean, ddt_sigma, sigma_v_measurement, j_model, error_cov_measurement,
                 error_cov_j_sqrt, sigma_sys_error_include=False):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param ddt_mean: mean Ddt [Mpc] of time-delay and lens model likelihood
        :param ddt_sigma: 1-sigma uncertainty in Ddt
        :param sigma_v_measurement: numpy array, velocity dispersion measured
        :param j_model: numpy array of the predicted dimensionless dispersion on the IFU's
        :param error_cov_measurement: covariance matrix of the measured velocity dispersions in the IFU's
        :param error_cov_j_sqrt: covariance matrix of sqrt(J) of the model predicted dimensionless dispersion on the IFU's
        :param sigma_sys_error_include: bool, if True will include a systematic error in the velocity dispersion
         measurement (if sampled from), otherwise this sampled value is ignored.
        """
        self._ddt_gauss_likelihood = DdtGaussianLikelihood(z_lens, z_source, ddt_mean=ddt_mean, ddt_sigma=ddt_sigma)
        self._kinlikelihood = KinLikelihood(z_lens, z_source, sigma_v_measurement, j_model, error_cov_measurement,
                                            error_cov_j_sqrt, sigma_sys_error_include=sigma_sys_error_include)
        self.num_data = self._ddt_gauss_likelihood.num_data + self._kinlikelihood.num_data

    def log_likelihood(self, ddt, dd, aniso_scaling=None, sigma_v_sys_error=None, sigma_v_sys_offset=None):
        """

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :param aniso_scaling: array of size of the velocity dispersion measurement or None, scaling of the predicted
         dimensionless quantity J (proportional to sigma_v^2) of the anisotropy model in the sampling relative to the
         anisotropy model used to derive the prediction and covariance matrix in the init of this class.
        :param sigma_v_sys_error: float (optional) added error on the velocity dispersion measurement in quadrature
        :param sigma_v_sys_offset: float (optional) for a fractional systematic offset in the kinematic measurement
         such that sigma_v = sigma_v_measured * (1 + sigma_v_sys_offset)
        :return: log likelihood given the single lens analysis
        """
        lnlikelihood = self._ddt_gauss_likelihood.log_likelihood(ddt) + self._kinlikelihood.log_likelihood(ddt, dd, aniso_scaling,
                                                                                                           sigma_v_sys_error=sigma_v_sys_error,
                                                                                                           sigma_v_sys_offset=sigma_v_sys_offset)
        return lnlikelihood
