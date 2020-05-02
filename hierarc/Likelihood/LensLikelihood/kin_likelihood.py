__author__ = 'sibirrer'
from lenstronomy.Util import constants as const
import numpy as np


class KinLikelihood(object):
    """
    likelihood to deal with IFU kinematics constraints with covariances in both the model and measured velocity dispersion
    """
    def __init__(self, z_lens, z_source, sigma_v_measurement, j_model, error_cov_measurement, error_cov_j_sqrt,
                 normalized=True, sigma_sys_error_include=False):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param sigma_v_measurement: numpy array, velocity dispersion measured
        :param j_model: numpy array of the predicted dimensionless dispersion on the IFU's
        :param error_cov_measurement: covariance matrix of the measured velocity dispersions in the IFU's
        :param error_cov_j_sqrt: covariance matrix of sqrt(J) of the model predicted dimensionless dispersion on the IFU's
        :param normalized: bool, if True, returns the normalized likelihood, if False, separates the constant prefactor
         (in case of a Gaussian 1/(sigma sqrt(2 pi)) ) to compute the reduced chi2 statistics
        :param sigma_sys_error_include: bool, if True will include a systematic error in the velocity dispersion
         measurement (if sampled from), otherwise this sampled value is ignored.
        """
        self._z_lens = z_lens
        self._j_model = j_model
        self._sigma_v_measured = sigma_v_measurement
        self._error_cov_measurement = error_cov_measurement
        self._error_cov_j_sqrt = error_cov_j_sqrt
        self.num_data = len(j_model)
        self._normalized = normalized
        self._sigma_sys_error_include = sigma_sys_error_include

    def log_likelihood(self, ddt, dd, aniso_scaling=None, sigma_v_sys_error=None, sigma_v_pert=None):
        """
        Note: kinematics + imaging data can constrain Ds/Dds. The input of Ddt, Dd is transformed here to match Ds/Dds

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :param sigma_v_sys_error: float (optional) added error on the velocity dispersion measurement in quadrature
        :return: log likelihood given the single lens analysis
        """
        ds_dds = np.maximum(ddt / dd / (1 + self._z_lens), 0)
        if aniso_scaling is None:
            scaling_ifu = 1
        else:
            scaling_ifu = aniso_scaling
        sigma_v_predict = self.sigma_v_model(ds_dds, scaling_ifu)
        delta = self.sigma_v_measured(sigma_v_pert) - sigma_v_predict
        cov_error = self.cov_error_measurement(sigma_v_sys_error) + self.cov_error_model(ds_dds, scaling_ifu)
        cov_error_inv = np.linalg.inv(cov_error)
        lnlikelihood = -delta.dot(cov_error_inv.dot(delta)) / 2.
        if self._normalized is True:
            sign_det, lndet = np.linalg.slogdet(cov_error)
            lnlikelihood -= 1 / 2. * (self.num_data * np.log(2 * np.pi) + lndet)
        return lnlikelihood

    def sigma_v_measured(self, sigma_v_pert=None):
        """

        :param sigma_v_pert: relative error in the measurement
        :return: corrected measured velocity dispersion
        """
        if sigma_v_pert is None:
            return self._sigma_v_measured
        else:
            return self._sigma_v_measured * (1 + sigma_v_pert)

    def sigma_v_model(self, ds_dds, aniso_scaling=1):
        """
        model predicted velocity dispersion for the iFU's

        :param ds_dds: Ds/Dds
        :param aniso_scaling: scaling of the anisotropy affecting sigma_v^2
        :return: array of predicted velocity dispersions
        """
        sigma_v_predict = np.sqrt(ds_dds * aniso_scaling * self._j_model) * const.c / 1000
        return sigma_v_predict

    def cov_error_model(self, ds_dds, scaling_ifu=1):
        """

        :param ds_dds: Ds/Dds
        :param scaling_ifu: scaling of the anisotropy affecting sigma_v^2
        :return: covariance matrix of the error in the predicted model (from mass model uncertainties)
        """
        scaling_matix = np.outer(np.sqrt(scaling_ifu), np.sqrt(scaling_ifu))
        return self._error_cov_j_sqrt * scaling_matix * ds_dds * (const.c / 1000)**2

    def cov_error_measurement(self, sigma_v_sys_error=None):
        """

        :param sigma_v_sys_error: float (optional) added error on the velocity dispersion measurement in quadrature
        :return: error covariance matrix of the velocity dispersion measurements
        """
        if self._sigma_sys_error_include and sigma_v_sys_error is not None:
            return self._error_cov_measurement + np.outer(self._sigma_v_measured * sigma_v_sys_error,
                                                          self._sigma_v_measured * sigma_v_sys_error)
        else:
            return self._error_cov_measurement
