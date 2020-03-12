__author__ = 'sibirrer'
from lenstronomy.Util import constants as const
import numpy as np
from hierarc.Likelihood.anisotropy_scaling import AnisotropyScalingIFU


class IFUKinCov(AnisotropyScalingIFU):
    """
    likelihood to deal with IFU kinematics constraints with covariances in both the model and measured velocity dispersion
    """
    def __init__(self, z_lens, z_source, sigma_v_measurement, j_mean_list, error_cov_measurement, error_cov_j_sqrt,
                 ani_param_array, ani_scaling_array_list):
        """

        :param z_lens:
        :param z_source:
        :param sigma_v_measurement: numpy array, velocity dispersion measured
        :param j_mean_list: numpy array of the predicted dimensionless dispersion on the IFU's
        :param error_cov_measurement: covariance matrix of the measured velocity dispersions in the IFU's
        :param error_cov_j_sqrt: covariance matrix of sqrt(J) of the model predicted dimensionless dispersion on the JFU's
        :param ani_param_array:
        :param ani_scaling_array_list:
        """
        self._z_lens = z_lens
        self._j_mean_list = j_mean_list
        self._sigma_v_measured = sigma_v_measurement
        self._error_cov_measurement = error_cov_measurement
        self._error_cov_j_sqrt = error_cov_j_sqrt
        AnisotropyScalingIFU.__init__(self, ani_param_array=ani_param_array, ani_scaling_array_list=ani_scaling_array_list)

    def log_likelihood(self, ddt, dd, aniso_param_array=None):
        """
        Note: kinematics + imaging data can constrain Ds/Dds. The input of Ddt, Dd is transformed here to match Ds/Dds

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :return: log likelihood given the single lens analysis
        """
        ds_dds = ddt / dd / (1 + self._z_lens)
        scaling_ifu = self.ani_scaling(aniso_param_array)
        sigma_v_predict = np.sqrt(ds_dds * scaling_ifu * self._j_mean_list) * const.c / 1000
        delta = self._sigma_v_measured - sigma_v_predict
        scaling_matix = np.outer(np.sqrt(scaling_ifu), np.sqrt(scaling_ifu))
        cov_error = np.linalg.inv(self._error_cov_measurement + self._error_cov_j_sqrt * scaling_matix * ds_dds * (const.c / 1000)**2)
        return -delta.dot(cov_error.dot(delta)) / 2.
