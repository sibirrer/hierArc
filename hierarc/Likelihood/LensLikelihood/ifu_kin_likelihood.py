__author__ = 'sibirrer'
from lenstronomy.Util import constants as const
import numpy as np
from hierarc.Likelihood.anisotropy_scaling import AnisotropyScalingIFU


class IFUKinCov(object):
    """
    likelihood to deal with IFU kinematics constraints with covariances in both the model and measured velocity dispersion
    """
    def __init__(self, z_lens, z_source, sigma_v_measurement, j_mean_list, error_cov_measurement, error_cov_j_sqrt):
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

    def log_likelihood(self, ddt, dd, aniso_scaling=None):
        """
        Note: kinematics + imaging data can constrain Ds/Dds. The input of Ddt, Dd is transformed here to match Ds/Dds

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :return: log likelihood given the single lens analysis
        """
        ds_dds = ddt / dd / (1 + self._z_lens)
        if aniso_scaling is None:
            scaling_ifu = 1
        else:
            scaling_ifu = aniso_scaling
        sigma_v_predict = np.sqrt(ds_dds * scaling_ifu * self._j_mean_list) * const.c / 1000
        delta = self._sigma_v_measured - sigma_v_predict
        scaling_matix = np.outer(np.sqrt(scaling_ifu), np.sqrt(scaling_ifu))
        cov_error = np.linalg.inv(self._error_cov_measurement + self._error_cov_j_sqrt * scaling_matix * ds_dds * (const.c / 1000)**2)
        return -delta.dot(cov_error.dot(delta)) / 2.
