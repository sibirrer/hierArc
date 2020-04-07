__author__ = 'sibirrer'
from lenstronomy.Util import constants as const
import numpy as np


class KinLikelihood(object):
    """
    likelihood to deal with IFU kinematics constraints with covariances in both the model and measured velocity dispersion
    """
    def __init__(self, z_lens, z_source, sigma_v_measurement, j_model, error_cov_measurement, error_cov_j_sqrt):
        """

        :param z_lens:
        :param z_source:
        :param sigma_v_measurement: numpy array, velocity dispersion measured
        :param j_model: numpy array of the predicted dimensionless dispersion on the IFU's
        :param error_cov_measurement: covariance matrix of the measured velocity dispersions in the IFU's
        :param error_cov_j_sqrt: covariance matrix of sqrt(J) of the model predicted dimensionless dispersion on the JFU's
        """
        self._z_lens = z_lens
        self._j_model = j_model
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
        sigma_v_predict = self.sigma_v_model(ds_dds, scaling_ifu)
        delta = self._sigma_v_measured - sigma_v_predict
        cov_error = np.linalg.inv(self._error_cov_measurement + self.cov_error_model(ds_dds, scaling_ifu))
        return -delta.dot(cov_error.dot(delta)) / 2.

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
