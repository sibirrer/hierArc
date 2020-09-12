import numpy as np


class MagnificationLikelihood(object):
    """
    likelihood of an unlensed apprarent source magnification given a measurement of the magnified brightness
    This can i.e. be applied to lensed SNIa on the population level

    """
    def __init__(self, amp_measured, cov_amp_measured, mag_model, cov_model):
        """

        :param amp_measured: array, amplitudes of measured fluxes of image positions
        :param cov_amp_measured: 2d array, error covariance matrix of the measured amplitudes
        :param mag_model: mean magnification of the model prediction
        :param cov_model: 2d array (image amplitudes); model lensing magnification covariances
        """

        self._data_vector = amp_measured
        self._cov_amp_measured = np.array(cov_amp_measured)
        # check sizes of covariances matches
        n_tot = len(self._data_vector)
        assert n_tot == len(cov_model)
        self._cov_data = self._cov_amp_measured
        self._model_tot = np.array(mag_model)
        self._cov_model = cov_model
        self.num_data = n_tot

    def log_likelihood(self, mu_intrinsic):
        """

        :param mu_intrinsic: intrinsic brightness of the source (already incorporating the inverse MST transform)
        :return: log likelihood of the measured magnified images given the source brightness
        """
        # compute model predicted magnified image amplitude and time delay
        model_vector = mu_intrinsic * self._model_tot
        # scale model covariance matrix with model_scale vector (in quadrature)
        cov_model = self._cov_model * mu_intrinsic**2
        # combine data and model covariance matrix
        cov_tot = self._cov_data + cov_model
        # invert matrix
        try:
            cov_tot_inv = np.linalg.inv(cov_tot)
        except:
            return -np.inf
        # difference to data vector
        delta = self._data_vector - model_vector
        # evaluate likelihood
        lnlikelihood = -delta.dot(cov_tot_inv.dot(delta)) / 2.
        return lnlikelihood

