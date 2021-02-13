import numpy as np
from lenstronomy.Util.data_util import magnitude2cps


class MagnificationLikelihood(object):
    """
    likelihood of an unlensed apprarent source magnification given a measurement of the magnified brightness
    This can i.e. be applied to lensed SNIa on the population level

    """
    def __init__(self, amp_measured, cov_amp_measured, magnification_model, cov_magnification_model,
                 magnitude_zero_point=20):
        """

        :param amp_measured: array, amplitudes of measured fluxes of image positions
        :param cov_amp_measured: 2d array, error covariance matrix of the measured amplitudes, in linear space
        for given magnitude zero point
        :param magnitude_zero_point: magnitude zero point for which the image amplitudes and covariance matrix are
        defined
        :param magnification_model: mean magnification of the model prediction (array with number of images)
        :param cov_magnification_model: 2d array (image amplitudes); model lensing magnification covariances
        """

        self._amp_measured = amp_measured
        self._cov_amp_measured = np.array(cov_amp_measured)
        # check sizes of covariances matches
        n_tot = len(self._amp_measured)
        assert n_tot == len(cov_magnification_model)
        self._mean_magnification_model = np.array(magnification_model)
        self._cov_magnification_model = np.array(cov_magnification_model)
        self.num_data = n_tot
        self._magnitude_zero_point = magnitude_zero_point

    def log_likelihood(self, mu_intrinsic):
        """

        :param mu_intrinsic: intrinsic brightness of the source (already incorporating the inverse MST transform)
        :return: log likelihood of the measured magnified images given the source brightness
        """
        model_vector, cov_tot = self._scale_model(mu_intrinsic)
        # invert matrix
        try:
            cov_tot_inv = np.linalg.inv(cov_tot)
        except:
            return -np.inf
        # difference to data vector
        delta = self._amp_measured - model_vector
        # evaluate likelihood
        lnlikelihood = -delta.dot(cov_tot_inv.dot(delta)) / 2.
        sign_det, lndet = np.linalg.slogdet(cov_tot)
        lnlikelihood -= 1 / 2. * (self.num_data * np.log(2 * np.pi) + lndet)
        return lnlikelihood

    def _scale_model(self, mu_intrinsic):
        """

        :param mu_intrinsic: intrinsic brightness of the source (already incorporating the inverse MST transform)
        :return:
        """
        amp_intrinsic = magnitude2cps(magnitude=mu_intrinsic, magnitude_zero_point=self._magnitude_zero_point)
        # compute model predicted magnified image amplitude and time delay
        model_vector = amp_intrinsic * self._mean_magnification_model
        # scale model covariance matrix with model_scale vector (in quadrature)
        cov_model = self._cov_magnification_model * amp_intrinsic ** 2
        # combine data and model covariance matrix
        cov_tot = self._cov_amp_measured + cov_model
        return model_vector, cov_tot

