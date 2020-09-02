import numpy as np


class MagnificationLikelihood(object):
    """
    likelihood of an unlensed apprarent source magnification given a measurement of the magnified brightness
    This can i.e. be applied to lensed SNIa on the population level

    # TODO: compute likelihood given intrinsic brightness and MST transform of the magnification
    """
    def __init__(self, z_lens, z_source, magnitude_measured, magnitude_measured_cov, magnification_model,
                 magnification_model_cov, normalized=True):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param magnitude_measured: array, brightness measured for each individual image
        :param magnitude_measured_cov: covariance matrix of the measurement
        :param magnification_model: array, mean modeled magnification
        :param magnification_model_cov: covariance matrix of the model uncertainties
        :param normalized: bool, if True, returns the normalized likelihood, if False, separates the constant prefactor
         (in case of a Gaussian 1/(sigma sqrt(2 pi)) ) to compute the reduced chi2 statistics
        """

        self._magnitude_measured = magnitude_measured
        self._magnitude_measured_cov = magnitude_measured_cov
        self._magnification_model = magnification_model
        self._magnification_model_cov = magnification_model_cov
        self._cov_tot = magnification_model_cov + magnitude_measured_cov
        self._normalized = normalized
        self.num_data = len(magnitude_measured)

        try:
            self._cov_tot_inv = np.linalg.inv(self._cov_tot)
            sign_det, self._lndet = np.linalg.slogdet(self._cov_tot)
        except:
            raise ValueError('Error covariance matrix could not be inverted!')

    def log_likelihood(self, mu_intrinsic):
        """

        :param mu_intrinsic: intrinsic brightness of the source (already incorporating the inverse MST transform)
        :return: log likelihood of the measured magnified images given the source brightness
        """
        # compute modeled magnified images
        amp_model = self._magnification_model * mu_intrinsic
        # difference to measurements
        delta = self._magnitude_measured - amp_model
        # evaluate likelihood
        lnlikelihood = -delta.dot(self._cov_tot_inv.dot(delta)) / 2.
        if self._normalized is True:
            lnlikelihood -= 1 / 2. * (self.num_data * np.log(2 * np.pi) + self._lndet)
        return lnlikelihood
