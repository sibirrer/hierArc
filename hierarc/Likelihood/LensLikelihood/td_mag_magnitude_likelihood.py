import numpy as np
from lenstronomy.Util import constants as const


class TDMagMagnitudeLikelihood(object):
    """
    likelihood of time delays and magnification likelihood

    This likelihood uses astronomical magnitude units in flux measurement and lensing magnification
    and Gaussian uncertainties in this space.

    """

    def __init__(self, time_delay_measured, cov_td_measured, magnitude_measured, cov_magnitude_measured,
                 fermat_diff, magnification_model, cov_model):
        """

        :param time_delay_measured: array, relative time delays (relative to the first image) [days]
        :param cov_td_measured: 2d array, error covariance matrix of time delay measurement [days^2]
        :param magnitude_measured: array, astronomical magnitude of measured fluxes of image positions
        :param cov_magnitude_measured: 2d array, error covariance matrix of the measured amplitudes
        :param fermat_diff: mean Fermat potential differences (relative to the first image) in arcsec^2
        :param magnification_model: mean lensing magnification of the model prediction
         in units of astronomical magnitudes (array of length of the images, (mean of - 2.5 * log10(mu))
        :param cov_model: 2d array (length relative time delays + image magnifications);
         model fermat potential differences and lensing magnification in astronomical magnitudes covariances
        """

        self._data_vector = np.append(time_delay_measured, magnitude_measured)
        self._cov_td_measured = np.array(cov_td_measured)
        self._cov_magnitude_measured = np.array(cov_magnitude_measured)
        # check sizes of covariances matches
        n_tot = len(self._data_vector)
        self._n_td = len(time_delay_measured)
        self._n_amp = len(magnitude_measured)
        assert self._n_td == len(cov_td_measured)
        assert self._n_amp == len(cov_magnitude_measured)
        assert n_tot == len(cov_model)
        # merge data covariance matrices from time delay and image amplitudes
        self._cov_data = np.zeros((n_tot, n_tot))
        self._cov_data[:self._n_td, :self._n_td] = self._cov_td_measured
        self._cov_data[self._n_td:, self._n_td:] = self._cov_magnitude_measured
        # self._fermat_diff = fermat_diff   # in units arcsec^2
        self._fermat_unit_conversion = const.Mpc / const.c / const.day_s * const.arcsec ** 2
        # self._mag_model = mag_model
        self._model_tot = np.append(fermat_diff, magnification_model)
        self._cov_model = cov_model
        self.num_data = n_tot

    def log_likelihood(self, ddt, mu_intrinsic):
        """

        :param ddt: time-delay distance (physical Mpc)
        :param mu_intrinsic: intrinsic brightness of the source (already incorporating the inverse MST transform)
        :return: log likelihood of the measured magnified images given the source brightness
        """
        model_vector, cov_tot = self._model_cov(ddt, mu_intrinsic)
        # invert matrix
        try:
            cov_tot_inv = np.linalg.inv(cov_tot)
        except:
            return -np.inf
        # difference to data vector
        delta = self._data_vector - model_vector
        # evaluate likelihood
        lnlikelihood = -delta.dot(cov_tot_inv.dot(delta)) / 2.
        sign_det, lndet = np.linalg.slogdet(cov_tot)
        lnlikelihood -= 1 / 2. * (self.num_data * np.log(2 * np.pi) + lndet)
        return lnlikelihood

    def _model_cov(self, ddt, mu_intrinsic):
        """
        combined covariance matrix of the data and model when marginialized over the Gaussian model uncertainties
        in the Fermat potential and magnification.

        :param ddt: time-delay distance (physical Mpc)
        :param mu_intrinsic: intrinsic brightness of the source (already incorporating the inverse MST transform)
        :return: model vector, combined covariance matrix
        """
        # compute model predicted magnified image amplitude and time delay

        model_scale = np.append(ddt * self._fermat_unit_conversion * np.ones(self._n_td), np.ones(self._n_amp))
        model_vector = np.zeros_like(self._model_tot)
        # time delay prediction
        model_vector[:self._n_td] = ddt * self._fermat_unit_conversion * self._model_tot[:self._n_td]
        # lensed astronomical magnitude prediction
        model_vector[self._n_td:] = self._model_tot[self._n_td:] + mu_intrinsic
        # scale model covariance matrix with model_scale vector (in quadrature),
        # shift in astronomical magnitudes do not change covariance matrix
        cov_model = model_scale * (self._cov_model * model_scale).T
        # combine data and model covariance matrix
        cov_tot = self._cov_data + cov_model
        return model_vector, cov_tot
