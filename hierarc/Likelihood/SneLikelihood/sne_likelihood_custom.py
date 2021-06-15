import numpy as np

_twopi = 2 * np.pi


class CustomSneLikelihood(object):
    """
    class method for an arbitrary apparent magnitude likelihood of a Sne sample where the error and systematic
    covariance matrix is described in astronomical magnitude space
    """

    def __init__(self, mag_mean, cov_mag, zhel, zcmb, no_intrinsic_scatter=False):
        """

        :param mag_mean: array of mean astronomical magnitudes of the sample of Sne
        :param cov_mag: error covariance matrix of the magnitudes to result in relative distance moduli
         (including measurement and systematic uncertainties)
        :param zhel: array, heliocentric redshift of the exploding shell
        :param zcmb: array, CMB-corrected redshift of the Sne
        :param no_intrinsic_scatter: boolean; if True, ignores possible additional intrinsic scatter parameters in the
        likelihood and does not require a re-inversion of the covariance matrix at each evaluation of the likelihood
        """
        self.zhel = zhel
        self.zcmb = zcmb
        self.mag = mag_mean
        self._cov_mag = cov_mag
        self._inv_cov_mag_input = np.linalg.inv(cov_mag)
        self.num_sne = len(mag_mean)
        self._no_intrinsic_scatter = no_intrinsic_scatter

    def log_likelihood_lum_dist(self, lum_dists, estimated_scriptm=None, sigma_m_z=None):
        """

        :param lum_dists: numpy array of luminosity distances to the measured supernovae bins
         (units do not matter since normalization is subtracted off for the likelihood)
        :param estimated_scriptm: mean magnitude at lum_dist=0 (optional)
        :param sigma_m_z: 1-sigma scatter in magnitude in the intrinsic SNe brightness distribution not accounted-for
         by the covariance matrix
        :return: log likelihood of the data given the luminosity distances
        """
        cov_mag, inv_cov = self._inverse_covariance_matrix(sigma_m_z)
        pre_vars = cov_mag.diagonal()
        invvars = 1.0 / pre_vars
        wtval = np.sum(invvars)

        # uncertainty weighted estimated normalization of magnitude (maximum likelihood value)
        if estimated_scriptm is None:
            estimated_scriptm = np.sum((self.mag - lum_dists) * invvars) / wtval
        diffmag = self.mag - lum_dists - estimated_scriptm

        lnlikelihood = -diffmag.dot(inv_cov.dot(diffmag)) / 2.
        sign_det, lndet = np.linalg.slogdet(cov_mag)
        lnlikelihood -= 1 / 2. * (self.num_sne * np.log(2 * np.pi) + lndet)
        return lnlikelihood

    def _inverse_covariance_matrix(self, sigma_m_z=None):
        """
        inverse error covariance matrix. Combines redshift uncertainties (to first order) and magnitude uncertainties
        as well as intrinsic scatter uncertainties

        :param sigma_m_z: float, 1-sigma additional intrinsic magnitude uncertainty of the distribution, not
        accounted-for in the original covariance matrix
        :return: covariance matrix, inverse covariance matrix (2d numpy array)
        """
        # here is the option for adding an additional covariance matrix term of the calibration and/or systematic
        # errors in the evolution of the Sne population
        if sigma_m_z is None or self._no_intrinsic_scatter:
            return self._cov_mag, self._inv_cov_mag_input
        # cov_mag_diag = self._cov_mag.diagonal()
        cov_mag = self._cov_mag + np.diag(np.ones(self.num_sne) * sigma_m_z**2)
        # np.fill_diagonal(self._cov_mag, cov_mag_diag + np.ones(self.num_sne) * sigma_m_z**2)
        invcov = np.linalg.inv(cov_mag)
        return cov_mag, invcov
