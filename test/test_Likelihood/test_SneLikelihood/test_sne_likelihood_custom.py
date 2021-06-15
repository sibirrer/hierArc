from hierarc.Likelihood.SneLikelihood.sne_likelihood_custom import CustomSneLikelihood
import numpy as np
import numpy.testing as npt
import pytest


class TestCustomSneLikelihood(object):

    def setup(self):
        np.random.seed(42)
        # define redshifts
        num = 30  # number of Sne
        zcmb = np.linspace(start=0.01, stop=0.8, num=num)
        zhel = zcmb

        # define cosmology
        from astropy.cosmology import FlatLambdaCDM

        om_mean, om_sigma = 0.284, 0.012
        cosmo_true = FlatLambdaCDM(H0=70, Om0=om_mean)

        # define apparent magnitudes
        m_apparent = 18

        # compute luminosity distances
        angular_diameter_distances = cosmo_true.angular_diameter_distance(zcmb).value
        lum_dists_true = (5 * np.log10((1 + zhel) * (1 + zcmb) * angular_diameter_distances))

        # draw from scatter
        sigma_m_z = 0.1
        cov_mag = np.ones((num, num)) * 0.05**2 + np.diag(np.ones(num) * 0.1**2)  # full covariance matrix of systematics
        cov_mag_measure = cov_mag + np.diag(np.ones(num) * sigma_m_z**2)
        mags = m_apparent + lum_dists_true

        mag_mean = np.random.multivariate_normal(mags, cov_mag_measure)

        self.likelihood = CustomSneLikelihood(mag_mean, cov_mag, zhel, zcmb)
        self.lum_dists_true = lum_dists_true
        self.m_apparent_true = m_apparent
        self.sigma_m_true = sigma_m_z

    def test_log_likelihood_lum_dist(self):

        # check whether truth as input results in a chi2 around 1
        logL = self.likelihood.log_likelihood_lum_dist(self.lum_dists_true, estimated_scriptm=self.m_apparent_true,
                                                       sigma_m_z=self.sigma_m_true)



        cov_mag, inv_cov = self.likelihood._inverse_covariance_matrix(sigma_m_z=self.sigma_m_true)
        sign_det, lndet = np.linalg.slogdet(cov_mag)
        logL_unnorm = logL + 1 / 2. * (len(self.lum_dists_true) * np.log(2 * np.pi) + lndet)
        npt.assert_almost_equal(-logL_unnorm * 2 / len(self.lum_dists_true), 1, decimal=0)

        logL_low = self.likelihood.log_likelihood_lum_dist(self.lum_dists_true, estimated_scriptm=self.m_apparent_true,
                                                           sigma_m_z=0)
        assert logL_low < logL

        logL_high = self.likelihood.log_likelihood_lum_dist(self.lum_dists_true, estimated_scriptm=self.m_apparent_true,
                                                           sigma_m_z=self.sigma_m_true + 0.1)
        assert logL_high < logL

        # check whether estimated apparent magnitude matches with the truth
        logL_no_m = self.likelihood.log_likelihood_lum_dist(self.lum_dists_true, estimated_scriptm=None,
                                                       sigma_m_z=self.sigma_m_true)
        npt.assert_almost_equal(logL, logL_no_m, decimal=1)

        # check whether output with sigma_m_z=0 and =None are identical
        logL_sigma_0 = self.likelihood.log_likelihood_lum_dist(self.lum_dists_true,
                                                               estimated_scriptm=self.m_apparent_true, sigma_m_z=0)
        logL_sigma_none = self.likelihood.log_likelihood_lum_dist(self.lum_dists_true,
                                                               estimated_scriptm=self.m_apparent_true, sigma_m_z=None)
        npt.assert_almost_equal(logL_sigma_0, logL_sigma_none, decimal=5)


if __name__ == '__main__':
    pytest.main()
