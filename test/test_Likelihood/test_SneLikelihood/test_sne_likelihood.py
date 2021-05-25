from hierarc.Likelihood.SneLikelihood.sne_likelihood import SneLikelihood
import pytest
import numpy as np


class TestSnePantheon(object):

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
        z_pivot = 0.1

        # compute luminosity distances
        angular_diameter_distances = cosmo_true.angular_diameter_distance(zcmb).value
        lum_dists_true = (5 * np.log10((1 + zhel) * (1 + zcmb) * angular_diameter_distances))

        angular_diameter_distance_pivot = cosmo_true.angular_diameter_distance(z_pivot).value
        lum_dist_pivot = (5 * np.log10((1 + z_pivot) * (1 + z_pivot) * angular_diameter_distance_pivot))

        # draw from scatter

        sigma_m_z = 0.1
        cov_mag = np.ones((num, num)) * 0.05 ** 2 + np.diag(
            np.ones(num) * 0.1 ** 2)  # full covariance matrix of systematics
        cov_mag_measure = cov_mag + np.diag(np.ones(num) * sigma_m_z ** 2)
        mags = m_apparent + lum_dists_true - lum_dist_pivot

        mag_mean = np.random.multivariate_normal(mags, cov_mag_measure)
        kwargs_sne_likelihood = {'mag_mean': mag_mean, 'cov_mag': cov_mag, 'zhel': zhel, 'zcmb': zcmb}

        self.likelihood = SneLikelihood(sample_name='CUSTOM', **kwargs_sne_likelihood)
        self.lum_dists_true = lum_dists_true
        self.m_apparent_true = m_apparent
        self.sigma_m_true = sigma_m_z
        self.cosmo_true = cosmo_true
        self.z_anchor = z_pivot

    def test_log_likelihood(self):
        logL = self.likelihood.log_likelihood(self.cosmo_true, apparent_m_z=self.m_apparent_true,
                                              sigma_m_z=self.sigma_m_true, z_anchor=self.z_anchor)

        logL_high = self.likelihood.log_likelihood(self.cosmo_true, apparent_m_z=self.m_apparent_true + 0.2,
                                              sigma_m_z=self.sigma_m_true, z_anchor=self.z_anchor)
        assert logL > logL_high

        logL_low = self.likelihood.log_likelihood(self.cosmo_true, apparent_m_z=self.m_apparent_true - 0.2,
                                                   sigma_m_z=self.sigma_m_true, z_anchor=self.z_anchor)
        assert logL > logL_low


if __name__ == '__main__':
    pytest.main()
