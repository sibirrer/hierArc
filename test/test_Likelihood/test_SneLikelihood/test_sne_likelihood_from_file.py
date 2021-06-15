import numpy.testing as npt
import pytest
import os
import numpy as np
from astropy.cosmology import FlatLambdaCDM

from hierarc.Likelihood.SneLikelihood.sne_likelihood_from_file import SneLikelihoodFromFile
from hierarc.Likelihood.SneLikelihood.sne_likelihood import SneLikelihood


class TestSneLikelihoodFromFile(object):

    def setup(self):
        self.pantheon_binned_likelihood = SneLikelihoodFromFile(sample_name='Pantheon_binned')
        self.pantheon_full_likelihood = SneLikelihoodFromFile(sample_name='Pantheon')
        self.zcmb_bin = self.pantheon_binned_likelihood.zcmb
        self.zcmb_full = self.pantheon_full_likelihood.zcmb
        self.zhel_bin = self.pantheon_binned_likelihood.zhel
        self.zhel_full = self.pantheon_full_likelihood.zhel

    def test_import_pantheon(self):

        assert os.path.exists(self.pantheon_binned_likelihood._data_file)
        assert len(self.pantheon_binned_likelihood.zcmb) == 40
        assert len(self.pantheon_binned_likelihood.zhel) == 40
        assert len(self.pantheon_binned_likelihood.mag) == 40
        assert len(self.pantheon_binned_likelihood.mag_var) == 40
        assert len(self.pantheon_binned_likelihood.z_var) == 40

        assert os.path.exists(self.pantheon_full_likelihood._data_file)
        assert len(self.pantheon_full_likelihood.zcmb) == 1048
        assert len(self.pantheon_full_likelihood.zhel) == 1048
        assert len(self.pantheon_full_likelihood.mag) == 1048
        assert len(self.pantheon_full_likelihood.mag_var) == 1048
        assert len(self.pantheon_full_likelihood.z_var) == 1048

    def test_log_likelihood_lum_dist(self):
        from astropy.cosmology import WMAP9 as cosmo
        angular_diameter_distances = cosmo.angular_diameter_distance(self.zcmb_bin).value
        lum_dists_binned = (5 * np.log10((1 + self.zhel_bin) * (1 + self.zcmb_bin) * angular_diameter_distances))

        # here we test some default floatings
        logL = self.pantheon_binned_likelihood.log_likelihood_lum_dist(lum_dists_binned, estimated_scriptm=None)
        npt.assert_almost_equal(logL, -29.447965959089437, decimal=4)

        angular_diameter_distances = cosmo.angular_diameter_distance(self.zcmb_full).value
        lum_dists_full = (5 * np.log10((1 + self.zhel_full) * (1 + self.zcmb_full) * angular_diameter_distances))
        logL = self.pantheon_full_likelihood.log_likelihood_lum_dist(lum_dists_full, estimated_scriptm=None)
        npt.assert_almost_equal(logL, -490.7138344241936, decimal=4)

        # here we use the apparent magnitude at z=0.1 as part of the likelihood. We are using the best fit value and
        # demand the same outcome as having solved for it.
        z_pivot = 0.1
        angular_diameter_distance_pivot = cosmo.angular_diameter_distance(z_pivot).value
        lum_dist_pivot = (5 * np.log10((1 + z_pivot) * (1 + z_pivot) * angular_diameter_distance_pivot))
        apparent_mag_sne_z01 = 18.963196264371216

        logL_with_mag = self.pantheon_full_likelihood.log_likelihood_lum_dist(lum_dists_full - lum_dist_pivot, estimated_scriptm=apparent_mag_sne_z01)
        npt.assert_almost_equal(logL_with_mag / logL, 1, decimal=3)

        # and here we test that if we change the apparent magnitude, the likelihood gets off
        logL_with_mag = self.pantheon_full_likelihood.log_likelihood_lum_dist(lum_dists_full - lum_dist_pivot, estimated_scriptm=apparent_mag_sne_z01 + 10)
        assert logL_with_mag / logL > 20

    def test_log_likelihood_cosmo(self):
        pantheon_binned_likelihood = SneLikelihood(sample_name='Pantheon_binned')
        pantheon_full_likelihood = SneLikelihood(sample_name='Pantheon')

        # cosmo instance

        # here we demand the 1-sigma difference in the Om constraints to be reflected in the likelihood
        # for the binned data (no systematics!!!) Scolnic et al. 2018 gets 0.284 ± 0.012 in FLCDM
        om_mean, om_sigma = 0.284, 0.012
        cosmo_mean = FlatLambdaCDM(H0=70, Om0=om_mean)
        logL_mean = pantheon_binned_likelihood.log_likelihood(cosmo=cosmo_mean)
        cosmo_sigma_plus = FlatLambdaCDM(H0=70, Om0=om_mean+om_sigma)
        logL_sigma_plus = pantheon_binned_likelihood.log_likelihood(cosmo=cosmo_sigma_plus)
        npt.assert_almost_equal(logL_sigma_plus - logL_mean, -1/2., decimal=1)
        cosmo_sigma_neg = FlatLambdaCDM(H0=70, Om0=om_mean - om_sigma)
        logL_sigma_neg = pantheon_binned_likelihood.log_likelihood(cosmo=cosmo_sigma_neg)
        npt.assert_almost_equal(logL_sigma_neg - logL_mean, -1 / 2., decimal=1)

        # for the full sample, including systematics, Scolnic et al. 2018 gets 0.298 ± 0.022 in FLCDM
        om_mean, om_sigma = 0.298, 0.022
        cosmo_mean = FlatLambdaCDM(H0=70, Om0=om_mean)
        logL_mean = pantheon_full_likelihood.log_likelihood(cosmo=cosmo_mean)
        cosmo_sigma_plus = FlatLambdaCDM(H0=70, Om0=om_mean + om_sigma)
        logL_sigma_plus = pantheon_full_likelihood.log_likelihood(cosmo=cosmo_sigma_plus)
        npt.assert_almost_equal(logL_sigma_plus - logL_mean, -1 / 2., decimal=1)
        cosmo_sigma_neg = FlatLambdaCDM(H0=70, Om0=om_mean - om_sigma)
        logL_sigma_neg = pantheon_full_likelihood.log_likelihood(cosmo=cosmo_sigma_neg)
        npt.assert_almost_equal(logL_sigma_neg - logL_mean, -1 / 2., decimal=1)

    def test_roman_forecast(self):
        roman_binned_likelihood = SneLikelihood(sample_name='Roman_forecast')
        om_mean, om_sigma = 0.284, 0.012
        cosmo_mean = FlatLambdaCDM(H0=70, Om0=om_mean)
        logL_mean = roman_binned_likelihood.log_likelihood(cosmo=cosmo_mean)
        npt.assert_almost_equal(logL_mean, -22.395743577818184, decimal=3)

    def test_raise(self):
        npt.assert_raises(ValueError, SneLikelihoodFromFile, sample_name='BAD')


if __name__ == '__main__':
    pytest.main()
