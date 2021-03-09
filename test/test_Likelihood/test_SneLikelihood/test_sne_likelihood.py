from hierarc.Likelihood.SneLikelihood.sne_likelihood import SneLikelihood
import os
import unittest
import pytest
import numpy.testing as npt


class TestSnePantheon(object):

    def setup(self):
        self.pantheon_binned_likelihood = SneLikelihood(sample_name='Pantheon_binned')
        self.pantheon_full_likelihood = SneLikelihood(sample_name='Pantheon')

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

    def test_log_likelihood(self):

        # cosmo instance
        from astropy.cosmology import WMAP9 as cosmo
        from astropy.cosmology import FlatLambdaCDM

        # here we test some default floatings
        logL = self.pantheon_binned_likelihood.log_likelihood(cosmo=cosmo)
        npt.assert_almost_equal(logL, -29.447965959089437, decimal=4)

        logL = self.pantheon_full_likelihood.log_likelihood(cosmo=cosmo)
        npt.assert_almost_equal(logL, -490.7138344241936, decimal=4)

        # here we use the apparent magnitude at z=0.1 as part of the likelihood. We are using the best fit value and
        # demand the same outcome as having solved for it.
        apparent_mag_sne_z01 = 18.963196264371216
        logL_with_mag = self.pantheon_full_likelihood.log_likelihood(cosmo=cosmo, apparent_m_z=apparent_mag_sne_z01)
        npt.assert_almost_equal(logL_with_mag /logL, 1, decimal=3)

        # and here we test that if we change the apparent magnitude, the likelihood gets off
        logL_with_mag = self.pantheon_full_likelihood.log_likelihood(cosmo=cosmo, apparent_m_z=apparent_mag_sne_z01 + 10)
        assert logL_with_mag / logL > 20

        # here we demand the 1-sigma difference in the Om constraints to be reflected in the likelihood
        # for the binned data (no systematics!!!) Scolnic et al. 2018 gets 0.284 ± 0.012 in FLCDM
        om_mean, om_sigma = 0.284, 0.012
        cosmo_mean = FlatLambdaCDM(H0=70, Om0=om_mean)
        logL_mean = self.pantheon_binned_likelihood.log_likelihood(cosmo=cosmo_mean)
        cosmo_sigma_plus = FlatLambdaCDM(H0=70, Om0=om_mean+om_sigma)
        logL_sigma_plus = self.pantheon_binned_likelihood.log_likelihood(cosmo=cosmo_sigma_plus)
        npt.assert_almost_equal(logL_sigma_plus - logL_mean, -1/2., decimal=1)
        cosmo_sigma_neg = FlatLambdaCDM(H0=70, Om0=om_mean - om_sigma)
        logL_sigma_neg = self.pantheon_binned_likelihood.log_likelihood(cosmo=cosmo_sigma_neg)
        npt.assert_almost_equal(logL_sigma_neg - logL_mean, -1 / 2., decimal=1)

        # for the full sample, including systematics, Scolnic et al. 2018 gets 0.298 ± 0.022 in FLCDM
        om_mean, om_sigma = 0.298, 0.022
        cosmo_mean = FlatLambdaCDM(H0=70, Om0=om_mean)
        logL_mean = self.pantheon_full_likelihood.log_likelihood(cosmo=cosmo_mean)
        cosmo_sigma_plus = FlatLambdaCDM(H0=70, Om0=om_mean + om_sigma)
        logL_sigma_plus = self.pantheon_full_likelihood.log_likelihood(cosmo=cosmo_sigma_plus)
        npt.assert_almost_equal(logL_sigma_plus - logL_mean, -1 / 2., decimal=1)
        cosmo_sigma_neg = FlatLambdaCDM(H0=70, Om0=om_mean - om_sigma)
        logL_sigma_neg = self.pantheon_full_likelihood.log_likelihood(cosmo=cosmo_sigma_neg)
        npt.assert_almost_equal(logL_sigma_neg - logL_mean, -1 / 2., decimal=1)


class TestRaise(unittest.TestCase):

    def test_raise(self):

        with self.assertRaises(ValueError):
            base = SneLikelihood(sample_name='BAD')


if __name__ == '__main__':
    pytest.main()
