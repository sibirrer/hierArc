from hierarc.Likelihood.SneLikelihood.sne_likelihood import SneLikelihood
import os
import unittest
import pytest
import numpy.testing as npt


class TestSnePantheon(object):

    def setup(self):
        self.pantheon_likelihood = SneLikelihood(sample_name='Pantheon_binned')

    def test_import_pantheon(self):

        assert os.path.exists(self.pantheon_likelihood._data_file)
        assert len(self.pantheon_likelihood.zcmb) == 40
        assert len(self.pantheon_likelihood.zhel) == 40
        assert len(self.pantheon_likelihood.mag) == 40
        assert len(self.pantheon_likelihood.mag_var) == 40
        assert len(self.pantheon_likelihood.z_var) == 40

    def test_log_likelihood(self):

        # cosmo instance
        from astropy.cosmology import WMAP9 as cosmo
        logL = self.pantheon_likelihood.log_likelihood(cosmo=cosmo)
        npt.assert_almost_equal(logL, -29.447965959089437, decimal=4)


class TestRaise(unittest.TestCase):

    def test_raise(self):

        with self.assertRaises(ValueError):
            base = SneLikelihood(sample_name='BAD')


if __name__ == '__main__':
    pytest.main()
