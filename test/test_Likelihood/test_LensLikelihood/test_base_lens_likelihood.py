import numpy as np
import pytest
import numpy.testing as npt
import unittest
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from hierarc.Likelihood.LensLikelihood.base_lens_likelihood import LensLikelihoodBase
from astropy.cosmology import FlatLambdaCDM


class TestLensLikelihood(object):

    def setup(self):
        np.random.seed(seed=41)
        self.z_L = 0.8
        self.z_S = 3.0

        self.H0_true = 70
        self.omega_m_true = 0.3
        self.cosmo = FlatLambdaCDM(H0=self.H0_true, Om0=self.omega_m_true, Ob0=0.05)
        lensCosmo = LensCosmo(self.z_L, self.z_S, cosmo=self.cosmo)
        self.dd_true = lensCosmo.dd
        self.ddt_true = lensCosmo.ddt

        self.sigma_Dd = 100
        self.sigma_Ddt = 100
        num_samples = 10000
        self.ddt_samples = np.random.normal(self.ddt_true, self.sigma_Ddt, num_samples)
        self.dd_samples = np.random.normal(self.dd_true, self.sigma_Dd, num_samples)
        self.kwargs_likelihood = {'z_lens': self.z_L, 'z_source': self.z_S, 'likelihood_type': 'DdtDdKDE',
                                  'dd_sample': self.dd_samples, 'ddt_sample': self.ddt_samples,
                                  'kde_type': 'scipy_gaussian', 'bandwidth': 1}

    def test_log_likelihood(self):
        lens = LensLikelihoodBase(**self.kwargs_likelihood)
        logl = lens.log_likelihood(ddt=self.ddt_true, dd=self.dd_true)
        cosmo = FlatLambdaCDM(H0=self.H0_true*0.99, Om0=self.omega_m_true, Ob0=0.05)
        lensCosmo = LensCosmo(self.z_L, self.z_S, cosmo=cosmo)
        ddt = lensCosmo.ddt
        dd = lensCosmo.dd
        logl_sigma = lens.log_likelihood(ddt, dd)
        npt.assert_almost_equal(logl - logl_sigma, 0.12, decimal=2)


class TestRaise(unittest.TestCase):

    def test_raise(self):

        with self.assertRaises(ValueError):
            LensLikelihoodBase(z_lens=0.5, z_source=2, likelihood_type='BAD')


if __name__ == '__main__':
    pytest.main()
