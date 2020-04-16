import numpy as np
import pytest
import numpy.testing as npt
import unittest
import copy
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from hierarc.Likelihood.lens_sample_likelihood import LensSampleLikelihood
from hierarc.Likelihood.lens_likelihood import LensLikelihoodBase, TDKinLikelihoodKDE
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import lognorm


class TestLensLikelihood(object):

    def setup(self):
        np.random.seed(seed=41)
        self.z_L = 0.8
        self.z_S = 3.0

        self.H0_true = 70
        self.omega_m_true = 0.3
        self.cosmo = FlatLambdaCDM(H0=self.H0_true, Om0=self.omega_m_true, Ob0=0.05)
        lensCosmo = LensCosmo(self.z_L, self.z_S, cosmo=self.cosmo)
        self.Dd_true = lensCosmo.dd
        self.D_dt_true = lensCosmo.ddt

        self.sigma_Dd = 100
        self.sigma_Ddt = 100
        num_samples = 10000
        self.D_dt_samples = np.random.normal(self.D_dt_true, self.sigma_Ddt, num_samples)
        self.D_d_samples = np.random.normal(self.Dd_true, self.sigma_Dd, num_samples)
        ani_param_array = np.linspace(0, 2, 10)
        ani_scaling_array = np.ones_like(ani_param_array)
        self.kwargs_lens_list = [{'z_lens': self.z_L, 'z_source': self.z_S, 'likelihood_type': 'TDKinKDE',
                                  'dd_sample': self.D_d_samples, 'ddt_sample': self.D_dt_samples,
                                  'kde_type': 'scipy_gaussian', 'bandwidth': 1},
                                 {'z_lens': self.z_L, 'z_source': self.z_S, 'likelihood_type': 'KinGaussian',
                                  'ds_dds_mean': lensCosmo.ds/lensCosmo.dds, 'ds_dds_sigma': 1,
                                  'ani_param_array': ani_param_array, 'ani_scaling_array': ani_scaling_array}]

    def test_log_likelihood(self):
        lens = LensSampleLikelihood(kwargs_lens_list=self.kwargs_lens_list)
        kwargs_lens = {'kappa_ext': 0, 'gamma_ppn': 1}
        kwargs_kin = {'a_ani': 1}
        logl = lens.log_likelihood(self.cosmo, kwargs_lens=kwargs_lens, kwargs_kin=kwargs_kin)
        cosmo = FlatLambdaCDM(H0=self.H0_true*0.99, Om0=self.omega_m_true, Ob0=0.05)
        logl_sigma = lens.log_likelihood(cosmo, kwargs_lens=kwargs_lens, kwargs_kin=kwargs_kin)
        npt.assert_almost_equal(logl - logl_sigma, 0.12, decimal=2)

    def test_log_likelihood_interp(self):
        kwargs_lens_list = copy.deepcopy(self.kwargs_lens_list)
        kwargs_lens_list[0]['interpol'] = True
        kwargs_lens = {'kappa_ext': 0, 'gamma_ppn': 1}
        kwargs_kin = {'a_ani': 1}
        lens = LensSampleLikelihood(kwargs_lens_list=self.kwargs_lens_list)
        logl = lens.log_likelihood(self.cosmo, kwargs_lens=kwargs_lens, kwargs_kin=kwargs_kin)
        cosmo = FlatLambdaCDM(H0=self.H0_true * 0.99, Om0=self.omega_m_true, Ob0=0.05)
        logl_sigma = lens.log_likelihood(cosmo, kwargs_lens=kwargs_lens, kwargs_kin=kwargs_kin)
        npt.assert_almost_equal(logl - logl_sigma, 0.12, decimal=2)


class TestTDKinLikelihoodKDE(object):

    def setup(self):
        np.random.seed(seed=41)
        self.z_L = 0.8
        self.z_S = 3.0

        self.H0_true = 70
        self.omega_m_true = 0.3
        self.cosmo = FlatLambdaCDM(H0=self.H0_true, Om0=self.omega_m_true, Ob0=0.05)
        lensCosmo = LensCosmo(self.z_L, self.z_S, cosmo=self.cosmo)
        self.Dd_true = lensCosmo.dd
        self.D_dt_true = lensCosmo.ddt

        self.sigma_Dd = 100
        self.sigma_Ddt = 500
        num_samples = 10000
        self.D_dt_samples = np.random.normal(self.D_dt_true, self.sigma_Ddt, num_samples)
        self.D_d_samples = np.random.normal(self.Dd_true, self.sigma_Dd, num_samples)

        self.kwargs_lens = {'z_lens': self.z_L, 'z_source': self.z_S,
                                  'dd_sample': self.D_d_samples, 'ddt_sample': self.D_dt_samples,
                                  'kde_type': 'scipy_gaussian', 'bandwidth': 1}

    def test_log_likelihood(self):
        tdkin = TDKinLikelihoodKDE(**self.kwargs_lens)
        kwargs_lens = copy.deepcopy(self.kwargs_lens)
        kwargs_lens['interpol'] = True
        tdkin_interp = TDKinLikelihoodKDE(**kwargs_lens)

        delta_ddt = 1
        delta_dd = 200
        logl = tdkin.log_likelihood(ddt=self.D_dt_true+delta_ddt, dd=self.Dd_true+delta_dd)
        logl_interp = tdkin_interp.log_likelihood(ddt=self.D_dt_true+delta_ddt, dd=self.Dd_true+delta_dd)
        print(logl, logl_interp)
        npt.assert_almost_equal(logl, logl_interp, decimal=3)
        logl_interp = tdkin_interp.log_likelihood(ddt=self.D_dt_true + 100000, dd=self.Dd_true+100000)
        #assert logl_interp == -np.inf

class TestTDLikelihoodLogNorm(object):

    def setup(self):
        self.z_L = 0.8
        self.z_S = 3.0
        self.ddt_mu = 3.5
        self.ddt_sigma = 0.2
        self.kwargs_post = {'z_lens': self.z_L, 'z_source': self.z_S, 'ddt_mu': self.ddt_mu, 'ddt_sigma': self.ddt_sigma, 'likelihood_type': 'TDLogNorm'}
        self.ddt_grid = np.arange(1, 10000, 50)
        self.scipy_lognorm = lognorm(scale=np.exp(self.ddt_mu), s=self.ddt_sigma)
        self.ll_object = LensLikelihoodBase(**self.kwargs_post)._lens_type

    def test_log_likelihood(self):
        ll = self.ll_object.log_likelihood(self.ddt_grid)
        scipy_ll = self.scipy_lognorm.logpdf(self.ddt_grid) # with the constant term included
        npt.assert_almost_equal(ll, scipy_ll + 0.5*np.log(2*np.pi), decimal=7)

class TestRaise(unittest.TestCase):

    def test_raise(self):

        with self.assertRaises(ValueError):
            LensLikelihoodBase(z_lens=0.5, z_source=2, likelihood_type='BAD')


if __name__ == '__main__':
    pytest.main()
