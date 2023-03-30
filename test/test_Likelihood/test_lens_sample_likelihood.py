import numpy as np
import pytest
import numpy.testing as npt
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from hierarc.Likelihood.lens_sample_likelihood import LensSampleLikelihood
from astropy.cosmology import FlatLambdaCDM
from hierarc.Likelihood.LensLikelihood.double_source_plane import beta_double_source_plane


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
        self.kwargs_lens_list = [{'z_lens': self.z_L, 'z_source': self.z_S, 'likelihood_type': 'DdtDdKDE',
                                  'dd_samples': self.D_d_samples, 'ddt_samples': self.D_dt_samples,
                                  'kde_type': 'scipy_gaussian', 'bandwidth': 1},
                                 {'z_lens': self.z_L, 'z_source': self.z_S, 'likelihood_type': 'DsDdsGaussian',
                                  'ds_dds_mean': lensCosmo.ds/lensCosmo.dds, 'ds_dds_sigma': 1,
                                  'ani_param_array': ani_param_array, 'ani_scaling_array': ani_scaling_array}]
        self.likelihood = LensSampleLikelihood(kwargs_lens_list=self.kwargs_lens_list)

    def test_log_likelihood(self):
        kwargs_lens = {'kappa_ext': 0, 'gamma_ppn': 1}
        kwargs_kin = {'a_ani': 1}
        logl = self.likelihood.log_likelihood(self.cosmo, kwargs_lens=kwargs_lens, kwargs_kin=kwargs_kin)
        cosmo = FlatLambdaCDM(H0=self.H0_true*0.99, Om0=self.omega_m_true, Ob0=0.05)
        logl_sigma = self.likelihood.log_likelihood(cosmo, kwargs_lens=kwargs_lens, kwargs_kin=kwargs_kin)
        npt.assert_almost_equal(logl - logl_sigma, 0.12, decimal=2)

    def test_num_data(self):
        num_data = self.likelihood.num_data()
        assert num_data == 3

    def test_double_source_plane(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        zl = 0.5
        zs1 = 1
        zs2 = 2
        beta = beta_double_source_plane(z_lens=zl, z_source_1=zs1, z_source_2=zs2, cosmo=cosmo)
        sigma_beta = 0.1

        kwargs_lens_list = [{'z_lens': zl, 'z_source_1': zs1, 'z_source_2': zs2,
                             'beta_dspl': beta, 'sigma_beta_dspl': sigma_beta,
                             'likelihood_type': 'DSPL'}]
        likelihood = LensSampleLikelihood(kwargs_lens_list=kwargs_lens_list)
        log_l = likelihood.log_likelihood(cosmo=cosmo)
        npt.assert_almost_equal(log_l, 0, decimal=5)


if __name__ == '__main__':
    pytest.main()
