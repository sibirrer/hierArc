from hierarc.Likelihood.hierarchy_likelihood import LensLikelihood
from astropy.cosmology import FlatLambdaCDM
import pytest
import numpy as np
import numpy.testing as npt


class TestLensLikelihood(object):

    def setup(self):
        z_lens = 0.5
        z_source = 1.5
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        dd = self.cosmo.angular_diameter_distance(z=z_lens).value
        ds = self.cosmo.angular_diameter_distance(z=z_source).value
        dds = self.cosmo.angular_diameter_distance_z1z2(z1=z_lens, z2=z_source).value
        ddt = (1. + z_lens) * dd * ds / dds

        ani_param_array = np.linspace(start=0, stop=5, num=10)
        ani_scaling_array = ani_param_array

        kwargs_likelihood = {'ddt_mean': ddt, 'ddt_sigma': ddt/20, 'dd_mean': dd, 'dd_sigma': dd/10}
        self.likelihood = LensLikelihood(z_lens, z_source, name='name', likelihood_type='DdtDdGaussian',
                                    anisotropy_model='OM', ani_param_array=ani_param_array, ani_scaling_array_list=None,
                                    ani_scaling_array=ani_scaling_array, num_distribution_draws=200, kappa_ext_bias=True,
                                    kappa_pdf=None, kappa_bin_edges=None, mst_ifu=True, **kwargs_likelihood)
        self.likelihood_single = LensLikelihood(z_lens, z_source, name='name', likelihood_type='DdtDdGaussian',
                                    anisotropy_model='OM', ani_param_array=ani_param_array, ani_scaling_array_list=None,
                                    ani_scaling_array=ani_scaling_array, num_distribution_draws=200, kappa_ext_bias=False,
                                    kappa_pdf=None, kappa_bin_edges=None, mst_ifu=False, **kwargs_likelihood)
        self.likelihood_zero_dist = LensLikelihood(z_lens, z_source, name='name', likelihood_type='DdtDdGaussian',
                                    anisotropy_model='OM', ani_param_array=ani_param_array, ani_scaling_array_list=None,
                                    ani_scaling_array=ani_scaling_array, num_distribution_draws=0, kappa_ext_bias=True,
                                    kappa_pdf=None, kappa_bin_edges=None, mst_ifu=True, **kwargs_likelihood)

        kappa_posterior = np.random.normal(loc=0, scale=0.03, size=100000)
        kappa_pdf, kappa_bin_edges = np.histogram(kappa_posterior, density=True)
        self.likelihood_kappa_ext = LensLikelihood(z_lens, z_source, name='name', likelihood_type='DdtDdGaussian',
                                         anisotropy_model='OM', ani_param_array=ani_param_array,
                                         ani_scaling_array_list=None,
                                         ani_scaling_array=ani_scaling_array, num_distribution_draws=200,
                                         kappa_ext_bias=True,
                                         kappa_pdf=kappa_pdf, kappa_bin_edges=kappa_bin_edges, mst_ifu=False, **kwargs_likelihood)

    def test_lens_log_likelihood(self):
        np.random.seed(42)
        kwargs_lens = {'lambda_mst': 1, 'lambda_mst_sigma': 0.01, 'kappa_ext': 0, 'kappa_ext_sigma': 0.03,
                       'lambda_ifu': 1, 'lambda_ifu_sigma': 0.01}
        kwargs_kin = {'a_ani': 1, 'a_ani_sigma': 0.1}
        ln_likelihood = self.likelihood.lens_log_likelihood(self.cosmo, kwargs_lens=kwargs_lens, kwargs_kin=kwargs_kin)
        npt.assert_almost_equal(ln_likelihood, -0.5, decimal=1)

        ln_likelihood_zero = self.likelihood_zero_dist.lens_log_likelihood(self.cosmo, kwargs_lens=kwargs_lens, kwargs_kin=kwargs_kin)
        assert ln_likelihood_zero == -np.inf

        ln_likelihood_kappa_ext = self.likelihood_kappa_ext.lens_log_likelihood(self.cosmo, kwargs_lens=kwargs_lens, kwargs_kin=kwargs_kin)
        npt.assert_almost_equal(ln_likelihood, ln_likelihood_kappa_ext, decimal=1)

        kwargs_lens = {'lambda_mst': 1000000, 'lambda_mst_sigma': 0, 'kappa_ext': 0, 'kappa_ext_sigma': 0,
                       'lambda_ifu': 1, 'lambda_ifu_sigma': 0}
        kwargs_kin = {'a_ani': 1, 'a_ani_sigma': 0}
        ln_inf = self.likelihood_single.lens_log_likelihood(self.cosmo, kwargs_lens=kwargs_lens, kwargs_kin=kwargs_kin)
        assert ln_inf < -10000000

        kwargs_test = self.likelihood._kwargs_init(kwargs=None)
        assert type(kwargs_test) is dict


if __name__ == '__main__':
    pytest.main()
