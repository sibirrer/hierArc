import unittest
import pytest
import numpy.testing as npt
from hierarc.Sampling.ParamManager.param_manager import ParamManager


class TestParamManager(object):

    def setup(self):
        cosmology_list = ['FLCDM', "FwCDM", "w0waCDM", "oLCDM"]
        kwargs_lower_cosmo = {'h0': 10, 'om': 0., 'ok': -0.5, 'w': -2, 'wa': -1, 'w0': -2, 'gamma_ppn': 0}
        kwargs_lower_lens = {'lambda_mst': 0, 'lambda_mst_sigma': 0.1, 'kappa_ext': -0.2, 'kappa_ext_sigma': 0}
        kwargs_lower_kin = {'a_ani': 0.1, 'a_ani_sigma': 0.1}
        kwargs_lower_source = {'mu_sne': 0, 'sigma_sne': 0}

        kwargs_upper_cosmo = {'h0': 200, 'om': 1, 'ok': 0.5, 'w': 0, 'wa': 1, 'w0': 1, 'gamma_ppn': 5}
        kwargs_upper_lens = {'lambda_mst': 2, 'lambda_mst_sigma': 0.1, 'kappa_ext': 0.2, 'kappa_ext_sigma': 1}
        kwargs_upper_kin = {'a_ani': 0.1, 'a_ani_sigma': 0.1}
        kwargs_upper_source = {'mu_sne': 100, 'sigma_sne': 10}

        kwargs_fixed_cosmo = {'h0': 70, 'om': 0.3, 'ok': 0., 'w': -1, 'wa': -0, 'w0': -0, 'gamma_ppn': 1}
        kwargs_fixed_lens = {'lambda_mst': 1, 'lambda_mst_sigma': 0.1, 'kappa_ext': 0, 'kappa_ext_sigma': 0}
        kwargs_fixed_kin = {'a_ani': 0.1, 'a_ani_sigma': 0.1}
        kwargs_fixed_source = {'mu_sne': 1, 'sigma_sne': 0.1}

        param_list = []
        for cosmology in cosmology_list:
            param_list.append(ParamManager(cosmology=cosmology, ppn_sampling=True, lambda_mst_sampling=True,
                         lambda_mst_distribution='GAUSSIAN', anisotropy_distribution='GAUSSIAN',
                         anisotropy_sampling=True, anisotropy_model='OM', kwargs_lower_cosmo=kwargs_lower_cosmo,
                                           kappa_ext_sampling=True, kappa_ext_distribution='GAUSSIAN',
                                           sne_apparent_m_sampling=True, sne_distribution='GAUSSIAN',
                         kwargs_upper_cosmo=kwargs_upper_cosmo,
                         kwargs_fixed_cosmo=kwargs_fixed_cosmo, kwargs_lower_lens=kwargs_lower_lens,
                         kwargs_upper_lens=kwargs_upper_lens, kwargs_fixed_lens=kwargs_fixed_lens,
                         kwargs_lower_kin=kwargs_lower_kin, kwargs_upper_kin=kwargs_upper_kin,
                                           kwargs_fixed_kin=kwargs_fixed_kin,
                                           kwargs_fixed_source=kwargs_fixed_source, kwargs_lower_source=kwargs_lower_source,
                                           kwargs_upper_source=kwargs_upper_source))

            param_list.append(ParamManager(cosmology=cosmology, ppn_sampling=True, lambda_mst_sampling=True,
                                           lambda_mst_distribution='GAUSSIAN', anisotropy_distribution='GAUSSIAN',
                         anisotropy_sampling=True, anisotropy_model='OM', kappa_ext_sampling=True, kappa_ext_distribution='GAUSSIAN',
                                           sne_apparent_m_sampling=True, sne_distribution='GAUSSIAN',
                                           kwargs_lower_cosmo=kwargs_lower_cosmo,
                         kwargs_upper_cosmo=kwargs_upper_cosmo,
                         kwargs_fixed_cosmo=None, kwargs_lower_lens=kwargs_lower_lens,
                         kwargs_upper_lens=kwargs_upper_lens, kwargs_fixed_lens=None,
                         kwargs_lower_kin=kwargs_lower_kin, kwargs_upper_kin=kwargs_upper_kin,
                                           kwargs_fixed_kin=None, kwargs_lower_source=kwargs_lower_source,
                                           kwargs_upper_source=kwargs_upper_source, kwargs_fixed_source=None))
        self.param_list = param_list

    def test_num_param(self):
        list = self.param_list[0].param_list(latex_style=False)
        assert len(list) == 0
        num = self.param_list[1].num_param
        assert num == 11
        for param in self.param_list:
            list = param.param_list(latex_style=True)
            list = param.param_list(latex_style=False)

    def test_kwargs2args(self):
        kwargs_cosmo = {'h0': 70, 'om': 0.3, 'ok': 0., 'w': -1, 'wa': -0, 'w0': -0, 'gamma_ppn': 1}
        kwargs_lens = {'lambda_mst': 1, 'lambda_mst_sigma': 0, 'kappa_ext': 0, 'kappa_ext_sigma': 0}
        kwargs_kin = {'a_ani': 1, 'a_ani_sigma': 0.3}
        kwargs_source = {'mu_sne': 2, 'sigma_sne': 0.2}
        for param in self.param_list:
            args = param.kwargs2args(kwargs_cosmo=kwargs_cosmo, kwargs_lens=kwargs_lens, kwargs_kin=kwargs_kin,
                                     kwargs_source=kwargs_source)
            kwargs_cosmo_new, kwargs_lens_new, kwargs_kin_new, kwargs_source_new = param.args2kwargs(args)
            args_new = param.kwargs2args(kwargs_cosmo_new, kwargs_lens_new, kwargs_kin_new, kwargs_source_new)
            npt.assert_almost_equal(args_new, args)

    def test_cosmo(self):
        kwargs_cosmo = {'h0': 70, 'om': 0.3, 'ok': 0., 'w': -1, 'wa': -0, 'w0': -0, 'gamma_ppn': 1}
        for param in self.param_list:
            cosmo = param.cosmo(kwargs_cosmo)
            assert hasattr(cosmo, 'H0')

    def test_param_bounds(self):
        lower_limit, upper_limit = self.param_list[0].param_bounds
        assert len(lower_limit) == 0
        lower_limit, upper_limit = self.param_list[1].param_bounds
        print(self.param_list[1].param_list())
        assert len(lower_limit) == 11


class TestRaise(unittest.TestCase):

    def test_raise(self):

        with self.assertRaises(ValueError):
            ParamManager(cosmology='wrong', ppn_sampling=False, lambda_mst_sampling=False, lambda_mst_distribution='delta',
                 anisotropy_sampling=False, anisotropy_model='OM', kwargs_lower_cosmo=None, kwargs_upper_cosmo=None,
                 kwargs_fixed_cosmo={}, kwargs_lower_lens=None, kwargs_upper_lens=None, kwargs_fixed_lens={},
                 kwargs_lower_kin=None, kwargs_upper_kin=None, kwargs_fixed_kin={})


if __name__ == '__main__':
    pytest.main()
