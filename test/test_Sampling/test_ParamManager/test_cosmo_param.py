from hierarc.Sampling.ParamManager.cosmo_param import CosmoParam
import numpy.testing as npt
import pytest
import unittest


class TestCosmoParamFLCDM(object):

    def setup(self):
        cosmology_list = ['FLCDM', "FwCDM", "w0waCDM", "oLCDM", "NONE"]
        kwargs_fixed = {'h0': 70, 'om': 0.3, 'ok': 0., 'w': -1, 'wa': -0, 'w0': -0, 'gamma_ppn': 1}
        self._param_list = []
        self._param_list_fixed = []
        for cosmology in cosmology_list:
            self._param_list.append(CosmoParam(cosmology, ppn_sampling=True, kwargs_fixed=None))
            self._param_list_fixed.append(CosmoParam(cosmology, ppn_sampling=True, kwargs_fixed=kwargs_fixed))
        self.cosmology_list = cosmology_list

    def test_param_list(self):
        num_param_list = [3, 4, 5, 4, 1]  # number of parameters for the cosmological models cosmology_list
        for i, param in enumerate(self._param_list):
            param_list = param.param_list(latex_style=False)
            assert len(param_list) == num_param_list[i]
            param_list = param.param_list(latex_style=True)
            assert len(param_list) == num_param_list[i]
        for i, param in enumerate(self._param_list_fixed):
            param_list = param.param_list(latex_style=False)
            assert len(param_list) == 0
            param_list = param.param_list(latex_style=True)
            assert len(param_list) == 0

    def test_args2kwargs(self):
        kwargs = {'h0': 70, 'om': 0.3, 'ok': 0., 'w': -1, 'wa': -0, 'w0': -0, 'gamma_ppn': 1}
        for i, param in enumerate(self._param_list):
            args = param.kwargs2args(kwargs)
            kwargs_new, i = param.args2kwargs(args, i=0)
            args_new = param.kwargs2args(kwargs_new)
            npt.assert_almost_equal(args_new, args)
        for i, param in enumerate(self._param_list_fixed):
            args = param.kwargs2args(kwargs)
            kwargs_new, i = param.args2kwargs(args, i=0)
            args_new = param.kwargs2args(kwargs_new)
            npt.assert_almost_equal(args_new, args)

    def test_cosmo(self):
        kwargs_cosmo = {'h0': 70, 'om': 0.3, 'ok': 0., 'w': -1, 'wa': -0, 'w0': -0, 'gamma_ppn': 1}
        for i, param in enumerate(self._param_list):
            cosmo = param.cosmo(kwargs_cosmo)
            if self.cosmology_list[i] != "NONE":
                assert hasattr(cosmo, 'H0')


class TestRaise(unittest.TestCase):

    def test_raise(self):

        with self.assertRaises(ValueError):
            param = CosmoParam(cosmology='FLCDM', ppn_sampling=True, kwargs_fixed={})
            param._cosmology = 'bad'
            kwargs_cosmo = {'h0': 70, 'om': 0.3, 'ok': 0., 'w': -1, 'wa': -0, 'w0': -0, 'gamma_ppn': 1}
            param.cosmo(kwargs_cosmo)


if __name__ == '__main__':
    pytest.main()
