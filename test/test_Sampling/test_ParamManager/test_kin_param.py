from hierarc.Sampling.ParamManager.kin_param import KinParam
import numpy.testing as npt
import pytest


class TestKinParam(object):

    def setup(self):
        self._param = KinParam(anisotropy_sampling=True, anisotropy_model='GOM', distribution_function='GAUSSIAN',
                 sigma_v_systematics=True, kwargs_fixed={})

        self._param_log_scatter = KinParam(anisotropy_sampling=True, anisotropy_model='GOM', distribution_function='GAUSSIAN',
                               sigma_v_systematics=True, log_scatter=True, kwargs_fixed={})

        kwargs_fixed = {'a_ani': 1, 'a_ani_sigma': 0.1, 'beta_inf': 1., 'beta_inf_sigma': 0.2,
                        'sigma_v_sys_error': 0.05}
        self._param_fixed = KinParam(anisotropy_sampling=True, anisotropy_model='GOM', distribution_function='GAUSSIAN',
                                     sigma_v_systematics=True, kwargs_fixed=kwargs_fixed)

    def test_param_list(self):
        param_list = self._param.param_list(latex_style=False)
        assert len(param_list) == 5
        param_list = self._param.param_list(latex_style=True)
        assert len(param_list) == 5

        param_list = self._param_log_scatter.param_list(latex_style=False)
        assert len(param_list) == 5
        param_list = self._param_log_scatter.param_list(latex_style=True)
        assert len(param_list) == 5

        param_list = self._param_fixed.param_list(latex_style=False)
        assert len(param_list) == 0
        param_list = self._param_fixed.param_list(latex_style=True)
        assert len(param_list) == 0

    def test_args2kwargs(self):
        kwargs = {'a_ani': 1, 'a_ani_sigma': 0.1, 'beta_inf': 1., 'beta_inf_sigma': 0.2, 'sigma_v_sys_error': 0.05}
        args = self._param.kwargs2args(kwargs)
        kwargs_new, i = self._param.args2kwargs(args, i=0)
        args_new = self._param.kwargs2args(kwargs_new)
        npt.assert_almost_equal(args_new, args)

        args = self._param_log_scatter.kwargs2args(kwargs)
        kwargs_new, i = self._param_log_scatter.args2kwargs(args, i=0)
        args_new = self._param_log_scatter.kwargs2args(kwargs_new)
        npt.assert_almost_equal(args_new, args)

        args = self._param_fixed.kwargs2args(kwargs)
        kwargs_new, i = self._param_fixed.args2kwargs(args, i=0)
        args_new = self._param_fixed.kwargs2args(kwargs_new)
        npt.assert_almost_equal(args_new, args)


if __name__ == '__main__':
    pytest.main()
