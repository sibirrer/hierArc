from hierarc.Sampling.ParamManager.lens_param import LensParam
import numpy.testing as npt


class TestLensParam(object):
    def setup_method(self):
        self._param = LensParam(
            lambda_mst_sampling=True,
            lambda_mst_distribution="GAUSSIAN",
            lambda_ifu_sampling=True,
            lambda_ifu_distribution="GAUSSIAN",
            alpha_lambda_sampling=True,
            beta_lambda_sampling=True,
            kwargs_fixed={},
            gamma_pl_num=2,
        )

        kwargs_fixed = {
            "lambda_mst": 1,
            "lambda_mst_sigma": 0.1,
            "lambda_ifu": 1.1,
            "lambda_ifu_sigma": 0.2,
            "alpha_lambda": 0,
            "beta_lambda": 0,
        }
        self._param_fixed = LensParam(
            lambda_mst_sampling=True,
            lambda_mst_distribution="GAUSSIAN",
            lambda_ifu_sampling=True,
            lambda_ifu_distribution="GAUSSIAN",
            alpha_lambda_sampling=True,
            beta_lambda_sampling=True,
            kwargs_fixed=kwargs_fixed,
        )
        self._param_log_scatter = LensParam(
            lambda_mst_sampling=True,
            lambda_mst_distribution="GAUSSIAN",
            lambda_ifu_sampling=True,
            lambda_ifu_distribution="GAUSSIAN",
            alpha_lambda_sampling=True,
            beta_lambda_sampling=True,
            log_scatter=True,
            kwargs_fixed={},
        )

    def test_param_list(self):
        param_list = self._param.param_list(latex_style=False)
        assert len(param_list) == 8
        param_list = self._param.param_list(latex_style=True)
        assert len(param_list) == 8

        param_list = self._param_log_scatter.param_list(latex_style=False)
        assert len(param_list) == 6
        param_list = self._param_log_scatter.param_list(latex_style=True)
        assert len(param_list) == 6

        param_list = self._param_fixed.param_list(latex_style=False)
        assert len(param_list) == 0
        param_list = self._param_fixed.param_list(latex_style=True)
        assert len(param_list) == 0

    def test_args2kwargs(self):
        kwargs = {
            "lambda_mst": 1.1,
            "lambda_mst_sigma": 0.1,
            "lambda_ifu": 1.1,
            "lambda_ifu_sigma": 0.2,
            "kappa_ext": 0.01,
            "kappa_ext_sigma": 0.03,
            "alpha_lambda": 0.1,
            "beta_lambda": 0.1,
            "gamma_pl_list": [2, 2.5],
        }
        args = self._param.kwargs2args(kwargs)
        print(args, "test args")
        kwargs_new, i = self._param.args2kwargs(args, i=0)
        print(kwargs_new, "test kwargs_new")
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


class TestLensParamGammaInnerM2l(object):
    def setup_method(self):
        self._param = LensParam(
            gamma_in_sampling=True,
            gamma_in_distribution="GAUSSIAN",
            log_m2l_sampling=True,
            log_m2l_distribution="GAUSSIAN",
            alpha_gamma_in_sampling=True,
            alpha_log_m2l_sampling=True,
            kwargs_fixed={},
            log_scatter=False,
        )

        kwargs_fixed = {
            "gamma_in": 1,
            "gamma_in_sigma": 0.1,
            "log_m2l": 5,
            "log_m2l_sigma": 1,
            "alpha_gamma_in": 0.1,
            "alpha_log_m2l": -0.5,
        }
        self._param_fixed = LensParam(
            gamma_in_sampling=True,
            gamma_in_distribution="GAUSSIAN",
            log_m2l_sampling=True,
            log_m2l_distribution="GAUSSIAN",
            alpha_gamma_in_sampling=True,
            alpha_log_m2l_sampling=True,
            kwargs_fixed=kwargs_fixed,
            log_scatter=False,
        )
        self._param_log_scatter = LensParam(
            gamma_in_sampling=True,
            gamma_in_distribution="GAUSSIAN",
            log_m2l_sampling=True,
            log_m2l_distribution="GAUSSIAN",
            alpha_gamma_in_sampling=True,
            alpha_log_m2l_sampling=True,
            kwargs_fixed={},
            log_scatter=True,
        )

    def test_param_list(self):
        param_list = self._param.param_list(latex_style=False)
        assert len(param_list) == 6
        param_list = self._param.param_list(latex_style=True)
        assert len(param_list) == 6

        param_list = self._param_log_scatter.param_list(latex_style=False)
        assert len(param_list) == 6
        param_list = self._param_log_scatter.param_list(latex_style=True)
        assert len(param_list) == 6

        param_list = self._param_fixed.param_list(latex_style=False)
        assert len(param_list) == 0
        param_list = self._param_fixed.param_list(latex_style=True)
        assert len(param_list) == 0

    def test_args2kwargs(self):
        kwargs = {
            "gamma_in": 1,
            "gamma_in_sigma": 0.1,
            "log_m2l": 5,
            "log_m2l_sigma": 1,
            "alpha_gamma_in": 0.1,
            "alpha_log_m2l": -0.5,
        }
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
