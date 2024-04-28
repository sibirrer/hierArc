from hierarc.Sampling.ParamManager.los_param import LOSParam
import numpy.testing as npt
import pytest


class TestLOSParam(object):
    def setup_method(self):
        self._param = LOSParam(
            los_sampling=True,
            los_distributions=["GEV"],
            kwargs_fixed=None,
        )

        self._param_gauss = LOSParam(
            los_sampling=True,
            los_distributions=["GAUSSIAN"],
            kwargs_fixed=None,
        )

        kwargs_fixed = [
            {
                "mean": 0,
                "sigma": 0.05,
                "xi": 0.1,
            }
        ]
        self._param_fixed = LOSParam(
            los_sampling=True,
            los_distributions=["GEV"],
            kwargs_fixed=kwargs_fixed,
        )

    def test_param_list(self):
        param_list = self._param.param_list(latex_style=False)
        assert len(param_list) == 3
        param_list = self._param.param_list(latex_style=True)
        assert len(param_list) == 3

        param_list = self._param_gauss.param_list(latex_style=False)
        assert len(param_list) == 2
        param_list = self._param_gauss.param_list(latex_style=True)
        assert len(param_list) == 2

        param_list = self._param_fixed.param_list(latex_style=False)
        assert len(param_list) == 0
        param_list = self._param_fixed.param_list(latex_style=True)
        assert len(param_list) == 0

    def test_args2kwargs(self):
        kwargs = [
            {
                "mean": 0.1,
                "sigma": 0.1,
                "xi": 0.2,
            }
        ]

        kwargs_gauss = [
            {
                "mean": 0.1,
                "sigma": 0.1,
            }
        ]
        args = self._param.kwargs2args(kwargs)
        kwargs_new, i = self._param.args2kwargs(args, i=0)
        args_new = self._param.kwargs2args(kwargs_new)
        npt.assert_almost_equal(args_new, args)

        args = self._param_gauss.kwargs2args(kwargs_gauss)
        kwargs_new, i = self._param_gauss.args2kwargs(args, i=0)
        args_new = self._param_gauss.kwargs2args(kwargs_new)
        npt.assert_almost_equal(args_new, args)

        args = self._param_fixed.kwargs2args(kwargs)
        kwargs_new, i = self._param_fixed.args2kwargs(args, i=0)
        args_new = self._param_fixed.kwargs2args(kwargs_new)
        npt.assert_almost_equal(args_new, args)


if __name__ == "__main__":
    pytest.main()
