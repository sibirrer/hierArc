from hierarc.Sampling.ParamManager.source_param import SourceParam
import numpy.testing as npt
import pytest
import unittest


class TestSourceParam(object):

    def setup(self):
        self._param = SourceParam(sne_distribution='GAUSSIAN', sne_apparent_m_sampling=True, kwargs_fixed=None)

        kwargs_fixed = {'mu_sne': 1, 'sigma_sne': 0.1}
        self._param_fixed = SourceParam(sne_distribution='GAUSSIAN', sne_apparent_m_sampling=True, kwargs_fixed=kwargs_fixed)

    def test_param_list(self):
        param_list = self._param.param_list(latex_style=False)
        assert len(param_list) == 2
        param_list = self._param.param_list(latex_style=True)
        assert len(param_list) == 2

        param_list = self._param_fixed.param_list(latex_style=False)
        assert len(param_list) == 0
        param_list = self._param_fixed.param_list(latex_style=True)
        assert len(param_list) == 0

    def test_args2kwargs(self):
        kwargs = {'mu_sne': 1, 'sigma_sne': 0.1}
        args = self._param.kwargs2args(kwargs)
        kwargs_new, i = self._param.args2kwargs(args, i=0)
        args_new = self._param.kwargs2args(kwargs_new)
        npt.assert_almost_equal(args_new, args)

        args = self._param_fixed.kwargs2args(kwargs)
        kwargs_new, i = self._param_fixed.args2kwargs(args, i=0)
        args_new = self._param_fixed.kwargs2args(kwargs_new)
        npt.assert_almost_equal(args_new, args)


class TestRaise(unittest.TestCase):

    def test_raise(self):

        with self.assertRaises(ValueError):
            param = SourceParam(sne_apparent_m_sampling=True, sne_distribution='BAD')


if __name__ == '__main__':
    pytest.main()
