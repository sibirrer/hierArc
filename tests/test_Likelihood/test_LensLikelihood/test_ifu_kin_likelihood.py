import numpy as np
import pytest
import numpy.testing as npt
import unittest
from hierarc.Likelihood.LensLikelihood.ifu_kin_likelihood import IFUKinCov
from lenstronomy.Util import  constants as const


class TestIFUKinLikelihood(object):

    def setup(self):
        np.random.seed(seed=41)

    def test_log_likelihood(self):
        num_ifu = 10
        z_lens = 0.5
        z_source = 2
        ddt, dd = 2., 1.
        ds_dds = ddt / dd / (1 + z_lens)
        j_mean_list = np.ones(num_ifu)
        scaling_ifu = 1
        sigma_v_measurement = np.sqrt(ds_dds * scaling_ifu * j_mean_list) * const.c / 1000
        sigma_v_sigma = sigma_v_measurement/10.
        error_cov_measurement = np.diag(sigma_v_sigma ** 2)
        error_cov_j_sqrt = np.diag(np.zeros_like(sigma_v_measurement))
        ifu_likelihood = IFUKinCov(z_lens, z_source, sigma_v_measurement, j_mean_list, error_cov_measurement,
                                   error_cov_j_sqrt, ani_param_array=None, ani_scaling_array_list=None)
        logl = ifu_likelihood.log_likelihood(ddt, dd, a_ani=None)
        npt.assert_almost_equal(logl, 0, decimal=5)
        logl = ifu_likelihood.log_likelihood(ddt*0.9**2, dd, a_ani=None)
        npt.assert_almost_equal(logl, -num_ifu/2, decimal=5)


class TestRaise(unittest.TestCase):

    def test_raise(self):

        with self.assertRaises(ValueError):
            raise ValueError()


if __name__ == '__main__':
    pytest.main()
