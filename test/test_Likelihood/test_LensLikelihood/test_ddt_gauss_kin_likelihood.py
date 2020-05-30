from hierarc.Likelihood.LensLikelihood.ddt_gauss_likelihood import DdtGaussianLikelihood
from hierarc.Likelihood.LensLikelihood.kin_likelihood import KinLikelihood
from hierarc.Likelihood.LensLikelihood.ddt_gauss_kin_likelihood import DdtGaussKinLikelihood
import numpy as np
import numpy.testing as npt
from lenstronomy.Util import constants as const
import pytest


class TestDdtGaussKinLikelihood(object):

    def setup(self):
        self.z_lens = 0.8
        self.z_source = 3.0

        self.ddt_mean = 10
        self.ddt_sigma = 0.1

        self.dd_mean = 1.

        kwargs_ddt_gauss = {'z_lens': self.z_lens, 'z_source': self.z_source,
                            'ddt_mean': self.ddt_mean, 'ddt_sigma': self.ddt_sigma}

        ds_dds = self.ddt_mean / self.dd_mean / (1 + self.z_lens)
        j_mean_list = np.ones(1)
        scaling_ifu = 1
        sigma_v_measurement = np.sqrt(ds_dds * scaling_ifu * j_mean_list) * const.c / 1000
        sigma_v_sigma = sigma_v_measurement / 10.
        error_cov_measurement = np.diag(sigma_v_sigma ** 2)
        error_cov_j_sqrt = np.diag(np.zeros_like(sigma_v_measurement))
        self.kin_likelihood = KinLikelihood(self.z_lens, self.z_source, sigma_v_measurement, j_mean_list,
                                            error_cov_measurement, error_cov_j_sqrt, normalized=True)
        self.ddt_gauss_likelihood = DdtGaussianLikelihood(**kwargs_ddt_gauss)

        self.ddt_gauss_kin_likelihood = DdtGaussKinLikelihood(self.z_lens, self.z_source, self.ddt_mean, self.ddt_sigma,
                                                              sigma_v_measurement, j_mean_list, error_cov_measurement,
                                                              error_cov_j_sqrt, sigma_sys_error_include=False)
        self.sigma_v_measurement = sigma_v_measurement
        self.error_cov_measurement = error_cov_measurement

    def test_log_likelihood(self):
        ddt, dd = 9, 0.9
        lnlog_tot = self.ddt_gauss_kin_likelihood.log_likelihood(ddt, dd, aniso_scaling=None, sigma_v_sys_error=None)
        lnlog_ddt = self.ddt_gauss_likelihood.log_likelihood(ddt, dd)
        lnlog_kin = self.kin_likelihood.log_likelihood(ddt, dd, aniso_scaling=None, sigma_v_sys_error=None)
        npt.assert_almost_equal(lnlog_tot, lnlog_ddt + lnlog_kin, decimal=5)

    def test_ddt_measurement(self):
        ddt_mean, ddt_sigma = self.ddt_gauss_kin_likelihood.ddt_measurement()
        assert ddt_mean == self.ddt_mean
        assert ddt_sigma == self.ddt_sigma

    def test_sigma_v_measurement(self):

        sigma_v_measurement_, error_cov_measurement_ = self.ddt_gauss_kin_likelihood.sigma_v_measurement()
        assert sigma_v_measurement_[0] == self.sigma_v_measurement[0]
        assert error_cov_measurement_[0, 0] == self.error_cov_measurement[0, 0]

        sigma_v_predict, error_cov_predict = self.ddt_gauss_kin_likelihood.sigma_v_prediction(self.ddt_mean, self.dd_mean, aniso_scaling=1)
        assert sigma_v_predict[0] == self.sigma_v_measurement[0]
        assert error_cov_predict[0, 0] == 0


if __name__ == '__main__':
    pytest.main()
