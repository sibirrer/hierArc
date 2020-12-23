from hierarc.Likelihood.LensLikelihood.ddt_hist_likelihood import DdtHistLikelihood
from hierarc.Likelihood.LensLikelihood.ddt_hist_kin_likelihood import DdtHistKinLikelihood
from hierarc.Likelihood.LensLikelihood.kin_likelihood import KinLikelihood
from lenstronomy.Util import constants as const
import numpy as np
import numpy.testing as npt
import pytest


class TestDdtHistKinHist(object):

    def setup(self):
        self._ddt, self._dd = 2000., 1000.

        self._sigma = 0.1
        ddt_samples = np.random.normal(loc=self._ddt, scale=self._ddt * self._sigma, size=1000000)
        ddt_weights = None  # np.random.uniform(low=0, high=1, size=100000)

        self._ddthist = DdtHistLikelihood(z_lens=None, z_source=None, ddt_samples=ddt_samples,
                                          ddt_weights=ddt_weights, nbins_hist=400)

        self._num_ifu = 10
        z_lens = 0.5
        z_source = 2
        ds_dds = self._ddt / self._dd / (1 + z_lens)
        j_model = np.ones(self._num_ifu)
        scaling_ifu = 1
        sigma_v_measurement = np.sqrt(ds_dds * scaling_ifu * j_model) * const.c / 1000
        sigma_v_sigma = sigma_v_measurement * self._sigma
        error_cov_measurement = np.diag(sigma_v_sigma ** 2)
        error_cov_j_sqrt = np.diag(np.zeros_like(sigma_v_measurement))
        self._kin_likelihood = KinLikelihood(z_lens, z_source, sigma_v_measurement, j_model, error_cov_measurement,
                                             error_cov_j_sqrt)
        self._ddt_kin_likelihood = DdtHistKinLikelihood(z_lens, z_source, ddt_samples, sigma_v_measurement,
                                                        j_model, error_cov_measurement, error_cov_j_sqrt, ddt_weights=ddt_weights,
                                                        kde_kernel='gaussian', bandwidth=20, nbins_hist=400)
        self.sigma_v_measurement = sigma_v_measurement
        self.error_cov_measurement = error_cov_measurement

    def test_log_likelihood(self):

        logl_max = self._ddt_kin_likelihood.log_likelihood(ddt=self._ddt, dd=self._dd)
        logl = self._ddt_kin_likelihood.log_likelihood(self._ddt, self._dd / (1 + self._sigma) ** 2, aniso_scaling=None)
        npt.assert_almost_equal(logl - logl_max, -self._num_ifu / 2, decimal=5)

    def test_ddt_measurement(self):

        ddt_mean, ddt_sigma = self._ddthist.ddt_measurement()
        npt.assert_almost_equal(ddt_mean / self._ddt, 1, decimal=3)
        npt.assert_almost_equal(ddt_sigma / (self._sigma * self._ddt), 1, decimal=3)

    def test_sigma_v_measurement(self):
        sigma_v_measurement_, error_cov_measurement_ = self._ddt_kin_likelihood.sigma_v_measurement()
        assert sigma_v_measurement_[0] == self.sigma_v_measurement[0]
        assert error_cov_measurement_[0, 0] == self.error_cov_measurement[0, 0]

        sigma_v_predict, error_cov_predict = self._ddt_kin_likelihood.sigma_v_prediction(self._ddt, self._dd, aniso_scaling=1)
        assert sigma_v_predict[0] == self.sigma_v_measurement[0]
        assert error_cov_predict[0, 0] == 0


if __name__ == '__main__':
    pytest.main()
