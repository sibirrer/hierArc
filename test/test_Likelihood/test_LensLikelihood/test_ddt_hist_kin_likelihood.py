from hierarc.Likelihood.LensLikelihood.ddt_hist_likelihood import DdtHistLikelihood
from hierarc.Likelihood.LensLikelihood.ddt_hist_kin_likelihood import DdtHistKinLikelihood
from hierarc.Likelihood.LensLikelihood.kin_likelihood import KinLikelihood
from lenstronomy.Util import constants as const
import numpy as np
import numpy.testing as npt


class TestDdtHistKinHist(object):

    def setup(self):
        self._ddt, self._dd = 2000., 1000.

        self._sigma = 0.1
        ddt_samples = np.random.normal(loc=self._ddt, scale=self._ddt * self._sigma, size=1000000)
        ddt_weights = None  # np.random.uniform(low=0, high=1, size=100000)

        self._ddthist = DdtHistLikelihood(z_lens=None, z_source=None, ddt_samples=ddt_samples,
                                          kde_kernel='gaussian', ddt_weights=ddt_weights, bandwidth=20, nbins_hist=400)

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
        self._ddt_kin_likelihood = DdtHistKinLikelihood(z_lens, z_source, ddt_samples, ddt_weights, sigma_v_measurement,
                                                        j_model, error_cov_measurement, error_cov_j_sqrt,
                                                        kde_kernel='gaussian', bandwidth=20, nbins_hist=400)

    def test_log_likelihood(self):

        logl_max = self._ddt_kin_likelihood.log_likelihood(ddt=self._ddt, dd=self._dd)
        logl = self._ddt_kin_likelihood.log_likelihood(self._ddt, self._dd / (1 + self._sigma) ** 2, aniso_scaling=None)
        npt.assert_almost_equal(logl - logl_max, -self._num_ifu / 2, decimal=5)
