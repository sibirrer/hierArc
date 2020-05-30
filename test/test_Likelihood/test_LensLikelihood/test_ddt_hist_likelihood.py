from hierarc.Likelihood.LensLikelihood.ddt_hist_likelihood import DdtHistLikelihood, DdtHistKDELikelihood
import numpy as np
import numpy.testing as npt
import pytest


class TestDdtHist(object):

    def setup(self):
        self._ddt = 2000
        self._sigma = 0.1
        ddt_samples = np.random.normal(loc=self._ddt, scale=self._sigma*self._ddt, size=1000000)
        weights = None  # np.random.uniform(low=0, high=1, size=100000)
        self._ddthist = DdtHistLikelihood(z_lens=None, z_source=None, ddt_samples=ddt_samples,
                                          kde_kernel='gaussian', ddt_weights=weights, bandwidth=20, nbins_hist=400,
                                          normalized=False)

    def test_log_likelihood(self):

        logl_max = self._ddthist.log_likelihood(ddt=self._ddt, dd=None)
        npt.assert_almost_equal(logl_max, 0, decimal=1)
        logl_sigma = self._ddthist.log_likelihood(ddt=self._ddt * (1 + self._sigma), dd=None)
        npt.assert_almost_equal(logl_sigma - logl_max, -1/2., decimal=1)

        logl_sigma = self._ddthist.log_likelihood(ddt=self._ddt * (1 + 2* self._sigma), dd=None)
        npt.assert_almost_equal(logl_sigma - logl_max, -2**2 / 2., decimal=0)

        logl_sigma = self._ddthist.log_likelihood(ddt=self._ddt*10, dd=None)
        npt.assert_almost_equal(logl_sigma, -np.inf, decimal=0)
        assert np.exp(logl_sigma) == 0

    def test_ddt_measurement(self):

        ddt_mean, ddt_sigma = self._ddthist.ddt_measurement()
        npt.assert_almost_equal(ddt_mean / self._ddt, 1, decimal=3)
        npt.assert_almost_equal(ddt_sigma / (self._sigma * self._ddt), 1, decimal=3)


class TestDdtHistKDELikelihood(object):
    def setup(self):
        self._ddt = 10
        self._sigma = 1
        ddt_samples = np.random.normal(loc=self._ddt, scale=self._sigma, size=1000000)
        weights = None  # np.random.uniform(low=0, high=1, size=100000)
        bandwidth = 0.1
        self._ddthist_normed = DdtHistKDELikelihood(z_lens=None, z_source=None, ddt_samples=ddt_samples,
                                             kde_kernel='gaussian', ddt_weights=weights, bandwidth=bandwidth,
                                             nbins_hist=400, normalized=True)
        self._ddthist = DdtHistKDELikelihood(z_lens=None, z_source=None, ddt_samples=ddt_samples,
                                             kde_kernel='gaussian', ddt_weights=weights, bandwidth=bandwidth,
                                             nbins_hist=400, normalized=False)

    def test_log_likelihood(self):

        logl_max = self._ddthist_normed.log_likelihood(ddt=self._ddt, dd=None)
        logl_sigma = self._ddthist_normed.log_likelihood(ddt=self._ddt + self._sigma, dd=None)
        npt.assert_almost_equal(logl_sigma - logl_max, -1 / 2., decimal=1)

        logl_sigma = self._ddthist_normed.log_likelihood(ddt=self._ddt + 2 * self._sigma, dd=None)
        npt.assert_almost_equal(logl_sigma - logl_max, -2 ** 2 / 2., decimal=0)

        logl_max = self._ddthist.log_likelihood(ddt=self._ddt, dd=None)
        print(logl_max, np.log(1 / self._sigma / np.sqrt(2 * np.pi)))
        npt.assert_almost_equal(logl_max, 0, decimal=2)
        logl_sigma = self._ddthist.log_likelihood(ddt=self._ddt + self._sigma, dd=None)
        npt.assert_almost_equal(logl_sigma - logl_max, -1 / 2., decimal=1)

        logl_sigma = self._ddthist.log_likelihood(ddt=self._ddt + 2 * self._sigma, dd=None)
        npt.assert_almost_equal(logl_sigma - logl_max, -2 ** 2 / 2., decimal=0)

    def test_ddt_measurement(self):

        ddt_mean, ddt_sigma = self._ddthist.ddt_measurement()
        npt.assert_almost_equal(ddt_mean / self._ddt, 1, decimal=3)
        npt.assert_almost_equal(ddt_sigma / self._sigma, 1, decimal=3)


if __name__ == '__main__':
    pytest.main()
