from hierarc.Likelihood.LensLikelihood.ddt_hist_likelihood import DdtHistLikelihood, DdtHistKDELikelihood
import numpy as np
import numpy.testing as npt


class TestDdtHist(object):

    def setup(self):
        self._ddt = 2000
        self._sigma = 0.1
        ddt_samples = np.random.normal(loc=self._ddt, scale=self._sigma*self._ddt, size=1000000)
        weights = None  # np.random.uniform(low=0, high=1, size=100000)
        self._ddthist = DdtHistLikelihood(z_lens=None, z_source=None, ddt_samples=ddt_samples,
                                          kde_kernel='gaussian', ddt_weights=weights, bandwidth=20, nbins_hist=400)

    def test_log_likelihood(self):

        logl_max = self._ddthist.log_likelihood(ddt=self._ddt, dd=None)
        logl_sigma = self._ddthist.log_likelihood(ddt=self._ddt * (1 + self._sigma), dd=None)
        npt.assert_almost_equal(logl_sigma - logl_max, -1/2., decimal=1)

        logl_sigma = self._ddthist.log_likelihood(ddt=self._ddt * (1 + 2* self._sigma), dd=None)
        npt.assert_almost_equal(logl_sigma - logl_max, -2**2 / 2., decimal=0)

        logl_sigma = self._ddthist.log_likelihood(ddt=self._ddt*10, dd=None)
        npt.assert_almost_equal(logl_sigma, -np.inf, decimal=0)
        assert np.exp(logl_sigma) == 0


class TestDdtHistKDELikelihood(object):
    def setup(self):
        self._ddt = 2000
        self._sigma = 0.1
        ddt_samples = np.random.normal(loc=self._ddt, scale=self._sigma*self._ddt, size=1000000)
        weights = None  # np.random.uniform(low=0, high=1, size=100000)
        self._ddthist = DdtHistKDELikelihood(z_lens=None, z_source=None, ddt_samples=ddt_samples,
                                          kde_kernel='gaussian', ddt_weights=weights, bandwidth=20, nbins_hist=400)

    def test_log_likelihood(self):

        logl_max = self._ddthist.log_likelihood(ddt=self._ddt, dd=None)
        logl_sigma = self._ddthist.log_likelihood(ddt=self._ddt * (1 + self._sigma), dd=None)
        npt.assert_almost_equal(logl_sigma - logl_max, -1/2., decimal=1)

        logl_sigma = self._ddthist.log_likelihood(ddt=self._ddt * (1 + 2* self._sigma), dd=None)
        npt.assert_almost_equal(logl_sigma - logl_max, -2**2 / 2., decimal=0)
