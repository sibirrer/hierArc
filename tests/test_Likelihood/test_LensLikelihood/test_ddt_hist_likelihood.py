from hierarc.Likelihood.LensLikelihood.ddt_hist_likelihood import DdtHist
import numpy as np
import numpy.testing as npt


class TestDdtHist(object):

    def setup(self):
        self._sigma = 1.
        ddt_samples = np.random.normal(loc=0, scale=self._sigma, size=1000000)
        weights = None  # np.random.uniform(low=0, high=1, size=100000)
        self._ddthist = DdtHist(z_lens=None, z_source=None, ddt_samples=ddt_samples,
                                kde_kernel='gaussian', weights=weights, bandwidth=20, nbins_hist=400)

    def test_log_likelihood(self):

        logl_max = self._ddthist.log_likelihood(ddt=0, dd=None)
        logl_sigma = self._ddthist.log_likelihood(ddt=1 * self._sigma, dd=None)
        npt.assert_almost_equal(logl_sigma - logl_max, -1/2., decimal=1)

        logl_sigma = self._ddthist.log_likelihood(ddt=2 * self._sigma, dd=None)
        npt.assert_almost_equal(logl_sigma - logl_max, -2**2 / 2., decimal=0)
