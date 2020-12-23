from hierarc.Likelihood.LensLikelihood.ddt_hist_likelihood import DdtHistLikelihood, DdtHistKDELikelihood
import numpy as np
import numpy.testing as npt
import unittest

def log_likelihood(hist_obj, mean, sig_factor):
    logl_max = hist_obj.log_likelihood(ddt=mean, dd=None)
    npt.assert_almost_equal(logl_max, 0, decimal=1)
    logl_sigma = hist_obj.log_likelihood(ddt=mean*(1+sig_factor), dd=None)
    npt.assert_almost_equal(logl_sigma-logl_max, -1/2., decimal=1)
    logl_sigma = hist_obj.log_likelihood(ddt=mean*(1+2*sig_factor), dd=None)
    npt.assert_almost_equal(logl_sigma-logl_max, -2**2/2., decimal=0)
    logl_sigma = hist_obj.log_likelihood(ddt=mean*(1+1000*sig_factor), dd=None)
    npt.assert_array_less(logl_sigma, -1e6) # some very small number
    assert np.exp(logl_sigma) == 0

def ddt_measurement(hist_obj, mean, sig_factor):
    ddt_mean, ddt_sigma = hist_obj.ddt_measurement()
    npt.assert_almost_equal(ddt_mean/mean, 1, decimal=3)
    npt.assert_almost_equal(ddt_sigma/(sig_factor*mean), 1, decimal=3)

class TestDdtHist(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(123)
        cls._ddt = 2000
        cls._sigma = 0.1
        ddt_samples = np.random.normal(loc=cls._ddt, 
                                       scale=cls._sigma*cls._ddt, 
                                       size=1000000)
        weights = None  # np.random.uniform(low=0, high=1, size=100000)
        cls.test_log_likelihood = staticmethod(log_likelihood)
        cls.test_ddt_measurement = staticmethod(ddt_measurement)
        cls._ddthist = DdtHistLikelihood(z_lens=None, z_source=None, 
                                          ddt_samples=ddt_samples,
                                          ddt_weights=weights, 
                                          nbins_hist=400,
                                          normalized=False)
        cls._ddthist_scott = DdtHistLikelihood(z_lens=None, z_source=None, 
                                               ddt_samples=ddt_samples,
                                               ddt_weights=weights, 
                                               binning_method='scott',
                                               normalized=False)
        cls._ddthist_silverman = DdtHistLikelihood(z_lens=None, z_source=None, 
                                               ddt_samples=ddt_samples,
                                               ddt_weights=weights, 
                                               binning_method='silverman',
                                               normalized=False)
        cls._ddthist_bw = DdtHistLikelihood(z_lens=None, z_source=None, 
                                            ddt_samples=ddt_samples,
                                            ddt_weights=weights, 
                                            # Very small bandwidth
                                            binning_method=cls._ddt*cls._sigma*0.001,
                                            normalized=False)

    def test_log_likelihood_nbins(self):
        self.test_log_likelihood(self._ddthist, self._ddt, self._sigma)

    def test_ddt_measurement_nbins(self):
        self.test_ddt_measurement(self._ddthist, self._ddt, self._sigma)

    def test_log_likelihood_scott(self):
        self.test_log_likelihood(self._ddthist_scott, self._ddt, self._sigma)

    def test_ddt_measurement_scott(self):
        self.test_ddt_measurement(self._ddthist_scott, self._ddt, self._sigma)

    def test_log_likelihood_silverman(self):
        self.test_log_likelihood(self._ddthist_silverman, self._ddt, self._sigma)

    def test_ddt_measurement_silverman(self):
        self.test_ddt_measurement(self._ddthist_silverman, self._ddt, self._sigma)

    def test_log_likelihood_bw(self):
        self.test_log_likelihood(self._ddthist_bw, self._ddt, self._sigma)

    def test_ddt_measurement_bw(self):
        self.test_ddt_measurement(self._ddthist_bw, self._ddt, self._sigma)

class TestDdtHistKDELikelihood(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._ddt = 10
        cls._sigma = 1
        ddt_samples = np.random.normal(loc=cls._ddt, scale=cls._sigma, size=1000000)
        weights = None  # np.random.uniform(low=0, high=1, size=100000)
        bandwidth = 0.1
        cls._ddthist_normed = DdtHistKDELikelihood(z_lens=None, z_source=None, ddt_samples=ddt_samples,
                                             kde_kernel='gaussian', ddt_weights=weights, bandwidth=bandwidth,
                                             nbins_hist=400, normalized=True)
        cls._ddthist = DdtHistKDELikelihood(z_lens=None, z_source=None, ddt_samples=ddt_samples,
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
    unittest.main()
