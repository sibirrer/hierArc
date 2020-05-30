from hierarc.Likelihood.LensLikelihood.ddt_gauss_likelihood import DdtGaussianLikelihood
import numpy as np
import numpy.testing as npt
import pytest


class TestDdtDdGaussianLikelihood(object):

    def setup(self):
        np.random.seed(seed=41)
        self.z_lens = 0.8
        self.z_source = 3.0

        self.ddt_mean = 10
        self.ddt_sigma = 0.1

        self.kwargs_lens = {'z_lens': self.z_lens, 'z_source': self.z_source,
                            'ddt_mean': self.ddt_mean, 'ddt_sigma': self.ddt_sigma}

    def test_log_likelihood(self):
        likelihood = DdtGaussianLikelihood(**self.kwargs_lens)
        lnlog = likelihood.log_likelihood(ddt=self.ddt_mean, dd=None)
        npt.assert_almost_equal(lnlog, 0, decimal=5)

        lnlog = likelihood.log_likelihood(ddt=self.ddt_mean + self.ddt_sigma, dd=None)
        npt.assert_almost_equal(lnlog, -0.5, decimal=5)

    def test_ddt_measurement(self):
        likelihood = DdtGaussianLikelihood(**self.kwargs_lens)
        ddt_mean, ddt_sigma = likelihood.ddt_measurement()
        assert ddt_mean == self.ddt_mean
        assert ddt_sigma == self.ddt_sigma


if __name__ == '__main__':
    pytest.main()
