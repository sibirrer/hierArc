import numpy as np
import numpy.testing as npt
from hierarc.Likelihood.LensLikelihood.ddt_lognorm_likelihood import DdtLogNormLikelihood
from scipy.stats import lognorm
import pytest


class TestTDLikelihoodLogNorm(object):

    def setup(self):
        self.z_L = 0.8
        self.z_S = 3.0
        self.ddt_mu = 3.5
        self.ddt_sigma = 0.2
        self.kwargs_post = {'z_lens': self.z_L, 'z_source': self.z_S, 'ddt_mu': self.ddt_mu, 'ddt_sigma': self.ddt_sigma}
        self.ddt_grid = np.arange(1, 10000, 50)
        self.scipy_lognorm = lognorm(scale=np.exp(self.ddt_mu), s=self.ddt_sigma)
        self.ll_object = DdtLogNormLikelihood(**self.kwargs_post)

    def test_log_likelihood(self):
        ll = self.ll_object.log_likelihood(self.ddt_grid)
        scipy_ll = self.scipy_lognorm.logpdf(self.ddt_grid)  # with the constant term included
        npt.assert_almost_equal(ll, scipy_ll + 0.5*np.log(2*np.pi), decimal=7)


if __name__ == '__main__':
    pytest.main()
