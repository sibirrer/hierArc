from hierarc.Likelihood.LensLikelihood.ds_dds_gauss_likelihood import DsDdsGaussianLikelihood
import numpy as np
import numpy.testing as npt
import pytest


class TestDdtDdGaussianLikelihood(object):

    def setup(self):
        np.random.seed(seed=41)
        self.z_lens = 0.8
        self.z_source = 3.0

        self.ds_dds_mean = 2.
        self.ds_dds_sigma = 0.2

        self.kwargs_lens = {'z_lens': self.z_lens, 'z_source': self.z_source,
                            'ds_dds_mean': self.ds_dds_mean, 'ds_dds_sigma': self.ds_dds_sigma}

    def test_log_likelihood(self):
        likelihood = DsDdsGaussianLikelihood(**self.kwargs_lens)

        dd = 1.
        ddt = self.ds_dds_mean * dd * (1 + self.z_lens)
        ddt_1_sigma = (self.ds_dds_mean + self.ds_dds_sigma) * dd * (1 + self.z_lens)

        lnlog = likelihood.log_likelihood(ddt=ddt, dd=dd)
        npt.assert_almost_equal(lnlog, 0, decimal=5)
        lnlog = likelihood.log_likelihood(ddt=ddt_1_sigma, dd=dd)
        npt.assert_almost_equal(lnlog, -0.5, decimal=5)

        aniso_scaling = [1./(1 + self.ds_dds_sigma/self.ds_dds_mean)]
        lnlog = likelihood.log_likelihood(ddt=ddt, dd=dd, aniso_scaling=aniso_scaling)
        npt.assert_almost_equal(lnlog, -0.5, decimal=5)


if __name__ == '__main__':
    pytest.main()
