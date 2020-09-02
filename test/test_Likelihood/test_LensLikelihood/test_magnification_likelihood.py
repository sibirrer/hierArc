import pytest
import unittest
import numpy as np
import numpy.testing as npt
from hierarc.Likelihood.LensLikelihood.magnification_likelihood import MagnificationLikelihood


class TestMagnificationLikelihood(object):

    def setup(self):
        pass

    def test_log_likelihood(self):
        num = 4
        magnification_model = np.ones(num)
        magnification_model_cov = np.diag((magnification_model / 10) ** 2)
        amp_int = 10
        magnitude_measured = magnification_model * amp_int
        magnitude_measured_cov = np.diag((magnitude_measured/10)**2)

        # un-normalized likelihood
        likelihood = MagnificationLikelihood(z_lens=None, z_source=None, magnitude_measured=magnitude_measured,
                                             magnitude_measured_cov=magnitude_measured_cov,
                                             magnification_model=magnification_model,
                                             magnification_model_cov=magnification_model_cov, normalized=False)

        logl = likelihood.log_likelihood(mu_intrinsic=amp_int)
        npt.assert_almost_equal(logl, 0, decimal=6)

        # normalized likelihood
        likelihood = MagnificationLikelihood(z_lens=None, z_source=None, magnitude_measured=magnitude_measured,
                                             magnitude_measured_cov=magnitude_measured_cov,
                                             magnification_model=magnification_model,
                                             magnification_model_cov=magnification_model_cov, normalized=True)

        logl = likelihood.log_likelihood(mu_intrinsic=amp_int)
        assert logl < 0


class TestRaise(unittest.TestCase):

    def test_raise(self):

        with self.assertRaises(ValueError):
            num = 4
            magnification_model = np.ones(num)
            magnification_model_cov = np.zeros((num, num))
            amp_int = 10
            magnitude_measured = magnification_model * amp_int
            magnitude_measured_cov = np.zeros((num, num))

            likelihood = MagnificationLikelihood(z_lens=None, z_source=None, magnitude_measured=magnitude_measured,
                                                 magnitude_measured_cov=magnitude_measured_cov,
                                                 magnification_model=magnification_model,
                                                 magnification_model_cov=magnification_model_cov, normalized=True)


if __name__ == '__main__':
    pytest.main()
