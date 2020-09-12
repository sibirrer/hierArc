import pytest
import numpy as np
import numpy.testing as npt
from hierarc.Likelihood.LensLikelihood.mag_likelihood import MagnificationLikelihood


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
        likelihood = MagnificationLikelihood(amp_measured=magnitude_measured,
                                             cov_amp_measured=magnitude_measured_cov,
                                             mag_model=magnification_model,
                                             cov_model=magnification_model_cov)

        logl = likelihood.log_likelihood(mu_intrinsic=amp_int)
        npt.assert_almost_equal(logl, 0, decimal=6)

        num = 4
        magnification_model = np.ones(num)
        magnification_model_cov = np.zeros((num, num))
        amp_int = 10
        magnitude_measured = magnification_model * amp_int
        magnitude_measured_cov = np.zeros((num, num))

        likelihood = MagnificationLikelihood(amp_measured=magnitude_measured,
                                             cov_amp_measured=magnitude_measured_cov,
                                             mag_model=magnification_model,
                                             cov_model=magnification_model_cov)
        logl = likelihood.log_likelihood(mu_intrinsic=1)
        assert logl == -np.inf


if __name__ == '__main__':
    pytest.main()
