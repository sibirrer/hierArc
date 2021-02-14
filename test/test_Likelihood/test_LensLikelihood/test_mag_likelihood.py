import pytest
import numpy as np
import numpy.testing as npt
from hierarc.Likelihood.LensLikelihood.mag_likelihood import MagnificationLikelihood
from lenstronomy.Util.data_util import magnitude2cps


class TestMagnificationLikelihood(object):

    def setup(self):
        pass

    def test_log_likelihood(self):
        num = 4
        magnitude_intrinsic = 19
        magnitude_zero_point = 20

        amp_int = magnitude2cps(magnitude=magnitude_intrinsic, magnitude_zero_point=magnitude_zero_point)
        magnification_model = np.ones(num)
        magnification_model_cov = np.diag((magnification_model / 10) ** 2)

        magnitude_measured = magnification_model * amp_int
        magnitude_measured_cov = np.diag((magnitude_measured/10)**2)

        # un-normalized likelihood
        likelihood = MagnificationLikelihood(amp_measured=magnitude_measured,
                                             cov_amp_measured=magnitude_measured_cov,
                                             magnification_model=magnification_model,
                                             cov_magnification_model=magnification_model_cov,
                                             magnitude_zero_point=magnitude_zero_point)

        logl = likelihood.log_likelihood(mu_intrinsic=magnitude_intrinsic)
        _, cov_tot = likelihood._scale_model(mu_intrinsic=magnitude_intrinsic)
        sign_det, lndet = np.linalg.slogdet(cov_tot)
        logl_test = -1 / 2. * (likelihood.num_data * np.log(2 * np.pi) + lndet)
        npt.assert_almost_equal(logl, logl_test, decimal=6)

        num = 4
        magnification_model = np.ones(num)
        magnification_model_cov = np.zeros((num, num))
        amp_int = 10
        magnitude_measured = magnification_model * amp_int
        magnitude_measured_cov = np.zeros((num, num))

        likelihood = MagnificationLikelihood(amp_measured=magnitude_measured,
                                             cov_amp_measured=magnitude_measured_cov,
                                             magnification_model=magnification_model,
                                             cov_magnification_model=magnification_model_cov)
        logl = likelihood.log_likelihood(mu_intrinsic=1)
        assert logl == -np.inf


if __name__ == '__main__':
    pytest.main()
