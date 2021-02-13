import pytest
import numpy as np
import numpy.testing as npt
from lenstronomy.Util import constants as const
from hierarc.Likelihood.LensLikelihood.td_mag_likelihood import TDMagLikelihood
from lenstronomy.Util.data_util import magnitude2cps


class TestMagnificationLikelihood(object):

    def setup(self):
        pass

    def test_log_likelihood(self):
        ddt = 1000  # time-delay distance in units Mpc
        num = 4
        magnification_model = np.ones(num)
        magnitude_intrinsic = 19
        magnitude_zero_point = 20

        amp_int = magnitude2cps(magnitude=magnitude_intrinsic, magnitude_zero_point=magnitude_zero_point)

        amp_measured = magnification_model * amp_int
        cov_amp_measured = np.diag((amp_measured/10)**2)
        time_delay_measured = np.ones(num - 1) * 10
        cov_td_measured = np.ones((num-1, num-1))
        fermat_unit_conversion = const.Mpc / const.c / const.day_s * const.arcsec ** 2
        fermat_diff = time_delay_measured / fermat_unit_conversion / ddt
        model_vector = np.append(fermat_diff, magnification_model)
        cov_model = np.diag((model_vector/10)**2)

        # un-normalized likelihood
        likelihood = TDMagLikelihood(time_delay_measured, cov_td_measured, amp_measured, cov_amp_measured,
                 fermat_diff, magnification_model, cov_model, magnitude_zero_point=magnitude_zero_point)

        logl = likelihood.log_likelihood(ddt=ddt, mu_intrinsic=magnitude_intrinsic)
        model_vector, cov_tot = likelihood._model_cov(ddt, mu_intrinsic=magnitude_intrinsic)
        sign_det, lndet = np.linalg.slogdet(cov_tot)
        logl_norm = -1 / 2. * (likelihood.num_data * np.log(2 * np.pi) + lndet)
        npt.assert_almost_equal(logl, logl_norm, decimal=6)

        num = 4
        magnification_model = np.ones(num)
        cov_td_measured = np.zeros((num - 1, num - 1))
        cov_amp_measured = np.zeros((num, num))
        amp_int = 10
        amp_measured = magnification_model * amp_int
        cov_model = np.zeros((num + num - 1, num + num - 1))

        likelihood = TDMagLikelihood(time_delay_measured, cov_td_measured, amp_measured, cov_amp_measured,
                 fermat_diff, magnification_model, cov_model, magnitude_zero_point=magnitude_zero_point)
        logl = likelihood.log_likelihood(ddt=ddt, mu_intrinsic=1)
        assert logl == -np.inf


if __name__ == '__main__':
    pytest.main()
