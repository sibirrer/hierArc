import numpy as np
import numpy.testing as npt
import pytest
from hierarc.JAM.jam_anisotropy import JAMAnisotropy


class TestJAMAnisotropy:

    @staticmethod
    def logistic_function(r, beta_0, beta_inf, r_ani, alpha):
        return beta_0 + (beta_inf - beta_0) / (1 + (r_ani / r) ** alpha)

    def test_const(self):
        jam_ani = JAMAnisotropy("const")
        beta_params = jam_ani.beta_params({"beta": 0.2}, n_gauss=10)
        npt.assert_almost_equal(beta_params, np.full(10, 0.2))

    def test_radial(self):
        jam_ani = JAMAnisotropy("radial")
        beta_params = jam_ani.beta_params({}, n_gauss=10)
        npt.assert_almost_equal(beta_params, np.full(10, 1.0))

    def test_isotropic(self):
        jam_ani = JAMAnisotropy("isotropic")
        beta_params = jam_ani.beta_params({}, n_gauss=10)
        npt.assert_almost_equal(beta_params, np.full(10, 0.0))

    def test_om(self):
        r_test = np.logspace(-2, 1, 100)
        jam_ani = JAMAnisotropy("OM")
        r_ani = 2.0
        beta_params = jam_ani.beta_params({"r_ani": r_ani})
        beta_r = self.logistic_function(
            r_test, *beta_params
        )
        expected_beta_r = r_test**2 / (r_test**2 + r_ani**2)
        npt.assert_almost_equal(beta_r, expected_beta_r)

    def test_gom(self):
        r_test = np.logspace(-2, 1, 100)
        jam_ani = JAMAnisotropy("GOM")
        r_ani = 2.0
        beta_inf = 0.8
        beta_params = jam_ani.beta_params({"r_ani": r_ani, "beta_inf": beta_inf})
        beta_r = self.logistic_function(
            r_test, *beta_params
        )
        expected_beta_r = beta_inf * r_test**2 / (r_test**2 + r_ani**2)
        npt.assert_almost_equal(beta_r, expected_beta_r)

    def test_colin(self):
        r_test = np.logspace(-2, 1, 100)
        jam_ani = JAMAnisotropy("Colin")
        r_ani = 3.0
        beta_params = jam_ani.beta_params({"r_ani": r_ani})
        beta_r = self.logistic_function(
            r_test, *beta_params
        )
        expected_beta_r = 0.5 * r_test / (r_test + r_ani)
        npt.assert_almost_equal(beta_r, expected_beta_r)

    def test_logistic(self):
        r_test = np.logspace(-2, 1, 100)
        jam_ani = JAMAnisotropy("logistic")
        beta_0 = -0.3
        beta_inf = 0.8
        r_ani = 2.0
        alpha = 3.0
        beta_params = jam_ani.beta_params(
            {"beta_0": beta_0, "beta_inf": beta_inf, "r_ani": r_ani, "alpha": alpha}
        )
        beta_r = self.logistic_function(
            r_test, *beta_params
        )
        expected_beta_r = beta_0 + (beta_inf - beta_0) / (1 + (r_ani / r_test) ** alpha)
        npt.assert_almost_equal(beta_r, expected_beta_r)


if __name__ == "__main__":
    pytest.main()
