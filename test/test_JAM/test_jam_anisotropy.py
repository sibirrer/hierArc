import numpy as np
import numpy.testing as npt
import pytest
from hierarc.JAM.jam_anisotropy import JAMAnisotropy


class TestJAMAnisotropy(object):

    def setup_method(self):
        self.r_test = np.logspace(-2, 1, 100)
        self.r_ani = 2.0
        self.beta_0 = -0.3
        self.beta_inf = 0.8
        self.alpha = 3.0

    @staticmethod
    def logistic_function(r, r_ani, beta_0, beta_inf, alpha):
        return beta_0 + (beta_inf - beta_0) / (1 + (r_ani / r) ** alpha)

    def test_const(self):
        jam_ani = JAMAnisotropy("const")
        beta_params = jam_ani.beta_params({"beta": 0.2})
        npt.assert_almost_equal(beta_params, 0.2)

    def test_radial(self):
        jam_ani = JAMAnisotropy("radial")
        beta_params = jam_ani.beta_params({})
        npt.assert_almost_equal(beta_params, 1.0, decimal=2)

    def test_isotropic(self):
        jam_ani = JAMAnisotropy("isotropic")
        beta_params = jam_ani.beta_params({})
        npt.assert_almost_equal(beta_params, 0.0)

    def test_om(self):
        jam_ani = JAMAnisotropy("OM")
        beta_params = jam_ani.beta_params({"r_ani": self.r_ani})
        beta_r = self.logistic_function(self.r_test, *beta_params)
        expected_beta_r = self.r_test**2 / (self.r_test**2 + self.r_ani**2)
        npt.assert_almost_equal(beta_r, expected_beta_r)

    def test_gom(self):
        jam_ani = JAMAnisotropy("GOM")
        beta_params = jam_ani.beta_params(
            {"r_ani": self.r_ani, "beta_inf": self.beta_inf}
        )
        beta_r = self.logistic_function(self.r_test, *beta_params)
        expected_beta_r = (
            self.beta_inf * self.r_test**2 / (self.r_test**2 + self.r_ani**2)
        )
        npt.assert_almost_equal(beta_r, expected_beta_r)

    def test_colin(self):
        jam_ani = JAMAnisotropy("Colin")
        beta_params = jam_ani.beta_params({"r_ani": self.r_ani})
        beta_r = self.logistic_function(self.r_test, *beta_params)
        expected_beta_r = 0.5 * self.r_test / (self.r_test + self.r_ani)
        npt.assert_almost_equal(beta_r, expected_beta_r)

    def test_logistic(self):
        jam_ani = JAMAnisotropy("logistic")
        beta_params = jam_ani.beta_params(
            {
                "beta_0": self.beta_0,
                "beta_inf": self.beta_inf,
                "r_ani": self.r_ani,
                "alpha": self.alpha,
            }
        )
        beta_r = self.logistic_function(self.r_test, *beta_params)
        expected_beta_r = self.beta_0 + (self.beta_inf - self.beta_0) / (
            1 + (self.r_ani / self.r_test) ** self.alpha
        )
        npt.assert_almost_equal(beta_r, expected_beta_r)

    def test_bad_type(self):
        with npt.assert_raises(ValueError):
            JAMAnisotropy("BAD_TYPE")


if __name__ == "__main__":
    pytest.main()
