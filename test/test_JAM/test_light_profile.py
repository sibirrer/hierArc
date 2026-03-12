import numpy as np
import numpy.testing as npt
import pytest
from hierarc.JAM.light_profile import LightProfile
from lenstronomy.LightModel.Profiles.hernquist import Hernquist


class TestLightProfile:
    def setup_method(self):
        self.kw_sersic = {
            "R_sersic": 1.5,
            "amp": 1.0,
            "n_sersic": 2.5,
            "e1": 0.1,
            "e2": 0.01,
            "center_x": 0.2,
            "center_y": -0.1,
        }
        self.kw_hernquist = {"Rs": 0.8, "amp": 1.0, "center_x": 0.2, "center_y": -0.1}
        self.kw_gaussian = {"sigma": 0.5, "amp": 1.0}

    def test_radial_surface_brightness(self):
        lp = LightProfile(["SERSIC", "GAUSSIAN"])
        kwargs_list = [
            dict(
                self.kw_sersic,
                **{"center_x": 0.1, "center_y": -0.1, "e1": 0.1, "e2": -0.1}
            ),
            dict(self.kw_gaussian, **{"center_x": 0.1, "center_y": -0.1}),
        ]
        r = np.array([0, 1, 2])
        val = lp.radial_surface_brightness(r, kwargs_list)
        npt.assert_almost_equal(val, np.array([97.2906, 2.0985, 0.5659]), decimal=4)

    def test_effective_radius(self):
        lp_sersic = LightProfile(["SERSIC"])
        assert (
            lp_sersic.effective_radius([self.kw_sersic]) == self.kw_sersic["R_sersic"]
        )
        lp_hern = LightProfile(["HERNQUIST"])
        npt.assert_almost_equal(
            lp_hern.effective_radius([self.kw_hernquist]),
            1.8153 * self.kw_hernquist["Rs"],
            decimal=4,
        )
        lp = LightProfile(["SERSIC", "HERNQUIST"])
        kwargs_list = [
            dict(self.kw_sersic, **{"center_x": 0.2, "center_y": -0.1}),
            self.kw_hernquist,
        ]
        r_eff = lp.effective_radius(kwargs_list)
        npt.assert_almost_equal(r_eff, 0.849914, decimal=3)

    def test_mge_lum(self):
        lp = LightProfile(["HERNQUIST"])
        r_mge = np.logspace(  # this must be in logspace
            np.log10(1e-4),
            np.log10(400),
            500,
        )
        r_test = np.logspace(  # this must be in logspace
            np.log10(1e-2),
            np.log10(100),
            300,
        )
        surf_lum, sigma_lum = lp.mge_lum_tracer(
            r_mge, [self.kw_hernquist], n_gauss=20, linear_solver=True
        )
        mge_surf_1d = self._mge(r_test, surf_lum, sigma_lum)
        hernq = Hernquist()
        hernq_surf_1d = hernq.function(
            x=r_test, y=0, Rs=self.kw_hernquist["Rs"], amp=self.kw_hernquist["amp"]
        )
        npt.assert_allclose(mge_surf_1d, hernq_surf_1d, rtol=0.1)

    def test_mge_lum_mge_prof(self):
        lp = LightProfile(["MULTI_GAUSSIAN"])
        kw_mge = {"amp": np.arange(5), "sigma": np.arange(1, 6)}
        r_test = np.logspace(  # this must be in logspace
            np.log10(1e-2),
            np.log10(100),
            300,
        )
        surf_lum, sigma_lum = lp.mge_lum_tracer(
            r_test, [kw_mge], n_gauss=20, linear_solver=True
        )
        mge_surf_1d = self._mge(r_test, surf_lum, sigma_lum)
        expected_surf_1d = self._mge(
            r_test,
            kw_mge["amp"] / (2 * np.pi * kw_mge["sigma"] ** 2),
            kw_mge["sigma"],
        )
        npt.assert_allclose(mge_surf_1d, expected_surf_1d, rtol=1e-2)

    def test_parse_kwargs(self):
        lp = LightProfile(["SERSIC_ELLIPSE", "GAUSSIAN"])
        kw_sersic = self.kw_sersic.copy()
        kw_sersic.pop("e1")
        kw_sersic.pop("e2")
        kw_gauss = self.kw_gaussian | {"e1": 0, "e2": 0}
        kwargs_list = [kw_sersic, kw_gauss]
        kwargs_list_parsed = lp._parse_kwargs(kwargs_list)
        assert "e1" in kwargs_list_parsed[0]
        assert "e2" in kwargs_list_parsed[0]
        assert "e1" not in kwargs_list_parsed[1]
        assert "e2" not in kwargs_list_parsed[1]

    @staticmethod
    def _gaussian(r, amp, sigma):
        return amp * np.exp(-0.5 * (r / sigma) ** 2)

    def _mge(self, r, amps, sigmas):
        total = np.zeros_like(r)
        for amp, sigma in zip(amps, sigmas):
            total += self._gaussian(r, amp, sigma)
        return total


if __name__ == "__main__":
    pytest.main()
