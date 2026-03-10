import numpy as np
import numpy.testing as npt
import pytest
from hierarc.JAM.light_profile import LightProfile


class TestLightProfile:
    def setup_method(self):
        self.kw_sersic = {"R_sersic": 1.5, "amp": 1.0, "n_sersic": 2.5}
        self.kw_hernquist = {"Rs": 0.8, "amp": 1.0}
        self.kw_gaussian = {"sigma": 0.5, "amp": 1.0}
        self.kw_with_centers = {"center_x": 0.2, "center_y": -0.1, "e1": 0.3, "e2": -0.2}

    def test_init(self):
        lp = LightProfile(["SERSIC_ELLIPSE", "GAUSSIAN"])
        assert any("SERSIC" == p for p in lp.profile_list)
        assert not any("SERSIC_ELLIPSE" == p for p in lp.profile_list)
        assert len(lp.light_model.func_list) == len(lp.profile_list)

    def test_radial_surface_brightness(self):
        lp = LightProfile(["SERSIC", "GAUSSIAN"])
        kwargs_list = [dict(self.kw_sersic, **{"center_x": 0.1, "center_y": -0.1, "e1": 0.1, "e2": -0.1}),
                       dict(self.kw_gaussian, **{"center_x": 0.1, "center_y": -0.1})]
        r = np.array([0, 1, 2])
        val = lp.radial_surface_brightness(r, kwargs_list)
        npt.assert_almost_equal(val, np.array([97.2906,  2.0985,  0.5659]), decimal=4)

    def test_effective_radius(self):
        lp_sersic = LightProfile(["SERSIC"])
        assert lp_sersic.effective_radius([self.kw_sersic]) == self.kw_sersic["R_sersic"]
        lp_hern = LightProfile(["HERNQUIST"])
        npt.assert_almost_equal(lp_hern.effective_radius([self.kw_hernquist]),
                                1.8153 * self.kw_hernquist["Rs"], decimal=4)
        lp = LightProfile(["SERSIC", "HERNQUIST"])
        kwargs_list = [dict(self.kw_sersic, **{"center_x": 0.2, "center_y": -0.1}), self.kw_hernquist]
        r_eff = lp.effective_radius(kwargs_list)
        npt.assert_almost_equal(r_eff, 0.866799, decimal=3)

    def test_circularize_kwargs(self):
        lp = LightProfile(["SERSIC", "SERSIC_ELLIPSE", "GAUSSIAN"])
        kw_list = [
            dict(self.kw_sersic, **{"center_x": 0.3, "center_y": -0.2, "e1": 0.1, "e2": -0.1}),
            dict(self.kw_sersic, **{"center_x": 0.0, "center_y": 0.0, "e1": 0.2, "e2": 0.0}),
            dict(self.kw_gaussian, **{"center_x": -0.1, "center_y": 0.1}),
        ]
        circ = lp._circularize_kwargs(kw_list)
        for c in circ:
            assert "center_x" not in c and "center_y" not in c
            assert "e1" not in c and "e2" not in c
        assert all(isinstance(c, dict) for c in circ)


class TestRaise:
    def test_invalid_profile_name_raises(self):
        with pytest.raises(Exception):
            LightProfile(["INVALID_PROFILE_NAME"])


if __name__ == "__main__":
    pytest.main()
