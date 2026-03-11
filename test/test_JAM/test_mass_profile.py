# test_mass_profile_simple.py
import numpy as np
import numpy.testing as npt
import pytest
from hierarc.JAM.mass_profile import MassProfile
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis


class TestMassProfile:
    def setup_method(self):
        self.kw_sie = {"theta_E": 1.0, "center_x": 0.0, "center_y": 0.0}
        self.kw_gauss = {"sigma": 0.5, "amp": 1.0, "center_x": 0.0, "center_y": 0.0}

    def test_init(self):
        mp = MassProfile(["SIE_ELLIPSE", "CONVERGENCE", "SHEAR", "NFW_ELLIPSE_CSE"])
        assert "CONVERGENCE" not in mp.profile_list
        assert "SHEAR" not in mp.profile_list
        assert not any(p.endswith("_ELLIPSE") for p in mp.profile_list)
        assert not any(p.endswith("_CSE") for p in mp.profile_list)
        assert len(mp.mass_model.func_list) == len(mp.profile_list)

    def test_radial_density(self):
        mp = MassProfile(["SIE", "GAUSSIAN"])
        kwargs_list = [dict(self.kw_sie), dict(self.kw_gauss)]
        r = np.array([0.1, 0.5, 1.0])
        dens = mp.radial_density(r, kwargs_list)
        npt.assert_almost_equal(dens, np.array([16.41339,  0.94471,  0.2279]), decimal=3)

    def test_einstein_radius(self):
        mp = MassProfile(["SIE"])
        theta = 2.5
        assert mp.einstein_radius([{"theta_E": theta}]) == theta

        mp2 = MassProfile(["SIS", "GAUSSIAN"])
        kwargs_list = [dict(self.kw_sie), dict(self.kw_gauss)]
        theta_composite = mp2.einstein_radius(kwargs_list)
        npt.assert_almost_equal(theta_composite, 1.2422, decimal=3)

    def test_circularize_kwargs(self):
        mp = MassProfile(["SIE", "GAUSSIAN"])
        kwargs_list = [
            {"e1": 0.2, "e2": 0.1, "center_x": 0.3, "center_y": -0.1, "theta_E": 2},
            {"center_x": 0.0, "center_y": 0.0, "amp": 1.0, "sigma": 0.5},
        ]
        circ = mp._circularize_kwargs(kwargs_list)
        for c in circ:
            assert "center_x" not in c and "center_y" not in c
        for i, entry in enumerate(circ):
            func = mp.mass_model.func_list[i]
            if "e1" in func.param_names:
                assert entry.get("e1", 0.0) == 0.0
            else:
                assert "e1" not in entry
            if "e2" in func.param_names:
                assert entry.get("e2", 0.0) == 0.0
            else:
                assert "e2" not in entry


class TestRaise:
    def test_invalid_profile_name_raises(self):
        with pytest.raises(Exception):
            MassProfile(["INVALID_PROFILE_NAME"])


if __name__ == "__main__":
    pytest.main()
