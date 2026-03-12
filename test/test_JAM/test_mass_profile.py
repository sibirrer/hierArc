import numpy as np
import numpy.testing as npt
import pytest
from hierarc.JAM.mass_profile import MassProfile
from lenstronomy.LensModel.Profiles.sie import SIE


class TestMassProfile:
    def setup_method(self):
        self.kw_sie = {"theta_E": 1.0, "center_x": 0.0, "center_y": 0.0, 'e1': 0.1, 'e2': 0.01}
        self.kw_gauss = {"sigma": 0.5, "amp": 1.0, "center_x": 0.0, "center_y": 0.0}

    def test_radial_convergence(self):
        mp = MassProfile(["SIE", "GAUSSIAN"])
        kwargs_list = [self.kw_sie, self.kw_gauss]
        r = np.array([0.1, 0.5, 1.0])
        dens = mp.radial_convergence(r, kwargs_list)
        npt.assert_almost_equal(dens, np.array([5.63832, 1.38899, 0.58759]), decimal=3)

    def test_einstein_radius(self):
        mp = MassProfile(["SIE"])
        theta = 2.5
        assert mp.einstein_radius([{"theta_E": theta}]) == theta

        mp2 = MassProfile(["SIE", "GAUSSIAN"])
        kwargs_list = [self.kw_sie, self.kw_gauss]
        theta_composite = mp2.einstein_radius(kwargs_list)
        npt.assert_almost_equal(theta_composite, 1.24952, decimal=3)

    def test_mge_mass(self):
        mp = MassProfile(["SIE"])
        mp.use_3d_density = False
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
        surf_mass, sigma_mass = mp.mge_mass(r_mge, [self.kw_sie], n_gauss=20, linear_solver=True)
        mge_surf_1d = self._mge(r_test, surf_mass, sigma_mass)
        theta_E = self.kw_sie["theta_E"]
        rho0 = SIE.theta2rho(theta_E)
        sie_surf_1d = SIE.density_2d(r_test, 0, rho0)
        npt.assert_allclose(mge_surf_1d, sie_surf_1d, rtol=0.1)

    def test_mge_mass_mge_prof(self):
        mp = MassProfile(["MULTI_GAUSSIAN"])
        kw_mge = {'amp': np.arange(5), 'sigma': np.arange(1, 6)}
        r_test = np.logspace(  # this must be in logspace
            np.log10(1e-2),
            np.log10(100),
            300,
        )
        surf_lum, sigma_lum = mp.mge_mass(r_test, [kw_mge],
                                                n_gauss=20, linear_solver=True)
        mge_surf_1d = self._mge(r_test, surf_lum, sigma_lum)
        expected_surf_1d = self._mge(
            r_test,
            kw_mge["amp"]
            / (2 * np.pi * kw_mge["sigma"] ** 2),
            kw_mge["sigma"],
        )
        npt.assert_allclose(mge_surf_1d, expected_surf_1d, rtol=1e-2)

    def test_parse_kwargs(self):
        mp = MassProfile(["SIE", "GAUSSIAN"])
        kw_sie = self.kw_sie.copy()
        kw_sie.pop('e1')
        kw_sie.pop('e2')
        kw_gauss = self.kw_gauss | {'e1': 0, 'e2': 0}
        kwargs_list = [kw_sie, kw_gauss]
        kwargs_list_parsed = mp._parse_kwargs(kwargs_list)
        assert 'e1' in kwargs_list_parsed[0]
        assert 'e2' in kwargs_list_parsed[0]
        assert 'e1' not in kwargs_list_parsed[1]
        assert 'e2' not in kwargs_list_parsed[1]

    @staticmethod
    def _gaussian(r, amp, sigma):
        return amp * np.exp(-0.5 * (r / sigma) ** 2)

    def _mge(self, r, amps, sigmas):
        total = np.zeros_like(r)
        for amp, sigma in zip(amps, sigmas):
            total += self._gaussian(r, amp, sigma)
        return total


class TestRaise:
    def test_invalid_profile_name_raises(self):
        with pytest.raises(Exception):
            MassProfile(["INVALID_PROFILE_NAME"])


if __name__ == "__main__":
    pytest.main()
