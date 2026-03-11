import numpy as np
from numpy import testing as npt
import pytest
from astropy.cosmology import FlatLambdaCDM
from numpy.testing import assert_allclose
from hierarc.JAM.jam_kinematics_api import JAMKinematicsAPI
from lenstronomy.Util import param_util
import warnings


class TestJAMKinematicsAPI:
    def setup_method(self):
        # example values you provided
        self.z_lens = 0.5
        self.z_source = 1.5

        lens_light_model_list = ["HERNQUIST_ELLIPSE"]
        lens_model_list = ["EPL"]
        self.kwargs_model = {
            "lens_model_list": lens_model_list,
            "lens_light_model_list": lens_light_model_list,
        }

        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.theta_E = 1.0
        self.r_eff = 1.0
        self.gamma = 2.1

        self.anisotropy_beta = 0.1

        q_intrinsic = 0.85
        q_observed = 0.86
        # phi_q2_ellipticity returns e1,e2 for an angle and q
        e1, e2 = param_util.phi_q2_ellipticity(0.0, q_observed)

        self.kwargs_lens = [
            {
                "theta_E": self.theta_E,
                "gamma": self.gamma,
                "center_x": 0.0,
                "center_y": 0.0,
                "e1": e1,
                "e2": e2,
            }
        ]
        self.kwargs_lens_light = [
            {
                "Rs": self.r_eff * 0.551,
                "amp": 1.0,
                "center_x": 0.0,
                "center_y": 0.0,
                "e1": e1,
                "e2": e2,
            }
        ]
        self.kwargs_anisotropy = {"beta": self.anisotropy_beta}

        self.kwargs_aperture = {
            "aperture_type": "shell",
            "r_in": 0.0,
            "r_out": 3.0 / 2.0,
            "center_ra": 0.0,
            "center_dec": 0.0,
        }
        self.kwargs_seeing = {"psf_type": "GAUSSIAN", "fwhm": 1.4}

        self.kwargs_numerics_jampy = {"mge_n_gauss_light": 20, "mge_n_gauss_mass": 20}

        # instantiate the API (this will create lenstronomy classes inside)
        self.api = JAMKinematicsAPI(
            z_lens=self.z_lens,
            z_source=self.z_source,
            kwargs_model=self.kwargs_model,
            kwargs_aperture=self.kwargs_aperture,
            kwargs_seeing=self.kwargs_seeing,
            anisotropy_model="const",
            axial_symmetry="axi_sph",
            cosmo=self.cosmo,
            lens_model_kinematics_bool=None,
            light_model_kinematics_bool=None,
            multi_observations=False,
            multi_light_profile=False,
            kwargs_numerics_jam=self.kwargs_numerics_jampy,
        )

    def test_constructor_created_expected_attributes(self):
        assert hasattr(self.api, "lensCosmo")
        assert hasattr(self.api, "_lens_light_model_list")
        assert hasattr(self.api, "_lens_model_list")
        # cosmology distances present
        assert set(self.api._kwargs_cosmo.keys()) == {"d_d", "d_s", "d_ds"}

    def test_transform_kappa_ext_numeric(self):
        sig = np.array([10.0, 20.0, 30.0])
        out = JAMKinematicsAPI.transform_kappa_ext(sig, kappa_ext=0.2)
        expected = sig * np.sqrt(1.0 - 0.2)
        assert_allclose(out, expected)

    def test_copy_centers_fills(self):
        kw1 = [{"a": 1.0}]
        kw2 = [{"center_x": 0.7, "center_y": -0.3}]
        res = self.api._copy_centers(kw1, kw2)
        assert "center_x" in res[0] and "center_y" in res[0]
        assert res[0]["center_x"] == pytest.approx(0.7)
        assert res[0]["center_y"] == pytest.approx(-0.3)

    def test_kinematic_lens_profiles(self):
        kwargs_lens = [dict(self.kwargs_lens[0])]
        mass_list, kwargs_profile = self.api.kinematic_lens_profiles(
            kwargs_lens, model_kinematics_bool=None
        )
        assert isinstance(mass_list, list)
        assert isinstance(kwargs_profile, list)
        assert len(mass_list) == len(kwargs_profile)
        # names should be a subset of the original lens_model_list
        for name in mass_list:
            assert name in self.api._lens_model_list

    def test_kinematic_light_profile(self):
        # single-mode
        kwargs_lens_light = [dict(self.kwargs_lens_light[0])]
        light_list, kwargs_light = self.api.kinematic_light_profile(
            kwargs_lens_light, r_eff=1.0, model_kinematics_bool=None
        )
        assert isinstance(light_list, list)
        # kwargs_light either list or dict depending on implementation; ensure non-empty
        assert kwargs_light is not None

    def test_kinematic_multi_light_profile(self):
        # multi-light-profile mode: create an API with multi_light_profile True and multi_observations True
        api_multi = JAMKinematicsAPI(
            z_lens=self.z_lens,
            z_source=self.z_source,
            kwargs_model=self.kwargs_model,
            kwargs_aperture=[self.kwargs_aperture, self.kwargs_aperture],
            kwargs_seeing=[self.kwargs_seeing, self.kwargs_seeing],
            anisotropy_model="const",
            axial_symmetry="axi_sph",
            cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
            lens_model_kinematics_bool=None,
            light_model_kinematics_bool=None,
            multi_observations=True,
            multi_light_profile=True,
            kwargs_numerics_jam=self.kwargs_numerics_jampy,
        )
        kwargs_lens_light_multi = [
            [dict(self.kwargs_lens_light[0])],
            [dict(self.kwargs_lens_light[0])],
        ]
        light_list_m, kwargs_light_m = api_multi.kinematic_light_profile(
            kwargs_lens_light_multi, r_eff=1.0, model_kinematics_bool=None
        )
        assert isinstance(light_list_m, list)
        assert isinstance(kwargs_light_m, list)
        assert len(kwargs_light_m) == 2

        sigma = api_multi.velocity_dispersion(
            self.kwargs_lens, kwargs_lens_light_multi, self.kwargs_anisotropy
        )
        assert len(sigma) == 2
        npt.assert_almost_equal(sigma, 215.139, decimal=1)

    def test_kinematics_modeling_settings(self):
        api = JAMKinematicsAPI(
            z_lens=self.z_lens,
            z_source=self.z_source,
            kwargs_model=self.kwargs_model,
            kwargs_aperture=self.kwargs_aperture,
            kwargs_seeing=self.kwargs_seeing,
            anisotropy_model="const",
            axial_symmetry="axi_sph",
            cosmo=self.cosmo,
            lens_model_kinematics_bool=None,
            light_model_kinematics_bool=None,
            multi_observations=False,
            multi_light_profile=False,
            kwargs_numerics_jam=self.kwargs_numerics_jampy,
        )
        api.kinematics_modeling_settings(
            "NEW_ANISO", axial_symmetry="spherical", kwargs_numerics_jam={"n": 1}
        )
        assert api._anisotropy_model == "NEW_ANISO"
        assert api._axial_symmetry == "spherical"
        assert api._kwargs_numerics_kin == {"n": 1}

    def test_jam_settings(self):
        jam_models, kwargs_profile, kwargs_light = self.api.jam_settings(
            kwargs_lens=self.kwargs_lens,
            kwargs_lens_light=self.kwargs_lens_light,
        )
        assert isinstance(jam_models, list)
        assert isinstance(kwargs_profile, list)
        assert isinstance(kwargs_light, list)
        assert len(jam_models) == 1
        assert len(kwargs_profile) == 1
        assert len(kwargs_light) == 1

    def test_velocity_dispersion(self):
        kwargs_lens = [dict(self.kwargs_lens[0])]
        kwargs_lens_light = [dict(self.kwargs_lens_light[0])]
        kwargs_anisotropy = self.kwargs_anisotropy
        sigma = self.api.velocity_dispersion(
            kwargs_lens, kwargs_lens_light, kwargs_anisotropy, r_eff=self.r_eff
        )
        npt.assert_almost_equal(sigma, 215.139, decimal=1)

    def test_velocity_dispersion_map(self):
        kwargs_lens = [dict(self.kwargs_lens[0])]
        kwargs_lens_light = [dict(self.kwargs_lens_light[0])]
        kwargs_anisotropy = self.kwargs_anisotropy
        sigma = self.api.velocity_dispersion_map(
            kwargs_lens, kwargs_lens_light, kwargs_anisotropy, r_eff=self.r_eff
        )
        npt.assert_almost_equal(sigma, 215.139, decimal=1)

    def test_warn_supersampling(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.api.velocity_dispersion_map(
                kwargs_lens=self.kwargs_lens,
                kwargs_lens_light=self.kwargs_lens_light,
                kwargs_anisotropy=self.kwargs_anisotropy,
                r_eff=self.r_eff,
                theta_E=self.theta_E,
                gamma=self.gamma,
                kappa_ext=0.0,
                q_intrinsic=0.85,
                supersampling_factor=2,
                voronoi_bins=None,
            )
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)


if __name__ == "__main__":
    pytest.main()
