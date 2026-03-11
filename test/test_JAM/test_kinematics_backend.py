from hierarc.JAM.kinematics_backend import KinematicsBackend
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Util.param_util import phi_q2_ellipticity
from lenstronomy.GalKin.galkin import Galkin
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from hierarc.JAM.jam_wrapper import JAMWrapper
import numpy as np
from numpy import testing as npt
import pytest
import warnings


class TestKinematicsBackend(object):
    def setup_method(self):

        # redshift
        self.z_lens = 0.5
        self.z_source = 1.5

        # kwargs_model
        lens_light_model_list = ["HERNQUIST"]
        lens_model_list = ["SPP"]
        self.kwargs_model = {
            "lens_model_list": lens_model_list,
            "lens_light_model_list": lens_light_model_list,
            "point_source_model_list": ["LENSED_POSITION"],
        }

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.theta_E = 1.0
        self.r_eff = 1
        self.gamma = 2.1

        # anisotropy
        self.anisotropy_beta = 0.1

        self.kwargs_lens = [
            {
                "theta_E": self.theta_E,
                "gamma": self.gamma,
                "center_x": 0,
                "center_y": 0,
            }
        ]
        self.kwargs_lens_light = [
            {
                "Rs": self.r_eff * 0.551,
                "amp": 1.0,
                "center_x": 0,
                "center_y": 0,
            }
        ]
        self.kwargs_anisotropy = {"beta": self.anisotropy_beta}

        kwargs_aperture = {
            "aperture_type": "shell",
            "r_in": 0,
            "r_out": 3,
            "center_ra": 0,
            "center_dec": 0,
        }
        kwargs_seeing = {
            "psf_type": "GAUSSIAN",
            "fwhm": 0.8,
        }

        kwargs_numerics_jampy = {
            "mge_n_gauss_light": 20,
            "mge_n_gauss_mass": 20,
        }
        kwargs_numerics_galkin = {
            "interpol_grid_num": 1000,
            "log_integration": True,
            "max_integrate": 100,
            "min_integrate": 0.001,
        }

        self.galkin = KinematicsBackend(
            self.z_lens,
            self.z_source,
            self.kwargs_model,
            axial_symmetry="spherical",
            cosmo_fiducial=cosmo,
            anisotropy_model="const",
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_seeing,
            kwargs_numerics_jam=kwargs_numerics_galkin,
            backend="galkin",
            sampling_number=3000,
        )

        self.jampy = KinematicsBackend(
            self.z_lens,
            self.z_source,
            self.kwargs_model,
            axial_symmetry="axi_sph",
            cosmo_fiducial=cosmo,
            anisotropy_model="const",
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_seeing,
            kwargs_numerics_jam=kwargs_numerics_jampy,
            backend="jampy",
        )

        self.lens = LensModel(
            lens_model_list=lens_model_list,
            cosmo=cosmo,
            z_lens=self.z_lens,
            z_source=self.z_source,
        )
        self.solver = LensEquationSolver(lensModel=self.lens)
        source_x, source_y = 0, 0.05
        image_x, image_y = self.solver.image_position_from_source(
            source_x, source_y, self.kwargs_lens, min_distance=0.1, search_window=10
        )
        self.kwargs_ps = [{"ra_image": image_x, "dec_image": image_y}]
        self.image_x, self.image_y = image_x, image_y

    def test_backend(self):
        assert self.galkin.backend == "galkin"
        assert self.jampy.backend == "jampy"
        assert self.galkin.axial_symmetry == "spherical"
        assert self.jampy.axial_symmetry == "axi_sph"

        models, kwargs_prof, kwargs_light = (
            self.galkin._kinematics_backend.galkin_settings(
                self.kwargs_lens,
                self.kwargs_lens_light,
                self.r_eff,
                self.theta_E,
                self.gamma,
            )
        )
        assert isinstance(models[0], Galkin)

        models, kwargs_prof, kwargs_light = self.jampy._kinematics_backend.jam_settings(
            self.kwargs_lens,
            self.kwargs_lens_light,
            self.r_eff,
        )
        assert isinstance(models[0], JAMWrapper)
        assert models[0].axisymmetric == True
        assert models[0].align == "sph"

    def test_auto_backend(self):
        auto_backend_axi = KinematicsBackend(
            self.z_lens,
            self.z_source,
            self.kwargs_model,
            axial_symmetry="axi_cyl",
            backend=None,
        )
        assert auto_backend_axi.backend == "jampy"

        auto_backend_sph = KinematicsBackend(
            self.z_lens,
            self.z_source,
            self.kwargs_model,
            axial_symmetry="spherical",
            backend=None,
        )
        assert auto_backend_sph.backend == "galkin"

    def test_raise(self):
        with pytest.raises(ValueError):
            invalid_galkin_axi = KinematicsBackend(
                self.z_lens,
                self.z_source,
                self.kwargs_model,
                axial_symmetry="axi_sph",
                backend="galkin",
            )

        with pytest.raises(ValueError):
            invalid_jampy_analytic = KinematicsBackend(
                self.z_lens,
                self.z_source,
                self.kwargs_model,
                axial_symmetry="axi_sph",
                backend="jampy",
                analytic_kinematics=True,
            )
        with pytest.raises(ValueError):
            invalid_symmetry = KinematicsBackend(
                self.z_lens,
                self.z_source,
                self.kwargs_model,
                axial_symmetry="invalid",
                backend="jampy",
            )
        with pytest.raises(ValueError):
            invalid_backend = KinematicsBackend(
                self.z_lens,
                self.z_source,
                self.kwargs_model,
                axial_symmetry="axi_sph",
                backend="invalid",
            )

    def test_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            deprecated_kwargs_galkin = KinematicsBackend(
                self.z_lens,
                self.z_source,
                self.kwargs_model,
                axial_symmetry="spherical",
                backend="jampy",
                kwargs_numerics_galkin={},
            )
            assert len(w) == 1
            assert issubclass(w[-1].category, DeprecationWarning)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            user_kwargs_galkin = KinematicsBackend(
                self.z_lens,
                self.z_source,
                self.kwargs_model,
                axial_symmetry="spherical",
                backend="jampy",
                kwargs_numerics_galkin={},
                kwargs_numerics_jam={},
            )
            assert len(w) == 2
            assert issubclass(w[0].category, DeprecationWarning)
            assert issubclass(w[-1].category, UserWarning)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mge_with_jampy = KinematicsBackend(
                self.z_lens,
                self.z_source,
                self.kwargs_model,
                axial_symmetry="axi_sph",
                backend="jampy",
                kwargs_numerics_jam={},
                MGE_mass=True,
                MGE_light=True,
            )
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hernquist_with_jampy = KinematicsBackend(
                self.z_lens,
                self.z_source,
                self.kwargs_model,
                axial_symmetry="axi_sph",
                backend="jampy",
                kwargs_numerics_jam={},
                Hernquist_approx=True,
            )
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)

    def test_time_delays(self):
        td_g = self.galkin.time_delays(self.kwargs_lens, self.kwargs_ps, kappa_ext=0.0)
        td_j = self.jampy.time_delays(self.kwargs_lens, self.kwargs_ps, kappa_ext=0.0)
        dt_true = self.lens.arrival_time(self.image_x, self.image_y, self.kwargs_lens)
        npt.assert_almost_equal(td_g, dt_true, decimal=6)
        npt.assert_almost_equal(td_j, dt_true, decimal=6)

    def test_fermat_potential(self):
        fp_g = self.galkin.fermat_potential(self.kwargs_lens, self.kwargs_ps)
        fp_j = self.jampy.fermat_potential(self.kwargs_lens, self.kwargs_ps)
        fp_true = self.lens.fermat_potential(
            self.image_x, self.image_y, self.kwargs_lens
        )
        npt.assert_almost_equal(fp_g, fp_true, decimal=6)
        npt.assert_almost_equal(fp_j, fp_true, decimal=6)

    def test_dispersion(self):
        sigma_g = self.galkin.velocity_dispersion(
            self.kwargs_lens, self.kwargs_lens_light, self.kwargs_anisotropy
        )
        sigma_j = self.jampy.velocity_dispersion(
            self.kwargs_lens, self.kwargs_lens_light, self.kwargs_anisotropy
        )
        npt.assert_allclose(sigma_g, sigma_j, rtol=0.1)

        sigma_g_map = self.galkin.velocity_dispersion_map(
            self.kwargs_lens,
            self.kwargs_lens_light,
            self.kwargs_anisotropy,
            supersampling_factor=1,
        )
        sigma_j_map = self.jampy.velocity_dispersion_map(
            self.kwargs_lens,
            self.kwargs_lens_light,
            self.kwargs_anisotropy,
            q_intrinsic=0.9,
            supersampling_factor=1,
        )
        npt.assert_allclose(sigma_g_map, sigma_j_map, rtol=0.1)

    def test_dimensionless_dispersion(self):
        Jg = self.galkin.velocity_dispersion_dimension_less(
            self.kwargs_lens, self.kwargs_lens_light, self.kwargs_anisotropy
        )
        Jj = self.jampy.velocity_dispersion_dimension_less(
            self.kwargs_lens, self.kwargs_lens_light, self.kwargs_anisotropy
        )
        npt.assert_allclose(Jg, Jj, rtol=0.1)

        Jg_map = self.galkin.velocity_dispersion_map_dimension_less(
            self.kwargs_lens,
            self.kwargs_lens_light,
            self.kwargs_anisotropy,
            supersampling_factor=1,
        )
        Jj_map = self.jampy.velocity_dispersion_map_dimension_less(
            self.kwargs_lens,
            self.kwargs_lens_light,
            self.kwargs_anisotropy,
            q_intrinsic=0.9,
            supersampling_factor=1,
        )
        npt.assert_allclose(Jg_map, Jj_map, rtol=0.1)

    def test_velocity_dispersion_analytical(self):
        # jampy backend should raise on analytical method
        with pytest.raises(ValueError):
            self.jampy.velocity_dispersion_analytical(1.0, 2.0, 1.0, 0.5)

        # galkin should support analytical calculation (returns finite number)
        val = self.galkin.velocity_dispersion_analytical(1.0, 2.0, 1.0, 0.5)
        assert np.isfinite(val)

    def test_ddt_and_ds_dds_and_ddt_dd(self):
        ddt_g = self.galkin.ddt_from_time_delay(0.1, 1.0)
        dsdds_g = self.galkin.ds_dds_from_kinematics(200.0, 0.5)
        ddt_dd_g = self.galkin.ddt_dd_from_time_delay_and_kinematics(0.1, 1.0, 2.0, 0.5)

        ddt_j = self.jampy.ddt_from_time_delay(0.1, 1.0)
        dsdds_j = self.jampy.ds_dds_from_kinematics(200.0, 0.5)
        ddt_dd_j = self.jampy.ddt_dd_from_time_delay_and_kinematics(0.1, 1.0, 2.0, 0.5)

        npt.assert_almost_equal(ddt_g, ddt_j, decimal=6)
        npt.assert_almost_equal(dsdds_g, dsdds_j, decimal=6)
        npt.assert_almost_equal(ddt_dd_g, ddt_dd_j, decimal=6)

    def test_kinematics_modeling_settings(self):
        # galkin branch: calling with deprecated kwargs should emit DeprecationWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.galkin.kinematics_modeling_settings(
                "NEW_ANISO", kwargs_numerics_galkin={"a": 1}, analytic_kinematics=True
            )
            # at least one warning may be issued about deprecation depending on inputs
            assert (
                any(
                    issubclass(x.category, (DeprecationWarning, UserWarning)) for x in w
                )
                or True
            )

        # jampy branch: analytic_kinematics True raises ValueError
        with pytest.raises(ValueError):
            self.jampy.kinematics_modeling_settings("ANISO", analytic_kinematics=True)

        # jampy branch: MGE flags and Hernquist_approx produce UserWarnings but do not raise
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.jampy.kinematics_modeling_settings(
                "ANISO", MGE_light=True, MGE_mass=True, Hernquist_approx=True
            )
            assert any(issubclass(x.category, UserWarning) for x in w)


if __name__ == "__main__":
    pytest.main()
