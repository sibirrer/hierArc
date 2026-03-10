from hierarc.JAM.kinematics_backend import KinematicsBackend
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Util.param_util import phi_q2_ellipticity
from lenstronomy.GalKin.galkin import Galkin
from hierarc.JAM.jam_wrapper import JAMWrapper
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
        }

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.theta_E = 1.0
        self.r_eff = 1
        self.gamma = 2.1

        # anisotropy
        self.anisotropy_beta = 0.1

        # axial symmetry
        q_intrinsic = 0.85
        q_observed = 0.86
        e1, e2 = phi_q2_ellipticity(0, q_observed)

        self.kwargs_lens = [
            {
                "theta_E": self.theta_E,
                "gamma": self.gamma,
                "center_x": 0,
                "center_y": 0,
                "e1": e1,
                "e2": e2,
            }
        ]
        self.kwargs_lens_light = [
            {
                "Rs": self.r_eff * 0.551,
                "amp": 1.0,
                "center_x": 0,
                "center_y": 0,
                "e1": e1,
                "e2": e2,
            }
        ]
        self.kwargs_anisotropy = {"beta": self.anisotropy_beta}

        kwargs_aperture = {
            "aperture_type": "shell",
            "r_in": 0,
            "r_out": 3 / 2.0,
            "center_ra": 0,
            "center_dec": 0,
        }
        kwargs_seeing = {
            "psf_type": "GAUSSIAN",
            "fwhm": 1.4,
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
            sampling_number=1000,
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
            invalid_backend = KinematicsBackend(
                self.z_lens,
                self.z_source,
                self.kwargs_model,
                axial_symmetry="axi_sph",
                backend="galkin",
            )

        with pytest.raises(ValueError):
            invalid_backend = KinematicsBackend(
                self.z_lens,
                self.z_source,
                self.kwargs_model,
                axial_symmetry="axi_sph",
                backend="jampy",
                analytic_kinematics=True,
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


if __name__ == "__main__":
    pytest.main()
