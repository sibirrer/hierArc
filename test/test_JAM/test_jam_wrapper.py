import numpy as np
import numpy.testing as npt
import pytest

from hierarc.JAM.jam_wrapper import JAMWrapper
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.GalKin.galkin import Galkin
from lenstronomy.GalKin.galkin_shells import GalkinShells


class TestJAMWrapperSpherical(object):
    """Test JAMWrapper against Lenstronomy Galkin module for spherical symmetry."""

    def setup_method(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        lens_cosmo = LensCosmo(0.5, 1.2, cosmo=cosmo)

        center_x = 0.0
        center_y = 0.0
        self.kwargs_light = [
            {"Rs": 1.0, "amp": 1.0, "center_x": center_x, "center_y": center_y}
        ]
        self.kwargs_lens_mass = [
            {"theta_E": 1.5, "gamma": 2.1, "center_x": center_x, "center_y": center_y}
        ]
        self.kwargs_anisotropy = {"beta": 0.3}
        kwargs_psf = {
            "psf_type": "GAUSSIAN",
            "fwhm": 0.5,
        }
        kwargs_cosmo = {
            "d_d": lens_cosmo.dd,
            "d_s": lens_cosmo.ds,
            "d_ds": lens_cosmo.dds,
        }
        kwargs_numerics_lenstronomy = {
            "interpol_grid_num": 2000,
            "log_integration": True,
            "max_integrate": 1e3,
            "min_integrate": 1e-5,
        }
        kwargs_numerics_mge = {
            "mge_n_gauss": 50,
            "mge_min_r": 1e-4,
            "mge_max_r": 300,
            "mge_n_radial": 500,
            "mge_linear_solver": True,
        }
        kwargs_numeric_jam = kwargs_numerics_lenstronomy | kwargs_numerics_mge
        kwargs_model = {
            "mass_profile_list": ["SPP"],
            "light_profile_list": ["HERNQUIST"],
            "anisotropy_model": "const",
        }

        x = y = np.linspace(-5, 5, 20)
        x_grid, y_grid = np.meshgrid(x, y)
        kwargs_aperture_grid = {
            "aperture_type": "IFU_grid",
            "x_grid": x_grid,
            "y_grid": y_grid,
        }
        self.jam_spherical_grid = JAMWrapper(
            kwargs_model=kwargs_model | {"symmetry": "spherical"},
            kwargs_aperture=kwargs_aperture_grid,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numeric_jam,
        )
        self.galkin_grid = Galkin(
            kwargs_model=kwargs_model,
            kwargs_aperture=kwargs_aperture_grid,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numerics_lenstronomy,
            analytic_kinematics=False,
        )

        kwargs_aperture_slit = {
            "aperture_type": "slit",
            "length": 3.0,
            "width": 0.5,
            "center_ra": center_x - 0.3,
            "center_dec": center_y + 0.4,
            "angle": np.deg2rad(30),
        }
        self.jam_spherical_slit = JAMWrapper(
            kwargs_model=kwargs_model | {"symmetry": "spherical"},
            kwargs_aperture=kwargs_aperture_slit,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numeric_jam,
        )
        self.galkin_slit = Galkin(
            kwargs_model=kwargs_model,
            kwargs_aperture=kwargs_aperture_slit,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numerics_lenstronomy,
            analytic_kinematics=False,
        )

        r_bins = np.linspace(0.1, 5, 11)
        kwargs_aperture_ifu_shells = {
            "aperture_type": "IFU_shells",
            "r_bins": r_bins,
            "center_ra": center_x,
            "center_dec": center_y,
        }
        self.jam_spherical_shells = JAMWrapper(
            kwargs_model=kwargs_model | {"symmetry": "spherical"},
            kwargs_aperture=kwargs_aperture_ifu_shells,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numeric_jam,
        )
        self.galkin_shells = GalkinShells(
            kwargs_model=kwargs_model,
            kwargs_aperture=kwargs_aperture_ifu_shells,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numerics_lenstronomy,
            analytic_kinematics=False,
        )

        kwargs_psf_multi_gaussian = {
            "psf_type": "MULTI-GAUSSIAN",
            "fwhm": 0.5,
            "sigmas": np.arange(1, 6),
            "amplitudes": np.arange(1, 6) / np.sum(np.arange(1, 6)),
        }
        self.jam_spherical_grid_multi_gaussian = JAMWrapper(
            kwargs_model=kwargs_model | {"symmetry": "spherical"},
            kwargs_aperture=kwargs_aperture_grid,
            kwargs_psf=kwargs_psf_multi_gaussian,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numeric_jam,
        )

        pix_kernel = np.zeros((11, 11))
        pix_kernel[5, 5] = 1
        kwargs_psf_pixel = {
            "psf_type": "PIXEL",
            "fwhm": 0.5,
            "kernel": pix_kernel,
            "supersampling_factor": 2,
        }
        self.jam_spherical_grid_pixel = JAMWrapper(
            kwargs_model=kwargs_model | {"symmetry": "spherical"},
            kwargs_aperture=kwargs_aperture_grid,
            kwargs_psf=kwargs_psf_pixel,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numeric_jam,
        )

    def test_spherical_dispersion_grid(self):
        sigma_v_jam = self.jam_spherical_grid.dispersion(
            self.kwargs_lens_mass,
            self.kwargs_light,
            self.kwargs_anisotropy,
            convolved=True,
        )
        sigma_v_galkin = self.galkin_grid.dispersion_map_grid_convolved(
            self.kwargs_lens_mass,
            self.kwargs_light,
            self.kwargs_anisotropy,
            supersampling_factor=5,
        )
        npt.assert_allclose(sigma_v_jam, sigma_v_galkin, rtol=1e-2)

    def test_spherical_dispersion_slit(self):
        np.random.seed(0)
        sigma_v_jam = self.jam_spherical_slit.dispersion(
            self.kwargs_lens_mass,
            self.kwargs_light,
            self.kwargs_anisotropy,
            convolved=True,
        )
        sigma_v_galkin = self.galkin_slit.dispersion(
            self.kwargs_lens_mass,
            self.kwargs_light,
            self.kwargs_anisotropy,
            sampling_number=5000,
        )
        npt.assert_allclose(sigma_v_jam, sigma_v_galkin, rtol=1e-2)

    def test_spherical_dispersion_shells(self):
        sigma_v_jam = self.jam_spherical_shells.dispersion(  # self.jam_spherical_shells.dispersion(
            self.kwargs_lens_mass,
            self.kwargs_light,
            self.kwargs_anisotropy,
            convolved=True,
        )
        sigma_v_galkin = self.galkin_shells.dispersion_map(
            self.kwargs_lens_mass,
            self.kwargs_light,
            self.kwargs_anisotropy,
        )
        npt.assert_allclose(sigma_v_jam, sigma_v_galkin, rtol=1e-2)

    def test_spherical_voronoi(self):
        pass
        # TODO: implement this test

    def test_spherical_multi_gaussian_psf(self):
        sigma_v_jam = self.jam_spherical_grid_multi_gaussian.dispersion(  # self.jam_spherical_shells.dispersion(
            self.kwargs_lens_mass,
            self.kwargs_light,
            self.kwargs_anisotropy,
            convolved=True,
        )
        # TODO: test against JamPy model

    def test_spherical_pixel_psf(self):
        sigma_v_jam = self.jam_spherical_grid_pixel.dispersion(  # self.jam_spherical_shells.dispersion(
            self.kwargs_lens_mass,
            self.kwargs_light,
            self.kwargs_anisotropy,
            convolved=True,
        )
        # TODO: test against JamPy model


class TestJAMWrapperAxiSph(object):
    """Test JAMWrapper with axisymmetric-spherical symmetry but in the spherical limit
    q=1, against Lenstronomy Galkin module for spherical symmetry."""

    def setup_method(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        lens_cosmo = LensCosmo(0.5, 1.2, cosmo=cosmo)

        self.ellipticities = {"e1": 0.0, "e2": 0.0}
        self.kwargs_light_spherical = {
            "Rs": 1.0,
            "amp": 1.0,
            "center_x": 0.0,
            "center_y": 0.0,
        }
        self.kwargs_lens_mass_spherical = {
            "theta_E": 1.5,
            "gamma": 2.1,
            "center_x": 0.0,
            "center_y": 0.0,
        }
        self.kwargs_anisotropy = {"beta": 0.3}
        self.inclination = 80.0
        kwargs_psf = {
            "psf_type": "GAUSSIAN",
            "fwhm": 0.5,
        }
        kwargs_cosmo = {
            "d_d": lens_cosmo.dd,
            "d_s": lens_cosmo.ds,
            "d_ds": lens_cosmo.dds,
        }
        kwargs_numerics_lenstronomy = {
            "interpol_grid_num": 2000,
            "log_integration": True,
            "max_integrate": 1e3,
            "min_integrate": 1e-3,
        }
        kwargs_numerics_mge = {
            "mge_n_gauss": 50,
            "mge_min_r": 1e-4,
            "mge_max_r": 300,
            "mge_n_radial": 500,
            "mge_linear_solver": True,
        }
        kwargs_numeric_jam = kwargs_numerics_lenstronomy | kwargs_numerics_mge
        kwargs_model = {
            "mass_profile_list": ["SPP"],
            "light_profile_list": ["HERNQUIST"],
            "anisotropy_model": "const",
        }

        x = y = np.linspace(-5, 5, 20)
        x_grid, y_grid = np.meshgrid(x, y)
        kwargs_aperture_grid = {
            "aperture_type": "IFU_grid",
            "x_grid": x_grid,
            "y_grid": y_grid,
        }
        self.jam_axi_sph_grid = JAMWrapper(
            kwargs_model=kwargs_model | {"symmetry": "axi_sph"},
            kwargs_aperture=kwargs_aperture_grid,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numeric_jam,
        )
        self.galkin_grid = Galkin(
            kwargs_model=kwargs_model,
            kwargs_aperture=kwargs_aperture_grid,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numerics_lenstronomy,
            analytic_kinematics=False,
        )

        r_bins = np.linspace(0.1, 5, 11)
        kwargs_aperture_ifu_shells = {
            "aperture_type": "IFU_shells",
            "r_bins": r_bins,
            "center_ra": 0.0,
            "center_dec": 0.0,
        }
        self.jam_axi_sph_shells = JAMWrapper(
            kwargs_model=kwargs_model | {"symmetry": "axi_sph"},
            kwargs_aperture=kwargs_aperture_ifu_shells,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numeric_jam,
        )
        self.galkin_shells = GalkinShells(
            kwargs_model=kwargs_model,
            kwargs_aperture=kwargs_aperture_ifu_shells,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numerics_lenstronomy,
            analytic_kinematics=False,
        )

        kwargs_aperture_slit = {
            "aperture_type": "slit",
            "length": 3.0,
            "width": 0.5,
            "center_ra": 0.0,
            "center_dec": 0.0,
            "angle": np.deg2rad(0),
        }
        self.jam_axi_sph_slit = JAMWrapper(
            kwargs_model=kwargs_model | {"symmetry": "axi_sph"},
            kwargs_aperture=kwargs_aperture_slit,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numeric_jam,
        )
        self.galkin_slit = Galkin(
            kwargs_model=kwargs_model,
            kwargs_aperture=kwargs_aperture_slit,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numerics_lenstronomy,
            analytic_kinematics=False,
        )

    def test_axi_dispersion_grid(self):
        sigma_v_jam = self.jam_axi_sph_grid.dispersion(
            [self.kwargs_lens_mass_spherical | self.ellipticities],
            [self.kwargs_light_spherical | self.ellipticities],
            self.kwargs_anisotropy,
            convolved=True,
        )
        sigma_v_galkin = self.galkin_grid.dispersion_map_grid_convolved(
            [self.kwargs_lens_mass_spherical],
            [self.kwargs_light_spherical],
            self.kwargs_anisotropy,
            supersampling_factor=5,
        )
        npt.assert_allclose(sigma_v_jam, sigma_v_galkin, rtol=1e-2)

    def test_axi_dispersion_slit(self):
        np.random.seed(0)
        sigma_v_jam = self.jam_axi_sph_slit.dispersion(
            [self.kwargs_lens_mass_spherical | self.ellipticities],
            [self.kwargs_light_spherical | self.ellipticities],
            self.kwargs_anisotropy,
            convolved=True,
        )
        sigma_v_galkin = self.galkin_slit.dispersion(
            [self.kwargs_lens_mass_spherical],
            [self.kwargs_light_spherical],
            self.kwargs_anisotropy,
            sampling_number=5000,
        )
        npt.assert_allclose(sigma_v_jam, sigma_v_galkin, rtol=1e-2)

    def test_axi_dispersion_shells(self):
        sigma_v_jam = self.jam_axi_sph_shells.dispersion(
            [self.kwargs_lens_mass_spherical | self.ellipticities],
            [self.kwargs_light_spherical | self.ellipticities],
            self.kwargs_anisotropy,
            convolved=True,
        )
        sigma_v_galkin = self.galkin_shells.dispersion_map(
            [self.kwargs_lens_mass_spherical],
            [self.kwargs_light_spherical],
            self.kwargs_anisotropy,
        )
        npt.assert_allclose(sigma_v_jam, sigma_v_galkin, rtol=1e-2)

    def test_axi_voronoi(self):
        pass


class TestJAMWrapperAxiCylIso(object):
    """Test JAMWrapper with axisymmetric-cylindrical symmetry but in the spherical and
    isotropic limit q=1, beta=0, against Lenstronomy Galkin module for spherical
    symmetry."""

    def setup_method(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        lens_cosmo = LensCosmo(0.5, 1.2, cosmo=cosmo)

        self.ellipticities = {"e1": 0.0, "e2": 0.0}
        self.kwargs_light_spherical = {
            "Rs": 1.0,
            "amp": 1.0,
            "center_x": 0.0,
            "center_y": 0.0,
        }
        self.kwargs_lens_mass_spherical = {
            "theta_E": 1.5,
            "gamma": 2.1,
            "center_x": 0.0,
            "center_y": 0.0,
        }
        self.kwargs_anisotropy = {}
        self.inclination = 80.0
        kwargs_psf = {
            "psf_type": "GAUSSIAN",
            "fwhm": 0.5,
        }
        kwargs_cosmo = {
            "d_d": lens_cosmo.dd,
            "d_s": lens_cosmo.ds,
            "d_ds": lens_cosmo.dds,
        }
        kwargs_numerics_lenstronomy = {
            "interpol_grid_num": 2000,
            "log_integration": True,
            "max_integrate": 1e3,
            "min_integrate": 1e-3,
        }
        kwargs_numerics_mge = {
            "mge_n_gauss": 50,
            "mge_min_r": 1e-4,
            "mge_max_r": 300,
            "mge_n_radial": 500,
            "mge_linear_solver": True,
        }
        kwargs_numeric_jam = kwargs_numerics_lenstronomy | kwargs_numerics_mge
        kwargs_model = {
            "mass_profile_list": ["SPP"],
            "light_profile_list": ["HERNQUIST"],
            "anisotropy_model": "isotropic",
        }

        x = y = np.linspace(-5, 5, 20)
        x_grid, y_grid = np.meshgrid(x, y)
        kwargs_aperture_grid = {
            "aperture_type": "IFU_grid",
            "x_grid": x_grid,
            "y_grid": y_grid,
        }
        self.jam_axi_cyl_grid = JAMWrapper(
            kwargs_model=kwargs_model | {"symmetry": "axi_cyl"},
            kwargs_aperture=kwargs_aperture_grid,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numeric_jam,
        )
        self.galkin_grid = Galkin(
            kwargs_model=kwargs_model,
            kwargs_aperture=kwargs_aperture_grid,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numerics_lenstronomy,
            analytic_kinematics=False,
        )

        r_bins = np.linspace(0.1, 5, 11)
        kwargs_aperture_ifu_shells = {
            "aperture_type": "IFU_shells",
            "r_bins": r_bins,
            "center_ra": 0.0,
            "center_dec": 0.0,
        }
        self.jam_axi_cyl_shells = JAMWrapper(
            kwargs_model=kwargs_model | {"symmetry": "axi_cyl"},
            kwargs_aperture=kwargs_aperture_ifu_shells,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numeric_jam,
        )
        self.galkin_shells = GalkinShells(
            kwargs_model=kwargs_model,
            kwargs_aperture=kwargs_aperture_ifu_shells,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numerics_lenstronomy,
            analytic_kinematics=False,
        )

        kwargs_aperture_slit = {
            "aperture_type": "slit",
            "length": 3.0,
            "width": 0.5,
            "center_ra": 0.0,
            "center_dec": 0.0,
            "angle": np.deg2rad(0),
        }
        self.jam_axi_cyl_slit = JAMWrapper(
            kwargs_model=kwargs_model | {"symmetry": "axi_cyl"},
            kwargs_aperture=kwargs_aperture_slit,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numeric_jam,
        )
        self.galkin_slit = Galkin(
            kwargs_model=kwargs_model,
            kwargs_aperture=kwargs_aperture_slit,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numerics_lenstronomy,
            analytic_kinematics=False,
        )

    def test_axi_dispersion_grid(self):
        sigma_v_jam = self.jam_axi_cyl_grid.dispersion(
            [self.kwargs_lens_mass_spherical | self.ellipticities],
            [self.kwargs_light_spherical | self.ellipticities],
            self.kwargs_anisotropy,
            convolved=True,
        )
        sigma_v_galkin = self.galkin_grid.dispersion_map_grid_convolved(
            [self.kwargs_lens_mass_spherical],
            [self.kwargs_light_spherical],
            self.kwargs_anisotropy,
            supersampling_factor=5,
        )
        npt.assert_allclose(sigma_v_jam, sigma_v_galkin, rtol=1e-2)

    def test_axi_dispersion_slit(self):
        np.random.seed(0)
        sigma_v_jam = self.jam_axi_cyl_slit.dispersion(
            [self.kwargs_lens_mass_spherical | self.ellipticities],
            [self.kwargs_light_spherical | self.ellipticities],
            self.kwargs_anisotropy,
            convolved=True,
        )
        sigma_v_galkin = self.galkin_slit.dispersion(
            [self.kwargs_lens_mass_spherical],
            [self.kwargs_light_spherical],
            self.kwargs_anisotropy,
            sampling_number=5000,
        )
        npt.assert_allclose(sigma_v_jam, sigma_v_galkin, rtol=1e-2)

    def test_axi_dispersion_shells(self):
        sigma_v_jam = self.jam_axi_cyl_shells.dispersion(
            [self.kwargs_lens_mass_spherical | self.ellipticities],
            [self.kwargs_light_spherical | self.ellipticities],
            self.kwargs_anisotropy,
            convolved=True,
        )
        sigma_v_galkin = self.galkin_shells.dispersion_map(
            [self.kwargs_lens_mass_spherical],
            [self.kwargs_light_spherical],
            self.kwargs_anisotropy,
        )
        npt.assert_allclose(sigma_v_jam, sigma_v_galkin, rtol=1e-2)

    def test_axi_voronoi(self):
        pass


# TODO: implement test against JamPy analytical Hernquist x2 - OM case

# TODO: implement test for non-spherical cases with simulated data comparison

# TODO: implement test for voronoi binning

if __name__ == "__main__":
    pytest.main()
