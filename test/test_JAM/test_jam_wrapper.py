import numpy as np
import numpy.testing as npt
import pytest

from hierarc.JAM.jam_wrapper import JAMWrapper
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.GalKin.galkin import Galkin
from lenstronomy.LensModel.Profiles.spp import SPP

class TestJAMWrapper(object):

    def setup_method(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        lens_cosmo = LensCosmo(0.5, 1.2, cosmo=cosmo)

        theta_E = 1.0
        r_eff = 1
        gamma = 2.1

        x = y = np.linspace(-5, 5, 20)
        x_grid, y_grid = np.meshgrid(x, y)

        kwargs_psf = {
                "psf_type": "GAUSSIAN",
                "fwhm": 0.5  # TODO: seems that galkin shifts based on the FWHM value
            }
        kwargs_cosmo = {
                'd_d': lens_cosmo.dd, 'd_s': lens_cosmo.ds, 'd_ds': lens_cosmo.dds
            }
        kwargs_numerics_lenstronomy = {
                "interpol_grid_num": 1000,
                "log_integration": True,
                "max_integrate": 100,
                "min_integrate": 0.001,
            }
        kwargs_numerics_mge = {
            "mge_n_gauss": 50,
            "mge_min_r": 1e-3,
            "mge_max_r": 100,
            "mge_n_radial": 500,
            "mge_log_spacing": True,
        }
        kwargs_numeric_jam = kwargs_numerics_lenstronomy | kwargs_numerics_mge
        kwargs_aperture_shell = {
                "aperture_type": "shell",
                "r_in": 0,
                "r_out": 3 / 2.0,
                "center_ra": 0.0,
                "center_dec": 0,
            }
        kwargs_aperture_grid = {
            "aperture_type": "IFU_grid",
            "x_grid": x_grid,
            "y_grid": y_grid,
        }
        kwargs_model = {
            "mass_profile_list": ["SPP"],
            "light_profile_list": ["HERNQUIST"],
            "anisotropy_model": "OM",
        }

        self.jam_spherical_om_grid = JAMWrapper(
            kwargs_model=kwargs_model | {"symmetry": "spherical"},
            kwargs_aperture=kwargs_aperture_grid,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numeric_jam,
        )
        self.kwargs_light_spherical = [{"Rs": r_eff, "amp": 1.0}]
        self.kwargs_lens_mass_spherical = [
            {"theta_E": theta_E, "gamma": gamma, "center_x": 2.0, "center_y": -1.0}
        ]
        self.kwargs_anisotropy_om = {"r_ani": 1.0}

        self.galkin_om_grid = Galkin(
            kwargs_model=kwargs_model,
            kwargs_aperture=kwargs_aperture_grid,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numerics_lenstronomy,
            analytic_kinematics=False,
        )
        self.spp = SPP()
        galkin_psf = self.galkin_om_grid._get_convolution_kernel(fwhm_factor=3, supersampling_factor=1)
        print("galkin_psf =", galkin_psf.shape)

    def test_mge_lum(self):
        surf_lum, sigma_lum = self.jam_spherical_om_grid.mge_lum_tracer(self.kwargs_light_spherical)
        r_test = self.jam_spherical_om_grid._mge_radial_points
        mge_surf_1d = self._mge(r_test, surf_lum, sigma_lum)
        galkin_surf_1d = self.galkin_om_grid.numerics.lightProfile.light_2d(r_test, self.kwargs_light_spherical)
        npt.assert_allclose(mge_surf_1d, galkin_surf_1d, rtol=1e-2)

    def test_mge_mass(self):
        surf_mass, sigma_mass = self.jam_spherical_om_grid.mge_mass(self.kwargs_lens_mass_spherical)
        r_test = self.jam_spherical_om_grid._mge_radial_points
        mge_surf_1d = self._mge(r_test, surf_mass, sigma_mass)
        theta_E = self.kwargs_lens_mass_spherical[0]['theta_E']
        gamma = self.kwargs_lens_mass_spherical[0]['gamma']
        rho0 = self.spp.theta2rho(theta_E, gamma)
        spp_surf_1d = self.spp.density_2d(r_test, 0, rho0, gamma)
        # larger tolerance as MGE does not fit well EPL at large radii
        npt.assert_allclose(mge_surf_1d, spp_surf_1d, rtol=1e-1)

    def test_spherical_dispersion_slit(self):
        pass

    def test_spherical_dispersion_shells(self):
        pass

    def test_spherical_dispersion_grid(self):
        sigma_v_jam = self.jam_spherical_om_grid.dispersion_grid(
            self.kwargs_lens_mass_spherical,
            self.kwargs_light_spherical,
            self.kwargs_anisotropy_om,
            convolved=True,
        )
        sigma_v_galkin = self.galkin_om_grid.dispersion_map_grid_convolved(
            self.kwargs_lens_mass_spherical,
            self.kwargs_light_spherical,
            self.kwargs_anisotropy_om,
        )
        import matplotlib.pyplot as plt
        plt.figure(figsize=(18, 6))
        plt.subplot(131)
        plt.imshow(sigma_v_galkin / sigma_v_galkin.mean(),
                   extent=[-5, 5, -5, 5], origin='lower')
        plt.scatter(self.kwargs_lens_mass_spherical[0]['center_x'],
                    self.kwargs_lens_mass_spherical[0]['center_y'],
                    color='red')
        plt.subplot(132)
        plt.imshow(sigma_v_jam / sigma_v_jam.mean(),
                   extent=[-5, 5, -5, 5], origin='lower')
        plt.scatter(self.kwargs_lens_mass_spherical[0]['center_x'],
                    self.kwargs_lens_mass_spherical[0]['center_y'],
                    color='red')
        plt.subplot(133)
        plt.imshow(sigma_v_jam / sigma_v_jam.mean() - sigma_v_galkin / sigma_v_galkin.mean(),
                   extent=[-5, 5, -5, 5], origin='lower', cmap='coolwarm')
        plt.scatter(self.kwargs_lens_mass_spherical[0]['center_x'],
                    self.kwargs_lens_mass_spherical[0]['center_y'],
                    color='red')
        plt.show()

        npt.assert_allclose(sigma_v_jam, sigma_v_galkin, rtol=1e-2)

    def test_spherical_voronoi(self):
        pass

    @staticmethod
    def _gaussian(r, amp, sigma):
        return amp / np.sqrt(2 * np.pi) / sigma * np.exp(-0.5 * (r / sigma) ** 2)

    def _mge(self, r, amps, sigmas):
        total = np.zeros_like(r)
        for amp, sigma in zip(amps, sigmas):
            total += self._gaussian(r, amp, sigma)
        return total

if __name__ == "__main__":
    pytest.main()
