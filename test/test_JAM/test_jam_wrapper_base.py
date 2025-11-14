import numpy as np
import numpy.testing as npt
import pytest

from hierarc.JAM.jam_wrapper import JAMWrapper
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.GalKin.galkin import Galkin
from lenstronomy.LensModel.Profiles.spp import SPP


class TestJAMWrapperBase(object):

    def setup_method(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        lens_cosmo = LensCosmo(0.5, 1.2, cosmo=cosmo)

        kwargs_psf = {
                "psf_type": "GAUSSIAN",
                "fwhm": 0.5  # TODO: seems that galkin shifts based on the FWHM value
            }
        kwargs_cosmo = {
                'd_d': lens_cosmo.dd, 'd_s': lens_cosmo.ds, 'd_ds': lens_cosmo.dds
            }
        kwargs_numerics_lenstronomy = {
                "interpol_grid_num": 2000,
                "log_integration": True,
                "max_integrate": 1e3,
                "min_integrate": 1e-3,
            }
        kwargs_numerics_mge = {
            "mge_n_gauss": 50,
            "mge_min_r": 1e-3,
            "mge_max_r": 100,
            "mge_n_radial": 500,
            "mge_log_spacing": True,
        }
        kwargs_numeric_jam = kwargs_numerics_lenstronomy | kwargs_numerics_mge
        kwargs_aperture = { # not used in this test
            "aperture_type": "slit",
            "length": 3,
            "width": 0.2,
        }
        kwargs_model = {
            "mass_profile_list": ["SPP"],
            "light_profile_list": ["HERNQUIST"],
            "anisotropy_model": "OM",
        }

        self.jam_spherical = JAMWrapper(
            kwargs_model=kwargs_model | {"symmetry": "spherical"},
            kwargs_aperture=kwargs_aperture,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numeric_jam,
        )
        self.kwargs_light = [{"Rs": 0.5, "amp": 1.0}]
        self.kwargs_lens_mass = [{"theta_E": 1.0, "gamma": 2.1}]
        self.kwargs_anisotropy = {"r_ani": 1.0}

        self.galkin_analytic = Galkin(
            kwargs_model=kwargs_model,
            kwargs_aperture=kwargs_aperture,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numerics_lenstronomy,
            analytic_kinematics=False,
        )
        self.spp = SPP()

    def test_mge_lum(self):
        surf_lum, sigma_lum = self.jam_spherical.mge_lum_tracer(self.kwargs_light)
        r_test = self.jam_spherical._mge_radial_points
        mge_surf_1d = self._mge(r_test, surf_lum, sigma_lum)
        galkin_surf_1d = self.galkin_analytic.numerics.lightProfile.light_2d(r_test, self.kwargs_light)
        npt.assert_allclose(mge_surf_1d, galkin_surf_1d, rtol=1e-2)

    def test_mge_mass(self):
        surf_mass, sigma_mass = self.jam_spherical.mge_mass(self.kwargs_lens_mass)
        r_test = self.jam_spherical._mge_radial_points
        mge_surf_1d = self._mge(r_test, surf_mass, sigma_mass)
        theta_E = self.kwargs_lens_mass[0]['theta_E']
        gamma = self.kwargs_lens_mass[0]['gamma']
        rho0 = self.spp.theta2rho(theta_E, gamma)
        galkin_surf_1d = self.spp.density_2d(r_test, 0, rho0, gamma)
        # tolerance is larger as the MGE fit is not perfect for a power-law profile
        npt.assert_allclose(mge_surf_1d, galkin_surf_1d, rtol=1e-1)
        # check with smaller tolerance skipping the edges
        npt.assert_allclose(mge_surf_1d[10:-30], galkin_surf_1d[10:-30], rtol=1e-2)

    def test_dispersion_points_unconvolved(self):
        r_test = np.logspace(-2, 3, 100)
        sigma_v_jam = self.jam_spherical.dispersion_points(
            x=r_test,
            y=np.zeros_like(r_test),
            kwargs_mass=self.kwargs_lens_mass,
            kwargs_light=self.kwargs_light,
            kwargs_anisotropy=self.kwargs_anisotropy,
            convolved=False,
        )
        sigma2_IR_galkin, IR_galkin = self.galkin_analytic.numerics.I_R_sigma2_and_IR(
            r_test,
            kwargs_mass=self.kwargs_lens_mass,
            kwargs_light=self.kwargs_light,
            kwargs_anisotropy=self.kwargs_anisotropy
        )
        sigma_v_galkin = np.sqrt(sigma2_IR_galkin / IR_galkin) / 1000
        import matplotlib.pyplot as plt
        plt.plot(r_test, sigma_v_jam)
        plt.plot(r_test, sigma_v_galkin, '--')
        plt.xscale('log')
        plt.show()
        npt.assert_allclose(sigma_v_jam, sigma_v_galkin, rtol=1e-3)

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
