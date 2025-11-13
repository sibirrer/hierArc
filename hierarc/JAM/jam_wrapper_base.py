__author__ = "furcelay,sbirrer"

from astropy.stats import gaussian_fwhm_to_sigma
from lenstronomy.GalKin.observation import GalkinObservation
from lenstronomy.GalKin.cosmo import Cosmo
from lenstronomy.Util.param_util import ellipticity2phi_q
from lenstronomy.Util import constants as const
from hierarc.JAM.mass_profile import MassProfile
from hierarc.JAM.light_profile import LightProfile
from hierarc.JAM.jam_anisotropy import JAMAnisotropy
import mgefit as mge
import jampy as jam
import numpy as np
import copy


__all__ = ["JAMWrapperBase"]


class JAMWrapperBase(GalkinObservation):
    """
    Wrapper class to use jampy JAM functionality similar to lenstronomy's Galkin class.

    :param kwargs_model: keyword arguments describing the model components
    :param kwargs_aperture: keyword arguments describing the spectroscopic aperture, see Aperture() class
    :param kwargs_psf: keyword argument specifying the PSF of the observation
    :param kwargs_cosmo: keyword arguments that define the cosmology in terms of the angular diameter distances
     involved
    :param kwargs_numerics: numerics keyword arguments
    """
    def __init__(
        self,
        kwargs_model,
        kwargs_aperture,
        kwargs_psf,
        kwargs_cosmo,
        kwargs_numerics=None,
    ):
        mass_profile_list = kwargs_model.get("mass_profile_list")
        light_profile_list = kwargs_model.get("light_profile_list")
        anisotropy_model = kwargs_model.get("anisotropy_model")
        self._light_profile = LightProfile(light_profile_list)
        self._mass_profile = MassProfile(mass_profile_list)
        self._anisotropy = JAMAnisotropy(anisotropy_model)
        self.cosmo = Cosmo(**kwargs_cosmo)

        self.symmetry = kwargs_model.get("symmetry", "spherical")  # 'spherical', 'axi_sph', 'axi_cyl'
        self.align = None
        self.axisymmetric = False
        if self.symmetry == 'spherical':
            self.align = 'sph'
            self.axisymmetric = False
        elif self.symmetry == 'axi_sph':
            self.axisymmetric = True
            self.align = 'sph'
        elif self.symmetry == 'axi_cyl':
            self.axisymmetric = True
            self.align = 'cyl'
        else:
            raise ValueError("Invalid symmetry type for JAMWrapper, "
                             "options are 'spherical', 'axi_sph' or 'axi_cyl'.")
        GalkinObservation.__init__(
            self, kwargs_aperture=kwargs_aperture, kwargs_psf=kwargs_psf
        )
        self.cosmo = Cosmo(**kwargs_cosmo)

        if kwargs_numerics is None:
            kwargs_numerics = {
            }

        self._mge_n_gauss = kwargs_numerics.get("mge_n_gauss", 20)  # TODO: split into mass and light
        self._mge_min_r = kwargs_numerics.get("mge_min_r", 1e-3)
        self._mge_max_r = kwargs_numerics.get("mge_max_r", 100)
        self._mge_n_radial = kwargs_numerics.get("mge_n_radial", 300)
        self._mge_log_spacing = kwargs_numerics.get("mge_log_spacing", True) # TODO: remove option
        self._mge_kwargs_lum = kwargs_numerics.get("mge_kwargs_lum", {})
        self._mge_kwargs_mass = kwargs_numerics.get("mge_kwargs_mass", {"outer_slope": 2})
        if kwargs_numerics.get("mge_log_spacing", True):
            self._mge_radial_points = np.logspace(
                np.log10(self._mge_min_r),
                np.log10(self._mge_max_r),
                self._mge_n_radial,
            )
        else:
            self._mge_radial_points = np.linspace(
                self._mge_min_r,
                self._mge_max_r,
                self._mge_n_radial,
            )

    def dispersion_points(
        self,
        x,
        y,
        kwargs_mass,
        kwargs_light,
        kwargs_anisotropy,
        inclination=90.0,
        convolved=False,
        psf_supersampling_factor=1,
        jam_kwargs=None,
        ):
        """Computes the LOS velocity dispersion at given points (not convolved).
        :param x: array of x positions where to compute the dispersion [arcsec]
        :param y: array of y positions where to compute the dispersion [arcsec].
        :param kwargs_mass: mass model parameters (following lenstronomy lens model
            conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light
            model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to
            anisotropy type chosen.
        :param inclination: inclination angle of the system [degrees]
        :param psf_supersampling_factor: int, supersampling factor for PSF convolution
        :param convolved: bool, if True the PSF convolution is applied
        :param jam_kwargs: keyword arguments for JAM call
        :return: array of LOS velocity dispersion at each (x,y) position [km/s]
        """
        # TODO: read light and mass mge parameters from cached values if available

        surf_lum, sigma_lum = self.mge_lum_tracer(kwargs_light)
        surm_mass, sigma_mass = self.mge_mass(kwargs_mass)
        beta = self._anisotropy.beta_params(
            kwargs_anisotropy,
            n_gauss=self._mge_n_gauss
        )
        phi_lum, q_lum = ellipticity2phi_q(*self._extract_ellipticity(kwargs_light))
        phi_mass, q_mass = ellipticity2phi_q(*self._extract_ellipticity(kwargs_mass))

        if convolved:
            seeing_fwhm = self._psf.fwhm
            delta_x, delta_y = self._delta_pix_xy()
            delta_pix = (delta_x + delta_y) / 2.0 / psf_supersampling_factor
        else:
            seeing_fwhm = 0.0
            delta_pix = 0.0

        vrms = self.call_jampy(
            surf_lum, sigma_lum, surm_mass, sigma_mass,
            x=x, y=y,
            q_lum=q_lum * np.ones_like(surf_lum),
            q_mass=q_mass * np.ones_like(surm_mass),
            inclination=inclination,
            beta=beta,
            sigma_psf=seeing_fwhm * gaussian_fwhm_to_sigma,
            pix_size=delta_pix,
            jam_kwargs=jam_kwargs,
        )
        return vrms

    def mge_lum_tracer(self, kwargs_light):
        # TODO: cache the MGE fit for repeated calls with same kwargs_light
        if self._light_profile.profile_list == ['MULTI_GAUSSIAN']:
            surf_lum = kwargs_light[0]['amp']
            sigma_lum = kwargs_light[0]['sigma']
        else:
            light_1d = self._light_profile.radial_surface_brightness(
                self._mge_radial_points,
                kwargs_light
            )
            mge_lum = mge.fit_1d(
                self._mge_radial_points,
                light_1d,
                ngauss=self._mge_n_gauss,
                plot=False, quiet=True,
                **self._mge_kwargs_lum,
            )
            surf_lum = mge_lum.sol[0]
            sigma_lum = mge_lum.sol[1]
            if len(surf_lum) < self._mge_n_gauss: # TODO: maybe not needed
                # pad with zeros
                n_missing = self._mge_n_gauss - len(surf_lum)
                surf_lum = np.concatenate([surf_lum, np.zeros(n_missing)])
                sigma_lum = np.concatenate([sigma_lum, np.ones(n_missing)])
        return surf_lum, sigma_lum

    def mge_mass(self, kwargs_mass):
        # TODO: cache the MGE fit for repeated calls with same kwargs_mass
        radial_convergence = self._mass_profile.radial_convergence(
            self._mge_radial_points,
            kwargs_mass
        )
        # times Sigma_crit to get surface mass density
        radial_convergence *= self.cosmo.epsilon_crit * const.arcsec ** 2
        mge_mass = mge.fit_1d(
            self._mge_radial_points,
            radial_convergence,
            ngauss=self._mge_n_gauss,
            plot=False, quiet=True,
            **self._mge_kwargs_mass,
        )
        surf_mass = mge_mass.sol[0]
        sigma_mass = mge_mass.sol[1]
        if len(surf_mass) < self._mge_n_gauss:
            # pad with zeros
            n_missing = self._mge_n_gauss - len(surf_mass)
            surf_mass = np.concatenate([surf_mass, np.zeros(n_missing)])
            sigma_mass = np.concatenate([sigma_mass, np.ones(n_missing)])
        return surf_mass, sigma_mass

    def call_jampy(
        self,
        surf_lum,
        sigma_lum,
        surf_mass,
        sigma_mass,
        x,
        y=None,
        q_lum=None,
        q_mass=None,
        inclination=90.0,
        beta=None,
        sigma_psf=0.0,
        pix_size=0.0,
        jam_kwargs=None,
    ):
        x = np.asarray(x)
        if y is None:
            y = np.zeros_like(x)
        y = np.asarray(y)
        x_shape = x.shape
        x = x.flatten()
        y = y.flatten()
        if jam_kwargs is None:
            jam_kwargs = {}
        if "mbh" not in jam_kwargs:
            jam_kwargs["mbh"] = 0.0
        if self.axisymmetric:
            return self.call_jampy_axi(
                surf_lum, sigma_lum, surf_mass, sigma_mass,
                x, y, q_lum, q_mass, inclination, beta,
                sigma_psf, pix_size, jam_kwargs
            ).reshape(x_shape)
        else:
            # TODO: evaluate at fixed radius and then interpolate for speed
            r = np.sqrt(x**2 + y**2)
            return self.call_jampy_sph(
                surf_lum, sigma_lum, surf_mass, sigma_mass,
                r, beta, sigma_psf, pix_size, jam_kwargs
            ).reshape(x_shape)

    def call_jampy_axi(
        self,
        surf_lum,
        sigma_lum,
        surf_mass,
        sigma_mass,
        x,
        y,
        q_lum=None,
        q_mass=None,
        inclination=90.0,
        beta=None,
        sigma_psf=0.0,
        pix_size=0.0,
        jam_kwargs=None,
    ):
        return jam.axi.proj(
            surf_lum,
            sigma_lum,
            q_lum,
            surf_mass,
            sigma_mass,
            q_mass,
            xbin=x,
            ybin=y,
            inc=inclination,
            align=self.align,
            distance=self.cosmo.dd,
            beta=beta,
            logistic=self._anisotropy.use_logistic,
            sigmapsf=sigma_psf,
            pixsize=pix_size,
            quiet=True,
            plot=False,
            **jam_kwargs,
        ).model

    def call_jampy_sph(
        self,
        surf_lum,
        sigma_lum,
        surf_mass,
        sigma_mass,
        r,
        beta=None,
        sigma_psf=0.0,
        pix_size=0.0,
        jam_kwargs=None,
    ):
        return jam.sph.proj(
                surf_lum,
                sigma_lum,
                surf_mass,
                sigma_mass,
                rad=r,
                distance=self.cosmo.dd,
                beta=beta,
                logistic=self._anisotropy.use_logistic,
                sigmapsf=sigma_psf,
                pixsize=pix_size,
                quiet=True,
                plot=False,
                **jam_kwargs,
            ).model

    @staticmethod
    def _extract_center(kwargs):
        if not isinstance(kwargs, dict):
            if "center_x" in kwargs[0]:
                return kwargs[0]["center_x"], kwargs[0]["center_y"]
            else:
                return 0, 0
        else:
            if "center_x" in kwargs:
                return kwargs["center_x"], kwargs["center_y"]
            else:
                return 0, 0

    @staticmethod
    def _extract_ellipticity(kwargs):
        if not isinstance(kwargs, dict):
            if "e1" in kwargs[0]:
                return kwargs[0]["e1"], kwargs[0]["e2"]
            else:
                return 0, 0
        else:
            if "e1" in kwargs:
                return kwargs["e1"], kwargs["e2"]
            else:
                return 0, 0

    @staticmethod
    def _rotate_grid(x_grid, y_grid, e1, e2):
        """Rotate the grid according to the ellipticity parameters.

        :param x_grid: x grid
        :param y_grid: y grid
        :param e1: e1 ellipticity parameter
        :param e2: e2 ellipticity parameter
        :return: x_rotated, y_rotated
        """
        phi = 0.5 * np.arctan2(e2, e1)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        x_rotated = cos_phi * x_grid + sin_phi * y_grid
        y_rotated = -sin_phi * x_grid + cos_phi * y_grid

        return x_rotated, y_rotated

    def _delta_pix_xy(self):
        """Get the pixel scale of the grid.

        :return: delta_x, delta_y
        """
        x_grid = self._aperture.x_grid
        y_grid = self._aperture.y_grid
        delta_x = x_grid[0, 1] - x_grid[0, 0]
        delta_y = y_grid[1, 0] - y_grid[0, 0]

        return delta_x, delta_y

    def _get_grid(self, kwargs_mass, supersampling_factor=1):
        """Compute the grid to compute the dispersion map on.

        The grid is supersampled and also shifted and rotated to align with the galaxy

        :param kwargs_mass: keyword arguments of the mass model
        :param supersampling_factor: sampling factor for the grid to do the 2D
            convolution on
        :return: x_grid, y_grid, log10_radial_distance_from_center
        """
        mass_center_x, mass_center_y = self._extract_center(kwargs_mass)

        delta_x, delta_y = self._delta_pix_xy()
        assert np.abs(delta_x) == np.abs(delta_y)

        x_grid = self._aperture.x_grid
        y_grid = self._aperture.y_grid

        new_delta_x = delta_x / supersampling_factor
        new_delta_y = delta_y / supersampling_factor
        x_start = x_grid[0, 0] - delta_x / 2.0 * (1 - 1 / supersampling_factor)
        x_end = x_grid[0, -1] + delta_x / 2.0 * (1 - 1 / supersampling_factor)
        y_start = y_grid[0, 0] - delta_y / 2.0 * (1 - 1 / supersampling_factor)
        y_end = y_grid[-1, 0] + delta_y / 2.0 * (1 - 1 / supersampling_factor)

        xs = np.arange(x_start, x_end * (1 + 1e-6), new_delta_x)
        ys = np.arange(y_start, y_end * (1 + 1e-6), new_delta_y)

        x_grid_supersampled, y_grid_supersmapled = np.meshgrid(xs, ys)

        # shift to mass center
        x_grid_supersampled -= mass_center_x
        y_grid_supersmapled -= mass_center_y

        # rotate grid according to mass ellipticity
        e1_mass, e2_mass = self._extract_ellipticity(kwargs_mass)
        x_grid_supersampled, y_grid_supersmapled = self._rotate_grid(
            x_grid_supersampled, y_grid_supersmapled, e1_mass, e2_mass
        )

        log10_radial_distance_from_center = np.log10(
            np.sqrt(x_grid_supersampled ** 2 + y_grid_supersmapled  ** 2)
        )

        return (
            x_grid_supersampled,
            y_grid_supersmapled,
            log10_radial_distance_from_center,
        )

    def _downsample_to_aperture(
        self,
        high_res_map,
        supersampling_factor,
    ):
        """Downsamples a high-resolution map to the aperture grid.
        """
        num_pix_x, num_pix_y = self._aperture.num_segments
        return high_res_map.reshape(num_pix_y, supersampling_factor,
                                    num_pix_x, supersampling_factor).mean(axis=(1, 3))
