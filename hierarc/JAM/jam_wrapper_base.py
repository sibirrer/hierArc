__author__ = "furcelay,sbirrer"

from astropy.stats import gaussian_fwhm_to_sigma
from lenstronomy.GalKin.observation import GalkinObservation
from lenstronomy.GalKin.cosmo import Cosmo
from lenstronomy.Util.param_util import ellipticity2phi_q
from hierarc.JAM.mass_profile import MassProfile
from hierarc.JAM.light_profile import LightProfile
from hierarc.JAM.jam_anisotropy import JAMAnisotropy
import mgefit as mge
import jampy as jam
import numpy as np


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
        self._mge_min_r = kwargs_numerics.get("mge_min_r", 1e-4)  # in arcsec
        self._mge_max_r = kwargs_numerics.get("mge_max_r", 300)   # in arcsec
        self._mge_n_radial = kwargs_numerics.get("mge_n_radial", 500)
        self._mge_linear_solver = kwargs_numerics.get("mge_linear_solver", True)  # use linear solver for MGE fit speed
        self._mge_kwargs_lum = kwargs_numerics.get("mge_kwargs_lum", {})
        self._mge_kwargs_mass = kwargs_numerics.get("mge_kwargs_mass", {"outer_slope": 2})
        self._mge_radial_points = np.logspace(  # this must be in logspace
            np.log10(self._mge_min_r),
            np.log10(self._mge_max_r),
            self._mge_n_radial,
        )
        if self.aperture_type == "IFU_grid":
            delta_x, delta_y = self._delta_pix_xy()
            self._delta_pix = (np.abs(delta_x) + np.abs(delta_y)) / 2
        else:
            # TODO: select the best value for other aperture type
            self._delta_pix = kwargs_numerics.get("delta_pix", 0.2)  # in arcsec

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
        :param y: array of y positions where to compute the dispersion [arcsec]
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
        surf_mass, sigma_mass = self.mge_mass(kwargs_mass)
        # convert to units of M_sun / pc^2
        surf_mass *= self.cosmo.epsilon_crit * 1e-12
        beta = self._anisotropy.beta_params(
            kwargs_anisotropy,
            n_gauss=self._mge_n_gauss
        )
        _, q_lum = ellipticity2phi_q(*self._extract_ellipticity(kwargs_light))
        _, q_mass = ellipticity2phi_q(*self._extract_ellipticity(kwargs_mass))
        if convolved:
            seeing_fwhm = self._psf.fwhm
            delta_pix = self._delta_pix / psf_supersampling_factor
        else:
            seeing_fwhm = 0.0
            delta_pix = 0.0

        vrms = self.call_jampy(
            surf_lum, sigma_lum, surf_mass, sigma_mass,
            x=x, y=y,
            q_lum=q_lum * np.ones_like(surf_lum),
            q_mass=q_mass * np.ones_like(surf_mass),
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
                linear=self._mge_linear_solver,
                plot=False, quiet=True,
                **self._mge_kwargs_lum,
            )
            sigma_lum = mge_lum.sol[1]  # in arcsec
            # convert to surface brightness
            surf_lum = mge_lum.sol[0] / (np.sqrt(2 * np.pi) * sigma_lum)
            if len(surf_lum) < self._mge_n_gauss: # TODO: maybe not needed
                # pad with zeros
                n_missing = self._mge_n_gauss - len(surf_lum)
                surf_lum = np.concatenate([surf_lum, np.zeros(n_missing)])
                sigma_lum = np.concatenate([sigma_lum, np.ones(n_missing)])
        return surf_lum, sigma_lum

    def mge_mass(self, kwargs_mass):
        # TODO: cache the MGE fit for repeated calls with same kwargs_mass
        radial_density = self._mass_profile.radial_density(
            self._mge_radial_points,
            kwargs_mass
        )
        mge_mass = mge.fit_1d(
            self._mge_radial_points,
            radial_density,
            ngauss=self._mge_n_gauss,
            linear=self._mge_linear_solver,
            plot=False, quiet=True,
            **self._mge_kwargs_mass,
        )
        surf_mass = mge_mass.sol[0]   # mass convergence
        sigma_mass = mge_mass.sol[1]  # in arcsec
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

    def _delta_pix_xy(self):
        """Get the pixel scale of the grid.

        :return: delta_x, delta_y
        """
        if self.aperture_type == "IFU_grid":
            x_grid = self._aperture.x_grid
            y_grid = self._aperture.y_grid
            delta_x = x_grid[0, 1] - x_grid[0, 0]
            delta_y = y_grid[1, 0] - y_grid[0, 0]
            return delta_x, delta_y
        else:
            return 0., 0.
