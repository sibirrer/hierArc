__author__ = "furcelay", "sibirrer"

from astropy.stats import gaussian_fwhm_to_sigma
from lenstronomy.GalKin.psf import PSF
from lenstronomy.GalKin.cosmo import Cosmo
from lenstronomy.Util.param_util import ellipticity2phi_q
from hierarc.JAM.aperture import Aperture
from hierarc.JAM.mass_profile import MassProfile
from hierarc.JAM.light_profile import LightProfile
from hierarc.JAM.jam_anisotropy import JAMAnisotropy
import mgefit as mge
import jampy as jam
import numpy as np


__all__ = ["JAMWrapperBase"]


class JAMWrapperBase(PSF, Aperture):
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
            msg = (f"Invalid symmetry type '{self.symmetry}' for JAMWrapper, "
                   f"options are 'spherical', 'axi_sph' or 'axi_cyl'.")
            raise ValueError(msg)
        self.psf_fwhm = kwargs_psf["fwhm"]
        if ("delta_pix" not in kwargs_aperture) and ("IFU" not in kwargs_aperture["aperture_type"]):
            # set the sampling of the aperture to FWHM/4
            kwargs_aperture = kwargs_aperture.copy()
            kwargs_aperture["delta_pix"] = min(self.psf_fwhm / 4, 0.1)
        Aperture.__init__(self, **kwargs_aperture)
        PSF.__init__(self, **kwargs_psf)
        self.cosmo = Cosmo(**kwargs_cosmo)

        if kwargs_numerics is None:
            kwargs_numerics = {
            }

        mge_n_gauss = kwargs_numerics.get("mge_n_gauss", 20)
        self._mge_n_gauss_mass = kwargs_numerics.get("mge_n_gauss_light", mge_n_gauss)
        self._mge_n_gauss_light = kwargs_numerics.get("mge_n_gauss_mass", mge_n_gauss)

        mge_min_r = kwargs_numerics.get("mge_min_r", 1e-4)  # TODO: check if this is ok
        mge_max_r = kwargs_numerics.get("mge_max_r", 300)   # TODO: check if this is ok
        mge_n_radial = kwargs_numerics.get("mge_n_radial", 500)  # TODO: check if this is ok
        mge_min_r_mass = kwargs_numerics.get("mge_min_r_mass", mge_min_r)  # relative to theta_E
        mge_max_r_mass = kwargs_numerics.get("mge_max_r_mass", mge_max_r)
        mge_n_radial_mass = kwargs_numerics.get("mge_n_radial_mass", mge_n_radial)
        mge_min_r_light = kwargs_numerics.get("mge_min_r_light", mge_min_r)  # relative to r_eff
        mge_max_r_light = kwargs_numerics.get("mge_max_r_light", mge_max_r)
        mge_n_radial_light = kwargs_numerics.get("mge_n_radial_light", mge_n_radial)
        self._mge_radial_points_mass = np.logspace(  # this must be in logspace
            np.log10(mge_min_r_mass),
            np.log10(mge_max_r_mass),
            mge_n_radial_mass,
        )
        self._mge_radial_points_light = np.logspace(
            np.log10(mge_min_r_light),
            np.log10(mge_max_r_light),
            mge_n_radial_light,
        )
        self._mge_linear_solver = kwargs_numerics.get("mge_linear_solver", True)  # use linear solver for MGE fit speed
        self._mge_kwargs_lum = kwargs_numerics.get("mge_kwargs_lum", {})
        self._mge_kwargs_mass = kwargs_numerics.get("mge_kwargs_mass", {"outer_slope": 2})


    def dispersion_points(
        self,
        x,
        y,
        kwargs_mass,
        kwargs_light,
        kwargs_anisotropy,
        inclination=90.0,
        convolved=False,
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
        :param convolved: bool, if True the PSF convolution is applied
        :param jam_kwargs: keyword arguments for JAM call
        :return: array of LOS velocity dispersion at each (x,y) position [km/s]
        """
        # TODO: read light and mass mge parameters from cached values if available

        surf_lum, sigma_lum = self.mge_lum_tracer(kwargs_light)
        surf_mass, sigma_mass = self.mge_mass(kwargs_mass)
        # convert to units of M_sun / pc^2
        surf_mass *= self.cosmo.epsilon_crit * 1e-12
        beta = self._anisotropy.beta_params(kwargs_anisotropy)
        if not self._anisotropy.use_logistic:
            beta = beta * np.ones_like(surf_lum)
        _, q_lum = ellipticity2phi_q(*self._extract_ellipticity(kwargs_light))
        _, q_mass = ellipticity2phi_q(*self._extract_ellipticity(kwargs_mass))
        if convolved:
            seeing_fwhm = self._psf.fwhm
            delta_pix = self.delta_pix
        else:
            seeing_fwhm = 0.0
            delta_pix = 0.0

        vrms, surf_bright = self.call_jampy(
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
        return vrms, surf_bright

    def mge_lum_tracer(self, kwargs_light):
        # TODO: cache the MGE fit for repeated calls with same kwargs_light
        profs = self._light_profile.profile_list
        if (len(profs) == 1) and (profs[0] in ['MULTI_GAUSSIAN', 'MULTI_GAUSSIAN_ELLIPSE']):
            sigma_lum = np.asarray(kwargs_light[0]['sigma'])
            surf_lum = np.asarray(kwargs_light[0]['amp']) / (2 * np.pi * sigma_lum**2)
        else:
            r_eff = self._light_profile.effective_radius(kwargs_light)
            light_1d = self._light_profile.radial_surface_brightness(
                self._mge_radial_points_light * r_eff,
                kwargs_light
            )
            mge_lum = mge.fit_1d(
                self._mge_radial_points_light * r_eff,
                light_1d,
                ngauss=self._mge_n_gauss_light,
                linear=self._mge_linear_solver,
                plot=False, quiet=True,
                **self._mge_kwargs_lum,
            )
            sigma_lum = mge_lum.sol[1]  # in arcsec
            # convert to surface brightness
            surf_lum = mge_lum.sol[0] / (np.sqrt(2 * np.pi) * sigma_lum)
        return surf_lum, sigma_lum

    def mge_mass(self, kwargs_mass):
        # TODO: cache the MGE fit for repeated calls with same kwargs_mass
        if self._mass_profile.profile_list == ['MULTI_GAUSSIAN']:
            sigma_mass = np.asarray(kwargs_mass[0]['sigma'])
            surf_mass = np.asarray(kwargs_mass[0]['amp']) / (2 * np.pi * sigma_mass)
        else:
            theta_E = self._mass_profile.einstein_radius(kwargs_mass)
            radial_density = self._mass_profile.radial_density(
                self._mge_radial_points_mass * theta_E,
                kwargs_mass
            )
            mge_mass = mge.fit_1d(
                self._mge_radial_points_mass * theta_E,
                radial_density,
                ngauss=self._mge_n_gauss_mass,
                linear=self._mge_linear_solver,
                plot=False, quiet=True,
                **self._mge_kwargs_mass,
            )
            surf_mass = mge_mass.sol[0]   # mass convergence
            sigma_mass = mge_mass.sol[1]  # in arcsec
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
            vrms, surf_bright = self.call_jampy_axi(
                surf_lum, sigma_lum, surf_mass, sigma_mass,
                x, y, q_lum, q_mass, inclination, beta,
                sigma_psf, pix_size, jam_kwargs
            )
        else:
            # TODO: evaluate at fixed radius and then interpolate for speed
            r = np.sqrt(x**2 + y**2)
            vrms, surf_bright = self.call_jampy_sph(
                surf_lum, sigma_lum, surf_mass, sigma_mass,
                r, beta, sigma_psf, pix_size, jam_kwargs
            )
        vrms = vrms.reshape(x_shape)
        surf_bright = surf_bright.reshape(x_shape)
        return vrms, surf_bright

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
        if jam_kwargs is None:
            jam_kwargs = {}
        jam_model = jam.axi.proj(
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
        )
        vrms = jam_model.model
        surf_bright = jam_model.flux
        return vrms, surf_bright

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
        if jam_kwargs is None:
            jam_kwargs = {}
        jam_model = jam.sph.proj(
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
            )
        vrms = jam_model.model
        surf_bright = jam_model.flux
        return vrms, surf_bright


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
    def _rotate_grid(x_grid, y_grid, phi):
        """Rotate the grid according to the ellipticity parameters.

        :param x_grid: x grid
        :param y_grid: y grid
        :param phi: angle in radians
        :return: x_rotated, y_rotated
        """
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        x_rotated = cos_phi * x_grid + sin_phi * y_grid
        y_rotated = -sin_phi * x_grid + cos_phi * y_grid

        return x_rotated, y_rotated

    def _shift_and_rotate(self, x, y, kwargs):
        center_x, center_y = self._extract_center(kwargs)
        x_shifted = x - center_x
        y_shifted = y - center_y
        e1, e2 = self._extract_ellipticity(kwargs)
        phi, q = ellipticity2phi_q(e1, e2)
        return self._rotate_grid(x_shifted, y_shifted, phi)
