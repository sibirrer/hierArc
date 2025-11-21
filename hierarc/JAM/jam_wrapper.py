__author__ = "furcelay,sbirrer"

from lenstronomy.Util.param_util import ellipticity2phi_q
from hierarc.JAM.jam_wrapper_base import JAMWrapperBase
import numpy as np
from lenstronomy.Util import util
from scipy import signal

__all__ = ["JAMWrapper"]


class JAMWrapper(JAMWrapperBase):
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

        super(JAMWrapper, self).__init__(
            kwargs_model,
            kwargs_aperture,
            kwargs_psf,
            kwargs_cosmo,
            kwargs_numerics
        )

    def dispersion(
        self,
        kwargs_mass,
        kwargs_light,
        kwargs_anisotropy,
        inclination=90.0,
        convolved=True,
        **kwargs,
    ):
        if self.aperture_type == "slit":
            return self.dispersion_slit(
                kwargs_mass,
                kwargs_light,
                kwargs_anisotropy,
                inclination=inclination,
                convolved=convolved,
                **kwargs,
            )
        elif self.aperture_type == "IFU_shells":
            if self.symmetry == "spherical":
                return self.dispersion_shells_spherical(
                    kwargs_mass,
                    kwargs_light,
                    kwargs_anisotropy,
                    inclination=inclination,
                    convolved=convolved,
                    **kwargs,
                )
            else:
                return self.dispersion_shells_axisymmetric(
                    kwargs_mass,
                    kwargs_light,
                    kwargs_anisotropy,
                    inclination=inclination,
                    convolved=convolved,
                    **kwargs
                )
        elif self.aperture_type == "IFU_grid":
            return self.dispersion_grid(
                kwargs_mass,
                kwargs_light,
                kwargs_anisotropy,
                inclination=inclination,
                convolved=convolved,
                **kwargs
            )
        else:
            raise ValueError("Invalid aperture type.")
        # TODO: add Voronoi binned case

    def dispersion_slit(
        self,
        kwargs_mass,
        kwargs_light,
        kwargs_anisotropy,
        inclination=90.0,
        convolved=True,
        sampling_number=1000
    ):
        """Computes the averaged LOS velocity dispersion in the slit (convolved)

        :param kwargs_mass: mass model parameters (following lenstronomy lens model
            conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light
            model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to
            anisotropy type chosen. We refer to the Anisotropy() class for details on
            the parameters.
        :param inclination: inclination angle of the system [degrees]
        :return: integrated LOS velocity dispersion in units [km/s]
        """

        x, y = self._draw_slit_points(
            kwargs_mass,
            sampling_number,
            convolved=convolved,
            fwhm_factor=3
        )

        IR = self._light_profile.light_model.surface_brightness(x, y, kwargs_light)
        vrms = self.dispersion_points(
            x,
            y,
            kwargs_mass,
            kwargs_light,
            kwargs_anisotropy,
            inclination=inclination,
            convolved=False,
            psf_supersampling_factor=1,
        )
        sigma2_IR = vrms**2 * IR
        return np.sqrt(sigma2_IR.sum() / IR.sum()), (x, y)
        # # old implementation below
        # x_slit, y_slit = self._get_slit_points(kwargs_mass, sampling_number)
        # x_slit_displaced, y_slit_displaced = [], []
        # for x, y in zip(x_slit.flatten(), y_slit.flatten()):
        #     x_disp, y_disp = self.displace_psf(x, y)
        #     x_slit_displaced.append(x_disp)
        #     y_slit_displaced.append(y_disp)
        # x_slit = np.array(x_slit_displaced)
        # y_slit = np.array(y_slit_displaced)
        # vrms_slit = self.dispersion_points(
        #     x_slit, y_slit,
        #     kwargs_mass,
        #     kwargs_light,
        #     kwargs_anisotropy,
        #     inclination=inclination,
        #     convolved=False,
        # )
        # return np.mean(vrms_slit)

    def dispersion_grid(
        self,
        kwargs_mass,
        kwargs_light,
        kwargs_anisotropy,
        inclination=90.0,
        convolved=True,
        supersampling_factor=1,
    ):
        """Computes the velocity dispersion in each Integral Field Unit.

        :param kwargs_mass: keyword arguments of the mass model
        :param kwargs_light: keyword argument of the light model
        :param kwargs_anisotropy: anisotropy keyword arguments
        :param inclination: inclination angle of the system [degrees]
        :param convolved: bool, if True the PSF convolution is applied
        :param supersampling_factor: sampling factor for the grid to do the 2D
            convolution on
        :return: ordered array of velocity dispersions [km/s] for each unit
        """
        x_sup, y_sup = self._get_IFU_grid(kwargs_mass, supersampling_factor)
        vrms_sup = self.dispersion_points(
            x_sup, y_sup,
            kwargs_mass,
            kwargs_light,
            kwargs_anisotropy,
            inclination=inclination,
            convolved=convolved,
            psf_supersampling_factor=supersampling_factor,
        )
        vrms = self._downsample_grid(
            vrms_sup,
            supersampling_factor=supersampling_factor
        )
        return vrms

    def dispersion_voronoi(
        self,
        kwargs_mass,
        kwargs_light,
        kwargs_anisotropy,
        inclination=90.0,
        convolved=True,
        supersampling_factor=1,
        voronoi_bins=None,
    ):
        pass

    def dispersion_shells_from_grid(
        self,
        kwargs_mass,
        kwargs_light,
        kwargs_anisotropy,
        inclination=90.0,
        convolved=True,
        supersampling_factor=1,
    ):
        # this is an experiment to make it closer to the GalkinShells implementation
        r_bins = self._aperture._r_bins
        r_max = np.max(r_bins)
        delta_pix = 1.5 * r_max * 2 / 100
        x_grid, y_grid = util.make_grid(numPix=100 * supersampling_factor, deltapix=delta_pix/ supersampling_factor)
        r_grid = np.sqrt(x_grid ** 2 + y_grid ** 2)
        vrms_sup = self.dispersion_points(
            x_grid, y_grid,
            kwargs_mass,
            kwargs_light,
            kwargs_anisotropy,
            inclination=inclination,
            convolved=convolved,
            psf_supersampling_factor=supersampling_factor,
        )
        # surf_bright = self._light_profile.light_model.surface_brightness(
        #     x_grid, y_grid, kwargs_light
        # )
        # kernel = self.convolution_kernel_grid(x_grid, y_grid)
        # surf_bright = signal.fftconvolve(surf_bright, kernel, mode="same")

        vrms = np.zeros(self._aperture.num_segments)
        # iterate over bin edges and average masked values
        for i in range(self._aperture.num_segments):
            mask = (r_grid >= r_bins[i]) & (r_grid < r_bins[i + 1])
            # vrms[i] = np.sum(vrms_sup[mask] * surf_bright[mask]) / np.sum(surf_bright[mask])
            vrms[i] = np.mean(vrms_sup[mask])
        return vrms

    def dispersion_shells_spherical(
        self,
        kwargs_mass,
        kwargs_light,
        kwargs_anisotropy,
        inclination=90.0,
        convolved=True,
        supersampling_factor=1,
    ):
        r_sup = self._get_shells_spherical(supersampling_factor)
        vrms_sup = self.dispersion_points(
            r_sup,
            None,
            kwargs_mass,
            kwargs_light,
            kwargs_anisotropy,
            inclination=inclination,
            convolved=convolved,
            psf_supersampling_factor=supersampling_factor,
        )
        vrms = self._downsample_shells_radially(
            vrms_sup,
            supersampling_factor
        )
        return vrms

    def dispersion_shells_axisymmetric(
        self,
        kwargs_mass,
        kwargs_light,
        kwargs_anisotropy,
        inclination=90.0,
        convolved=True,
        supersampling_factor=1,
    ):
        x_sup, y_sup, n_angular = self._get_shells_axisymmetric(kwargs_mass, supersampling_factor)
        vrms_sup2 = self.dispersion_points(
            x_sup,
            y_sup,
            kwargs_mass,
            kwargs_light,
            kwargs_anisotropy,
            inclination=inclination,
            convolved=convolved,
            psf_supersampling_factor=supersampling_factor,
        )
        vrms_sup = self._downsample_shells_angularly(
            vrms_sup2,
            n_angular=n_angular
        )
        vrms = self._downsample_shells_radially(
            vrms_sup,
            supersampling_factor
        )
        return vrms

    def _draw_one_sigma2(
        self,
        kwargs_mass,
        kwargs_light,
        kwargs_anisotropy,
        inclination=90.0,
        convolved=True,
        fwhm_factor=3
    ):
        """

        :param kwargs_mass: mass model parameters (following lenstronomy lens model conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to anisotropy type chosen.
            We refer to the Anisotropy() class for details on the parameters.
        :return: integrated LOS velocity dispersion in angular units for a single draw of the light distribution that
         falls in the aperture after displacing with the seeing
        """
        while True:
            x, y = np.random.uniform(-self._mge_max_r, self._mge_max_r, size=2)
            if convolved:
                x, y = self.displace_psf(x, y)
            bool_ap, _ = self.aperture_select(x, y)
            if bool_ap is True:
                break
        IR = self._light_profile.light_model.surface_brightness(x, y, kwargs_light)
        vrms = self.dispersion_points(
            x,
            y,
            kwargs_mass,
            kwargs_light,
            kwargs_anisotropy,
            inclination=inclination,
            convolved=False,
            psf_supersampling_factor=1,
        )
        sigma2_IR = vrms**2 * IR
        return sigma2_IR, IR

    def _draw_slit_points(self, kwargs_mass, sampling_number, convolved=True, fwhm_factor=3):

        def draw_point(dx, dy, slit_x, slit_y, slit_angle, lens_x, lens_y, lens_angle):
            while True:
                x, y = np.random.random(2)
                x = (x - 0.5) * dx
                y = (y - 0.5) * dy
                if convolved:
                    x, y = self.displace_psf(x, y)
                x, y = self._rotate_grid(x, y, slit_angle)
                x += slit_x
                y += slit_y
                bool_ap, _ = self.aperture_select(x, y)
                if bool_ap is True:
                    # shift to mass center
                    x -= lens_x
                    y -= lens_y
                    # rotate grid according to mass major axis
                    x, y = self._rotate_grid(x, y, -lens_angle)
                    return x, y

        x_samples, y_samples = [], []
        mass_center_x, mass_center_y = self._extract_center(kwargs_mass)
        mass_e1, mass_e2 = self._extract_center(kwargs_mass)
        phi_mass, q_mass = ellipticity2phi_q(mass_e1, mass_e2)
        length, width = self._aperture._length, self._aperture._width
        slit_ra, slit_dec = self._aperture._center_ra, self._aperture._center_dec
        slit_angle = self._aperture._angle

        # sample including PSF displacement
        sample_dx = length + fwhm_factor * self._psf._fwhm
        sample_dy = width + fwhm_factor * self._psf._fwhm

        for _ in range(sampling_number):
            x_draw, y_draw = draw_point(
                sample_dx, sample_dy,
                slit_ra, slit_dec, slit_angle,
                mass_center_x, mass_center_y, phi_mass
            )
            x_samples.append(x_draw)
            y_samples.append(y_draw)
        x_samples = np.array(x_samples)
        y_samples = np.array(y_samples)
        return x_samples, y_samples


    def _get_IFU_grid(self, kwargs_mass, supersampling_factor=1):
        """Compute the grid to compute the dispersion map on.

        The grid is supersampled and also shifted and rotated to align with the galaxy

        :param kwargs_mass: keyword arguments of the mass model
        :param supersampling_factor: sampling factor for the grid to do the 2D
            convolution on
        :return: x_grid, y_grid
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
        # TODO: maybe is better to use the light ellipticity?
        e1_mass, e2_mass = self._extract_ellipticity(kwargs_mass)
        phi_mass, q_mass = ellipticity2phi_q(e1_mass, e2_mass)
        x_grid_supersampled, y_grid_supersmapled = self._rotate_grid(
            x_grid_supersampled, y_grid_supersmapled, -phi_mass
        )
        return x_grid_supersampled, y_grid_supersmapled

    def _downsample_grid(
        self,
        high_res_map,
        supersampling_factor=1,
    ):
        """Downsamples a high-resolution map to the aperture grid.
        """
        num_pix_x, num_pix_y = self._aperture.num_segments
        return high_res_map.reshape(num_pix_y, supersampling_factor,
                                    num_pix_x, supersampling_factor).mean(axis=(1, 3))

    def _get_shells_spherical(self, supersampling_factor):
        """radial shells with equal spacing in r for spherical shells
        """
        # create a radial grid
        r_bins = self._aperture._r_bins
        r_grid = np.arange(np.min(r_bins), np.max(r_bins) * (1 + 1e-6), self._delta_pix / supersampling_factor)
        return r_grid

    def _get_shells_axisymmetric(self, kwargs_mass, supersampling_factor):
        """create a 2D grid in polar coordinates for spherical shells with axisymmetric model
        shells are then integrated angularly
        :param kwargs_mass: keyword arguments of the mass model
        :param supersampling_factor: sampling factor for the grid
        :return: x_grid, y_grid, n_angular (number of points per shell)
        """
        r_grid = self._get_shells_spherical(supersampling_factor)

        # TODO: implement for elliptical shells
        e1_mass, e2_mass = self._extract_ellipticity(kwargs_mass)
        phi_mass, q_mass = ellipticity2phi_q(e1_mass, e2_mass)
        x_grids, y_grids, n_angular = [], [], []
        for r_shell in r_grid:
            grid_x, grid_y = self._shells_grid_points(r_shell, supersampling_factor, phi_mass, q_shell=1.0)
            x_grids.append(grid_x)
            y_grids.append(grid_y)
            n_angular.append(len(grid_x))
        x_grids = np.concatenate(x_grids)
        y_grids = np.concatenate(y_grids)
        return x_grids, y_grids, n_angular

    def _shells_grid_points(self, r_shell, supersampling, pos_angle=0.0, q_shell=1.0):
        """Generate grid points in a shell at radius r_shell in elliptical polar coordinates."""
        a = r_shell
        b = a * q_shell
        n_points = 2 * np.pi * r_shell * supersampling / self._delta_pix
        if np.ceil(n_points) > 1:
            angle = np.linspace(0, np.pi / 2, int(np.ceil(n_points)))
        else:
            angle = np.array([np.pi / 4])
        x, y = a * np.cos(angle), b * np.sin(angle)
        return self._rotate_grid(x, y, pos_angle)

    def _downsample_shells_radially(
        self,
        high_res_map,
        supersampling_factor=1,
    ):
        """Downsamples a high-resolution map to the aperture grid.
        """
        r_shells = self._get_shells_spherical(supersampling_factor)
        r_bins = self._aperture._r_bins
        vel_disp = np.zeros(self._aperture.num_segments)
        # iterate over bin edges and average masked values
        for i in range(self._aperture.num_segments):
            mask = (r_shells >= r_bins[i]) & (r_shells < r_bins[i+1])
            vel_disp[i] = np.mean(high_res_map[mask])
        return vel_disp

    @staticmethod
    def _downsample_shells_angularly(
        high_res_map,
        n_angular
    ):
        """Downsamples a high-resolution map to the aperture grid.
        """
        i0 = i1 = 0
        vel_disp = []
        for n in n_angular:
            i1 += n
            map_i = high_res_map[i0:i1]
            vel_disp.append(np.mean(map_i))
            i0 = i1
        return np.array(vel_disp)

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
