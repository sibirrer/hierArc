from lenstronomy.GalKin.observation import GalkinObservation
import mgefit as mge
import jampy as jam
import numpy as np


__all__ = ["JAMWrapper"]

class JAMWrapper(GalkinObservation):
    def __init__(
        self,
        kwargs_model,
        kwargs_aperture,
        kwargs_psf,
        kwargs_cosmo,
        kwargs_numerics=None,
        analytic_kinematics=False,
    ):
        """
        Wrapper class to use jampy JAM functionality similar to lenstronomy's Galkin class.

        :param kwargs_model: keyword arguments describing the model components
        :param kwargs_aperture: keyword arguments describing the spectroscopic aperture, see Aperture() class
        :param kwargs_psf: keyword argument specifying the PSF of the observation
        :param kwargs_cosmo: keyword arguments that define the cosmology in terms of the angular diameter distances
         involved
        :param kwargs_numerics: numerics keyword arguments
        :param analytic_kinematics: bool, if True uses the analytic kinematic model
        """
        GalkinObservation.__init__(
            self, kwargs_aperture=kwargs_aperture, kwargs_psf=kwargs_psf
        )

    def dispersion(
        self, kwargs_mass, kwargs_light, kwargs_anisotropy, sampling_number=1000
    ):
        """Computes the averaged LOS velocity dispersion in the slit (convolved)

        :param kwargs_mass: mass model parameters (following lenstronomy lens model
            conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light
            model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to
            anisotropy type chosen. We refer to the Anisotropy() class for details on
            the parameters.
        :param sampling_number: int, number of spectral sampling of the light
            distribution
        :return: integrated LOS velocity dispersion in units [km/s]
        """
        pass

    def dispersion_map(
        self,
        kwargs_mass,
        kwargs_light,
        kwargs_anisotropy,
        num_kin_sampling=1000,
        num_psf_sampling=100,
    ):
        """Computes the velocity dispersion in each Integral Field Unit.

        :param kwargs_mass: keyword arguments of the mass model
        :param kwargs_light: keyword argument of the light model
        :param kwargs_anisotropy: anisotropy keyword arguments
        :param num_kin_sampling: int, number of draws from a kinematic prediction of a
            LOS
        :param num_psf_sampling: int, number of displacements/render from a spectra to
            be displaced on the IFU
        :return: ordered array of velocity dispersions [km/s] for each unit
        """
        pass

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

        log10_radial_distance_from_center = np.log10(
            np.sqrt(
                (x_grid_supersampled - mass_center_x) ** 2
                + (y_grid_supersmapled - mass_center_y) ** 2
            )
        )

        return (
            x_grid_supersampled,
            y_grid_supersmapled,
            log10_radial_distance_from_center,
        )

    def dispersion_map_grid_convolved(
        self,
        kwargs_mass,
        kwargs_light,
        kwargs_anisotropy,
        supersampling_factor=1,
        voronoi_bins=None,
    ):
        """Computes the velocity dispersion in each Integral Field Unit.

        :param kwargs_mass: keyword arguments of the mass model
        :param kwargs_light: keyword argument of the light model
        :param kwargs_anisotropy: anisotropy keyword arguments
        :param supersampling_factor: sampling factor for the grid to do the 2D
            convolution on
        :param voronoi_bins: mapping of the voronoi bins, bin indices should start from
            0, -1 values for pixels not binned
        :return: ordered array of velocity dispersions [km/s] for each unit
        """
        pass

    def _draw_one_sigma2(self, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """

        :param kwargs_mass: mass model parameters (following lenstronomy lens model conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to anisotropy type chosen.
            We refer to the Anisotropy() class for details on the parameters.
        :return: integrated LOS velocity dispersion in angular units for a single draw of the light distribution that
         falls in the aperture after displacing with the seeing
        """
        pass
        # return sigma2_IR, IR
