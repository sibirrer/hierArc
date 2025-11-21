__author__ = "sibirrer,furcelay"

import numpy as np


class Slit(object):
    """Slit aperture description."""

    def __init__(self, length, width, center_ra=0, center_dec=0, angle=0):
        """

        :param length: length of slit
        :param width: width of slit
        :param center_ra: center of slit
        :param center_dec: center of slit
        :param angle: orientation angle of slit, angle=0 corresponds length in RA direction
        """
        self._length = length
        self._width = width
        self._center_ra, self._center_dec = center_ra, center_dec
        self._angle = angle

    def aperture_select(self, ra, dec):
        """

        :param ra: angular coordinate of photon/ray
        :param dec: angular coordinate of photon/ray
        :return: bool, True if photon/ray is within the slit, False otherwise
        """
        return (
            slit_select(
                ra,
                dec,
                self._length,
                self._width,
                self._center_ra,
                self._center_dec,
                self._angle,
            ),
            0,
        )

    @property
    def num_segments(self):
        """Number of segments with separate measurements of the velocity dispersion.

        :return: int
        """
        return 1


def slit_select(ra, dec, length, width, center_ra=0, center_dec=0, angle=0):
    """

    :param ra: angular coordinate of photon/ray
    :param dec: angular coordinate of photon/ray
    :param length: length of slit
    :param width: width of slit
    :param center_ra: center of slit
    :param center_dec: center of slit
    :param angle: orientation angle of slit, angle=0 corresponds length in RA direction
    :return: bool, True if photon/ray is within the slit, False otherwise
    """
    ra_ = ra - center_ra
    dec_ = dec - center_dec
    x = np.cos(angle) * ra_ + np.sin(angle) * dec_
    y = -np.sin(angle) * ra_ + np.cos(angle) * dec_
    return (abs(x) < length / 2.0) & (abs(y) < width / 2.0)


class Frame(object):
    """Rectangular box with a hole in the middle (also rectangular), effectively a
    frame."""

    def __init__(self, width_outer, width_inner, center_ra=0, center_dec=0, angle=0):
        """

        :param width_outer: width of box to the outer parts
        :param width_inner: width of inner removed box
        :param center_ra: center of slit
        :param center_dec: center of slit
        :param angle: orientation angle of slit, angle=0 corresponds length in RA direction
        """
        self._width_outer = width_outer
        self._width_inner = width_inner
        self._center_ra, self._center_dec = center_ra, center_dec
        self._angle = angle

    def aperture_select(self, ra, dec):
        """

        :param ra: angular coordinate of photon/ray
        :param dec: angular coordinate of photon/ray
        :return: bool, True if photon/ray is within the slit, False otherwise
        """
        return (
            frame_select(
                ra,
                dec,
                self._width_outer,
                self._width_inner,
                self._center_ra,
                self._center_dec,
                self._angle,
            ),
            0,
        )

    @property
    def num_segments(self):
        """Number of segments with separate measurements of the velocity dispersion.

        :return: int
        """
        return 1


def frame_select(ra, dec, width_outer, width_inner, center_ra=0, center_dec=0, angle=0):
    """

    :param ra: angular coordinate of photon/ray
    :param dec: angular coordinate of photon/ray
    :param width_outer: width of box to the outer parts
    :param width_inner: width of inner removed box
    :param center_ra: center of slit
    :param center_dec: center of slit
    :param angle: orientation angle of slit, angle=0 corresponds length in RA direction
    :return: bool, True if photon/ray is within the box with a hole, False otherwise
    """
    ra_ = ra - center_ra
    dec_ = dec - center_dec
    x = np.cos(angle) * ra_ + np.sin(angle) * dec_
    y = -np.sin(angle) * ra_ + np.cos(angle) * dec_

    cond_outer = (abs(x) < width_outer / 2.0) & (abs(y) < width_outer / 2.0)
    cond_inner = (abs(x) < width_inner / 2.0) & (abs(y) < width_inner / 2.0)

    return cond_outer & ~cond_inner


class Shell(object):
    """Shell aperture."""

    def __init__(self, r_in, r_out, center_ra=0, center_dec=0):
        """

        :param r_in: innermost radius to be selected
        :param r_out: outermost radius to be selected
        :param center_ra: center of the sphere
        :param center_dec: center of the sphere
        """
        self._r_in, self._r_out = r_in, r_out
        self._center_ra, self._center_dec = center_ra, center_dec

    def aperture_select(self, ra, dec):
        """

        :param ra: angular coordinate of photon/ray
        :param dec: angular coordinate of photon/ray
        :return: bool, True if photon/ray is within the slit, False otherwise
        """
        return (
            shell_select(
                ra, dec, self._r_in, self._r_out, self._center_ra, self._center_dec
            ),
            0,
        )

    @property
    def num_segments(self):
        """Number of segments with separate measurements of the velocity dispersion.

        :return: int
        """
        return 1


def shell_select(ra, dec, r_in, r_out, center_ra=0, center_dec=0):
    """

    :param ra: angular coordinate of photon/ray
    :param dec: angular coordinate of photon/ray
    :param r_in: innermost radius to be selected
    :param r_out: outermost radius to be selected
    :param center_ra: center of the sphere
    :param center_dec: center of the sphere
    :return: boolean, True if within the radial range, False otherwise
    """
    x = ra - center_ra
    y = dec - center_dec
    r = np.sqrt(x**2 + y**2)
    return (r >= r_in) & (r < r_out)


class IFUShells(object):
    """Class for an Integral Field Unit spectrograph with azimuthal shells where the
    kinematics are measured."""

    def __init__(self, r_bins, center_ra=0, center_dec=0, ifu_grid=None):
        """

        :param r_bins: array of radial bins to average the dispersion spectra in ascending order.
         It starts with the innermost edge to the outermost edge.
        :param center_ra: center of the sphere
        :param center_dec: center of the sphere
        """
        self._r_bins = r_bins
        self._center_ra, self._center_dec = center_ra, center_dec
        if ifu_grid is None:
            ifu_num_pix = 100
            r_max = np.max(r_bins)
            delta_pix = 1.5 * r_max * 2 / ifu_num_pix
            ifu_x = ifu_y = np.linspace(
                -ifu_num_pix / 2 * delta_pix,
                ifu_num_pix / 2 * delta_pix,
                ifu_num_pix,
            )
            ifu_x_grid, ifu_y_grid = np.meshgrid(ifu_x, ifu_y)
        else:
            ifu_x_grid, ifu_y_grid = ifu_grid
        self._ifu_grid = IFUGrid(ifu_x_grid, ifu_y_grid)

    def aperture_select(self, ra, dec):
        """

        :param ra: angular coordinate of photon/ray
        :param dec: angular coordinate of photon/ray
        :return: bool, True if photon/ray is within the slit, False otherwise, index of shell
        """
        return shell_ifu_select(
            ra, dec, self._r_bins, self._center_ra, self._center_dec
        )

    def aperture_sample(self, supersampling_factor):
        x_grid, y_grid = self._ifu_grid.aperture_sample(supersampling_factor)
        return x_grid, y_grid

    def aperture_downsample(self, high_res_map, supersampling_factor):
        downsampled_map = np.zeros(self.num_segments)
        x_grid, y_grid = self._ifu_grid.aperture_sample(supersampling_factor)
        r_grid = np.sqrt(x_grid ** 2 + y_grid ** 2)
        r_bins = self._r_bins
        # iterate over bin edges and average masked values
        for i in range(self.num_segments):
            mask = (r_grid >= r_bins[i]) & (r_grid < r_bins[i + 1])
            downsampled_map[i] = np.mean(high_res_map[mask])
        return downsampled_map

    @property
    def num_segments(self):
        """Number of segments with separate measurements of the velocity dispersion
        :return: int."""
        return len(self._r_bins) - 1


class IFUGrid(object):
    """Class for an Integral Field Unit spectrograph with rectangular grid where the
    kinematics are measured."""

    def __init__(self, x_grid, y_grid):
        """

        :param x_grid: x coordinates of the grid
        :param y_grid: y coordinates of the grid
        """
        self._x_grid = x_grid
        self._y_grid = y_grid
        delta_x, delta_y = self.delta_pix_xy
        if np.abs(delta_x) != np.abs(delta_y):
            raise ValueError("IFU grid pixels must be square!")

    def aperture_select(self, ra, dec):
        """

        :param ra: angular coordinate of photon/ray
        :param dec: angular coordinate of photon/ray
        :return: bool, True if photon/ray is within the slit, False otherwise, index of shell
        """
        return grid_ifu_select(ra, dec, self._x_grid, self._y_grid)

    def aperture_sample(self, supersampling_factor):
        delta_x, delta_y = self.delta_pix_xy
        x_grid = self._x_grid
        y_grid = self._y_grid

        new_delta_x = delta_x / supersampling_factor
        new_delta_y = delta_y / supersampling_factor
        x_start = x_grid[0, 0] - delta_x / 2.0 * (1 - 1 / supersampling_factor)
        x_end = x_grid[0, -1] + delta_x / 2.0 * (1 - 1 / supersampling_factor)
        y_start = y_grid[0, 0] - delta_y / 2.0 * (1 - 1 / supersampling_factor)
        y_end = y_grid[-1, 0] + delta_y / 2.0 * (1 - 1 / supersampling_factor)

        xs = np.arange(x_start, x_end * (1 + 1e-6), new_delta_x)
        ys = np.arange(y_start, y_end * (1 + 1e-6), new_delta_y)

        x_grid_supersampled, y_grid_supersampled = np.meshgrid(xs, ys)
        return x_grid_supersampled, y_grid_supersampled

    def aperture_downsample(self, high_res_map, supersampling_factor):
        """Downsample a high-resolution map to the IFU grid by averaging over the
        supersampling factor.
        :param high_res_map: 2D array of high-resolution map to be downsampled
        :param supersampling_factor: int, factor by which the high-res map is sampled
        :return: 2D array of downsampled map
        """
        num_pix_x, num_pix_y = self.num_segments
        return high_res_map.reshape(num_pix_y, supersampling_factor,
                                    num_pix_x, supersampling_factor).mean(axis=(1, 3))

    @property
    def num_segments(self):
        """Number of segments with separate measurements of the velocity dispersion.

        :return: int
        """
        return self._x_grid.shape[0], self._x_grid.shape[1]

    @property
    def x_grid(self):
        """X coordinates of the grid."""
        return self._x_grid

    @property
    def y_grid(self):
        """Y coordinates of the grid."""
        return self._y_grid

    @property
    def delta_pix_xy(self):
        """Get the pixel scale of the grid.
        """
        delta_x = self._x_grid[0, 1] - self._x_grid[0, 0]
        delta_y = self._y_grid[1, 0] - self._y_grid[0, 0]
        return delta_x, delta_y


def grid_ifu_select(ra, dec, x_grid, y_grid):
    """

    :param ra: angular coordinate of photon/ray
    :param dec: angular coordinate of photon/ray
    :param x_grid: array of x_grid bins
    :param y_grid: array of y_grid bins
    :return: boolean, True if within the grid range, False otherwise
    """
    x_pixel_size = x_grid[0, 1] - x_grid[0, 0]
    y_pixel_size = y_grid[1, 0] - y_grid[0, 0]

    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            x_down = x_grid[i, j] - x_pixel_size / 2
            x_up = x_grid[i, j] + x_pixel_size / 2

            y_down = y_grid[i, j] - y_pixel_size / 2
            y_up = y_grid[i, j] + y_pixel_size / 2

            if (x_down <= ra <= x_up) and (y_down <= dec <= y_up):
                return True, (i, j)

    return False, None


def shell_ifu_select(ra, dec, r_bin, center_ra=0, center_dec=0):
    """

    :param ra: angular coordinate of photon/ray
    :param dec: angular coordinate of photon/ray
    :param r_bin: array of radial bins to average the dispersion spectra in ascending order.
     It starts with the inner-most edge to the outermost edge.
    :param center_ra: center of the sphere
    :param center_dec: center of the sphere
    :return: boolean, True if within the radial range, False otherwise
    """
    x = ra - center_ra
    y = dec - center_dec
    r = np.sqrt(x**2 + y**2)
    for i in range(0, len(r_bin) - 1):
        if (r >= r_bin[i]) and (r < r_bin[i + 1]):
            return True, i
    return False, None
