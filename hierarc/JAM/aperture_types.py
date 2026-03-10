__author__ = "furcelay,sibirrer"

import numpy as np


class GeneralAperture(object):
    """General aperture class."""

    def __init__(self, x_cords, y_cords, bin_ids=None, delta_pix=0.1):
        """

        :param x_cords: x coordinates to compute the kinematics
        :param y_cords: y coordinates to compute the kinematics
        :param bin_ids: array with the same shape as x_cords/y_cords
            defining the bin index for each coordinate
        :param delta_pix: pixel scale of the coordinates, needed for PSF convolution
        """
        self._x_cords = x_cords
        self._y_cords = y_cords
        self._bin_ids = bin_ids
        self._delta_pix = delta_pix

    def aperture_sample(self):
        return self._x_cords, self._y_cords

    def aperture_downsample(self, high_res_map):
        return _downsample_cords_to_bins_1d(
            high_res_map,
            self._bin_ids,
        )

    @property
    def num_segments(self):
        """Number of segments with separate measurements of the velocity dispersion.

        :return: int
        """
        return int(np.max(self._bin_ids)) + 1

    @property
    def delta_pix(self):
        return self._delta_pix


class Slit(GeneralAperture):
    """Slit aperture description."""

    def __init__(
        self, length, width, center_ra=0, center_dec=0, angle=0, delta_pix=0.1
    ):
        """

        :param length: length of slit
        :param width: width of slit
        :param center_ra: center of slit
        :param center_dec: center of slit
        :param angle: orientation angle of slit, angle=0 corresponds length in RA direction
        :param delta_pix: pixel scale of the coordinates, needed for PSF convolution
        """
        self._length = length
        self._width = width
        self._center_ra, self._center_dec = center_ra, center_dec
        self._angle = angle

        slit_grid_x, slit_grid_y = self._make_slit_grid(delta_pix)
        super().__init__(
            slit_grid_x.flatten(), slit_grid_y.flatten(), delta_pix=delta_pix
        )

    def _make_slit_grid(self, delta_pix):
        slit_x = np.arange((-self._length + delta_pix) / 2, self._length / 2, delta_pix)
        slit_y = np.arange((-self._width + delta_pix) / 2, self._width / 2, delta_pix)
        grid_x, grid_y = np.meshgrid(slit_x, slit_y)
        # rotate
        grid_x, grid_y = _rotate(grid_x, grid_y, angle=-self._angle)
        # shift
        grid_x = grid_x + self._center_ra
        grid_y = grid_y + self._center_dec
        return grid_x.flatten(), grid_y.flatten()

    def aperture_downsample(self, high_res_map):
        return np.sum(high_res_map)

    @property
    def num_segments(self):
        """Number of segments with separate measurements of the velocity dispersion.

        :return: int
        """
        return 1


class Frame(GeneralAperture):
    """Rectangular box with a hole in the middle (also rectangular), effectively a
    frame."""

    def __init__(
        self,
        width_outer,
        width_inner,
        center_ra=0,
        center_dec=0,
        angle=0,
        delta_pix=0.1,
    ):
        """

        :param width_outer: width of box to the outer parts
        :param width_inner: width of inner removed box
        :param center_ra: center of slit
        :param center_dec: center of slit
        :param angle: orientation angle of slit, angle=0 corresponds length in RA direction
        :param delta_pix: pixel scale of the coordinates, needed for PSF convolution
        """
        self._width_outer = width_outer
        self._width_inner = width_inner
        self._center_ra, self._center_dec = center_ra, center_dec
        self._angle = angle

        x_grid, y_grid = self.make_frame_grid(delta_pix)
        super().__init__(x_grid, y_grid, delta_pix=delta_pix)

    def make_frame_grid(self, delta_pix):
        """Make a grid of coordinates within the frame aperture first create a grid for
        the outer box, then mask out the inner box :return: x_grid, y_grid."""
        x_outer = np.arange(
            (-self._width_outer + delta_pix) / 2, self._width_outer / 2, delta_pix
        )
        y_outer = np.arange(
            (-self._width_outer + delta_pix) / 2, self._width_outer / 2, delta_pix
        )
        x_outer_grid, y_outer_grid = np.meshgrid(x_outer, y_outer)
        # rotate
        x_outer_grid, y_outer_grid = _rotate(
            x_outer_grid, y_outer_grid, angle=-self._angle
        )

        # create inner box mask
        mask_inner = (np.abs(x_outer_grid) < self._width_inner / 2) & (
            np.abs(y_outer_grid) < self._width_inner / 2
        )
        # apply mask
        x_grid = x_outer_grid[~mask_inner]
        y_grid = y_outer_grid[~mask_inner]
        # shift
        x_grid = x_grid + self._center_ra
        y_grid = y_grid + self._center_dec
        return x_grid, y_grid

    def aperture_downsample(self, high_res_map, *args, **kwargs):
        return np.sum(high_res_map)

    @property
    def num_segments(self):
        """Number of segments with separate measurements of the velocity dispersion.

        :return: int
        """
        return 1


class Shell(GeneralAperture):
    """Shell aperture."""

    def __init__(self, r_in, r_out, center_ra=0, center_dec=0, delta_pix=0.1):
        """

        :param r_in: innermost radius to be selected
        :param r_out: outermost radius to be selected
        :param center_ra: center of the sphere
        :param center_dec: center of the sphere
        :param delta_pix: pixel scale of the coordinates, needed for PSF convolution
        """
        self._r_in, self._r_out = r_in, r_out
        self._center_ra, self._center_dec = center_ra, center_dec

        shell_x, shell_y = self.make_shell_grid(delta_pix)
        super().__init__(shell_x, shell_y, delta_pix=delta_pix)

    def make_shell_grid(self, delta_pix):
        """Make a grid of coordinates within the shell aperture :return: x_grid,
        y_grid."""
        r_vals = np.arange(self._r_in, self._r_out, delta_pix)
        x_grid, y_grid = [], []
        for r in r_vals:
            x, y = _sample_circle_uniform(r, delta_pix)
            x_grid.append(x)
            y_grid.append(y)
        x_grid = np.concatenate(x_grid) + self._center_ra
        y_grid = np.concatenate(y_grid) + self._center_dec
        return x_grid, y_grid

    # def make_shell_grid(self, delta_pix):
    #     """
    #     make a grid of coordinates within the shell aperture
    #     :return: x_grid, y_grid
    #     """
    #     x = y = np.arange(-self._r_out + delta_pix / 2, self._r_out, delta_pix)
    #     x_grid, y_grid = np.meshgrid(x, y)
    #     r2_grid = x_grid**2 + y_grid**2
    #     mask_shell = (r2_grid > self._r_in**2) & (r2_grid < self._r_out**2)
    #     return x_grid[mask_shell] + self._center_ra, y_grid[mask_shell] + self._center_dec

    def aperture_downsample(self, high_res_map):
        return np.sum(high_res_map)

    @property
    def num_segments(self):
        """Number of segments with separate measurements of the velocity dispersion.

        :return: int
        """
        return 1


class IFUGrid(GeneralAperture):
    """Class for an Integral Field Unit spectrograph with rectangular grid where the
    kinematics are measured."""

    def __init__(self, x_grid, y_grid, supersampling_factor=1, padding_arcsec=0):
        """

        :param x_grid: x coordinates of the grid
        :param y_grid: y coordinates of the grid
        :param supersampling_factor: supersampling factor
        :param padding_arcsec: padding around the grid in arcsec
        """
        self._x_grid = x_grid
        self._y_grid = y_grid
        self._supersampling_factor = supersampling_factor
        delta_x, delta_y = self.delta_pix_xy
        if np.abs(delta_x) != np.abs(delta_y):
            raise ValueError("IFU grid pixels must be square!")
        delta_pix_sup = np.abs(delta_x) / supersampling_factor
        # padding in pixels
        self._padding = int(padding_arcsec / delta_pix_sup)

        x_grid_supersampled, y_grid_supersampled = self.make_supersampled_grid(
            supersampling_factor, self._padding
        )
        super().__init__(
            x_grid_supersampled, y_grid_supersampled, delta_pix=delta_pix_sup
        )

    def make_supersampled_grid(self, supersampling_factor, padding):
        """Creates a new grid, supersampled and with padding for PSF convolution."""

        delta_x, delta_y = self.delta_pix_xy
        x_grid = self._x_grid
        y_grid = self._y_grid

        # New (supersampled) pixel size
        new_delta_x = delta_x / supersampling_factor
        new_delta_y = delta_y / supersampling_factor

        # the padding is in supersampled pixels
        pad_x = padding * new_delta_x
        pad_y = padding * new_delta_y

        # grid bounds (pixel-centered)
        x_start = x_grid[0, 0] - 0.5 * delta_x * (1 - 1 / supersampling_factor) - pad_x
        x_end = x_grid[0, -1] + 0.5 * delta_x * (1 - 1 / supersampling_factor) + pad_x
        y_start = y_grid[0, 0] - 0.5 * delta_y * (1 - 1 / supersampling_factor) - pad_y
        y_end = y_grid[-1, 0] + 0.5 * delta_y * (1 - 1 / supersampling_factor) + pad_y

        xs = np.arange(x_start, x_end * (1 + 1e-6), new_delta_x)
        ys = np.arange(y_start, y_end * (1 + 1e-6), new_delta_y)

        x_grid_supersampled, y_grid_supersampled = np.meshgrid(xs, ys)
        return x_grid_supersampled, y_grid_supersampled

    def aperture_downsample(self, high_res_map):
        """Downsample a high-resolution map to the IFU grid by averaging over the
        supersampling factor.

        :param high_res_map: 2D array of high-resolution map to be downsampled
        :return: 2D array of downsampled map
        """
        num_pix_y, num_pix_x = self.grid_shape
        high_res_map = _unpad_map(high_res_map, self._padding)
        return high_res_map.reshape(
            num_pix_y, self._supersampling_factor, num_pix_x, self._supersampling_factor
        ).mean(axis=(1, 3))

    @property
    def num_segments(self):
        """Number of segments with separate measurements of the velocity dispersion.

        :return: int
        """
        return self._x_grid.size

    @property
    def grid_shape(self):
        """Shape of the IFU grid."""
        return self._x_grid.shape

    @property
    def x_grid(self):
        """X coordinates of the grid."""
        return self._x_grid

    @property
    def y_grid(self):
        """Y coordinates of the grid."""
        return self._y_grid

    @property
    def supersampling_factor(self):
        """Supersampling factor of the IFU grid."""
        return self._supersampling_factor

    @property
    def padding(self):
        """Padding around the grid for convolution."""
        return self._padding

    @property
    def delta_pix_xy(self):
        """Get the pixel scale of the grid."""
        delta_x = self._x_grid[0, 1] - self._x_grid[0, 0]
        delta_y = self._y_grid[1, 0] - self._y_grid[0, 0]
        return delta_x, delta_y


class IFUShells(IFUGrid):
    """Class for an Integral Field Unit spectrograph with azimuthal shells where the
    kinematics are measured."""

    def __init__(
        self, r_bins, center_ra=0, center_dec=0, ifu_grid_kwargs=None, delta_pix=None
    ):
        """

        :param r_bins: array of radial bins to average the dispersion spectra in ascending order.
         It starts with the innermost edge to the outermost edge.
        :param center_ra: center of the sphere
        :param center_dec: center of the sphere
        :param ifu_grid_kwargs: kwargs to create the IFU grid, if None a default grid is created.
        :param delta_pix: pixel scale of the IFU grid, only used if ifu_grid is None.
        """
        self._r_bins = r_bins
        self._center_ra, self._center_dec = center_ra, center_dec
        if ifu_grid_kwargs is None:
            # make an IFU grid
            r_max = np.max(r_bins)
            if delta_pix is None:
                # the same as in GalKin
                delta_pix = r_max * 1.5 * 2 / 100
            ifu_x = ifu_y = np.arange(
                -r_max + delta_pix / 2,
                r_max,
                delta_pix,
            )
            ifu_x_grid, ifu_y_grid = np.meshgrid(ifu_x, ifu_y)
            ifu_grid_kwargs = {
                "x_grid": ifu_x_grid,
                "y_grid": ifu_y_grid,
                "supersampling_factor": 1,
                "padding_arcsec": 0,
            }
        super().__init__(**ifu_grid_kwargs)

    def aperture_downsample(self, high_res_map):
        downsampled_map = np.zeros(self.num_segments)
        x_grid, y_grid = self.aperture_sample()
        x_grid -= self._center_ra
        y_grid -= self._center_dec
        r_grid = np.sqrt(x_grid**2 + y_grid**2)
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


class IFUBinned(IFUGrid):
    """Class for an Integral Field Unit spectrograph, with a binned (e.g. Voronoi)
    rectangular grid.

    It has the same grid definition as IFUGrid, and a matrix of bin ids, indicating to
    which bin each pixel belongs.
    """

    def __init__(self, x_grid, y_grid, bins):
        """
        :param x_grid: float array of shape (n_y, n_x) with the x coordinates of the grid
        :param y_grid: float array of shape (n_y, n_x) with the y coordinates of the grid
        :param bins: int array of shape (n_y, n_x) with the bin ids (0, 1, ...), and -1 for excluded pixels.
        """
        super(IFUBinned, self).__init__(x_grid, y_grid)
        self._bins = bins.astype(int)

    def aperture_downsample(self, high_res_map):
        downsampled_map = downsample_cords_to_bins(
            high_res_map, self._bins, self.supersampling_factor, self.padding
        )
        return downsampled_map

    @property
    def num_segments(self):
        """Number of segments with separate measurements of the velocity dispersion.
        This is the number of unique bin ids.

        :return: int.
        """
        unique_bins = np.unique(self._bins[self.bins > -1])
        return len(unique_bins)

    @property
    def bins(self):
        return self._bins


def _rotate(x, y, angle):
    x_rot = np.cos(angle) * x + np.sin(angle) * y
    y_rot = -np.sin(angle) * x + np.cos(angle) * y
    return x_rot, y_rot


def _sample_circle_uniform(r_shell, step):
    n_points = 2 * np.pi * r_shell / step
    if np.ceil(n_points) > 1:
        angle = np.linspace(0, 2 * np.pi, int(np.ceil(n_points)))
    else:
        angle = np.array([0])
    return r_shell * np.cos(angle), r_shell * np.sin(angle)


def _unpad_map(padded_map, padding):
    if padding > 0:
        return padded_map[padding:-padding, padding:-padding]
    else:
        return padded_map


def downsample_cords_to_bins(vrms_grid, bins, supersampling_factor=1, padding=0):
    # remove padding from the grid
    vrms_grid = _unpad_map(vrms_grid, padding)
    supersampled_bins = bins.repeat(supersampling_factor, axis=0).repeat(
        supersampling_factor, axis=1
    )
    return _downsample_cords_to_bins_1d(vrms_grid, supersampled_bins)

def _downsample_cords_to_bins_1d(vrms_grid, bins):
    n_bins = int(np.max(bins)) + 1
    vrms = np.zeros(n_bins)
    for n in range(n_bins):
        vrms[n] = np.mean(vrms_grid[bins == n])
    return vrms
