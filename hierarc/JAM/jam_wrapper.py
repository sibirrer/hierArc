__author__ = "furcelay", "sibirrer"

from hierarc.JAM.jam_wrapper_base import JAMWrapperBase
from hierarc.JAM.aperture import downsample_cords_to_bins
import numpy as np

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
        voronoi_bins=None,
    ):
        """Computes the velocity dispersion in the aperture.
        IF the aperture is a slit, frame or shell, the output is a single float.
        If the aperture is an IFU grid, the output is a 2D array of the same shape as the IFU grid.
        If the aperture is an IFU shells, the output is a 1D array with the number of shells.

        :param kwargs_mass: keyword arguments of the mass model
        :param kwargs_light: keyword argument of the light model
        :param kwargs_anisotropy: anisotropy keyword arguments
        :param inclination: inclination angle of the system [degrees]
        :param convolved: bool, if True the PSF convolution is applied
        :param voronoi_bins: None or 2D array with same shape as the IFU grid defining the Voronoi
            bins. If None, no Voronoi binning is applied. Only relevant if aperture is of type 'IFU_grid'.
        :return: ordered array of velocity dispersions [km/s] for each unit
        """
        x_sup, y_sup = self.aperture_sample()
        # shift and rotate to align with light profile
        x_gal_sup, y_gal_sup = self._shift_and_rotate(x_sup, y_sup, kwargs_light)
        vrms_sup, surf_bright_sup = self.dispersion_points(
            x_gal_sup, y_gal_sup,
            kwargs_mass,
            kwargs_light,
            kwargs_anisotropy,
            inclination=inclination,
            convolved=convolved,
        )
        sigma2_lum_weighted_sup = vrms_sup**2 * surf_bright_sup

        if voronoi_bins is not None:
            sigma2_lum_weighted = downsample_cords_to_bins(
                sigma2_lum_weighted_sup,
                voronoi_bins,
                supersampling_factor=1,
            )
            surf_bright = downsample_cords_to_bins(
                surf_bright_sup,
                voronoi_bins,
                supersampling_factor=1,
            )
            vrms = np.sqrt(sigma2_lum_weighted / surf_bright)
        else:
            sigma2_lum_weighted = self.aperture_downsample(
                sigma2_lum_weighted_sup,
            )
            surf_bright = self.aperture_downsample(
                surf_bright_sup,
            )
            vrms = np.sqrt(sigma2_lum_weighted / surf_bright)
        return vrms
