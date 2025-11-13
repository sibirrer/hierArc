__author__ = "furcelay,sbirrer"

from hierarc.JAM.jam_wrapper_base import JAMWrapperBase
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
        supersampling_factor=1,
    ):
        if self.aperture_type == "slit":
            return self.dispersion_slit(
                kwargs_mass,
                kwargs_light,
                kwargs_anisotropy,
            )
        elif self.aperture_type == "IFU_shells":
            return self.dispersion_shells(
                kwargs_mass,
                kwargs_light,
                kwargs_anisotropy,
            )
        elif self.aperture_type == "IFU_grid":
            return self.dispersion_grid(
                kwargs_mass,
                kwargs_light,
                kwargs_anisotropy,
                inclination=inclination,
                convolved=convolved,
                supersampling_factor=supersampling_factor,
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
        :param sampling_number: int, number of spectral sampling of the light
            distribution
        :return: integrated LOS velocity dispersion in units [km/s]
        """
        pass

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
        x_sup, y_sup, _ = self._get_grid(kwargs_mass, supersampling_factor=supersampling_factor)
        vrms_sup = self.dispersion_points(
            x_sup, y_sup,
            kwargs_mass,
            kwargs_light,
            kwargs_anisotropy,
            inclination=inclination,
            convolved=convolved,
            psf_supersampling_factor=supersampling_factor,
        )
        vrms = self._downsample_to_aperture(
            vrms_sup,
            supersampling_factor
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

    def dispersion_shells(
        self,
        kwargs_mass,
        kwargs_light,
        kwargs_anisotropy,
        inclination=90.0,
    ):
        pass

    def dispersion_map_grid_convolved(
        self,
        kwargs_mass,
        kwargs_light,
        kwargs_anisotropy,
        inclination=90.0,
        supersampling_factor=1,
        voronoi_bins=None,
    ):
        """Computes the velocity dispersion in each Integral Field Unit.

        :param kwargs_mass: keyword arguments of the mass model
        :param kwargs_light: keyword argument of the light model
        :param kwargs_anisotropy: anisotropy keyword arguments
        :param inclination: inclination angle of the system [degrees]
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
