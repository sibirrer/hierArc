__author__ = "sibirrer,furcelay"

from hierarc.JAM.aperture_types import GeneralAperture, Shell, Slit, IFUShells, Frame, IFUGrid, IFUBinned, \
    downsample_cords_to_bins

__all__ = ["Aperture", "downsample_cords_to_bins"]
"""Class that defines the aperture of the measurement (e.g. slit, integral field
spectroscopy regions etc).

Available aperture types:
-------------------------

'general': x_cords, y_cords, bin_ids, delta_pix
'slit': length, width, center_ra, center_dec, angle, delta_pix
'shell': r_in, r_out, center_ra, center_dec, delta_pix
'frame': width_outer, width_inner, center_ra, center_dec, angle, delta_pix
'IFU_grid': x_grid, y_grid, supersampling_factor, padding_arcsec
'IFU_shells': r_bins, center_ra, center_dec, ifu_grid_kwargs, delta_pix
'IFU_binned': x_grid, y_grid, bins, supersampling_factor, padding_arcsec
"""


class Aperture(object):
    """Defines mask(s) of spectra, can handle IFU and single slit/box type data."""

    def __init__(self, aperture_type, **kwargs_aperture):
        """

        :param aperture_type: string
        :param kwargs_aperture: keyword arguments reflecting the aperture type chosen.
         We refer to the specific class instances for documentation.
        """
        if aperture_type == "general":
            self._aperture = GeneralAperture(**kwargs_aperture)
        elif aperture_type == "slit":
            self._aperture = Slit(**kwargs_aperture)
        elif aperture_type == "shell":
            self._aperture = Shell(**kwargs_aperture)
        elif aperture_type == "IFU_shells":
            self._aperture = IFUShells(**kwargs_aperture)
        elif aperture_type == "frame":
            self._aperture = Frame(**kwargs_aperture)
        elif aperture_type == "IFU_grid":
            self._aperture = IFUGrid(**kwargs_aperture)
        elif aperture_type == "IFU_binned":
            self._aperture = IFUBinned(**kwargs_aperture)
        else:
            raise ValueError(
                "aperture type %s not implemented! Available are "
                "'general' 'slit', 'shell', 'IFU_grid', 'IFU_shells' 'IFU_binned'. "
                % aperture_type
            )
        self.aperture_type = aperture_type

    def aperture_sample(self):
        return self._aperture.aperture_sample()

    def aperture_downsample(self, hires_map):
        return self._aperture.aperture_downsample(hires_map)

    @property
    def num_segments(self):
        return self._aperture.num_segments

    @property
    def delta_pix(self):
        return self._aperture.delta_pix
