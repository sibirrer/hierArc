import numpy as np
import numpy.testing as npt
import pytest
from hierarc.JAM.aperture import Aperture
from lenstronomy.GalKin.psf import PSF
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.GalKin.galkin import Galkin


class TestJAMApertureSlit(object):

    def setup_method(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        lens_cosmo = LensCosmo(0.5, 1.2, cosmo=cosmo)

        center_x = 2.0
        center_y = -1.0
        self.kwargs_light = [
            {"Rs": 1.0, "amp": 1.0, "center_x": center_x, "center_y": center_y}
        ]
        self.kwargs_lens_mass = [
            {"theta_E": 1.5, "gamma": 2.1, "center_x": center_x, "center_y": center_y}
        ]
        self.kwargs_anisotropy = {"beta": 0.3}
        kwargs_psf = {
            "psf_type": "GAUSSIAN",
            "fwhm": 0.5,
        }
        kwargs_cosmo = {
            "d_d": lens_cosmo.dd,
            "d_s": lens_cosmo.ds,
            "d_ds": lens_cosmo.dds,
        }
        kwargs_numerics_lenstronomy = {
            "interpol_grid_num": 2000,
            "log_integration": True,
            "max_integrate": 1e3,
            "min_integrate": 1e-5,
        }
        kwargs_numerics_mge = {
            "mge_n_gauss": 50,
            "mge_min_r": 1e-4,
            "mge_max_r": 300,
            "mge_n_radial": 500,
            "mge_linear_solver": True,
        }
        kwargs_numeric_jam = kwargs_numerics_lenstronomy | kwargs_numerics_mge
        kwargs_model = {
            "mass_profile_list": ["SPP"],
            "light_profile_list": ["HERNQUIST"],
            "anisotropy_model": "const",
        }

        kwargs_aperture_grid = {}

        self.aperture_grid = Aperture(**kwargs_aperture_grid)

    def test_IFU_grid_aperture(self):
        pass

    def test_IFU_shells_aperture(self):
        pass

    def test_IFU_voronoi_aperture(self):
        pass


if __name__ == "__main__":
    pytest.main()
