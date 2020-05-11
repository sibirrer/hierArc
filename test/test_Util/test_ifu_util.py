import numpy as np
import numpy.testing as npt
import pytest


from hierarc.Util import ifu_util


class TestIFUUtil(object):

    def setup(self):
        pass

    def test_radial_dispersion(self):
        num = 10

        dispersion_map = np.zeros((num, num))
        weight_map_disp = np.ones((num, num))

        velocity_map = np.ones((num, num))
        weight_map_v = np.ones((num, num))

        r_bins = np.linspace(0, 5, 5)
        fiber_scale = 1

        flux_map = np.ones((num, num))

        disp_r, error_r = ifu_util.binned_total(dispersion_map, weight_map_disp, velocity_map, weight_map_v, flux_map, fiber_scale, r_bins)
        assert len(disp_r) == len(r_bins) - 1
        npt.assert_almost_equal(disp_r, 1, decimal=6)


if __name__ == '__main__':
    pytest.main()
