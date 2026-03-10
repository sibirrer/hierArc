import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from hierarc.JAM.aperture_types import (
    GeneralAperture,
    Slit,
    Frame,
    Shell,
    IFUGrid,
    IFUShells,
    IFUBinned,
    _rotate,
    _sample_circle_uniform,
    _unpad_map,
    downsample_cords_to_bins,
)


class TestApertureTypes(object):
    def setup_method(self):
        # small generic grid
        self.x = np.array([[0.0, 1.0], [0.0, 1.0]])
        self.y = np.array([[0.0, 0.0], [1.0, 1.0]])
        # bins for IFUBinned / GeneralAperture tests (shape matches x/y)
        self.bins = np.array([[0, 1], [2, -1]])
        # a simple high-res map (4x4) for downsampling tests
        self.hr_map = np.arange(16).reshape(4, 4).astype(float)

    def test_rotate_basic(self):
        # 90 degree rotation (pi/2)
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])
        x_r, y_r = _rotate(x, y, angle=np.pi / 2)
        # rotate formula used in module corresponds to: x' = cos a * x + sin a * y
        assert_allclose(x_r, np.cos(np.pi / 2) * x + np.sin(np.pi / 2) * y)
        assert_allclose(y_r, -np.sin(np.pi / 2) * x + np.cos(np.pi / 2) * y)

    def test_sample_circle_uniform(self):
        # small radius -> should return one point
        x, y = _sample_circle_uniform(0.1, step=1.0)
        assert x.shape == y.shape
        # For small r compared to step, expects single point at angle 0
        assert_allclose(x, np.array([0.1]))
        assert_allclose(y, np.array([0.0]))

        # larger radius -> multiple points around the circle should be produced
        x2, y2 = _sample_circle_uniform(2.0, step=0.5)
        assert x2.size == 26

    def test_unpad_map(self):
        padded = np.arange(25).reshape(5, 5)
        # padding 1 should remove first and last rows/cols
        unp = _unpad_map(padded, padding=1)
        assert unp.shape == (3, 3)
        assert unp[0, 0] == padded[1, 1]

        # padding 0 returns the same array
        assert_array_equal(_unpad_map(padded, 0), padded)

    def test_downsample_cords_to_bins(self):
        # use hr_map shape 4x4 and bins shape 2x2; use supersampling_factor=2 so each bin maps to 2x2
        bins = np.array([[0, 1], [2, 3]])
        # replicate to 4x4 by repeating each element 2x2
        supersampling_factor = 2
        v = downsample_cords_to_bins(
            self.hr_map, bins, supersampling_factor=supersampling_factor, padding=0
        )
        # expected vrms for each bin is mean of the corresponding 2x2 block:
        # blocks in hr_map:
        # block 0: hr_map[0:2,0:2] -> [0,1,4,5] mean=2.5
        # block 1: hr_map[0:2,2:4] -> [2,3,6,7] mean=4.5
        # block 2: hr_map[2:4,0:2] -> [8,9,12,13] mean=10.5
        # block 3: hr_map[2:4,2:4] -> [10,11,14,15] mean=12.5
        assert_allclose(v, np.array([2.5, 4.5, 10.5, 12.5]))

    def test_general_aperture(self):
        # create GeneralAperture with 4 coordinates and bin ids 0..2 (one bin id missing intentionally)
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 0.5, 1.0])
        bins = np.array([0, 1, 1])
        ga = GeneralAperture(x, y, bin_ids=bins, delta_pix=0.2)
        # aperture_sample returns original coordinates
        xa, ya = ga.aperture_sample()
        assert_array_equal(xa, x)
        assert_array_equal(ya, y)
        # delta_pix
        assert_allclose(ga.delta_pix, 0.2)
        # num_segments = max(bin)+1 = 1+1 = 2
        assert ga.num_segments == 2

    def test_slit(self):
        # small slit length & width -> will create few points
        slit = Slit(
            length=0.3,
            width=0.2,
            center_ra=0.1,
            center_dec=-0.1,
            angle=0.0,
            delta_pix=0.1,
        )
        # slit aperture returns flattened grid coordinates
        xs, ys = slit.aperture_sample()
        assert xs.ndim == 1 and ys.ndim == 1
        # slit.num_segments should be 1
        assert slit.num_segments == 1
        # slit.aperture_downsample returns sum of high_res_map
        assert_allclose(slit.aperture_downsample(self.hr_map), np.sum(self.hr_map))

    def test_frame(self):
        frame = Frame(
            width_outer=0.6,
            width_inner=0.2,
            center_ra=0.0,
            center_dec=0.0,
            angle=0.0,
            delta_pix=0.2,
        )
        xs, ys = frame.aperture_sample()
        # ensure inner box removed: no points with both |x|<width_inner/2 and |y|<width_inner/2
        inner_mask = (np.abs(xs) < 0.2 / 2) & (np.abs(ys) < 0.2 / 2)
        assert not np.any(inner_mask)
        # aperture_downsample returns sum
        assert_allclose(frame.aperture_downsample(self.hr_map), np.sum(self.hr_map))
        assert frame.num_segments == 1

    def test_shell(self):
        shell = Shell(r_in=0.5, r_out=1.1, center_ra=0.0, center_dec=0.0, delta_pix=0.5)
        xs, ys = shell.aperture_sample()
        rs = np.sqrt((xs) ** 2 + (ys) ** 2)
        # all radii should be within [r_in, r_out)
        assert rs.min() >= 0.5
        assert rs.max() < 1.1
        # aperture_downsample returns sum
        assert_allclose(shell.aperture_downsample(self.hr_map), np.sum(self.hr_map))
        assert shell.num_segments == 1

    def test_ifu_grid(self):
        # create a simple IFU grid 2x2, centered grid with delta 1.0
        xg = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        yg = np.array([[-0.5, -0.5], [0.5, 0.5]])
        ifu = IFUGrid(xg, yg, supersampling_factor=2, padding_arcsec=0.0)
        # grid_shape property matches input
        assert ifu.grid_shape == xg.shape
        # delta_pix_xy should be (1.0, 1.0)
        dx, dy = ifu.delta_pix_xy
        assert_allclose(np.abs(dx), np.abs(dy))
        # construct a high-res map that corresponds to supersampling_factor=2:
        # IFUGrid inits delta_pix_sup = abs(delta_x)/supersampling_factor, and creates a supersampled grid.
        # We will create a high-res map shaped like the internal supersampled grid:
        x_sup, y_sup = ifu.aperture_sample()
        # make a high-res map filled with ones: downsample should yield array of ones with shape (2,2)
        hr = (
            np.ones((x_sup.size // 2, 2))
            if False
            else np.ones((int(np.sqrt(x_sup.size)), int(np.sqrt(x_sup.size))))
        )
        # safer approach: create a high-res map that when unpadded and averaged yields a known shape:
        # The IFU.aperture_downsample uses high_res_map.reshape(num_pix_y, s, num_pix_x, s).mean(axis=(1,3))
        num_pix_y, num_pix_x = ifu.grid_shape
        s = ifu.supersampling_factor
        # create high_res_map with shape (num_pix_y*s, num_pix_x*s)
        hr_map = (
            np.arange(num_pix_y * s * num_pix_x * s)
            .reshape(num_pix_y * s, num_pix_x * s)
            .astype(float)
        )
        down = ifu.aperture_downsample(hr_map)
        assert down.shape == (num_pix_y, num_pix_x)
        # verify the (0,0) entry equals mean of top-left sxs block
        expected00 = hr_map[0:s, 0:s].mean()
        assert_allclose(down[0, 0], expected00)

    def test_ifu_shells(self):
        # create radial bins [0, 0.5, 1.0, 1.5]
        r_bins = np.array([0.0, 0.7, 1.4])
        ifu_shells = IFUShells(r_bins, center_ra=0.0, center_dec=0.0, delta_pix=0.5)
        # num_segments should be len(r_bins)-1
        assert ifu_shells.num_segments == len(r_bins) - 1
        # create a high-res map with values equal to radius for simplicity
        x_sup, y_sup = ifu_shells.aperture_sample()
        hr_vals = np.sqrt((x_sup) ** 2 + (y_sup) ** 2)
        # downsample: for each shell we expect mean radius to be between the bins; just check shape and finite values
        out = ifu_shells.aperture_downsample(hr_vals)
        assert out.shape == (ifu_shells.num_segments,)
        assert np.isfinite(out).all()

    def test_ifu_binned(self):
        # create IFUBinned with 2x2 grid and bins
        bins = np.array([[0, 1], [2, -1]])
        # Need grid with shape (2,2)
        xg = np.array([[0.0, 1.0], [0.0, 1.0]])
        yg = np.array([[0.0, 0.0], [1.0, 1.0]])
        ifu_b = IFUBinned(xg, yg, bins)
        assert_array_equal(ifu_b.bins, bins)
        # unique bins excluding -1 are 0,1,2 -> num_segments = 3
        assert ifu_b.num_segments == 3
        # create a small high-res map consistent with no supersampling and no padding (shape 2x2)
        hr = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = ifu_b.aperture_downsample(hr)
        # bin 0 -> hr[0,0]=1 ; bin1 -> hr[0,1]=2 ; bin2 -> hr[1,0]=3
        assert_allclose(out, np.array([1.0, 2.0, 3.0]))


class TestRaise(object):

    def test_raise_ifu_grid_not_square(self):
        xg = np.array([[0.0, 0.0], [0.0, 1.0]])  # x size is 1
        yg = np.array([[0.0, 0.0], [0.0, -0.5]])  # y size is 0.5
        with pytest.raises(ValueError):
            IFUGrid(xg, yg, supersampling_factor=1, padding_arcsec=0)


if __name__ == "__main__":
    pytest.main()
