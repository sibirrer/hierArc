import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.signal import convolve2d
from hierarc.JAM.psf import PSF
from lenstronomy.Util import kernel_util
import lenstronomy.Util.util as util


class TestPSF:
    def setup_method(self):
        self.fwhm = 0.7
        self.delta_pix = 0.05
        self.num_pix = 5
        self.data = np.arange(100).reshape(10, 10).astype(float)
        self.gaussian_kernel = kernel_util.kernel_gaussian(
            self.num_pix, self.delta_pix, self.fwhm
        )

    def test_psf_gaussian(self):
        g = PSF("GAUSSIAN", fwhm=self.fwhm)
        k = g.convolution_kernel(delta_pix=self.delta_pix, num_pix=self.num_pix)
        assert k.shape == (self.num_pix, self.num_pix)
        assert_allclose(k.sum(), 1.0, rtol=1e-6, atol=1e-8)
        # fwhm property
        assert_allclose(g.psf_fwhm, self.fwhm)
        assert_allclose(g.psf_sigmas, util.fwhm2sigma(self.fwhm))
        assert_allclose(g.psf_amplitudes, 1.0)
        assert g.psf_supersampling_factor == 1

    def test_psf_multigaussian(self):
        amps = np.array([0.6, 0.4])
        sigmas = np.array([0.3, 0.8])
        mg = PSF("MULTI-GAUSSIAN", fwhm=self.fwhm, amplitudes=amps, sigmas=sigmas)
        assert_allclose(np.sum(mg.psf_amplitudes), 1.0)
        assert_array_equal(mg.psf_sigmas, sigmas)
        assert_allclose(mg.psf_fwhm, self.fwhm)
        k_mg = mg.convolution_kernel(delta_pix=self.delta_pix, num_pix=self.num_pix)
        assert k_mg.shape == (self.num_pix, self.num_pix)
        assert_allclose(k_mg.sum(), 1.0, rtol=1e-6, atol=1e-8)

    def test_psfpixel_kernel_passthrough_and_properties(self):
        # create a pixel PSF from an external kernel and supersampling factor
        ss = 3
        pixel_psf = PSF(
            "PIXEL",
            fwhm=self.fwhm,
            kernel=self.gaussian_kernel,
            supersampling_factor=ss,
        )
        assert pixel_psf._psf._kernel_size == self.gaussian_kernel.shape[0]
        k_ret = pixel_psf.convolution_kernel()
        assert_allclose(k_ret, self.gaussian_kernel)
        assert_allclose(pixel_psf.psf_fwhm, self.fwhm)
        assert pixel_psf.psf_sigmas is None
        assert pixel_psf.psf_amplitudes is None
        assert pixel_psf.psf_supersampling_factor == ss

    def test_psf_convolve(self):
        psf = PSF("GAUSSIAN", fwhm=self.fwhm)
        kernel = psf.convolution_kernel(delta_pix=self.delta_pix, num_pix=self.num_pix)
        ref_kernel = self.gaussian_kernel
        assert_allclose(kernel, ref_kernel, rtol=1e-7, atol=1e-10)
        conv = psf.convolve(self.data, delta_pix=self.delta_pix, num_pix=self.num_pix)
        conv_ref = convolve2d(self.data, kernel, mode="same")
        assert_allclose(conv, conv_ref)


class TestRaise:
    def test_invalid_psf_type_raises(self):
        with pytest.raises(ValueError):
            PSF("UNKNOWN_TYPE")


if __name__ == "__main__":
    pytest.main()
