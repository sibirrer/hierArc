from lenstronomy.LightModel.Profiles.gaussian import Gaussian
import lenstronomy.Util.util as util
from scipy.signal import convolve2d
import numpy as np


class PSF(object):
    """General class to handle the PSF in the GalKin module for rendering the
    displacement of photons/spectro."""

    def __init__(self, psf_type, **kwargs_psf):
        """

        :param psf_type: string, point spread function type, current support for 'GAUSSIAN', 'MOFFAT', 'MULTI-GAUSSIAN',
            and 'PIXEL'
        :param kwargs_psf: keyword argument describing the relevant parameters of the PSF.
        """
        self.psf_type = psf_type
        if psf_type == "GAUSSIAN":
            self._psf = PSFGaussian(**kwargs_psf)
        elif psf_type == "MULTI-GAUSSIAN":
            self._psf = PSFMultiGaussian(**kwargs_psf)
        elif psf_type == "PIXEL":
            self._psf = PSFPixel(**kwargs_psf)
        else:
            raise ValueError("psf_type %s not supported for convolution!" % psf_type)

    def convolution_kernel(self, **kwargs):
        """Normalized convolution kernel."""
        return self._psf.convolution_kernel(**kwargs)

    def convolve(self, data, **kernel_kwargs):
        kernel = self.convolution_kernel(**kernel_kwargs)
        return convolve2d(data, kernel, mode="same")

    @property
    def psf_fwhm(self):
        return self._psf.fwhm

    @property
    def psf_sigmas(self):
        return self._psf.sigmas

    @property
    def psf_amplitudes(self):
        return self._psf.amplitudes

    @property
    def psf_supersampling_factor(self):
        """Retrieve supersampling factor if stored as a private variable."""
        return self._psf.supersampling_factor


class PSFGaussian(object):
    """Gaussian PSF."""

    def __init__(self, fwhm):
        """

        :param fwhm: full width at half maximum seeing condition
        """
        self._fwhm = fwhm

    def convolution_kernel(self, delta_pix, num_pix=21):
        """Normalized convolution kernel.

        :param delta_pix: pixel scale of kernel
        :param num_pix: number of pixels per axis of the kernel
        :return: 2d numpy array of kernel
        """
        kernel = _make_gaussian_psf_kernel(
            self.fwhm, delta_pix, num_pix, normalize=True
        )
        return kernel

    @property
    def fwhm(self):
        """Retrieve FWHM of PSF if stored as a private variable."""
        return self._fwhm

    @property
    def sigmas(self):
        """Retrieve sigma of PSF if stored as a private variable."""
        return util.fwhm2sigma(self._fwhm)

    @property
    def amplitudes(self):
        """Retrieve amplitude of PSF if stored as a private variable."""
        return 1.0

    @property
    def supersampling_factor(self):
        """Retrieve supersampling factor if stored as a private variable."""
        return 1


class PSFMultiGaussian(object):
    """Multi-Gaussian PSF."""

    def __init__(self, fwhm, amplitudes, sigmas):
        """

        :param fwhm: full width at half maximum seeing condition
        :param amplitudes: amplitudes of PSF as obtained by mgefit, they must sum to 1.
        :param sigmas: sigmas of PSF as obtained by mgefit and converted to arcseconds
        """
        self._fwhm = fwhm
        self._amplitudes = amplitudes / np.sum(amplitudes)
        self._sigmas = sigmas

    def convolution_kernel(self, delta_pix, num_pix=21):
        """Normalized convolution kernel.

        :param delta_pix: pixel scale of kernel
        :param num_pix: number of pixels per axis of the kernel
        :return: 2d numpy array of kernel
        """
        fwhms = util.sigma2fwhm(self.sigmas)
        kernel = np.zeros((num_pix, num_pix))
        for amp, fwhm in zip(self._amplitudes, fwhms):
            kernel += amp * _make_gaussian_psf_kernel(
                fwhm, delta_pix, num_pix, normalize=False
            )
        kernel /= np.sum(kernel)
        return kernel

    @property
    def fwhm(self):
        """Retrieve FWHM of PSF if stored as a private variable."""
        return self._fwhm

    @property
    def sigmas(self):
        """Retrieve sigmas of PSF if stored as a private variable."""
        return self._sigmas

    @property
    def amplitudes(self):
        """Retrieve amplitudes of PSF if stored as a private variable."""
        return self._amplitudes

    @property
    def supersampling_factor(self):
        """Retrieve supersampling factor if stored as a private variable."""
        return 1


class PSFPixel(object):
    """Pixelated PSF model over a supersampled grid."""

    def __init__(self, fwhm, kernel, supersampling_factor):
        self._fwhm = fwhm
        self._kernel = kernel
        self._kernel_size = kernel.shape[0]
        self._supersampling_factor = supersampling_factor

    def convolution_kernel(self, **kwargs):
        return self._kernel

    @property
    def fwhm(self):
        """Retrieve FWHM of PSF if stored as a private variable."""
        return self._fwhm

    @property
    def sigmas(self):
        """Retrieve sigmas of PSF if stored as a private variable."""
        return None

    @property
    def amplitudes(self):
        """Retrieve amplitudes of PSF if stored as a private variable."""
        return None

    @property
    def supersampling_factor(self):
        """Retrieve supersampling factor if stored as a private variable."""
        return self._supersampling_factor


def _make_gaussian_psf_kernel(fwhm, delta_pix, num_pix=21, normalize=True):
    """Helper function to make a Gaussian PSF kernel."""
    x_grid, y_grid = util.make_grid(num_pix, delta_pix)
    sigma = util.fwhm2sigma(fwhm)
    gaussian = Gaussian()
    kernel = gaussian.function(
        x_grid, y_grid, amp=1.0, sigma=sigma, center_x=0, center_y=0
    )
    kernel = util.array2image(kernel)
    if normalize:
        kernel /= np.sum(kernel)
    return kernel
