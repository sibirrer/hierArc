import pytest
import numpy as np
import numpy.testing as npt
from hierarc.Util import distribution_util
from hierarc.Util.distribution_util import PDFSampling
from lenstronomy.Util import prob_density


class TestProbDensity(object):

    def setup(self):
        np.random.seed(seed=42)

    def gauss(self, x, simga):
        return np.exp(-(x/(simga))**2/2)

    def test_approx_cdf_1d(self):
        bin_edges = np.linspace(start=-5, stop=5, num=501)
        x_array = (bin_edges[1:] + bin_edges[:-1]) / 2
        sigma = 1.
        pdf_array = self.gauss(x_array, simga=sigma)
        pdf_array /= np.sum(pdf_array)

        cdf_array, cdf_func, cdf_inv_func = distribution_util.approx_cdf_1d(bin_edges, pdf_array)
        npt.assert_almost_equal(cdf_array[0], 0, decimal=7)
        npt.assert_almost_equal(cdf_array[-1], 1, decimal=8)
        npt.assert_almost_equal(cdf_func(0), 0.5, decimal=2)
        npt.assert_almost_equal(cdf_inv_func(0.5), 0., decimal=2)

    def test_compute_lower_upper_errors(self):
        x_array = np.linspace(-5, 5, 1000)
        bin_edges = np.linspace(start=-5, stop=5, num=1000+1)
        sigma = 1.
        pdf_array = self.gauss(x_array, simga=sigma)
        approx = PDFSampling(bin_edges, pdf_array)
        np.random.seed(42)
        sample = approx.draw(n=20000)
        mean, [[lower_sigma1, upper_sigma1], [lower_sigma2, upper_sigma2], [lower_sigma3, upper_sigma3]] = prob_density.compute_lower_upper_errors(sample, num_sigma=3)
        npt.assert_almost_equal(mean, 0, decimal=2)
        npt.assert_almost_equal(lower_sigma1, sigma, decimal=2)
        npt.assert_almost_equal(lower_sigma2, 2*sigma, decimal=1)
        npt.assert_almost_equal(lower_sigma3, 3 * sigma, decimal=1)

        draw = approx.draw_one
        assert len(draw) == 1


if __name__ == '__main__':
    pytest.main()
