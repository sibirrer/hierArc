import numpy as np
from scipy.interpolate import interp1d


class PDFSampling(object):
    """
    class for approximations with a given pdf sample
    """
    def __init__(self, bin_edges, pdf_array):
        """

        :param bin_edges: bin edges of PDF values
        :param pdf_array: pdf array of given bins (len(bin_edges)-1)
        """
        assert len(bin_edges) == len(pdf_array) + 1
        self._cdf_array, self._cdf_func, self._cdf_inv_func = approx_cdf_1d(bin_edges, pdf_array)

    def draw(self, n=1):
        """

        :return:
        """
        p = np.random.uniform(0, 1, n)
        return self._cdf_inv_func(p)

    @property
    def draw_one(self):
        """

        :return:
        """
        return self.draw(n=1)


def approx_cdf_1d(bin_edges, pdf_array):

    """

    :param bin_edges: bin edges of PDF values
    :param pdf_array: pdf array of given bins (len(bin_edges)-1)
    :return: cdf, interp1d function of cdf, inverse interpolation function
    """
    assert len(bin_edges) == len(pdf_array) + 1
    norm_pdf = pdf_array/np.sum(pdf_array)
    cdf_array = np.zeros_like(bin_edges)
    cdf_array[0] = 0
    for i in range(0, len(norm_pdf)):
        cdf_array[i+1] = cdf_array[i] + norm_pdf[i]
    cdf_func = interp1d(bin_edges, cdf_array)
    cdf_inv_func = interp1d(cdf_array, bin_edges)
    return cdf_array, cdf_func, cdf_inv_func
