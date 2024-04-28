import numpy as np
from hierarc.Util.distribution_util import PDFSampling


class LOSDistribution(object):
    """
    line of sight distribution drawing
    """
    def __init__(self, kappa_pdf=None, kappa_bin_edges=None, global_los_distribution=False,
                 los_distributions=None):
        """

        :param global_los_distribution: if integer, will draw from the global kappa distribution specified in that
         integer. If False, will instead draw from the distribution specified in kappa_pdf.
        :type global_los_distribution: bool or int
        :param kappa_pdf: array of probability density function of the external convergence distribution
         binned according to kappa_bin_edges
        :param kappa_bin_edges: array of length (len(kappa_pdf)+1), bin edges of the kappa PDF
        :param los_distributions: list of all line of sight distributions parameterized
        :type los_distributions: list of str or None
        """

        self._global_los_distribution = global_los_distribution
        if isinstance(self._global_los_distribution, int) and self._global_los_distribution is not False:
            self._draw_kappa_global = True
            self._los_distribution = los_distributions[global_los_distribution]
        else:
            self._draw_kappa_global = False
        if kappa_pdf is not None and kappa_bin_edges is not None and not self._draw_kappa_global:
            print("test kappa pdf sampling")
            self._kappa_dist = PDFSampling(
                bin_edges=kappa_bin_edges, pdf_array=kappa_pdf
            )
            self._draw_kappa_individual = True
        else:
            self._draw_kappa_individual = False

    def draw_los(self, kwargs_los, size=1):
        """
        Draw from the distribution of line of sight convergence

        :param kwargs_los: line of sight parameters
        :type kwargs_los: list of dict
        :param size: how many samples to be drawn
        :type size: int>0
        :return: external convergence draw
        """
        if self._draw_kappa_individual is True:
            kappa_ext_draw = self._kappa_dist.draw(n=size)
        elif self._draw_kappa_global:
            kwargs_los_i = kwargs_los[self._global_los_distribution]
            if self._los_distribution == "GAUSSIAN":
                los_mean = kwargs_los_i["mean"]
                los_sigma = kwargs_los_i["sigma"]
                kappa_ext_draw = np.random.normal(los_mean, los_sigma, size=size)
            elif self._los_distribution == "GEV":
                mean = kwargs_los_i["mean"]
                sigma = kwargs_los_i["sigma"]
                xi = kwargs_los_i["xi"]
                from scipy.stats import genextreme
                kappa_ext_draw = genextreme.rvs(c=xi, loc=mean, scale=sigma, size=size)
            else:
                raise ValueError("line of sight distribution %s not valid." % self._los_distribution)
        else:
            kappa_ext_draw = 0
        return kappa_ext_draw

    def draw_bool(self, kwargs_los):
        """
        whether single-valued or extended distribution (need to draw from)

        :param kwargs_los: list of keyword arguments for line of sight distributions
        :return: boolean, True with samples need to be drawn, else False
        """
        if self._draw_kappa_individual is True:
            return True
        elif self._draw_kappa_global is True:
            if kwargs_los[self._global_los_distribution]["sigma"] != 0:
                return True
        return False
