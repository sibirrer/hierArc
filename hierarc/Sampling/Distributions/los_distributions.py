import numpy as np
from hierarc.Util.distribution_util import PDFSampling
from scipy.stats import genextreme


class LOSDistribution(object):
    """Line of sight distribution drawing."""

    def __init__(
        self,
        global_los_distribution=False,
        los_distributions=None,
        individual_distribution=None,
        kwargs_individual=None,
    ):
        """

        :param global_los_distribution: if integer, will draw from the global kappa distribution specified in that
         integer. If False, will instead draw from the distribution specified in kappa_pdf.
        :type global_los_distribution: bool or int
        :param los_distributions: list of all line of sight distributions parameterized
        :type los_distributions: list of str or None
        :param individual_distribution: name of the individual distribution ["GEV" and "PDF"]
        :type individual_distribution: str or None
        :param kwargs_individual: dictionary of the parameters of the individual distribution
         If individual_distribution is "PDF":
         "pdf_array": array of probability density function of the external convergence distribution
         binned according to kappa_bin_edges
         "bin_edges": array of length (len(kappa_pdf)+1), bin edges of the kappa PDF
         If individual_distribution is "GEV":
         "xi", "mean", "sigma"
        :type kwargs_individual: dict or None
        """

        self._global_los_distribution = global_los_distribution
        if (
            isinstance(self._global_los_distribution, int)
            and self._global_los_distribution is not False
        ):
            self._draw_kappa_global = True
            self._los_distribution = los_distributions[global_los_distribution]
        else:
            self._draw_kappa_global = False
        if (not self._draw_kappa_global and individual_distribution is not None
        ):
            if individual_distribution == "PDF":
                self._kappa_dist = PDFSampling(**kwargs_individual)
            elif individual_distribution == "GEV":
                self._kappa_dist = GEV(**kwargs_individual)
            else:
                raise ValueError("individual_distribution %s not supported. Chose among 'GEV' and 'PDF'")
            self._draw_kappa_individual = True
        else:
            self._draw_kappa_individual = False

    def draw_los(self, kwargs_los, size=1):
        """Draw from the distribution of line of sight convergence.

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
                kappa_ext_draw = genextreme.rvs(c=xi, loc=mean, scale=sigma, size=size)
            else:
                raise ValueError(
                    "line of sight distribution %s not valid." % self._los_distribution
                )
        else:
            kappa_ext_draw = 0
        return kappa_ext_draw

    def draw_bool(self, kwargs_los):
        """Whether single-valued or extended distribution (need to draw from)

        :param kwargs_los: list of keyword arguments for line of sight distributions
        :return: boolean, True with samples need to be drawn, else False
        """
        if self._draw_kappa_individual is True:
            return True
        elif self._draw_kappa_global is True:
            if kwargs_los[self._global_los_distribution]["sigma"] != 0:
                return True
        return False


class GEV(object):
    """
    draw from General Extreme Value distribution
    """
    def __init__(self, xi, mean, sigma):
        """

        :param xi: Xi value of GEV
        :param mean: mean of GEV
        :param sigma: sigma of GEV
        """
        self._xi = xi
        self._mean = mean
        self._sigma = sigma

    def draw(self, n=1):
        """
        draws from the PDF of the GEV distribution

        :param n: number of draws from distribution
        :type n: int
        :return: draws according to the PDF of the distribution
        """
        kappa_ext_draw = genextreme.rvs(c=self._xi, loc=self._mean, scale=self._sigma, size=n)
        return kappa_ext_draw
