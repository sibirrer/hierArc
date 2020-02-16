
from hierarc.Likelihood.hierarchy_likelihood import LensLikelihood


class LensSampleLikelihood(object):
    """
    class to evaluate the likelihood of a cosmology given a sample of angular diameter posteriors
    Currently this class does not include possible covariances between the lens samples
    """
    def __init__(self, kwargs_lens_list):
        """

        :param kwargs_lens_list: keyword argument list specifying the arguments of the LensLikelihood class
        """
        self._lens_list = []
        for kwargs_lens in kwargs_lens_list:
            self._lens_list.append(LensLikelihood(**kwargs_lens))

    def log_likelihood(self, cosmo, **kwargs):
        """

        :param cosmo: astropy.cosmology instance
        :param kwargs: keywords of the parameters
        :return: log likelihood of the combined lenses
        """
        logL = 0
        for lens in self._lens_list:
            logL += lens.lens_log_likelihood(cosmo=cosmo, **kwargs)
            print(logL, 'test logl')
        return logL
