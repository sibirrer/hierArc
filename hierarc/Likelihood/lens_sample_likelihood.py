import copy

from hierarc.Likelihood.hierarchy_likelihood import LensLikelihood
from hierarc.Likelihood.LensLikelihood.double_source_plane import DSPLikelihood


class LensSampleLikelihood(object):
    """
    class to evaluate the likelihood of a cosmology given a sample of angular diameter posteriors
    Currently this class does not include possible covariances between the lens samples
    """
    def __init__(self, kwargs_lens_list, normalized=False):
        """

        :param kwargs_lens_list: keyword argument list specifying the arguments of the LensLikelihood class
        :param normalized: bool, if True, returns the normalized likelihood, if False, separates the constant prefactor
         (in case of a Gaussian 1/(sigma sqrt(2 pi)) ) to compute the reduced chi2 statistics
        """
        self._lens_list = []
        for kwargs_lens in kwargs_lens_list:
            if kwargs_lens['likelihood_type'] == 'DSPL':
                _kwargs_lens = copy.deepcopy(kwargs_lens)
                _kwargs_lens.pop('likelihood_type')
                self._lens_list.append(DSPLikelihood(normalized=normalized, **_kwargs_lens))
            else:
                self._lens_list.append(LensLikelihood(normalized=normalized, **kwargs_lens))

    def log_likelihood(self, cosmo, kwargs_lens=None, kwargs_kin=None, kwargs_source=None):
        """

        :param cosmo: astropy.cosmology instance
        :param kwargs_lens: keywords of the parameters of the lens model
        :param kwargs_kin: keyword arguments of the kinematic model
        :param kwargs_source: keyword argument of the source model (such as SNe)
        :return: log likelihood of the combined lenses
        """
        log_likelihood = 0
        for lens in self._lens_list:
            log_likelihood += lens.lens_log_likelihood(cosmo=cosmo, kwargs_lens=kwargs_lens, kwargs_kin=kwargs_kin,
                                                       kwargs_source=kwargs_source)
        return log_likelihood

    def num_data(self):
        """
        number of data points across the lens sample

        :return: integer
        """
        num = 0
        for lens in self._lens_list:
            num += lens.num_data()
        return num
