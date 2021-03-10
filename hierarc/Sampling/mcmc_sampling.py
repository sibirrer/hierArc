import emcee
from hierarc.Likelihood.cosmo_likelihood import CosmoLikelihood
from lenstronomy.Util import sampling_util


class MCMCSampler(object):
    """
    class which executes the different sampling  methods
    """
    def __init__(self, *args, **kwargs):
        """
        initialise the classes of the chain and for parameter options
        :param args: positional arguments for the CosmoLikelihood() instance
        :param kwargs: keyword arguments for the CosmoLikelihood() instance

        """
        self.chain = CosmoLikelihood(*args, **kwargs)
        self.param = self.chain.param

    def mcmc_emcee(self, n_walkers, n_burn, n_run, kwargs_mean_start, kwargs_sigma_start, continue_from_backend=False,
                   **kwargs_emcee):
        """
        runs the EMCEE MCMC sampling

        :param n_walkers: number of walkers
        :param n_burn: number of iteration of burn in (not stored in the output sample
        :param n_run: number of iterations (after burn in) to be sampled
        :param kwargs_mean_start: keyword arguments of the mean starting position
        :param kwargs_sigma_start: keyword arguments of the spread in the initial particles per parameter
        :param continue_from_backend: bool, if True and 'backend' in kwargs_emcee, will continue a chain sampling from backend
        :param kwargs_emcee: keyword argument for the emcee (e.g. to specify backend)
        :return: samples of the EMCEE run
        """

        num_param = self.param.num_param
        sampler = emcee.EnsembleSampler(n_walkers, num_param, self.chain.likelihood, args=(), **kwargs_emcee)
        mean_start = self.param.kwargs2args(**kwargs_mean_start)
        sigma_start = self.param.kwargs2args(**kwargs_sigma_start)
        p0 = sampling_util.sample_ball(mean_start, sigma_start, n_walkers)
        backend = kwargs_emcee.get('backend', None)
        if backend is not None:
            if continue_from_backend:
                p0 = None
            else:
                backend.reset(n_walkers, num_param)
        sampler.run_mcmc(p0, n_burn+n_run, progress=True)
        flat_samples = sampler.get_chain(discard=n_burn, thin=1, flat=True)
        log_prob = sampler.get_log_prob(discard=n_burn, thin=1, flat=True)
        return flat_samples, log_prob

    def param_names(self, latex_style=False):
        """
        list of parameter names being sampled in the same order as the sampling

        :param latex_style: bool, if True returns strings in latex symbols, else in the convention of the sampler
        :return: list of strings
        """
        labels = self.param.param_list(latex_style=latex_style)
        return labels
