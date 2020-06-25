from hierarc.Likelihood.lens_sample_likelihood import LensSampleLikelihood
from hierarc.Sampling.ParamManager.param_manager import ParamManager
from hierarc.Likelihood.cosmo_interp import CosmoInterp
import numpy as np


class CosmoLikelihood(object):
    """
    this class contains the likelihood function of the Strong lensing analysis
    """

    def __init__(self, kwargs_likelihood_list, cosmology, kwargs_bounds, ppn_sampling=False,
                 lambda_mst_sampling=False, lambda_mst_distribution='delta', anisotropy_sampling=False,
                 kappa_ext_sampling=False, kappa_ext_distribution='NONE', alpha_lambda_sampling=False,
                 lambda_ifu_sampling=False, lambda_ifu_distribution='NONE', sigma_v_systematics=False,
                 log_scatter=False,
                 anisotropy_model='OM', anisotropy_distribution='NONE', custom_prior=None, interpolate_cosmo=True,
                 num_redshift_interp=100, cosmo_fixed=None):
        """

        :param kwargs_likelihood_list: keyword argument list specifying the arguments of the LensLikelihood class
        :param cosmology: string describing cosmological model
        :param kwargs_bounds: keyword arguments of the lower and upper bounds and parameters that are held fixed.
        Includes:
        'kwargs_lower_lens', 'kwargs_upper_lens', 'kwargs_fixed_lens',
        'kwargs_lower_kin', 'kwargs_upper_kin', 'kwargs_fixed_kin'
        'kwargs_lower_cosmo', 'kwargs_upper_cosmo', 'kwargs_fixed_cosmo'
        :param ppn_sampling:post-newtonian parameter sampling
        :param lambda_mst_sampling: bool, if True adds a global mass-sheet transform parameter in the sampling
        :param lambda_mst_distribution: string, defines the distribution function of lambda_mst
        :param lambda_ifu_sampling: bool, if True samples a separate lambda_mst for a second (e.g. IFU) data set
        independently
        :param lambda_ifu_distribution: string, distribution function of the lambda_ifu parameter
        :param alpha_lambda_sampling: bool, if True samples a parameter alpha_lambda, which scales lambda_mst linearly
         according to a predefined quantity of the lens
        :param kappa_ext_sampling: bool, if True samples a global external convergence parameter
        :param kappa_ext_distribution: string, distribution function of the kappa_ext parameter
        :param anisotropy_sampling: bool, if True adds a global stellar anisotropy parameter that alters the single lens
        kinematic prediction
        :param anisotropy_model: string, specifies the stellar anisotropy model
        :param anisotropy_distribution: string, distribution of the anisotropy parameters
        :param sigma_v_systematics: bool, if True samples paramaters relative to systematics in the velocity dispersion
         measurement
        :param log_scatter: boolean, if True, samples the Gaussian scatter amplitude in log space (and thus flat prior in log)
        :param custom_prior: None or a definition that takes the keywords from the CosmoParam conventions and returns a
        log likelihood value (e.g. prior)
        :param interpolate_cosmo: bool, if True, uses interpolated comoving distance in the calculation for speed-up
        :param num_redshift_interp: int, number of redshift interpolation steps
        :param cosmo_fixed: astropy.cosmology instance to be used and held fixed throughout the sampling
        """
        self._cosmology = cosmology
        self._kwargs_lens_list = kwargs_likelihood_list
        self._likelihoodLensSample = LensSampleLikelihood(kwargs_likelihood_list)
        self.param = ParamManager(cosmology, ppn_sampling=ppn_sampling, lambda_mst_sampling=lambda_mst_sampling,
                                  lambda_mst_distribution=lambda_mst_distribution,
                                  lambda_ifu_sampling=lambda_ifu_sampling,
                                  lambda_ifu_distribution=lambda_ifu_distribution,
                                  alpha_lambda_sampling=alpha_lambda_sampling,
                                  sigma_v_systematics=sigma_v_systematics,
                                  kappa_ext_sampling=kappa_ext_sampling, kappa_ext_distribution=kappa_ext_distribution,
                                  anisotropy_sampling=anisotropy_sampling, anisotropy_model=anisotropy_model,
                                  anisotropy_distribution=anisotropy_distribution, log_scatter=log_scatter,
                                  **kwargs_bounds)
        self._lower_limit, self._upper_limit = self.param.param_bounds
        self._prior_add = False
        if custom_prior is not None:
            self._prior_add = True
        self._custom_prior = custom_prior
        self._interpolate_cosmo = interpolate_cosmo
        self._num_redshift_interp = num_redshift_interp
        self._cosmo_fixed = cosmo_fixed
        z_max = 0
        for kwargs_lens in kwargs_likelihood_list:
            if kwargs_lens['z_source'] > z_max:
                z_max = kwargs_lens['z_source']
        self._z_max = z_max

    def likelihood(self, args):
        """

        :param args: list of sampled parameters
        :return: log likelihood of the combined lenses
        """
        for i in range(0, len(args)):
            if args[i] < self._lower_limit[i] or args[i] > self._upper_limit[i]:
                return -np.inf

        kwargs_cosmo, kwargs_lens, kwargs_kin = self.param.args2kwargs(args)
        if self._cosmology == "oLCDM":
            # assert we are not in a crazy cosmological situation that prevents computing the angular distance integral
            h0, ok, om = kwargs_cosmo['h0'], kwargs_cosmo['ok'], kwargs_cosmo['om']
            if np.any(
                    [ok * (1.0 + lens['z_source']) ** 2 + om * (1.0 + lens['z_source']) ** 3 + (1.0 - om - ok) <= 0 for lens in
                     self._kwargs_lens_list]):
                return -np.inf
            # make sure that Omega_DE is not negative...
            if 1.0 - om - ok <= 0:
                return -np.inf
        cosmo = self.cosmo_instance(kwargs_cosmo)
        logL = self._likelihoodLensSample.log_likelihood(cosmo=cosmo, kwargs_lens=kwargs_lens, kwargs_kin=kwargs_kin)

        if self._prior_add is True:
            logL += self._custom_prior(kwargs_cosmo, kwargs_lens, kwargs_kin)
        return logL

    def cosmo_instance(self, kwargs_cosmo):
        """

        :param kwargs_cosmo: cosmology parameter keyword argument list
        :return: astropy.cosmology (or equivalent interpolation scheme class)
        """
        if self._cosmo_fixed is None:
            cosmo = self.param.cosmo(kwargs_cosmo)
            if self._interpolate_cosmo is True:
                cosmo = CosmoInterp(cosmo=cosmo, z_stop=self._z_max, num_interp=self._num_redshift_interp)
        else:
            if self._interpolate_cosmo is True:
                if not hasattr(self, '_cosmo_fixed_interp'):
                    self._cosmo_fixed_interp = CosmoInterp(cosmo=self._cosmo_fixed, z_stop=self._z_max, num_interp=self._num_redshift_interp)
                cosmo = self._cosmo_fixed_interp
            else:
                cosmo = self._cosmo_fixed
        return cosmo
