from hierarc.Likelihood.lens_sample_likelihood import LensSampleLikelihood
from hierarc.Sampling.ParamManager.param_manager import ParamManager
from lenstronomy.Cosmo.cosmo_interp import CosmoInterp
from hierarc.Likelihood.SneLikelihood.sne_likelihood import SneLikelihood
from hierarc.Likelihood.KDELikelihood.kde_likelihood import KDELikelihood
from hierarc.Likelihood.KDELikelihood.chain import rescale_vector_to_unity
import numpy as np


class CosmoLikelihood(object):
    """
    this class contains the likelihood function of the Strong lensing analysis
    """

    def __init__(self, kwargs_likelihood_list, cosmology, kwargs_bounds, sne_likelihood=None,
                 kwargs_sne_likelihood=None, KDE_likelihood_chain=None, kwargs_kde_likelihood=None,
                 ppn_sampling=False,
                 lambda_mst_sampling=False, lambda_mst_distribution='delta', anisotropy_sampling=False,
                 kappa_ext_sampling=False, kappa_ext_distribution='NONE',
                 alpha_lambda_sampling=False, beta_lambda_sampling=False,
                 lambda_ifu_sampling=False, lambda_ifu_distribution='NONE', sigma_v_systematics=False,
                 sne_apparent_m_sampling=False, sne_distribution='GAUSSIAN', z_apparent_m_anchor=0.1,
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
        :param KDE_likelihood_chain: (Likelihood.chain.Chain). Chain object to be evaluated with a kernel density
         estimator
        :param kwargs_kde_likelihood: keyword argument for the KDE likelihood, see KDELikelihood module for options
        :param sne_likelihood: (string), optional. Sampling supernovae relative expansion history likelihood, see
         SneLikelihood module for options
        :param kwargs_sne_likelihood: keyword argument for the SNe likelihood, see SneLikelihood module for options
        :param ppn_sampling:post-newtonian parameter sampling
        :param lambda_mst_sampling: bool, if True adds a global mass-sheet transform parameter in the sampling
        :param lambda_mst_distribution: string, defines the distribution function of lambda_mst
        :param lambda_ifu_sampling: bool, if True samples a separate lambda_mst for a second (e.g. IFU) data set
        independently
        :param lambda_ifu_distribution: string, distribution function of the lambda_ifu parameter
        :param alpha_lambda_sampling: bool, if True samples a parameter alpha_lambda, which scales lambda_mst linearly
         according to the lens posterior kwargs 'lambda_scaling_property'
        :param beta_lambda_sampling: bool, if True samples a parameter beta_lambda, which scales lambda_mst linearly
         according to the lens posterior kwargs 'lambda_scaling_property_beta'
        :param kappa_ext_sampling: bool, if True samples a global external convergence parameter
        :param kappa_ext_distribution: string, distribution function of the kappa_ext parameter
        :param anisotropy_sampling: bool, if True adds a global stellar anisotropy parameter that alters the single lens
        kinematic prediction
        :param anisotropy_model: string, specifies the stellar anisotropy model
        :param anisotropy_distribution: string, distribution of the anisotropy parameters
        :param sigma_v_systematics: bool, if True samples paramaters relative to systematics in the velocity dispersion
         measurement
        :param sne_apparent_m_sampling: boolean, if True, samples/queries SNe unlensed magnitude distribution
         (not intrinsic magnitudes but apparent!)
        :param sne_distribution: string, apparent non-lensed brightness distribution (in linear space).
         Currently supports:
         'GAUSSIAN': Gaussian distribution
        :param z_apparent_m_anchor: redshift of pivot/anchor at which the apparent SNe brightness is defined relative to
        :param log_scatter: boolean, if True, samples the Gaussian scatter amplitude in log space
         (and thus flat prior in log)
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
                                  beta_lambda_sampling=beta_lambda_sampling,
                                  sne_apparent_m_sampling=sne_apparent_m_sampling,
                                  sne_distribution=sne_distribution, z_apparent_m_anchor=z_apparent_m_anchor,
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
        if sne_likelihood is not None:
            if kwargs_sne_likelihood is None:
                kwargs_sne_likelihood = {}
            self._sne_likelihood = SneLikelihood(sample_name=sne_likelihood, **kwargs_sne_likelihood)
            z_max = np.max(self._sne_likelihood.zcmb)
            self._sne_evaluate = True
        else:
            self._sne_evaluate = False

        if KDE_likelihood_chain is not None:
            if kwargs_kde_likelihood is None:
                kwargs_kde_likelihood = {}
            self._kde_likelihood = KDELikelihood(KDE_likelihood_chain, **kwargs_kde_likelihood)
            self._kde_evaluate = True
            self._chain_params = self._kde_likelihood.chain.list_params()
        else:
            self._kde_evaluate = False

        for kwargs_lens in kwargs_likelihood_list:
            if 'z_source' in kwargs_lens:
                if kwargs_lens['z_source'] > z_max:
                    z_max = kwargs_lens['z_source']
            if 'z_source_2' in kwargs_lens:
                if kwargs_lens['z_source_2'] > z_max:
                    z_max = kwargs_lens['z_source_2']
        self._z_max = z_max

    def likelihood(self, args, kwargs_cosmo_interp=None):
        """

        :param args: list of sampled parameters
        :param kwargs_cosmo_interp: interpolated angular diameter distances with
         'ang_diameter_distances' and 'redshifts', and optionally 'ok' and 'K' in none-flat scenarios
        :type kwargs_cosmo_interp: dict
        :return: log likelihood of the combined lenses
        """
        for i in range(0, len(args)):
            if args[i] < self._lower_limit[i] or args[i] > self._upper_limit[i]:
                return -np.inf

        kwargs_cosmo, kwargs_lens, kwargs_kin, kwargs_source = self.param.args2kwargs(args)
        if self._cosmology == "oLCDM":
            # assert we are not in a crazy cosmological situation that prevents computing the angular distance integral
            h0, ok, om = kwargs_cosmo['h0'], kwargs_cosmo['ok'], kwargs_cosmo['om']
            for lens in self._kwargs_lens_list:
                if 'z_source' in lens:
                    z = lens['z_source']
                elif 'z_source_2' in lens:
                    z = lens['z_source_2']
                else:
                    z = 1100
                cut = ok * (1.0 + z) ** 2 + om * (1.0 + z) ** 3 + (1.0 - om - ok)
                if cut <= 0:
                    return -np.inf
            # make sure that Omega_DE is not negative...
            if 1.0 - om - ok <= 0:
                return -np.inf
        if kwargs_cosmo_interp is not None:
            kwargs_cosmo = {**kwargs_cosmo, **kwargs_cosmo_interp}
        cosmo = self.cosmo_instance(kwargs_cosmo)
        log_l = self._likelihoodLensSample.log_likelihood(cosmo=cosmo, kwargs_lens=kwargs_lens, kwargs_kin=kwargs_kin,
                                                          kwargs_source=kwargs_source)

        if self._sne_evaluate is True:
            apparent_m_z = kwargs_source.get('mu_sne', None)
            z_apparent_m_anchor = kwargs_source['z_apparent_m_anchor']
            sigma_m_z = kwargs_source.get('sigma_sne', None)
            log_l += self._sne_likelihood.log_likelihood(cosmo=cosmo, apparent_m_z=apparent_m_z,
                                                         z_anchor=z_apparent_m_anchor, sigma_m_z=sigma_m_z)
        if self._kde_evaluate is True:
            # all chain_params must be in the kwargs_cosmo
            cosmo_params = np.array([[kwargs_cosmo[k] for k in self._chain_params]])
            cosmo_params = rescale_vector_to_unity(cosmo_params, self._kde_likelihood.chain.rescale_dic,
                                                   self._chain_params)
            log_l += self._kde_likelihood.kdelikelihood_samples(cosmo_params)[0]
        if self._prior_add is True:
            log_l += self._custom_prior(kwargs_cosmo, kwargs_lens, kwargs_kin, kwargs_source)
        return log_l

    def cosmo_instance(self, kwargs_cosmo):
        """

        :param kwargs_cosmo: cosmology parameter keyword argument list
        :return: astropy.cosmology (or equivalent interpolation scheme class)
        """
        if hasattr(kwargs_cosmo, 'ang_diameter_distances') and hasattr(kwargs_cosmo, 'redshifts'):
            # in that case we use directly the interpolation mode to approximate angular diameter distances
            cosmo = CosmoInterp(ang_dist_list=kwargs_cosmo['ang_diameter_distances'],
                                z_list=kwargs_cosmo['redshifts'],
                                Ok0=kwargs_cosmo.get('ok', 0),
                                K=kwargs_cosmo.get('K', None))
            return cosmo
        if self._cosmo_fixed is None:
            cosmo = self.param.cosmo(kwargs_cosmo)
            if self._interpolate_cosmo is True:
                cosmo = CosmoInterp(cosmo=cosmo, z_stop=self._z_max, num_interp=self._num_redshift_interp)
        else:
            if self._interpolate_cosmo is True:
                if not hasattr(self, '_cosmo_fixed_interp'):
                    self._cosmo_fixed_interp = CosmoInterp(cosmo=self._cosmo_fixed, z_stop=self._z_max,
                                                           num_interp=self._num_redshift_interp)
                cosmo = self._cosmo_fixed_interp
            else:
                cosmo = self._cosmo_fixed
        return cosmo
