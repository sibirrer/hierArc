from hierarc.Likelihood.transformed_cosmography import TransformedCosmography
from hierarc.Likelihood.LensLikelihood.base_lens_likelihood import LensLikelihoodBase
from hierarc.Likelihood.anisotropy_scaling import AnisotropyScalingIFU
from hierarc.Util.distribution_util import PDFSampling
import numpy as np
import copy


class LensLikelihood(TransformedCosmography, LensLikelihoodBase, AnisotropyScalingIFU):
    """
    master class containing the likelihood definitions of different analysis for s single lens
    """
    def __init__(self, z_lens, z_source, name='name', likelihood_type='TDKin', anisotropy_model='NONE',
                 ani_param_array=None, ani_scaling_array_list=None, ani_scaling_array=None,
                 num_distribution_draws=50, kappa_ext_bias=False, kappa_pdf=None, kappa_bin_edges=None, mst_ifu=False,
                 lambda_scaling_property=0, lambda_scaling_property_beta=0,
                 normalized=True, kwargs_lens_properties=None, **kwargs_likelihood):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param name: string (optional) to name the specific lens
        :param likelihood_type: string to specify the likelihood type
        :param ani_param_array: array of anisotropy parameter values for which the kinematics are predicted
        :param ani_scaling_array: velocity dispersion sigma**2 scaling (also J scaling) of anisotropy parameter relative
         to default prediction. The scaling corresponds to the ani_param_array parameter spacing
         (to generate an interpolation function). A value =1 in ani_scaling_array results in the value stored in the
         provided J() predictions.
        :param ani_scaling_array_list: list of array with the scalings of J() for each IFU
        :param num_distribution_draws: int, number of distribution draws from the likelihood that are being averaged
         over
        :param kappa_ext_bias: bool, if True incorporates the global external selection function into the likelihood.
        If False, the likelihood needs to incorporate the individual selection function with sufficient accuracy.
        :param kappa_pdf: array of probability density function of the external convergence distribution
         binned according to kappa_bin_edges
        :param kappa_bin_edges: array of length (len(kappa_pdf)+1), bin edges of the kappa PDF
        :param mst_ifu: bool, if True replaces the lambda_mst parameter by the lambda_ifu parameter (and distribution)
         in sampling this lens.
        :param lambda_scaling_property: float (optional), scaling of
         lambda_mst = lambda_mst_global + alpha * lambda_scaling_property
        :param lambda_scaling_property_beta: float (optional), scaling of
         lambda_mst = lambda_mst_global + beta * lambda_scaling_property_beta
        :param normalized: bool, if True, returns the normalized likelihood, if False, separates the constant prefactor
         (in case of a Gaussian 1/(sigma sqrt(2 pi)) ) to compute the reduced chi2 statistics
        :param kwargs_lens_properties: keyword arguments of the lens properties
        :param kwargs_likelihood: keyword arguments specifying the likelihood function,
        see individual classes for their use
        """
        TransformedCosmography.__init__(self, z_lens=z_lens, z_source=z_source)
        if ani_scaling_array_list is None and ani_scaling_array is not None:
            ani_scaling_array_list = [ani_scaling_array]
        AnisotropyScalingIFU.__init__(self, anisotropy_model=anisotropy_model, ani_param_array=ani_param_array,
                                      ani_scaling_array_list=ani_scaling_array_list)
        LensLikelihoodBase.__init__(self, z_lens=z_lens, z_source=z_source, likelihood_type=likelihood_type, name=name,
                                    normalized=normalized, kwargs_lens_properties=kwargs_lens_properties,
                                    **kwargs_likelihood)
        self._num_distribution_draws = int(num_distribution_draws)
        self._kappa_ext_bias = kappa_ext_bias
        self._mst_ifu = mst_ifu
        if kappa_pdf is not None and kappa_bin_edges is not None:
            self._kappa_dist = PDFSampling(bin_edges=kappa_bin_edges, pdf_array=kappa_pdf)
            self._draw_kappa = True
        else:
            self._draw_kappa = False
        self._lambda_scaling_property = lambda_scaling_property
        self._lambda_scaling_property_beta = lambda_scaling_property_beta

    def lens_log_likelihood(self, cosmo, kwargs_lens=None, kwargs_kin=None, kwargs_source=None):
        """
        log likelihood of the data of a lens given a model (defined with hyper-parameters) and cosmology

        :param cosmo: astropy.cosmology instance
        :param kwargs_lens: keywords of the hyper parameters of the lens model
        :param kwargs_kin: keyword arguments of the kinematic model hyper parameters
        :param kwargs_source: keyword argument of the source model (such as SNe)
        :return: log likelihood of the data given the model
        """

        # here we compute the unperturbed angular diameter distances of the lens system given the cosmology
        # Note: Distances are in physical units of Mpc. Make sure the posteriors to evaluate this likelihood is in the
        # same units
        ddt, dd = self.angular_diameter_distances(cosmo)
        kwargs_source = self._kwargs_init(kwargs_source)
        z_apparent_m_anchor = kwargs_source.get('z_apparent_m_anchor', 0.1)
        delta_lum_dist = self.luminosity_distance_modulus(cosmo, z_apparent_m_anchor)
        # here we effectively change the posteriors of the lens, but rather than changing the instance of the KDE we
        # displace the predicted angular diameter distances in the opposite direction
        return self.hyper_param_likelihood(ddt, dd, delta_lum_dist, kwargs_lens=kwargs_lens, kwargs_kin=kwargs_kin,
                                           kwargs_source=kwargs_source, cosmo=cosmo)

    def hyper_param_likelihood(self, ddt, dd, delta_lum_dist, kwargs_lens=None, kwargs_kin=None, kwargs_source=None,
                               cosmo=None):
        """
        log likelihood of the data of a lens given a model (defined with hyper-parameters) and cosmological distances

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :param delta_lum_dist: relative luminosity distance to pivot redshift
        :param kwargs_lens: keywords of the hyper parameters of the lens model
        :param kwargs_kin: keyword arguments of the kinematic model hyper parameters
        :param kwargs_source: keyword argument of the source model (such as SNe)
        :param cosmo: astropy.cosmology instance
        :return: log likelihood given the single lens analysis for the given hyper parameter
        """
        kwargs_lens = self._kwargs_init(kwargs_lens)
        kwargs_kin = self._kwargs_init(kwargs_kin)
        kwargs_source = self._kwargs_init(kwargs_source)
        kwargs_kin_copy = copy.deepcopy(kwargs_kin)
        sigma_v_sys_error = kwargs_kin_copy.pop('sigma_v_sys_error', None)

        if self.check_dist(kwargs_lens, kwargs_kin, kwargs_source):  # sharp distributions
            return self.log_likelihood_single(ddt, dd, delta_lum_dist, kwargs_lens, kwargs_kin_copy, kwargs_source,
                                              sigma_v_sys_error=sigma_v_sys_error)
        else:
            likelihood = 0
            for i in range(self._num_distribution_draws):
                logl = self.log_likelihood_single(ddt, dd, delta_lum_dist, kwargs_lens, kwargs_kin_copy, kwargs_source,
                                                  sigma_v_sys_error=sigma_v_sys_error)
                exp_logl = np.exp(logl)
                if np.isfinite(exp_logl) and exp_logl > 0:
                    likelihood += exp_logl
            if likelihood <= 0:
                return -np.inf
            return np.log(likelihood/self._num_distribution_draws)

    def log_likelihood_single(self, ddt, dd, delta_lum_dist, kwargs_lens, kwargs_kin, kwargs_source,
                              sigma_v_sys_error=None):
        """
        log likelihood of the data of a lens given a specific model (as a draw from hyper-parameters) and cosmological
        distances

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :param delta_lum_dist: relative luminosity distance to pivot redshift
        :param kwargs_lens: keywords of the hyper parameters of the lens model
        :param kwargs_kin: keyword arguments of the kinematic model hyper parameters
        :param kwargs_source: keyword arguments of source brightness
        :param sigma_v_sys_error: unaccounted uncertainty in the velocity dispersion measurement
        :return: log likelihood given the single lens analysis for a single (random) realization of the hyper parameter
         distribution
        """
        lambda_mst, kappa_ext, gamma_ppn = self.draw_lens(**kwargs_lens)
        # draw intrinsic source magnitude
        mag_source = self.draw_source(lum_dist=delta_lum_dist, **kwargs_source)
        ddt_, dd_, mag_source_ = self.displace_prediction(ddt, dd, gamma_ppn=gamma_ppn, lambda_mst=lambda_mst,
                                                          kappa_ext=kappa_ext, mag_source=mag_source)
        aniso_param_array = self.draw_anisotropy(**kwargs_kin)
        aniso_scaling = self.ani_scaling(aniso_param_array)

        lnlikelihood = self.log_likelihood(ddt_, dd_, aniso_scaling=aniso_scaling, sigma_v_sys_error=sigma_v_sys_error,
                                           mu_intrinsic=mag_source_)
        return np.nan_to_num(lnlikelihood)

    def angular_diameter_distances(self, cosmo):
        """
        time-delay distance Ddt, angular diameter distance to the lens (dd)

        :param cosmo: astropy.cosmology instance (or equivalent with interpolation)
        :return: ddt, dd, ds in units physical Mpc
        """
        dd = cosmo.angular_diameter_distance(z=self._z_lens).value
        ds = cosmo.angular_diameter_distance(z=self._z_source).value
        dds = cosmo.angular_diameter_distance_z1z2(z1=self._z_lens, z2=self._z_source).value
        ddt = (1. + self._z_lens) * dd * ds / dds
        return np.maximum(np.nan_to_num(ddt), 0.00001), np.maximum(np.nan_to_num(dd), 0.00001)

    def luminosity_distance_modulus(self, cosmo, z_apparent_m_anchor):
        """
        the difference in luminosity distance between a pivot redshift (z_apparent_m_anchor) and the source redshift
        (effectively the ratio as this is the magnitude transform)

        :param cosmo: astropy.cosmology instance (or equivalent with interpolation)
        :param z_apparent_m_anchor: redshift of pivot/anchor at which the apparent SNe brightness is defined relative to
        :return: lum_dist(z_source) - lum_dist(z_pivot)
        """
        angular_diameter_distances = np.maximum(np.nan_to_num(cosmo.angular_diameter_distance(self._z_source).value),
                                                0.00001)
        lum_dists = (5 * np.log10((1 + self._z_source) * (1 + self._z_source) * angular_diameter_distances))

        z_anchor = z_apparent_m_anchor
        ang_dist_anchor = np.maximum(np.nan_to_num(cosmo.angular_diameter_distance(z_anchor).value), 0.00001)
        lum_dist_anchor = (5 * np.log10((1 + z_anchor) * (1 + z_anchor) * ang_dist_anchor))
        delta_lum_dist = lum_dists - lum_dist_anchor
        return delta_lum_dist

    def check_dist(self, kwargs_lens, kwargs_kin, kwargs_source):
        """
        checks if the provided keyword arguments describe a distribution function of hyper parameters or are single
        values

        :param kwargs_lens: lens model hyper-parameter keywords
        :param kwargs_kin: kinematic model hyper-parameter keywords
        :param kwargs_source: source brightness hyper-parameter keywords
        :return: bool, True if delta function, else False
        """
        lambda_mst_sigma = kwargs_lens.get('lambda_mst_sigma', 0)  # scatter in MST
        kappa_ext_sigma = kwargs_lens.get('kappa_ext_sigma', 0)
        a_ani_sigma = kwargs_kin.get('a_ani_sigma', 0)
        beta_inf_sigma = kwargs_kin.get('beta_inf_sigma', 0)
        sne_sigma = kwargs_source.get('sigma_sne', 0)
        if a_ani_sigma == 0 and lambda_mst_sigma == 0 and kappa_ext_sigma == 0 and beta_inf_sigma == 0 \
           and sne_sigma == 0:
            if self._draw_kappa is False:
                return True
        return False

    def draw_lens(self, lambda_mst=1, lambda_mst_sigma=0, kappa_ext=0, kappa_ext_sigma=0, gamma_ppn=1, lambda_ifu=1,
                  lambda_ifu_sigma=0, alpha_lambda=0, beta_lambda=0):
        """
        draws a realization of a specific model from the hyper-parameter distribution

        :param lambda_mst: MST transform
        :param lambda_mst_sigma: spread in the distribution
        :param kappa_ext: external convergence mean in distribution
        :param kappa_ext_sigma: spread in the distribution
        :param gamma_ppn: Post-Newtonian parameter
        :param lambda_ifu: secondary lambda_mst parameter for subset of lenses specified for
        :param lambda_ifu_sigma: secondary lambda_mst_sigma parameter for subset of lenses specified for
        :param alpha_lambda: float, linear slope of the lambda_int scaling relation with lens quantity
         self._lambda_scaling_property
        :param beta_lambda: float, a second linear slope of the lambda_int scaling relation with lens quantity
         self._lambda_scaling_property_beta
        :return: draw from the distributions
        """
        if self._mst_ifu is True:
            lambda_lens = lambda_ifu + alpha_lambda * self._lambda_scaling_property \
                          + beta_lambda * self._lambda_scaling_property_beta
            lambda_mst_draw = np.random.normal(lambda_lens, lambda_ifu_sigma)
        else:
            lambda_lens = lambda_mst + alpha_lambda * self._lambda_scaling_property \
                          + beta_lambda * self._lambda_scaling_property_beta
            lambda_mst_draw = np.random.normal(lambda_lens, lambda_mst_sigma)
        if self._draw_kappa is True:
            kappa_ext_draw = self._kappa_dist.draw_one
        elif self._kappa_ext_bias is True:
            kappa_ext_draw = np.random.normal(kappa_ext, kappa_ext_sigma)
        else:
            kappa_ext_draw = 0
        return lambda_mst_draw, kappa_ext_draw, gamma_ppn

    @staticmethod
    def draw_source(mu_sne=1, sigma_sne=0, lum_dist=0, **kwargs):
        """
        draws a source magnitude from a distribution specified by population parameters

        :param mu_sne: mean magnitude of SNe
        :param sigma_sne: std of magnitude distribution of SNe relative to the mean magnitude
        :param lum_dist: luminosity distance
         (astronomical magnitude scaling of defined brightness to the source redshift)
        :return: realization of source amplitude given distribution
        """
        # draw apparent magnitude at pivot luminosity distance (z=0.1)
        mag_draw = np.random.normal(loc=mu_sne, scale=sigma_sne)
        # move apparent magnitude to redshift of source with relative luminosity distance
        mag_source = mag_draw + lum_dist
        # return linear amplitude with base log 10
        return mag_source

    def sigma_v_measured_vs_predict(self, cosmo, kwargs_lens=None, kwargs_kin=None):
        """
        mean and error covariance of velocity dispersion measurement
        mean and error covariance of velocity dispersion predictions

        :param cosmo: astropy.cosmology instance
        :param kwargs_lens: keywords of the hyper parameters of the lens model
        :param kwargs_kin: keyword arguments of the kinematic model hyper parameters
        :return: sigma_v_measurement, cov_error_measurement, sigma_v_predict_mean, cov_error_predict
        """
        # if no kinematics is provided, return None's
        if self.likelihood_type not in ['DdtHistKin', 'IFUKinCov', 'DdtGaussKin']:
            return None, None, None, None
        if kwargs_lens is None:
            kwargs_lens = {}
        if kwargs_kin is None:
            kwargs_kin = {}
        kwargs_kin_copy = copy.deepcopy(kwargs_kin)
        sigma_v_sys_error = kwargs_kin_copy.pop('sigma_v_sys_error', None)
        ddt, dd = self.angular_diameter_distances(cosmo)
        sigma_v_measurement, cov_error_measurement = self.sigma_v_measurement(sigma_v_sys_error=sigma_v_sys_error)
        sigma_v_predict_list = []
        sigma_v_predict_mean = np.zeros_like(sigma_v_measurement)
        cov_error_predict = np.zeros_like(cov_error_measurement)
        for i in range(self._num_distribution_draws):
            lambda_mst, kappa_ext, gamma_ppn = self.draw_lens(**kwargs_lens)
            ddt_, dd_, _ = self.displace_prediction(ddt, dd, gamma_ppn=gamma_ppn, lambda_mst=lambda_mst,
                                                    kappa_ext=kappa_ext)
            aniso_param_array = self.draw_anisotropy(**kwargs_kin_copy)
            aniso_scaling = self.ani_scaling(aniso_param_array)
            sigma_v_predict_i, cov_error_predict_i = self.sigma_v_prediction(ddt_, dd_, aniso_scaling=aniso_scaling)
            sigma_v_predict_mean += sigma_v_predict_i
            cov_error_predict += cov_error_predict_i
            sigma_v_predict_list.append(sigma_v_predict_i)

        sigma_v_predict_mean /= self._num_distribution_draws
        cov_error_predict /= self._num_distribution_draws
        sigma_v_predict_list = np.array(sigma_v_predict_list)
        # TODO: check whether covariance matrix is calculated properly
        cov_error_predict += np.cov(sigma_v_predict_list.T)
        # sigma_v_mean_std = np.std(sigma_v_predict_list, axis=0)
        # cov_error_predict += np.outer(sigma_v_mean_std, sigma_v_mean_std)
        return sigma_v_measurement, cov_error_measurement, sigma_v_predict_mean, cov_error_predict

    def ddt_dd_model_prediction(self, cosmo, kwargs_lens=None):
        """
        predicts the model uncertainty corrected ddt prediction of the applied model (e.g. power-law)

        :param cosmo: astropy.cosmology instance
        :param kwargs_lens: keywords of the hyper parameters of the lens model
        :return: ddt_model mean, ddt_model sigma, dd_model mean, dd_model sigma
        """
        if kwargs_lens is None:
            kwargs_lens = {}
        ddt, dd = self.angular_diameter_distances(cosmo)
        ddt_draws = []
        dd_draws = []
        for i in range(self._num_distribution_draws):
            lambda_mst, kappa_ext, gamma_ppn = self.draw_lens(**kwargs_lens)
            ddt_, dd_, _ = self.displace_prediction(ddt, dd, gamma_ppn=gamma_ppn, lambda_mst=lambda_mst,
                                                    kappa_ext=kappa_ext)
            ddt_draws.append(ddt_)
            dd_draws.append(dd_)
        return np.mean(ddt_draws), np.std(ddt_draws), np.mean(dd_draws), np.std(dd_draws)

    @staticmethod
    def _kwargs_init(kwargs=None):
        """

        :param kwargs: keyword argument or None
        :return: keyword argument
        """
        if kwargs is None:
            kwargs = {}
        return kwargs
