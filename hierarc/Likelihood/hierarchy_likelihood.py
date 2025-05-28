from hierarc.Likelihood.transformed_cosmography import TransformedCosmography
from hierarc.Likelihood.LensLikelihood.base_lens_likelihood import LensLikelihoodBase
from hierarc.Likelihood.kin_scaling import KinScaling
from hierarc.Likelihood.prior_likelihood import PriorLikelihood
from hierarc.Sampling.Distributions.los_distributions import LOSDistribution
from hierarc.Sampling.Distributions.anisotropy_distributions import (
    AnisotropyDistribution,
)
from hierarc.Util.distribution_util import PDFSampling, DistributionSampling
from hierarc.Sampling.Distributions.lens_distribution import LensDistribution
import numpy as np
import copy


class LensLikelihood(TransformedCosmography, LensLikelihoodBase, KinScaling):
    """Master class containing the likelihood definitions of different analysis for a
    single lens."""

    def __init__(
        self,
        # properties of the lens
        z_lens,
        z_source,
        name="name",
        likelihood_type="TDKin",
        lambda_scaling_property=0,
        lambda_scaling_property_beta=0,
        kwargs_lens_properties=None,
        # specific distribution settings for individual lenses
        global_los_distribution=False,
        mst_ifu=False,
        # global distributions
        anisotropy_model="NONE",
        anisotropy_sampling=False,
        anisotropy_distribution="NONE",  # make sure input is provided
        anisotropy_parameterization="beta",
        los_distributions=None,
        lambda_mst_distribution="NONE",
        gamma_in_sampling=False,
        gamma_in_distribution="NONE",
        log_m2l_sampling=False,
        log_m2l_distribution="NONE",
        alpha_lambda_sampling=False,
        beta_lambda_sampling=False,
        alpha_gamma_in_sampling=False,
        alpha_log_m2l_sampling=False,
        log_scatter=False,
        gamma_pl_index=None,
        gamma_pl_global_sampling=False,
        gamma_pl_global_dist="NONE",
        # kinematic model quantities
        kin_scaling_param_list=None,
        j_kin_scaling_param_axes=None,
        j_kin_scaling_grid_list=None,
        bin_edges_vel_disp_scaling=None,
        pdf_array_vel_disp_scaling=None,
        vel_disp_scaling_distributions=None,
        # likelihood evaluation quantities
        num_distribution_draws=50,
        normalized=True,
        # kappa quantities
        los_distribution_individual=None,
        kwargs_los_individual=None,
        # gamma_pl scaling
        gamma_pl_pivot=None,
        gamma_pl_ddt_slope=None,
        # priors
        prior_list=None,
        # specifics for each lens
        **kwargs_likelihood
    ):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param name: string (optional) to name the specific lens
        :param likelihood_type: string to specify the likelihood type
        :param j_kin_scaling_param_axes: array of parameter values for each axes of j_kin_scaling_grid
        :param j_kin_scaling_grid_list: list of array with the scalings of J() for each IFU
        :param j_kin_scaling_param_name_list: list of strings for the parameters as they are interpolated in the same
         order as j_kin_scaling_grid
        :param bin_edges_vel_disp_scaling: bin edges of histogram with PDF of scaling of sigma_axis/sigma_spherical
         in velocity dispersion prediction
        :param pdf_array_vel_disp_scaling: histogram for the bin edges to calculate the PDF of the
         sigma_axis/sigma_spherical velocity dispersion prediction
        :param vel_disp_scaling_distributions: list of samples that describes sigma_axis/sigma_spherical velocity
         dispersion prediction. Distribution can be single values or an array of length of the velocity dispersion
         measurements.
        :param num_distribution_draws: int, number of distribution draws from the likelihood that are being averaged
         over
        :param global_los_distribution: if integer, will draw from the global kappa distribution specified in that
         integer. If False, will instead draw from the distribution specified in kappa_pdf.
        :type global_los_distribution: bool or integer
        :param los_distribution_individual: name of the individual distribution ["GEV" and "PDF"]
        :type los_distribution_individual: str or None
        :param kwargs_los_individual: dictionary of the parameters of the individual distribution
         If individual_distribution is "PDF":
         "pdf_array": array of probability density function of the external convergence distribution
         binned according to kappa_bin_edges
         "bin_edges": array of length (len(kappa_pdf)+1), bin edges of the kappa PDF
         If individual_distribution is "GEV":
         "xi", "mean", "sigma"
        :type kwargs_los_individual: dict or None
        :param mst_ifu: bool, if True replaces the lambda_mst parameter by the lambda_ifu parameter (and distribution)
         in sampling this lens.
        :param lambda_scaling_property: float (optional), scaling of
         lambda_mst = lambda_mst_global + alpha * lambda_scaling_property
        :param lambda_scaling_property_beta: float (optional), scaling of
         lambda_mst = lambda_mst_global + beta * lambda_scaling_property_beta
        :param gamma_pl_index: index of gamma_pl parameter associated with this lens
        :type gamma_pl_index: int or None
        :param gamma_pl_global_sampling: if sampling a global power-law density slope distribution
        :type gamma_pl_global_sampling: bool
        :param gamma_pl_global_dist: distribution of global gamma_pl distribution ("GAUSSIAN" or "NONE")
        :param normalized: bool, if True, returns the normalized likelihood, if False, separates the constant prefactor
         (in case of a Gaussian 1/(sigma sqrt(2 pi)) ) to compute the reduced chi2 statistics
        :param kwargs_lens_properties: keyword arguments of the lens properties
        :param prior_list: list of [[name, mean, sigma], [],...] for priors on parameters being sampled for
         individual lenses
        :param kwargs_likelihood: keyword arguments specifying the likelihood function,
         see individual classes for their use
        :param los_distributions: list of all line of sight distributions parameterized
        :type los_distributions: list of str or None
        :param anisotropy_sampling: bool, if True adds a global stellar anisotropy parameter that alters the single lens
         kinematic prediction
        :param anisotropy_parameterization: model of parameterization (currently for constant anisotropy),
         ["beta" or "TAN_RAD"] supported
        :param gamma_pl_pivot: pivote power-law slope relative to which the Ddt scaling operates
        :param gamma_pl_ddt_slope: slope of Ddt / gamma_pl such that Ddt(gamma_pl) = Ddt_0 + slope * gamma_pl
        """
        TransformedCosmography.__init__(
            self, gamma_pl_pivot=gamma_pl_pivot, gamma_pl_ddt_slope=gamma_pl_ddt_slope
        )

        KinScaling.__init__(
            self,
            j_kin_scaling_param_axes=j_kin_scaling_param_axes,
            j_kin_scaling_grid_list=j_kin_scaling_grid_list,
            j_kin_scaling_param_name_list=kin_scaling_param_list,
        )

        LensLikelihoodBase.__init__(
            self,
            z_lens=z_lens,
            z_source=z_source,
            likelihood_type=likelihood_type,
            name=name,
            normalized=normalized,
            kwargs_lens_properties=kwargs_lens_properties,
            **kwargs_likelihood
        )
        self._num_distribution_draws = int(num_distribution_draws)

        self._los = LOSDistribution(
            global_los_distribution=global_los_distribution,
            los_distributions=los_distributions,
            individual_distribution=los_distribution_individual,
            kwargs_individual=kwargs_los_individual,
        )
        kwargs_min, kwargs_max = self.param_bounds_interpol()
        self._lens_distribution = LensDistribution(
            lambda_mst_distribution=lambda_mst_distribution,
            gamma_in_sampling=gamma_in_sampling,
            gamma_in_distribution=gamma_in_distribution,
            log_m2l_sampling=log_m2l_sampling,
            log_m2l_distribution=log_m2l_distribution,
            alpha_lambda_sampling=alpha_lambda_sampling,
            beta_lambda_sampling=beta_lambda_sampling,
            alpha_gamma_in_sampling=alpha_gamma_in_sampling,
            alpha_log_m2l_sampling=alpha_log_m2l_sampling,
            log_scatter=log_scatter,
            mst_ifu=mst_ifu,
            lambda_scaling_property=lambda_scaling_property,
            lambda_scaling_property_beta=lambda_scaling_property_beta,
            kwargs_min=kwargs_min,
            kwargs_max=kwargs_max,
            gamma_pl_index=gamma_pl_index,
            gamma_pl_global_sampling=gamma_pl_global_sampling,
            gamma_pl_global_dist=gamma_pl_global_dist,
        )
        self._aniso_distribution = AnisotropyDistribution(
            anisotropy_model=anisotropy_model,
            anisotropy_sampling=anisotropy_sampling,
            distribution_function=anisotropy_distribution,
            kwargs_anisotropy_min=kwargs_min,
            kwargs_anisotropy_max=kwargs_max,
            parameterization=anisotropy_parameterization,
        )
        self._prior = PriorLikelihood(prior_list=prior_list)
        if vel_disp_scaling_distributions is not None:
            self._inclination_sampling_class = DistributionSampling(
                distributions=vel_disp_scaling_distributions
            )
            self._inclination_sampling = True
        elif (
            bin_edges_vel_disp_scaling is not None
            and pdf_array_vel_disp_scaling is not None
        ):
            self._inclination_sampling = True
            self._inclination_sampling_class = PDFSampling(
                bin_edges=bin_edges_vel_disp_scaling,
                pdf_array=pdf_array_vel_disp_scaling,
            )
        else:
            self._inclination_sampling = False

    def info(self):
        """Information about the lens.

        :return: print statement
        """
        gamma_pl_index = self._lens_distribution.gamma_pl_index
        print("Name: ", self.name)
        print("likelihood type: ", self.likelihood_type)
        if gamma_pl_index is not None:
            print("gamma_pl_index", gamma_pl_index)
        print("========")

    def lens_log_likelihood(
        self,
        cosmo,
        kwargs_lens=None,
        kwargs_kin=None,
        kwargs_source=None,
        kwargs_los=None,
        verbose=False,
    ):
        """Log likelihood of the data of a lens given a model (defined with hyper-
        parameters) and cosmology.

        :param cosmo: astropy.cosmology instance
        :param kwargs_lens: keywords of the hyperparameters of the lens model
        :param kwargs_kin: keyword arguments of the kinematic model hyperparameters
        :param kwargs_source: keyword argument of the source model (such as SNe)
        :param kwargs_los: list of keyword arguments of global line of sight
            distributions
        :param verbose: If true, prints intermediate outputs of likelihood calculation
        :type verbose: bool
        :return: log likelihood of the data given the model
        """

        # here we compute the unperturbed angular diameter distances of the lens system given the cosmology
        # Note: Distances are in physical units of Mpc. Make sure the posteriors to evaluate this likelihood is in the
        # same units
        ddt, dd = self.angular_diameter_distances(cosmo)
        beta_dsp = self.beta_dsp(cosmo)
        kwargs_source = self._kwargs_init(kwargs_source)
        z_apparent_m_anchor = kwargs_source.get("z_apparent_m_anchor", 0.1)
        delta_lum_dist = self.luminosity_distance_modulus(cosmo, z_apparent_m_anchor)
        # here we effectively change the posteriors of the lens, but rather than changing the instance of the KDE we
        # displace the predicted angular diameter distances in the opposite direction
        a = self.hyper_param_likelihood(
            ddt,
            dd,
            delta_lum_dist,
            beta_dsp=beta_dsp,
            kwargs_lens=kwargs_lens,
            kwargs_kin=kwargs_kin,
            kwargs_source=kwargs_source,
            kwargs_los=kwargs_los,
            cosmo=cosmo,
        )
        if verbose:
            print("log likelihood of lens %s = %s" % (self.name, a))
        return np.nan_to_num(a)

    def hyper_param_likelihood(
        self,
        ddt,
        dd,
        delta_lum_dist,
        beta_dsp=None,
        kwargs_lens=None,
        kwargs_kin=None,
        kwargs_source=None,
        kwargs_los=None,
        cosmo=None,
    ):
        """Log likelihood of the data of a lens given a model (defined with
        hyperparameters) and cosmological distances.

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :param delta_lum_dist: relative luminosity distance to pivot redshift
        :param beta_dsp: Model prediction of ratio of Einstein radii theta_E_1 /
            theta_E_2
        :param kwargs_lens: keywords of the hyperparameters of the lens model
        :param kwargs_kin: keyword arguments of the kinematic model hyperparameters
        :param kwargs_source: keyword argument of the source model (such as SNe)
        :param kwargs_los: list of keyword arguments of global line of sight
            distributions
        :param cosmo: ~astropy.cosmology instance
        :return: log likelihood given the single lens analysis for the given
            hyperparameter
        """
        kwargs_lens = self._kwargs_init(kwargs_lens)
        kwargs_kin = self._kwargs_init(kwargs_kin)
        kwargs_source = self._kwargs_init(kwargs_source)
        kwargs_kin_copy = copy.deepcopy(kwargs_kin)
        sigma_v_sys_error = kwargs_kin_copy.pop("sigma_v_sys_error", None)

        if self.check_dist(
            kwargs_lens, kwargs_kin, kwargs_source, kwargs_los
        ):  # sharp distributions
            return self.log_likelihood_single(
                ddt,
                dd,
                delta_lum_dist,
                beta_dsp=beta_dsp,
                kwargs_lens=kwargs_lens,
                kwargs_kin=kwargs_kin_copy,
                kwargs_source=kwargs_source,
                kwargs_los=kwargs_los,
                sigma_v_sys_error=sigma_v_sys_error,
            )
        else:
            likelihood = 0
            for i in range(self._num_distribution_draws):
                logl = self.log_likelihood_single(
                    ddt,
                    dd,
                    delta_lum_dist,
                    beta_dsp=beta_dsp,
                    kwargs_lens=kwargs_lens,
                    kwargs_kin=kwargs_kin_copy,
                    kwargs_source=kwargs_source,
                    kwargs_los=kwargs_los,
                    sigma_v_sys_error=sigma_v_sys_error,
                )
                exp_logl = np.exp(logl)
                if np.isfinite(exp_logl) and exp_logl > 0:
                    likelihood += exp_logl
            if likelihood <= 0:
                return -np.inf
            return np.log(likelihood / self._num_distribution_draws)

    def log_likelihood_single(
        self,
        ddt,
        dd,
        delta_lum_dist,
        beta_dsp,
        kwargs_lens,
        kwargs_kin,
        kwargs_source,
        kwargs_los=None,
        sigma_v_sys_error=None,
    ):
        """Log likelihood of the data of a lens given a specific model (as a draw from
        hyperparameters) and cosmological distances.

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :param delta_lum_dist: relative luminosity distance to pivot redshift
        :param beta_dsp: Model prediction of ratio of Einstein radii theta_E_1 /
            theta_E_2
        :param kwargs_lens: keywords of the hyperparameters of the lens model
        :param kwargs_kin: keyword arguments of the kinematic model hyperparameters
        :param kwargs_source: keyword arguments of source brightness
        :param kwargs_los: line of sight list of dictionaries
        :param sigma_v_sys_error: unaccounted uncertainty in the velocity dispersion
            measurement
        :return: log likelihood given the single lens analysis for a single (random)
            realization of the hyperparameter distribution
        """
        kwargs_lens_draw = self._lens_distribution.draw_lens(**kwargs_lens)
        lambda_mst, gamma_ppn = (
            kwargs_lens_draw["lambda_mst"],
            kwargs_lens_draw["gamma_ppn"],
        )
        gamma_pl = kwargs_lens_draw.get("gamma_pl", 2)
        kappa_ext = self._los.draw_los(kwargs_los, size=1)[0]

        # draw intrinsic source magnitude
        mag_source = self.draw_source(lum_dist=delta_lum_dist, **kwargs_source)

        ddt_, dd_, mag_source_ = self.displace_prediction(
            ddt,
            dd,
            gamma_ppn=gamma_ppn,
            lambda_mst=lambda_mst,
            kappa_ext=kappa_ext,
            mag_source=mag_source,
            gamma_pl=gamma_pl,
        )

        kwargs_kin_draw = self._aniso_distribution.draw_anisotropy(**kwargs_kin)

        kwargs_param = {**kwargs_lens_draw, **kwargs_kin_draw}
        kin_scaling = self.kin_scaling(kwargs_param)
        if self._inclination_sampling is True:
            inclination_scaling = self._inclination_sampling_class.draw_one
            kin_scaling *= inclination_scaling**2
        lnlikelihood = self.log_likelihood(
            ddt_,
            dd_,
            beta_dsp=beta_dsp,
            kin_scaling=kin_scaling,
            sigma_v_sys_error=sigma_v_sys_error,
            mu_intrinsic=mag_source_,
            gamma_pl=gamma_pl,
            lambda_mst=lambda_mst,
        )
        lnlikelihood += self._prior.log_likelihood(kwargs_param)
        return lnlikelihood

    def angular_diameter_distances(self, cosmo):
        """Time-delay distance Ddt, angular diameter distance to the lens (dd)

        :param cosmo: astropy.cosmology instance (or equivalent with interpolation)
        :return: ddt, dd in units physical Mpc
        """
        if self.likelihood_type in ["DSPL"]:
            return 0, 0  # just returns some random numbers as not being used
        dd = cosmo.angular_diameter_distance(z=self.z_lens).value
        ds = cosmo.angular_diameter_distance(z=self.z_source).value
        dds = cosmo.angular_diameter_distance_z1z2(
            z1=self.z_lens, z2=self.z_source
        ).value
        ddt = (1.0 + self.z_lens) * dd * ds / dds
        return np.maximum(np.nan_to_num(ddt), 0.00001), np.maximum(
            np.nan_to_num(dd), 0.00001
        )

    def luminosity_distance_modulus(self, cosmo, z_apparent_m_anchor):
        """The difference in luminosity distance between a pivot redshift
        (z_apparent_m_anchor) and the source redshift (effectively the ratio as this is
        the magnitude transform)

        :param cosmo: astropy.cosmology instance (or equivalent with interpolation)
        :param z_apparent_m_anchor: redshift of pivot/anchor at which the apparent SNe brightness is defined relative to
        :return: lum_dist(z_source) - lum_dist(z_pivot)
        """
        if self.likelihood_type not in ["Mag", "TDMag", "TDMagMagnitude"]:
            return 0
        angular_diameter_distances = np.maximum(
            np.nan_to_num(cosmo.angular_diameter_distance(self.z_source).value),
            0.00001,
        )
        lum_dists = 5 * np.log10(
            (1 + self.z_source) * (1 + self.z_source) * angular_diameter_distances
        )

        z_anchor = z_apparent_m_anchor
        ang_dist_anchor = np.maximum(
            np.nan_to_num(cosmo.angular_diameter_distance(z_anchor).value), 0.00001
        )
        lum_dist_anchor = 5 * np.log10(
            (1 + z_anchor) * (1 + z_anchor) * ang_dist_anchor
        )
        delta_lum_dist = lum_dists - lum_dist_anchor
        return delta_lum_dist

    def check_dist(self, kwargs_lens, kwargs_kin, kwargs_source, kwargs_los):
        """Checks if the provided keyword arguments describe a distribution function of
        hyperparameters or are single values.

        :param kwargs_lens: lens model hyperparameter keywords
        :param kwargs_kin: kinematic model hyperparameter keywords
        :param kwargs_source: source brightness hyperparameter keywords
        :param kwargs_los: list of dictionaries for line of sight hyperparameters
        :return: bool, True if delta function, else False
        """
        lambda_mst_sigma = kwargs_lens.get("lambda_mst_sigma", 0)  # scatter in MST
        draw_kappa_bool = self._los.draw_bool(kwargs_los)
        a_ani_sigma = kwargs_kin.get("a_ani_sigma", 0)
        beta_inf_sigma = kwargs_kin.get("beta_inf_sigma", 0)
        sne_sigma = kwargs_source.get("sigma_sne", 0)
        gamma_pl_sigma = kwargs_lens.get("gamma_pl_sigma", 0)
        if (
            a_ani_sigma == 0
            and lambda_mst_sigma == 0
            and beta_inf_sigma == 0
            and sne_sigma == 0
            and gamma_pl_sigma == 0
            and not draw_kappa_bool
            and not self._inclination_sampling
        ):
            return True
        return False

    @staticmethod
    def draw_source(mu_sne=1, sigma_sne=0, lum_dist=0, **kwargs):
        """Draws a source magnitude from a distribution specified by population
        parameters.

        :param mu_sne: mean magnitude of SNe
        :param sigma_sne: std of magnitude distribution of SNe relative to the mean
            magnitude
        :param lum_dist: luminosity distance (astronomical magnitude scaling of defined
            brightness to the source redshift)
        :return: realization of source amplitude given distribution
        """
        # draw apparent magnitude at pivot luminosity distance (z=0.1)
        mag_draw = np.random.normal(loc=mu_sne, scale=sigma_sne)
        # move apparent magnitude to redshift of source with relative luminosity distance
        mag_source = mag_draw + lum_dist
        # return linear amplitude with base log 10
        return mag_source

    def sigma_v_measured_vs_predict(
        self, cosmo, kwargs_lens=None, kwargs_kin=None, kwargs_los=None
    ):
        """Mean and error covariance of velocity dispersion measurement mean and error
        covariance of velocity dispersion predictions.

        :param cosmo: astropy.cosmology instance
        :param kwargs_lens: keywords of the hyperparameters of the lens model
        :param kwargs_kin: keyword arguments of the kinematic model hyperparameters
        :param kwargs_los: line of sight parapers
        :return: sigma_v_measurement, cov_error_measurement, sigma_v_predict_mean,
            cov_error_predict
        """
        # if no kinematics is provided, return None's
        if self.likelihood_type not in ["DdtHistKin", "IFUKinCov", "DdtGaussKin"]:
            return None, None, None, None
        if kwargs_lens is None:
            kwargs_lens = {}
        if kwargs_kin is None:
            kwargs_kin = {}
        kwargs_kin_copy = copy.deepcopy(kwargs_kin)
        sigma_v_sys_error = kwargs_kin_copy.pop("sigma_v_sys_error", None)
        ddt, dd = self.angular_diameter_distances(cosmo)
        sigma_v_measurement, cov_error_measurement = self.sigma_v_measurement(
            sigma_v_sys_error=sigma_v_sys_error
        )
        sigma_v_predict_list = []
        sigma_v_predict_mean = np.zeros_like(sigma_v_measurement)
        cov_error_predict = np.zeros_like(cov_error_measurement)
        for i in range(self._num_distribution_draws):
            kwargs_lens_draw = self._lens_distribution.draw_lens(**kwargs_lens)
            lambda_mst, gamma_ppn, gamma_pl = (
                kwargs_lens_draw["lambda_mst"],
                kwargs_lens_draw["gamma_ppn"],
                kwargs_lens_draw.get("gamma_pl", None),
            )
            kappa_ext = self._los.draw_los(kwargs_los, size=1)[0]
            ddt_, dd_, _ = self.displace_prediction(
                ddt,
                dd,
                gamma_ppn=gamma_ppn,
                lambda_mst=lambda_mst,
                kappa_ext=kappa_ext,
                gamma_pl=gamma_pl,
            )
            kwargs_kin_draw = self._aniso_distribution.draw_anisotropy(
                **kwargs_kin_copy
            )
            kwargs_param = {**kwargs_lens_draw, **kwargs_kin_draw}
            kin_scaling = self.kin_scaling(kwargs_param)
            if self._inclination_sampling is True:
                inclination_scaling = self._inclination_sampling_class.draw_one
                kin_scaling *= inclination_scaling**2
            sigma_v_predict_i, cov_error_predict_i = self.sigma_v_prediction(
                ddt_, dd_, kin_scaling=kin_scaling
            )
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
        return (
            sigma_v_measurement,
            cov_error_measurement,
            sigma_v_predict_mean,
            cov_error_predict,
        )

    def ddt_dd_model_prediction(self, cosmo, kwargs_lens=None, kwargs_los=None):
        """Predicts the model uncertainty corrected ddt prediction of the applied model
        (e.g. power-law)

        :param cosmo: astropy.cosmology instance
        :param kwargs_lens: keywords of the hyper parameters of the lens model
        :param kwargs_los: line of slight list of dictionaries
        :return: ddt_model mean, ddt_model sigma, dd_model mean, dd_model sigma
        """
        if kwargs_lens is None:
            kwargs_lens = {}
        ddt, dd = self.angular_diameter_distances(cosmo)
        ddt_draws = []
        dd_draws = []
        for i in range(self._num_distribution_draws):
            kwargs_lens_draw = self._lens_distribution.draw_lens(**kwargs_lens)
            lambda_mst, gamma_ppn, gamma_pl = (
                kwargs_lens_draw["lambda_mst"],
                kwargs_lens_draw["gamma_ppn"],
                kwargs_lens_draw.get("gamma_pl", None),
            )
            kappa_ext = self._los.draw_los(kwargs_los)
            ddt_, dd_, _ = self.displace_prediction(
                ddt,
                dd,
                gamma_ppn=gamma_ppn,
                lambda_mst=lambda_mst,
                kappa_ext=kappa_ext,
                gamma_pl=gamma_pl,
            )
            ddt_draws.append(ddt_)
            dd_draws.append(dd_)
        return (
            np.mean(ddt_draws),
            np.std(ddt_draws),
            np.mean(dd_draws),
            np.std(dd_draws),
        )

    @staticmethod
    def _kwargs_init(kwargs=None):
        """

        :param kwargs: keyword argument or None
        :return: keyword argument
        """
        if kwargs is None:
            kwargs = {}
        return kwargs
