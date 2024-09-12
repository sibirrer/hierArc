__author__ = "ajshjib"

import copy

import numpy as np
from hierarc.LensPosterior.kin_constraints import KinConstraints
from lenstronomy.Util import constants as const
from lenstronomy.Analysis.light_profile import LightProfileAnalysis
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.Profiles.gnfw import GNFW


class KinConstraintsComposite(KinConstraints):
    def __init__(
        self,
        z_lens,
        z_source,
        gamma_in_array,
        log_m2l_array,
        alpha_Rs_array,
        r_s_angle_array,
        theta_E,
        theta_E_error,
        gamma,
        gamma_error,
        r_eff,
        r_eff_error,
        sigma_v_measured,
        kwargs_aperture,
        kwargs_seeing,
        kwargs_numerics_galkin,
        anisotropy_model,
        gamma_in_prior_mean=None,
        gamma_in_prior_std=None,
        sigma_v_error_independent=None,
        sigma_v_error_covariant=None,
        sigma_v_error_cov_matrix=None,
        kwargs_lens_light=None,
        lens_light_model_list=["HERNQUIST"],
        kwargs_mge_light=None,
        sampling_number=1000,
        num_psf_sampling=100,
        num_kin_sampling=1000,
        multi_observations=False,
        rho0_array=None,
        kappa_s_array=None,
        r_s_array=None,
        is_m2l_population_level=True,
    ):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param gamma_in_array: array of power-law slopes of the mass model
        :param log_m2l_array: array of log10(mass-to-light ratios) of the stellar component,
            needs to be in the unit/scaling such that m2l / sigma_crit * amp in the
            kwargs_lens_light provides the convergence amplitude of the stars
        :param alpha_Rs_array: array of the deflection (angular units) at projected Rs
        :param r_s_angle_array: array of halo scale radii in arcsecond
        :param theta_E: Einstein radius (in arc seconds)
        :param theta_E_error: 1-sigma error on Einstein radius
        :param gamma: power-law slope (2 = isothermal) estimated from imaging data
        :param gamma_error: 1-sigma uncertainty on power-law slope
        :param r_eff: half-light radius of the deflector (arc seconds)
        :param r_eff_error: uncertainty on half-light radius
        :param sigma_v_measured: numpy array of IFU velocity dispersion of the main
            deflector in km/s
        :param sigma_v_error_independent: numpy array of 1-sigma uncertainty in velocity
            dispersion of the IFU
         observation independent of each other
        :param sigma_v_error_covariant: covariant error in the measured kinematics
            shared among all IFU measurements
        :param sigma_v_error_cov_matrix: error covariance matrix in the sigma_v
            measurements (km/s)^2
        :type sigma_v_error_cov_matrix: nxn matrix with n the length of the
            sigma_v_measured array
        :param kwargs_aperture: spectroscopic aperture keyword arguments, see
            lenstronomy.Galkin.aperture for options
        :param kwargs_seeing: seeing condition of spectroscopic observation, corresponds
            to kwargs_psf in the GalKin module specified in lenstronomy.GalKin.psf
        :param kwargs_numerics_galkin: numerical settings for the integrated
            line-of-sight velocity dispersion
        :param anisotropy_model: type of stellar anisotropy model. See details in
            MamonLokasAnisotropy() class of lenstronomy.GalKin.anisotropy
        :param gamma_in_prior_mean: prior mean for inner power-law slope of the NFW profile, if available
        :param gamma_in_prior_std: standard deviation of the Gaussian prior for `gamma_in`
        :param kwargs_lens_light: keyword argument list of lens light model (optional)
        :param kwargs_mge_light: keyword arguments that go into the MGE decomposition
            routine
        :param multi_observations: bool, if True, interprets kwargs_aperture and
            kwargs_seeing as lists of multiple observations
        :param rho0_array: array of halo mass normalizations in M_sun / Mpc^3
        :param kappa_s_array: array of generalized NFW profile's convergence normalization at the scale radius
        :param r_s_array: array of halo scale radii in Mpc
        """

        if (
            len(lens_light_model_list) == 1
            and lens_light_model_list[0] == "MULTI_GAUSSIAN"
        ):
            kwargs_lens_light = kwargs_lens_light
        else:
            self._light_profile_analysis = LightProfileAnalysis(
                light_model=LightModel(light_model_list=lens_light_model_list)
            )

            (
                amps,
                sigmas,
                center_x,
                center_y,
            ) = self._light_profile_analysis.multi_gaussian_decomposition(
                kwargs_lens_light, r_h=r_eff, **kwargs_mge_light
            )

            lens_light_model_list = ["MULTI_GAUSSIAN"]
            kwargs_lens_light = [{"amp": amps, "sigma": sigmas}]

        lens_model_list = ["GNFW", "MULTI_GAUSSIAN"]

        # log_m2l is interpolated when sampled on the population level, otherwise marginalized
        if is_m2l_population_level:
            log_m2l_scaling = log_m2l_array
        else:
            log_m2l_scaling = None

        super(KinConstraintsComposite, self).__init__(
            z_lens,
            z_source,
            theta_E,
            theta_E_error,
            gamma,
            gamma_error,
            r_eff,
            r_eff_error,
            sigma_v_measured,
            kwargs_aperture,
            kwargs_seeing,
            kwargs_numerics_galkin,
            anisotropy_model,
            sigma_v_error_independent=sigma_v_error_independent,
            sigma_v_error_covariant=sigma_v_error_covariant,
            sigma_v_error_cov_matrix=sigma_v_error_cov_matrix,
            kwargs_lens_light=kwargs_lens_light,
            lens_light_model_list=lens_light_model_list,
            lens_model_list=lens_model_list,
            MGE_light=False,  # set False, as MGE is already done as default
            kwargs_mge_light=None,
            hernquist_approx=False,
            sampling_number=sampling_number,
            num_psf_sampling=num_psf_sampling,
            num_kin_sampling=num_kin_sampling,
            multi_observations=multi_observations,
            gamma_in_scaling=gamma_in_array,
            log_m2l_scaling=log_m2l_scaling,
        )

        if self._check_arrays(alpha_Rs_array, r_s_angle_array):
            self._alpha_Rs_or_kappa_s_array = alpha_Rs_array
            self._alpha_Rs_or_kappa_s_bool = True
            self._r_scale_angle_array = r_s_angle_array
        elif self._check_arrays(kappa_s_array, r_s_angle_array):
            self._alpha_Rs_or_kappa_s_array = kappa_s_array
            self._alpha_Rs_or_kappa_s_bool = False
            self._r_scale_angle_array = r_s_angle_array
        elif self._check_arrays(rho0_array, r_s_array):
            kappa_s_array, self._r_scale_angle_array = self.get_kappa_s_r_s_angle(
                rho0_array, r_s_array
            )
            self._alpha_Rs_or_kappa_s_array = kappa_s_array
            self._alpha_Rs_or_kappa_s_bool = False
        else:
            raise ValueError(
                "Both alpha_Rs_array and r_s_angle_array, kappa_s_array and r_s_angle_array, or rho0_array and r_s_array must be arrays of the same length!"
            )

        self.gamma_in_array = gamma_in_array
        self.log_m2l_array = log_m2l_array
        self._is_m2l_population_level = is_m2l_population_level

        self._gamma_in_prior_mean = gamma_in_prior_mean
        self._gamma_in_prior_std = gamma_in_prior_std

        if not is_m2l_population_level and not self._check_arrays(
            self._alpha_Rs_or_kappa_s_array, log_m2l_array
        ):
            raise ValueError(
                "log_m2l_array must have the same length as alpha_Rs_array, kappa_s_array, or rho0_array!"
            )

    @staticmethod
    def _check_arrays(array0, array1):
        """Checks if two arrays are valid sample distributions for pairs of parameters.

        :param array0, array1: arrays representing samples from respective distributions
        :return: bool, True if arrays are the same length and > 0; False otherwise
        """
        if array0 is None or array1 is None:
            return False
        return len(array0) == len(array1) and len(array0) > 0

    def get_kappa_s_r_s_angle(self, rho_s, r_scale):
        """Computes the surface mass density of the NFW halo at the scale radius.

        :param rho_s: halo mass normalization in M_sun / Mpc^3
        :param r_scale: halo scale radius in arc seconds
        :return: surface mass density divided by the critical density
        """
        r_s_angle = r_scale / self.lensCosmo.dd / const.arcsec  # Rs in arcsec
        kappa_s = rho_s * r_scale / self.lensCosmo.sigma_crit

        return kappa_s, r_s_angle

    def draw_lens(self, no_error=False):
        """Draws a lens model from the posterior.

        :param no_error: bool, if True, does not render from the uncertainty but uses
            the mean values instead
        """
        if no_error is True:
            if self._is_m2l_population_level:
                return (
                    np.mean(self._alpha_Rs_or_kappa_s_array),
                    np.mean(self._r_scale_angle_array),
                    self._r_eff,
                    1,
                )
            else:
                return (
                    np.mean(self._alpha_Rs_or_kappa_s_array),
                    np.mean(self._r_scale_angle_array),
                    np.mean(self.log_m2l_array),
                    self._r_eff,
                    1,
                )

        random_index = np.random.randint(
            low=0, high=len(self._alpha_Rs_or_kappa_s_array)
        )
        alpha_Rs_or_kappa_s_draw = self._alpha_Rs_or_kappa_s_array[random_index]
        r_scale_angle_draw = self._r_scale_angle_array[random_index]

        # we make sure no negative r_eff are being sampled
        delta_r_eff = np.maximum(
            np.random.normal(loc=1, scale=self._r_eff_error / self._r_eff), 0.001
        )
        r_eff_draw = delta_r_eff * self._r_eff

        if self._is_m2l_population_level:
            return alpha_Rs_or_kappa_s_draw, r_scale_angle_draw, r_eff_draw, delta_r_eff
        else:
            log_m2l_draw = self.log_m2l_array[random_index]
            return (
                alpha_Rs_or_kappa_s_draw,
                r_scale_angle_draw,
                log_m2l_draw,
                r_eff_draw,
                delta_r_eff,
            )

    def model_marginalization(self, num_sample_model=20):
        """

        :param num_sample_model: number of samples drawn from the lens and light model
            posterior to compute the dimensionless kinematic component J()
        :return: J() as array for each measurement prediction, covariance matrix in
            sqrt(J)
        """
        num_data = len(self._sigma_v_measured)
        j_kin_matrix = np.zeros(
            (num_sample_model, num_data)
        )  # matrix that contains the sampled J() distribution
        for i in range(num_sample_model):
            if self._is_m2l_population_level:
                j_kin = self.j_kin_draw_composite(
                    self.kwargs_anisotropy_base,
                    np.mean(self.gamma_in_array),
                    np.mean(self.log_m2l_array),
                    no_error=False,
                )
            else:
                j_kin = self.j_kin_draw_composite_m2l(
                    self.kwargs_anisotropy_base,
                    np.mean(self.gamma_in_array),
                    no_error=False,
                )
            j_kin_matrix[i, :] = j_kin

        error_cov_j_sqrt = np.cov(np.sqrt(j_kin_matrix.T))
        j_model_list = np.nanmean(j_kin_matrix, axis=0)
        return j_model_list, error_cov_j_sqrt

    def j_kin_draw_composite(
        self, kwargs_anisotropy, gamma_in, log_m2l, no_error=False
    ):
        """One simple sampling realization of the dimensionless kinematics of the model.

        :param kwargs_anisotropy: keyword argument of anisotropy setting
        :param gamma_in: power-law slope of the mass model
        :param log_m2l: log10(mass-to-light ratio) of the stellar component
        :param no_error: bool, if True, does not render from the uncertainty but uses
            the mean values instead
        :return: dimensionless kinematic component J() Birrer et al. 2016, 2019
        """
        alpha_Rs_or_kappa_s_draw, r_scale_angle_draw, r_eff_draw, delta_r_eff = (
            self.draw_lens(no_error=no_error)
        )

        kwargs_lens_stars = copy.deepcopy(self._kwargs_lens_light[0])

        kwargs_lens_stars["amp"] *= 10**log_m2l / self.lensCosmo.sigma_crit_angle

        kwargs_lens_stars["sigma"] *= delta_r_eff

        kwargs_light = copy.deepcopy(self._kwargs_lens_light)
        for kwargs in kwargs_light:
            kwargs["sigma"] *= delta_r_eff

        # Input is alpha_Rs
        if self._alpha_Rs_or_kappa_s_bool:
            alpha_Rs_draw = alpha_Rs_or_kappa_s_draw
        # Input is kappa_s
        else:
            alpha_Rs_draw = GNFW().kappa_s_to_alpha_Rs(
                alpha_Rs_or_kappa_s_draw, r_scale_angle_draw, gamma_in
            )

        kwargs_lens = [
            {
                "Rs": r_scale_angle_draw,
                "gamma_in": gamma_in,
                "alpha_Rs": alpha_Rs_draw,
                "center_x": 0,
                "center_y": 0,
            },
            kwargs_lens_stars,
        ]

        j_kin = self.velocity_dispersion_map_dimension_less(
            kwargs_lens=kwargs_lens,
            kwargs_lens_light=kwargs_light,
            kwargs_anisotropy=kwargs_anisotropy,
            r_eff=r_eff_draw,
            theta_E=self._theta_E,  # send this to avoid unnecessary recomputation
            gamma=self._gamma,  # send this to avoid unnecessary recomputation
        )
        return j_kin

    def j_kin_draw_composite_m2l(self, kwargs_anisotropy, gamma_in, no_error=False):
        """Similar function to j_kin_draw_composite, but now drawing from log_m2l
        distribution.

        :param kwargs_anisotropy: keyword argument of anisotropy setting
        :param gamma_in: power-law slope of the mass model
        :param no_error: bool, if True, does not render from the uncertainty but uses
            the mean values instead
        :return: dimensionless kinematic component J() Birrer et al. 2016, 2019
        """
        (
            alpha_Rs_or_kappa_s_draw,
            r_scale_angle_draw,
            log_m2l_draw,
            r_eff_draw,
            delta_r_eff,
        ) = self.draw_lens(no_error=no_error)

        kwargs_lens_stars = copy.deepcopy(self._kwargs_lens_light[0])

        kwargs_lens_stars["amp"] *= log_m2l_draw / self.lensCosmo.sigma_crit_angle

        kwargs_lens_stars["sigma"] *= delta_r_eff

        kwargs_light = copy.deepcopy(self._kwargs_lens_light)
        for kwargs in kwargs_light:
            kwargs["sigma"] *= delta_r_eff

        # Input is alpha_Rs
        if self._alpha_Rs_or_kappa_s_bool:
            alpha_Rs_draw = alpha_Rs_or_kappa_s_draw
        # Input is kappa_s
        else:
            alpha_Rs_draw = GNFW().kappa_s_to_alpha_Rs(
                alpha_Rs_or_kappa_s_draw, r_scale_angle_draw, gamma_in
            )

        kwargs_lens = [
            {
                "Rs": r_scale_angle_draw,
                "gamma_in": gamma_in,
                "alpha_Rs": alpha_Rs_draw,
                "center_x": 0,
                "center_y": 0,
            },
            kwargs_lens_stars,
        ]

        j_kin = self.velocity_dispersion_map_dimension_less(
            kwargs_lens=kwargs_lens,
            kwargs_lens_light=kwargs_light,
            kwargs_anisotropy=kwargs_anisotropy,
            r_eff=r_eff_draw,
            theta_E=self._theta_E,  # send this to avoid unnecessary recomputation
            gamma=self._gamma,  # send this to avoid unnecessary recomputation
        )
        return j_kin

    def hierarchy_configuration(self, num_sample_model=20):
        """Routine to configure the likelihood to be used in the hierarchical sampling.
        In particular, a default configuration is set to compute the Gaussian
        approximation of Ds/Dds by sampling the posterior and the estimate of the
        variance of the sample. The anisotropy scaling is then performed. Different
        anisotropy models are supported.

        :param num_sample_model: number of samples drawn from the lens and light model
            posterior to compute the dimensionless kinematic component J()
        :return: keyword arguments
        """

        j_model_list, error_cov_j_sqrt = self.model_marginalization(num_sample_model)
        ani_scaling_grid_list = self.anisotropy_scaling()

        error_cov_measurement = self.error_cov_measurement
        # configuration keyword arguments for the hierarchical sampling
        kwargs_likelihood = {
            "z_lens": self._z_lens,
            "z_source": self._z_source,
            "likelihood_type": "IFUKinCov",
            "sigma_v_measurement": self._sigma_v_measured,
            "anisotropy_model": self._anisotropy_model,
            "j_model": j_model_list,
            "error_cov_measurement": error_cov_measurement,
            "error_cov_j_sqrt": error_cov_j_sqrt,
            # "ani_param_array": self.kin_scaling_param_array,
            # "gamma_in_array": self.gamma_in_array,
            # "log_m2l_array": self.log_m2l_array,
            # "param_scaling_grid_list": ani_scaling_grid_list,
            "kin_scaling_param_list": self.param_name_list,
            "j_kin_scaling_param_axes": self.kin_scaling_param_array,
            "j_kin_scaling_grid_list": ani_scaling_grid_list,
        }
        prior_list = None
        if (
            self.gamma_in_array is not None
            and self._gamma_in_prior_mean is not None
            and self._gamma_in_prior_std is not None
        ):
            prior_list = [
                ["gamma_in", self._gamma_in_prior_mean, self._gamma_in_prior_std]
            ]
        kwargs_likelihood["prior_list"] = prior_list
        # if not self._is_m2l_population_level:
        #    kwargs_likelihood["log_m2l_array"] = None
        return kwargs_likelihood

    def anisotropy_scaling(self):
        """

        :return: anisotropy scaling grid along the axes defined by ani_param_array
        """
        if self._is_m2l_population_level:
            j_ani_0 = self.j_kin_draw_composite(
                self.kwargs_anisotropy_base,
                np.mean(self.gamma_in_array),
                np.mean(self.log_m2l_array),
                no_error=True,
            )
            return self._anisotropy_scaling_relative(j_ani_0)
        else:
            j_ani_0 = self.j_kin_draw_composite_m2l(
                self.kwargs_anisotropy_base,
                np.mean(self.gamma_in_array),
                no_error=True,
            )
            return self._anisotropy_scaling_relative_m2l(j_ani_0)

    def _anisotropy_scaling_relative(self, j_ani_0):
        """Anisotropy scaling relative to a default J prediction.

        :param j_ani_0: default J() prediction for default anisotropy
        :return: list of arrays (for the number of measurements) according to anisotropy
            scaling
        """
        num_data = len(self._sigma_v_measured)

        if self._anisotropy_model == "GOM":
            ani_scaling_grid_list = [
                np.zeros(
                    (
                        len(self.kin_scaling_param_array[0]),
                        len(self.kin_scaling_param_array[1]),
                        len(self.gamma_in_array),
                        len(self.log_m2l_array),
                    )
                )
                for _ in range(num_data)
            ]
            for i, a_ani in enumerate(self.kin_scaling_param_array[0]):
                for j, beta_inf in enumerate(self.kin_scaling_param_array[1]):
                    for k, g_in in enumerate(self.gamma_in_array):
                        for l, log_m2l in enumerate(self.log_m2l_array):
                            kwargs_anisotropy = self.anisotropy_kwargs(
                                a_ani=a_ani, beta_inf=beta_inf
                            )
                            j_kin_ani = self.j_kin_draw_composite(
                                kwargs_anisotropy, g_in, log_m2l, no_error=True
                            )

                            for m, j_kin in enumerate(j_kin_ani):
                                ani_scaling_grid_list[m][i, j, k, l] = (
                                    j_kin / j_ani_0[m]
                                )
                                # perhaps change the order
        elif self._anisotropy_model in ["OM", "const"]:
            ani_scaling_grid_list = [
                np.zeros(
                    (
                        len(self.kin_scaling_param_array[0]),
                        len(self.gamma_in_array),
                        len(self.log_m2l_array),
                    )
                )
                for _ in range(num_data)
            ]
            for i, a_ani in enumerate(self.kin_scaling_param_array[0]):
                for k, g_in in enumerate(self.gamma_in_array):
                    for l, log_m2l in enumerate(self.log_m2l_array):
                        kwargs_anisotropy = self.anisotropy_kwargs(a_ani)
                        j_kin_ani = self.j_kin_draw_composite(
                            kwargs_anisotropy, g_in, log_m2l, no_error=True
                        )
                        for m, j_kin in enumerate(j_kin_ani):
                            ani_scaling_grid_list[m][i, k, l] = j_kin / j_ani_0[m]
        else:
            raise ValueError("anisotropy model %s not valid." % self._anisotropy_model)
        return ani_scaling_grid_list

    def _anisotropy_scaling_relative_m2l(self, j_ani_0):
        """Similar function to _anisotropy_scaling_relative, but instead drawing log_m2l
        from a distribution.

        :param j_ani_0: default J() prediction for default anisotropy
        :return: list of arrays (for the number of measurements) according to anisotropy
            scaling
        """
        num_data = len(self._sigma_v_measured)

        if self._anisotropy_model == "GOM":
            ani_scaling_grid_list = [
                np.zeros(
                    (
                        len(self.kin_scaling_param_array[0]),
                        len(self.kin_scaling_param_array[1]),
                        len(self.gamma_in_array),
                    )
                )
                for _ in range(num_data)
            ]
            for i, a_ani in enumerate(self.kin_scaling_param_array[0]):
                for j, beta_inf in enumerate(self.kin_scaling_param_array[1]):
                    for k, g_in in enumerate(self.gamma_in_array):
                        kwargs_anisotropy = self.anisotropy_kwargs(
                            a_ani=a_ani, beta_inf=beta_inf
                        )
                        j_kin_ani = self.j_kin_draw_composite_m2l(
                            kwargs_anisotropy, g_in, no_error=True
                        )

                        for m, j_kin in enumerate(j_kin_ani):
                            ani_scaling_grid_list[m][i, j, k] = j_kin / j_ani_0[m]
                            # perhaps change the order
        elif self._anisotropy_model in ["OM", "const"]:
            ani_scaling_grid_list = [
                np.zeros(
                    (
                        len(self.kin_scaling_param_array[0]),
                        len(self.gamma_in_array),
                    )
                )
                for _ in range(num_data)
            ]
            for i, a_ani in enumerate(self.kin_scaling_param_array[0]):
                for k, g_in in enumerate(self.gamma_in_array):
                    kwargs_anisotropy = self.anisotropy_kwargs(a_ani)
                    j_kin_ani = self.j_kin_draw_composite_m2l(
                        kwargs_anisotropy, g_in, no_error=True
                    )
                    for m, j_kin in enumerate(j_kin_ani):
                        ani_scaling_grid_list[m][i, k] = j_kin / j_ani_0[m]
        else:
            raise ValueError("anisotropy model %s not valid." % self._anisotropy_model)
        return ani_scaling_grid_list
