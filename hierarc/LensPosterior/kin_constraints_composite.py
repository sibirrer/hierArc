import copy

import numpy as np
from hierarc.LensPosterior.base_config import BaseLensConfig
from hierarc.LensPosterior.kin_constraints import KinConstraints


class KinConstraintsComposite(KinConstraints):

    def __init__(self, z_lens, z_source, gamma_in_array, m2l_array,
                 r_scale_array, m200_array, theta_E,
                 theta_E_error,gamma, gamma_error, r_eff, r_eff_error,
                 sigma_v_measured, kwargs_aperture, kwargs_seeing,
                 kwargs_numerics_galkin, anisotropy_model,
                 sigma_v_error_independent=None, sigma_v_error_covariant=None,
                 sigma_v_error_cov_matrix=None,
                 kwargs_lens_light=None, lens_light_model_list = ['HERNQUIST'],
                 MGE_light=False, kwargs_mge_light = None, hernquist_approx = True,
                 sampling_number=1000,
                 num_psf_sampling=100, num_kin_sampling=1000, multi_observations=False):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param gamma_in_array: array of power-law slopes of the mass model
        :param m2l_array: array of mass-to-light ratios of the stellar component
        :param r_scale_array: array of halo scale radii in arc seconds
        :param m200_array: array of halo masses in M_sun
        :param theta_E: Einstein radius (in arc seconds)
        :param theta_E_error: 1-sigma error on Einstein radius
        :param gamma: power-law slope (2 = isothermal) estimated from imaging data
        :param gamma_error: 1-sigma uncertainty on power-law slope
        :param r_eff: half-light radius of the deflector (arc seconds)
        :param r_eff_error: uncertainty on half-light radius
        :param sigma_v_measured: numpy array of IFU velocity dispersion of the main deflector in km/s
        :param sigma_v_error_independent: numpy array of 1-sigma uncertainty in velocity dispersion of the IFU
         observation independent of each other
        :param sigma_v_error_covariant: covariant error in the measured kinematics shared among all IFU measurements
        :param sigma_v_error_cov_matrix: error covariance matrix in the sigma_v measurements (km/s)^2
        :type sigma_v_error_cov_matrix: nxn matrix with n the length of the sigma_v_measured array
        :param kwargs_aperture: spectroscopic aperture keyword arguments, see lenstronomy.Galkin.aperture for options
        :param kwargs_seeing: seeing condition of spectroscopic observation, corresponds to kwargs_psf in the GalKin
         module specified in lenstronomy.GalKin.psf
        :param kwargs_numerics_galkin: numerical settings for the integrated line-of-sight velocity dispersion
        :param anisotropy_model: type of stellar anisotropy model. See details in MamonLokasAnisotropy() class of
         lenstronomy.GalKin.anisotropy
        :param kwargs_lens_light: keyword argument list of lens light model (optional)
        :param kwargs_mge_light: keyword arguments that go into the MGE decomposition routine
        :param hernquist_approx: bool, if True, uses the Hernquist approximation for the light profile
        :param multi_observations: bool, if True, interprets kwargs_aperture and kwargs_seeing as lists of multiple
         observations
        """
        self._m200_array = m200_array
        self._r_scale_array = r_scale_array
        self.gamma_in_array = gamma_in_array
        self.m2l_array = m2l_array

        super(KinConstraintsComposite, self).__init__(z_lens, z_source, theta_E, theta_E_error, gamma, gamma_error, r_eff,
                                                      r_eff_error, sigma_v_measured, kwargs_aperture, kwargs_seeing,
                                                      kwargs_numerics_galkin,
                                                      anisotropy_model, sigma_v_error_independent=sigma_v_error_independent,
                                                      sigma_v_error_covariant=sigma_v_error_covariant,
                                                      sigma_v_error_cov_matrix=sigma_v_error_cov_matrix,
                                                      kwargs_lens_light=kwargs_lens_light,
                                                      lens_light_model_list=lens_light_model_list, MGE_light=MGE_light,
                                                      kwargs_mge_light=kwargs_mge_light, hernquist_approx=hernquist_approx,
                                                      sampling_number=sampling_number, num_psf_sampling=num_psf_sampling,
                                                      num_kin_sampling=num_kin_sampling,
                                                      multi_observations=multi_observations)

    def j_kin_draw_composite(self, kwargs_anisotropy, gamma_in, m2l, no_error=False):
        """
        one simple sampling realization of the dimensionless kinematics of the model

        :param kwargs_anisotropy: keyword argument of anisotropy setting
        :param gamma_in: power-law slope of the mass model
        :param m2l: mass-to-light ratio of the stellar component
        :param no_error: bool, if True, does not render from the uncertainty but uses the mean values instead
        :return: dimensionless kinematic component J() Birrer et al. 2016, 2019
        """
        m200_draw, r_scale_draw, r_eff_draw, delta_r_eff = self.draw_lens(
            no_error=no_error)
        kappa_s = # To-do: from m200_draw and r_scale draw
        kwargs_lens = [{'Rs': r_scale_draw, 'gamma_in': gamma_in, 'kappa_s': kappa_s,
                        'center_x': 0, 'center_y': 0},
                       {} # To-do: stellar mass distribution using m2l
                       ]

        kwargs_light = copy.deepcopy(self._kwargs_lens_light)

        for kwargs in kwargs_light:
            if 'Rs' in kwargs:
                kwargs['Rs'] *= delta_r_eff
            if 'R_sersic' in kwargs:
                kwargs['R_sersic'] *= delta_r_eff

        j_kin = self.velocity_dispersion_map_dimension_less(kwargs_lens=kwargs_lens,
                                                            kwargs_lens_light=kwargs_light,
                                                            kwargs_anisotropy=kwargs_anisotropy,
                                                            r_eff=r_eff_draw,
                                                            )
        return j_kin

    def galkin_settings(
        self, kwargs_lens, kwargs_lens_light, r_eff=None, theta_E=None, gamma=None
    ):
        """

        :param kwargs_lens: lens model keyword argument list
        :param kwargs_lens_light: deflector light keyword argument list
        :param r_eff: half-light radius (optional)
        :param theta_E: Einstein radius (optional)
        :param gamma: local power-law slope at the Einstein radius (optional)
        :return: Galkin() instance and mass and light profiles configured for the Galkin module
        """
        # to-do: implement this
        pass

    def velocity_dispersion_map_dimension_less(
        self,
        kwargs_lens,
        kwargs_lens_light,
        kwargs_anisotropy,
        r_eff=None,
        theta_E=None,
        gamma=None,
    ):
        """Sigma**2 = Dd/Dds * c**2 * J(kwargs_lens, kwargs_light, anisotropy) (Equation
        4.11 in Birrer et al. 2016 or Equation 6 in Birrer et al. 2019) J() is a
        dimensionless and cosmological independent quantity only depending on angular
        units. This function returns J given the lens and light parameters and the
        anisotropy choice without an external mass sheet correction. This routine
        computes the IFU map of the kinematic quantities.

        :param kwargs_lens: lens model keyword arguments
        :param kwargs_lens_light: lens light model keyword arguments
        :param kwargs_anisotropy: stellar anisotropy keyword arguments
        :param r_eff: projected half-light radius of the stellar light associated with
            the deflector galaxy, optional, if set to None will be computed in this
            function with default settings that may not be accurate.
        :return: dimensionless velocity dispersion (see e.g. Birrer et al. 2016, 2019)
        """
        # to-do: implement this
        pass

    def velocity_dispersion_map(
        self,
        kwargs_lens,
        kwargs_lens_light,
        kwargs_anisotropy,
        r_eff=None,
        theta_E=None,
        gamma=None,
        kappa_ext=0,
    ):
        """API for both, analytic and numerical JAM to compute the velocity dispersion
        map with IFU data [km/s]

        :param kwargs_lens: lens model keyword arguments
        :param kwargs_lens_light: lens light model keyword arguments
        :param kwargs_anisotropy: stellar anisotropy keyword arguments
        :param r_eff: projected half-light radius of the stellar light associated with
            the deflector galaxy, optional, if set to None will be computed in this
            function with default settings that may not be accurate.
        :param theta_E: circularized Einstein radius, optional, if not provided will
            either be computed in this function with default settings or not required
        :param gamma: power-law slope at the Einstein radius, optional
        :param kappa_ext: external convergence
        :return: velocity dispersion [km/s]
        """
        # to-do: implement this
        pass


    def hierarchy_configuration(self, num_sample_model=20):
        """
        routine to configure the likelihood to be used in the hierarchical sampling. In particular, a default
        configuration is set to compute the Gaussian approximation of Ds/Dds by sampling the posterior and the estimate
        of the variance of the sample. The anisotropy scaling is then performed. Different anisotropy models are
        supported.

        :param num_sample_model: number of samples drawn from the lens and light model posterior to compute the
         dimensionless kinematic component J()
        :return: keyword arguments
        """

        j_model_list, error_cov_j_sqrt = self.model_marginalization(num_sample_model)
        ani_scaling_array_grid = self.anisotropy_scaling()

        error_cov_measurement = self.error_cov_measurement
        # configuration keyword arguments for the hierarchical sampling
        kwargs_likelihood = {'z_lens': self._z_lens, 'z_source': self._z_source,
                             'likelihood_type': 'IFUKinCov',
                             'sigma_v_measurement': self._sigma_v_measured,
                             'anisotropy_model': self._anisotropy_model,
                             'j_model': j_model_list,
                             'error_cov_measurement': error_cov_measurement,
                             'error_cov_j_sqrt': error_cov_j_sqrt,
                             'ani_param_array': self.ani_param_array,
                             'gamma_in_array': self.gamma_in_array,
                             'm2l_array': self.m2l_array,
                             'ani_scaling_array_grid': ani_scaling_array_grid,
                             }
        return kwargs_likelihood

    def _anisotropy_scaling_relative(self, j_ani_0):
        """
        anisotropy scaling relative to a default J prediction

        :param j_ani_0: default J() prediction for default anisotropy
        :return: list of arrays (for the number of measurements) according to anisotropy scaling
        """
        num_data = len(self._sigma_v_measured)

        if self._anisotropy_model == 'GOM':
            ani_scaling_array_list = [np.zeros((len(self.gamma_in_array),
                                                len(self.m2l_array),
                                                len(self.ani_param_array[0]),
                                                len(self.ani_param_array[1]),
                                                ))
                                      for _ in range(num_data)]
            for i, a_ani in enumerate(self.ani_param_array[0]):
                for j, beta_inf in enumerate(self.ani_param_array[1]):
                    for k, g_in in enumerate(self.gamma_in_array):
                        for l, m2l in enumerate(self.m2l_array):
                            kwargs_anisotropy = self.anisotropy_kwargs(a_ani=a_ani, beta_inf=beta_inf)
                            j_kin_ani = self.j_kin_draw_composite(kwargs_anisotropy, g_in,
                                                                  m2l, no_error=True)

                            for m, j_kin in enumerate(j_kin_ani):
                                ani_scaling_array_list[m][k, l, i, j] = (j_kin /
                                                                       j_ani_0[m])
                                #perhaps change the order
        elif self._anisotropy_model in ['OM', 'const']:
            ani_scaling_array_list = [np.zeros((len(self.gamma_in_array),
                                                len(self.m2l_array),
                                                len(self.ani_param_array[0]),
                                                len(self.ani_param_array[1]),
                                                ))
                                      for _ in range(num_data)]
            for i, a_ani in enumerate(self.ani_param_array):
                    for k, g_in in enumerate(self.gamma_in_array):
                        for l, m2l in enumerate(self.m2l_array):
                            kwargs_anisotropy = self.anisotropy_kwargs(a_ani)
                            j_kin_ani = self.j_kin_draw_composite(kwargs_anisotropy,
                                                                  g_in,
                                                                  m2l, no_error=True)
                            for m, j_kin in enumerate(j_kin_ani):
                                    ani_scaling_array_list[m][k, l, i] = (j_kin /
                                                                       j_ani_0[m])
        else:
            raise ValueError('anisotropy model %s not valid.' % self._anisotropy_model)
        return ani_scaling_array_list



