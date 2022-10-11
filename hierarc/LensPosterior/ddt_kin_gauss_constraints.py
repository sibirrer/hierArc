from hierarc.LensPosterior.kin_constraints import KinConstraints


class DdtGaussKinConstraints(KinConstraints):
    """
    class for sampling Ds/Dds posteriors from imaging data and kinematic constraints with additional constraints on the
    time-delay distance Ddt
    """

    def __init__(self, z_lens, z_source, ddt_mean, ddt_sigma, theta_E, theta_E_error, gamma, gamma_error, r_eff,
                 r_eff_error, sigma_v_measured, kwargs_aperture, kwargs_seeing, kwargs_numerics_galkin, anisotropy_model,
                 sigma_v_error_independent=None, sigma_v_error_covariant=None, sigma_v_error_cov_matrix=None,
                 kwargs_lens_light=None, lens_light_model_list=['HERNQUIST'], MGE_light=False, kwargs_mge_light=None,
                 hernquist_approx=True, kappa_ext=0, kappa_ext_sigma=0, sampling_number=1000, num_psf_sampling=100,
                 num_kin_sampling=1000, multi_observations=False):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param ddt_mean: mean of time-delay distance likelihood
        :param ddt_sigma: 1-sigma Gaussian uncertainty on the time-delay distance likelihood
        :param sigma_v_measured: numpy array of IFU velocity dispersion of the main deflector in km/s
        :param sigma_v_error_independent: numpy array of 1-sigma uncertainty in velocity dispersion of the IFU
         observation independent of each other
        :param sigma_v_error_covariant: covariant error in the measured kinematics shared among all IFU measurements
        :param sigma_v_error_cov_matrix: error covariance matrix in the sigma_v measurements (km/s)^2
        :type sigma_v_error_cov_matrix: nxn matrix with n the length of the sigma_v_measured array
        :param kwargs_aperture: spectroscopic aperture keyword arguments, see lenstronomy.Galkin.aperture for options
        :param kwargs_seeing: seeing condition of spectroscopic observation, corresponds to kwargs_psf in the GalKin module specified in lenstronomy.GalKin.psf
        :param theta_E: Einstein radius (in arc seconds)
        :param theta_E_error: 1-sigma error on Einstein radius
        :param gamma: power-law slope (2 = isothermal) estimated from imaging data
        :param gamma_error: 1-sigma uncertainty on power-law slope
        :param r_eff: half-light radius of the deflector (arc seconds)
        :param r_eff_error: uncertainty on half-light radius
        :param kwargs_numerics_galkin: numerical settings for the integrated line-of-sight velocity dispersion
        :param anisotropy_model: type of stellar anisotropy model. See details in MamonLokasAnisotropy() class of lenstronomy.GalKin.anisotropy
        :param kwargs_lens_light: keyword argument list of lens light model (optional)
        :param kwargs_mge_light: keyword arguments that go into the MGE decomposition routine
        :param hernquist_approx: bool, if True, uses the Hernquist approximation for the light profile
        :param kappa_ext: mean of the external convergence from which the ddt constraints are coming from
        :param kappa_ext_sigma: 1-sigma distribution uncertainty from which the ddt constraints are coming from
        :param multi_observations: bool, if True, interprets kwargs_aperture and kwargs_seeing as lists of multiple
         observations
        """
        self._ddt_mean, self._ddt_sigma = ddt_mean, ddt_sigma
        self._kappa_ext_mean, self._kappa_ext_sigma = kappa_ext, kappa_ext_sigma
        super(DdtGaussKinConstraints, self).__init__(z_lens, z_source, theta_E, theta_E_error, gamma, gamma_error, r_eff,
                                                     r_eff_error, sigma_v_measured, kwargs_aperture, kwargs_seeing,
                                                     kwargs_numerics_galkin, anisotropy_model,
                                                     sigma_v_error_independent=sigma_v_error_independent,
                                                     sigma_v_error_covariant=sigma_v_error_covariant,
                                                     sigma_v_error_cov_matrix=sigma_v_error_cov_matrix,
                                                     kwargs_lens_light=kwargs_lens_light,
                                                     lens_light_model_list=lens_light_model_list, MGE_light=MGE_light,
                                                     kwargs_mge_light=kwargs_mge_light, hernquist_approx=hernquist_approx,
                                                     sampling_number=sampling_number, num_psf_sampling=num_psf_sampling,
                                                     num_kin_sampling=num_kin_sampling,
                                                     multi_observations=multi_observations)

    def hierarchy_configuration(self, num_sample_model=20):
        """
        routine to configure the likelihood to be used in the hierarchical sampling. In particular, a default
        configuration is set to compute the Gaussian approximation of Ds/Dds by sampling the posterior and the estimate
        of the variance of the sample. The anisotropy scaling is then performed. Different anisotropy models are
        supported.

        :param num_sample_model: number of samples drawn from the lens and light model posterior to compute the dimensionless
         kinematic component J()
        :return: keyword arguments
        """
        j_model_list, error_cov_j_sqrt = self.model_marginalization(num_sample_model)
        ani_scaling_array_list = self.anisotropy_scaling()
        error_cov_measurement = self.error_cov_measurement
        # configuration keyword arguments for the hierarchical sampling
        kwargs_likelihood = {'z_lens': self._z_lens, 'z_source': self._z_source, 'likelihood_type': 'DdtGaussKin',
                             'ddt_mean': self._ddt_mean, 'ddt_sigma': self._ddt_sigma,
                             'sigma_v_measurement': self._sigma_v_measured, 'anisotropy_model': self._anisotropy_model,
                             'j_model': j_model_list,  'error_cov_measurement': error_cov_measurement,
                             'error_cov_j_sqrt': error_cov_j_sqrt, 'ani_param_array': self.ani_param_array,
                             'ani_scaling_array_list': ani_scaling_array_list}
        return kwargs_likelihood
