import copy

import numpy as np
from hierarc.LensPosterior.base_config import BaseLensConfig


class KinConstraints(BaseLensConfig):
    """
    class that manages constraints from Integral Field Unit spectral observations.
    """
    def __init__(self, z_lens, z_source, theta_E, theta_E_error, gamma, gamma_error, r_eff, r_eff_error,
                 sigma_v_measured, kwargs_aperture, kwargs_seeing, kwargs_numerics_galkin, anisotropy_model,
                 sigma_v_error_independent=None, sigma_v_error_covariant=None, sigma_v_error_cov_matrix=None,
                 kwargs_lens_light=None, lens_light_model_list=['HERNQUIST'],
                 MGE_light=False, kwargs_mge_light=None, hernquist_approx=True, sampling_number=1000,
                 num_psf_sampling=100, num_kin_sampling=1000, multi_observations=False):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
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
        self._sigma_v_measured = np.array(sigma_v_measured)
        self._sigma_v_error_independent = np.array(sigma_v_error_independent)
        self._sigma_v_error_covariant = sigma_v_error_covariant
        self._sigma_v_error_cov_matrix = sigma_v_error_cov_matrix

        self._kwargs_lens_light = kwargs_lens_light
        self._anisotropy_model = anisotropy_model
        BaseLensConfig.__init__(self, z_lens, z_source, theta_E, theta_E_error, gamma, gamma_error, r_eff, r_eff_error,
                                kwargs_aperture, kwargs_seeing, kwargs_numerics_galkin,
                                anisotropy_model, kwargs_lens_light=kwargs_lens_light,
                                lens_light_model_list=lens_light_model_list, MGE_light=MGE_light,
                                kwargs_mge_light=kwargs_mge_light, hernquist_approx=hernquist_approx,
                                sampling_number=sampling_number, num_psf_sampling=num_psf_sampling,
                                num_kin_sampling=num_kin_sampling, multi_observations=multi_observations)

    def j_kin_draw(self, kwargs_anisotropy, no_error=False):
        """
        one simple sampling realization of the dimensionless kinematics of the model

        :param kwargs_anisotropy: keyword argument of anisotropy setting
        :param no_error: bool, if True, does not render from the uncertainty but uses the mean values instead
        :return: dimensionless kinematic component J() Birrer et al. 2016, 2019
        """
        theta_E_draw, gamma_draw, r_eff_draw, delta_r_eff = self.draw_lens(no_error=no_error)
        kwargs_lens = [{'theta_E': theta_E_draw, 'gamma': gamma_draw, 'center_x': 0, 'center_y': 0}]
        if self._kwargs_lens_light is None:
            kwargs_light = [{'Rs': r_eff_draw * 0.551, 'amp': 1.}]
        else:
            kwargs_light = copy.deepcopy(self._kwargs_lens_light)
            for kwargs in kwargs_light:
                if 'Rs' in kwargs:
                    kwargs['Rs'] *= delta_r_eff
                if 'R_sersic' in kwargs:
                    kwargs['R_sersic'] *= delta_r_eff
        j_kin = self.velocity_dispersion_map_dimension_less(kwargs_lens=kwargs_lens, kwargs_lens_light=kwargs_light,
                                                            kwargs_anisotropy=kwargs_anisotropy, r_eff=r_eff_draw,
                                                            theta_E=theta_E_draw, gamma=gamma_draw)
        return j_kin

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
        ani_scaling_array_list = self.anisotropy_scaling()
        error_cov_measurement = self.error_cov_measurement
        # configuration keyword arguments for the hierarchical sampling
        kwargs_likelihood = {'z_lens': self._z_lens, 'z_source': self._z_source, 'likelihood_type': 'IFUKinCov',
                             'sigma_v_measurement': self._sigma_v_measured, 'anisotropy_model': self._anisotropy_model,
                             'j_model': j_model_list,  'error_cov_measurement': error_cov_measurement,
                             'error_cov_j_sqrt': error_cov_j_sqrt, 'ani_param_array': self.ani_param_array,
                             'ani_scaling_array_list': ani_scaling_array_list}
        return kwargs_likelihood

    def model_marginalization(self, num_sample_model=20):
        """

        :param num_sample_model: number of samples drawn from the lens and light model posterior to compute the
         dimensionless kinematic component J()
        :return: J() as array for each measurement prediction, covariance matrix in sqrt(J)
        """
        num_data = len(self._sigma_v_measured)
        j_kin_matrix = np.zeros((num_sample_model, num_data))  # matrix that contains the sampled J() distribution
        for i in range(num_sample_model):
            j_kin = self.j_kin_draw(self.kwargs_anisotropy_base, no_error=False)
            j_kin_matrix[i, :] = j_kin

        error_cov_j_sqrt = np.cov(np.sqrt(j_kin_matrix.T))
        j_model_list = np.mean(j_kin_matrix, axis=0)
        return j_model_list, error_cov_j_sqrt

    @property
    def error_cov_measurement(self):
        """
        error covariance matrix of the measured velocity dispersion data points
        This is either calculated from the diagonal 'sigma_v_error_independent' and the off-diagonal
        'sigma_v_error_covariant' terms, or directly from the 'sigma_v_error_cov_matrix' if provided.

        :return: nxn matrix of the error covariances in the velocity dispersion measurements (km/s)^2
        """
        if self._sigma_v_error_cov_matrix is None:
            if self._sigma_v_error_independent is None or self._sigma_v_error_covariant is None:
                raise ValueError('sigma_v_error_independent and sigma_v_error_covariant need to be provided as arrays '
                                 'of the same length as sigma_v_measurement.')
            error_covariance_array = np.ones_like(self._sigma_v_error_independent) * self._sigma_v_error_covariant
            error_cov_measurement = np.outer(error_covariance_array, error_covariance_array) + np.diag(
                self._sigma_v_error_independent ** 2)
            return error_cov_measurement
        else:
            return self._sigma_v_error_cov_matrix

    def anisotropy_scaling(self):
        """

        :return: anisotropy scaling grid along the axes defined by ani_param_array
        """
        j_ani_0 = self.j_kin_draw(self.kwargs_anisotropy_base, no_error=True)
        return self._anisotropy_scaling_relative(j_ani_0)

    def _anisotropy_scaling_relative(self, j_ani_0):
        """
        anisotropy scaling relative to a default J prediction

        :param j_ani_0: default J() prediction for default anisotropy
        :return: list of arrays (for the number of measurements) according to anisotropy scaling
        """
        num_data = len(self._sigma_v_measured)

        if self._anisotropy_model == 'GOM':
            ani_scaling_array_list = [np.zeros((len(self.ani_param_array[0]), len(self.ani_param_array[1]))) for _ in
                                      range(num_data)]
            for i, a_ani in enumerate(self.ani_param_array[0]):
                for j, beta_inf in enumerate(self.ani_param_array[1]):
                    kwargs_anisotropy = self.anisotropy_kwargs(a_ani=a_ani, beta_inf=beta_inf)
                    j_kin_ani = self.j_kin_draw(kwargs_anisotropy, no_error=True)
                    for k, j_kin in enumerate(j_kin_ani):
                        ani_scaling_array_list[k][i, j] = j_kin / j_ani_0[k]  # perhaps change the order
        elif self._anisotropy_model in ['OM', 'const']:
            ani_scaling_array_list = [[] for _ in range(num_data)]
            for a_ani in self.ani_param_array:
                kwargs_anisotropy = self.anisotropy_kwargs(a_ani)
                j_kin_ani = self.j_kin_draw(kwargs_anisotropy, no_error=True)
                for i, j_kin in enumerate(j_kin_ani):
                    ani_scaling_array_list[i].append(j_kin / j_ani_0[i])
        else:
            raise ValueError('anisotropy model %s not valid.' % self._anisotropy_model)
        return ani_scaling_array_list
