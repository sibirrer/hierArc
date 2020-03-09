import numpy as np
from hierarc.LensPosterior.base_config import BaseLensConfig


class DsDdsConstraints(BaseLensConfig):
    """
    class for sampling Ds/Dds posteriors from imaging data and kinematic constraints
    """

    def __init__(self, z_lens, z_source, theta_E, theta_E_error, gamma, gamma_error, r_eff, r_eff_error, sigma_v,
                 sigma_v_error, kwargs_aperture, kwargs_seeing, kwargs_numerics_galkin, anisotropy_model,
                 kwargs_lens_light=None, lens_light_model_list=['HERNQUIST'], MGE_light=False, kwargs_mge_light=None,
                 hernquist_approx=True):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param theta_E: Einstein radius (in arc seconds)
        :param theta_E_error: 1-sigma error on Einstein radius
        :param gamma: power-law slope (2 = isothermal) estimated from imaging data
        :param gamma_error: 1-sigma uncertainty on power-law slope
        :param r_eff: half-light radius of the deflector (arc seconds)
        :param r_eff_error: uncertainty on half-light radius
        :param sigma_v: velocity dispersion of the main deflector in km/s
        :param sigma_v_error: 1-sigma uncertainty in velocity dispersion
        :param kwargs_aperture: spectroscopic aperture keyword arguments, see lenstronomy.Galkin.aperture for options
        :param kwargs_seeing: seeing condition of spectroscopic observation, corresponds to kwargs_psf in the GalKin module specified in lenstronomy.GalKin.psf
        :param kwargs_numerics_galkin: numerical settings for the integrated line-of-sight velocity dispersion
        :param anisotropy_model: type of stellar anisotropy model. See details in MamonLokasAnisotropy() class of lenstronomy.GalKin.anisotropy
        :param kwargs_lens_light: keyword argument list of lens light model (optional)
        :param kwargs_mge_light: keyword arguments that go into the MGE decomposition routine
        :param hernquist_approx: bool, if True, uses the Hernquist approximation for the light profile
        """
        BaseLensConfig.__init__(self, z_lens, z_source, theta_E, theta_E_error, gamma, gamma_error, r_eff, r_eff_error,
                                sigma_v, sigma_v_error, kwargs_aperture, kwargs_seeing, kwargs_numerics_galkin,
                                anisotropy_model, kwargs_lens_light=kwargs_lens_light,
                                lens_light_model_list=lens_light_model_list, MGE_light=MGE_light,
                                kwargs_mge_light=kwargs_mge_light, hernquist_approx=hernquist_approx)

    def draw_vel_disp(self, num=1, no_error=False):
        """
        produces realizations of measurements based on the uncertainty in the measurement of the velocity dispersion

        :param num: int, number of realization
        :param no_error: bool, if True, does not render from the uncertainty but uses the mean values instead
        :return: realizations of draws from the measured velocity dispersion
        """
        if no_error is True:
            return self._sigma_v
        return np.random.normal(loc=self._sigma_v, scale=self._sigma_v_error_independent, size=num)

    def ds_dds_realization(self, kwargs_anisotropy, no_error=False):
        """
        creates a realization of Ds/Dds from the measurement uncertainties

        :param kwargs_anisotropy: keyword argument of anisotropy setting
        :param no_error: bool, if True, does not render from the uncertainty but uses the mean values instead
        """

        # compute dimensionless kinematic quantity
        j_kin = self.j_kin_draw(kwargs_anisotropy, no_error)
        sigma_v_draw = self.draw_vel_disp(num=1, no_error=no_error)
        ds_dds = self.ds_dds_from_kinematics(sigma_v_draw, j_kin, kappa_s=0, kappa_ds=0)
        return ds_dds

    def j_kin_draw(self, kwargs_anisotropy, no_error=False, sampling_number=1000):
        """
        one simple sampling realization of the dimensionless kinematics of the model

        :param kwargs_anisotropy: keyword argument of anisotropy setting
        :param no_error: bool, if True, does not render from the uncertainty but uses the mean values instead
        :param sampling_number: int, number of spectral rendering (see lenstronomy GalKin module)
        :return: dimensionless kinematic component J() Birrer et al. 2016, 2019
        """
        theta_E_draw, gamma_draw, r_eff_draw = self.draw_lens(no_error=no_error)
        kwargs_lens = [{'theta_E': theta_E_draw, 'gamma': gamma_draw, 'center_x': 0, 'center_y': 0}]
        if self._kwargs_lens_light is None:
            kwargs_light = [{'Rs': r_eff_draw * 0.551, 'amp': 1.}]
        else:
            kwargs_light = self._kwargs_lens_light
        j_kin = self.velocity_dispersion_dimension_less(kwargs_lens=kwargs_lens, kwargs_lens_light=kwargs_light,
                                                        kwargs_anisotropy=kwargs_anisotropy, r_eff=r_eff_draw,
                                                        theta_E=theta_E_draw, gamma=gamma_draw,
                                                        sampling_number=sampling_number)
        return j_kin

    def ds_dds_sample(self, kwargs_anisotropy, num_sample_model=20, num_kin_measurements=50):
        """

        :param num_sample_model: number of samples drawn from the lens and light model posterior to compute the dimensionless
        kinematic component J()
        :param num_kin_measurements: number of draws from the velocity dispersion measurements to simple sample the
        posterior in Ds/Dds. The total number of posteriors is num_sample_model x num_kin_measurements
        :param kwargs_anisotropy: keyword argument with stellar anisotropy configuration parameters
        :return: numpy array of posterior values of Ds/Dds
        """
        ds_dds_list = []
        for i in range(num_sample_model):
            j_kin = self.j_kin_draw(kwargs_anisotropy, no_error=False)
            for j in range(num_kin_measurements):
                sigma_v_draw = self.draw_vel_disp(num=1, no_error=False)
                ds_dds = self.ds_dds_from_kinematics(sigma_v_draw, j_kin, kappa_s=0, kappa_ds=0)
                ds_dds_list.append(ds_dds)
        return np.array(ds_dds_list)

    def kin_constraints(self, num_sample_model=20, num_kin_measurements=50):
        """
        routine to configure the likelihood to be used in the hierarchical sampling. In particular, a default
        configuration is set to compute the Gaussian approximation of Ds/Dds by sampling the posterior and the estimate
        of the variance of the sample. The anisotropy scaling is then performed. Different anisotropy models are
        supported.

        :param num_sample_model: number of samples drawn from the lens and light model posterior to compute the dimensionless
        kinematic component J()
        :param num_kin_measurements: number of draws from the velocity dispersion measurements to simple sample the
        posterior in Ds/Dds. The total number of posteriors is num_sample_model x num_kin_measurements
        :return:
        """

        # here we simple sampling the default anisotropy configuration and compute the mean and std of the sample
        ds_dds_sample = self.ds_dds_sample(kwargs_anisotropy=self.kwargs_anisotropy_base, num_sample_model=num_sample_model,
                                           num_kin_measurements=num_kin_measurements)
        ds_dds_mean = np.mean(ds_dds_sample)
        ds_dds_sigma = np.std(ds_dds_sample)

        # here we loop through the possible anisotropy configuration within the model parameterization
        ds_dds_ani_0 = self.ds_dds_realization(self.kwargs_anisotropy_base, no_error=True)
        ani_scaling_array = []
        for a_ani in self.ani_param_array:
            kwargs_anisotropy = self.anisotropy_kwargs(a_ani)
            ds_dds_ani = self.ds_dds_realization(kwargs_anisotropy, no_error=True)
            ani_scaling_array.append(ds_dds_ani / ds_dds_ani_0)
        ani_scaling_array = np.array(ani_scaling_array)
        return ds_dds_mean, ds_dds_sigma, self.ani_param_array, ani_scaling_array

    def hierarchy_configuration(self, num_sample_model=20, num_kin_measurements=50):
        """
        routine to configure the likelihood to be used in the hierarchical sampling. In particular, a default
        configuration is set to compute the Gaussian approximation of Ds/Dds by sampling the posterior and the estimate
        of the variance of the sample. The anisotropy scaling is then performed. Different anisotropy models are
        supported.

        :param num_sample_model: number of samples drawn from the lens and light model posterior to compute the dimensionless
        kinematic component J()
        :param num_kin_measurements: number of draws from the velocity dispersion measurements to simple sample the
        posterior in Ds/Dds. The total number of posteriors is num_sample_model x num_kin_measurements
        :return: keyword arguments
        """

        ds_dds_mean, ds_dds_sigma, ani_param_array, ani_scaling_array = self.kin_constraints(num_sample_model,
                                                                                             num_kin_measurements)
        # configuration keyword arguments for the hierarchical sampling
        kwargs_likelihood = {'z_lens': self._z_lens, 'z_source': self._z_source, 'likelihood_type': 'KinGaussian',
                             'ds_dds_mean': ds_dds_mean,  'ds_dds_sigma': ds_dds_sigma,
                             'ani_param_array': ani_param_array, 'ani_scaling_array': ani_scaling_array}
        return kwargs_likelihood
