import numpy as np
from lenstronomy.Analysis.td_cosmography import TDCosmography
from hierarc.LensPosterior.imaging_constraints import ImageModelPosterior


class BaseLensConfig(TDCosmography, ImageModelPosterior):
    """
    this class contains and manages the base configurations of the lens posteriors and makes sure that they
    are universally applied consistently through the different likelihood definitions
    """
    def __init__(self, z_lens, z_source, theta_E, theta_E_error, gamma, gamma_error, r_eff, r_eff_error, sigma_v,
                 sigma_v_error, kwargs_aperture, kwargs_seeing, kwargs_numerics_galkin, anisotropy_model,
                 kwargs_lens_light=None, lens_light_model_list=['HERNQUIST'], MGE_light=False, kwargs_mge_light=None,
                 hernquist_approx=True, sampling_number=1000, num_psf_sampling=100, num_kin_sampling=1000):
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
        self._z_lens, self._z_source = z_lens, z_source
        kwargs_model = {'lens_model_list': ['SPP'], 'lens_light_model_list': lens_light_model_list}
        self._sigma_v, self._sigma_v_error_independent = sigma_v, sigma_v_error
        TDCosmography.__init__(self, z_lens, z_source, kwargs_model, cosmo_fiducial=None,
                                 lens_model_kinematics_bool=None, light_model_kinematics_bool=None,
                               kwargs_seeing=kwargs_seeing, kwargs_aperture=kwargs_aperture)

        analytic_kinematics = False
        self.kinematics_modeling_settings(anisotropy_model, kwargs_numerics_galkin,
                                          analytic_kinematics=analytic_kinematics,
                                          Hernquist_approx=hernquist_approx, MGE_light=MGE_light, MGE_mass=False,
                                          kwargs_mge_light=kwargs_mge_light, sampling_number=sampling_number,
                                          num_psf_sampling=num_psf_sampling, num_kin_sampling=num_kin_sampling)
        self._kwargs_lens_light = kwargs_lens_light
        ImageModelPosterior.__init__(self, theta_E, theta_E_error, gamma, gamma_error, r_eff, r_eff_error)

        self._anisotropy_model = anisotropy_model

        if self._anisotropy_model == 'OM':
            self._ani_param_array = np.array([0.1, 0.2, 0.5, 1, 2, 5])  # used for r_ani OsipkovMerritt anisotropy description
        elif self._anisotropy_model == 'GOM':
            self._ani_param_array = [np.array([0.1, 0.2, 0.5, 1, 2, 5]), np.array([0, 0.5, 0.8, 1])]
        elif self._anisotropy_model == 'const':
            self._ani_param_array = np.linspace(0, 1, 10)  # used for constant anisotropy description
        else:
            raise ValueError('anisotropy model %s not supported.' % self._anisotropy_model)

    @property
    def kwargs_anisotropy_base(self):
        """

        :return: keyword arguments of base anisotropy model configuration
        """
        if self._anisotropy_model == 'OM':
            a_ani_0 = 1
            r_ani = a_ani_0 * self._r_eff
            kwargs_anisotropy_0 = {'r_ani': r_ani}
        elif self._anisotropy_model == 'GOM':
            a_ani_0 = 1
            r_ani = a_ani_0 * self._r_eff
            beta_inf_0 = 1
            kwargs_anisotropy_0 = {'r_ani': r_ani, 'beta_inf': beta_inf_0}
        elif self._anisotropy_model == 'const':
            a_ani_0 = 0.1
            kwargs_anisotropy_0 = {'beta': a_ani_0}
        else:
            raise ValueError('anisotropy model %s not supported.' % self._anisotropy_model)
        return kwargs_anisotropy_0

    @property
    def ani_param_array(self):
        """

        :return: numpy array of anisotropy parameter values to be explored
        """
        return self._ani_param_array

    def anisotropy_kwargs(self, a_ani, beta_inf=None):
        """

        :param a_ani: anisotropy parameter
        :param beta_inf: anisotropy at infinity (only used for 'GOM' model)
        :return: list of anisotropy keyword arguments, value of anisotropy parameter list
        """

        if self._anisotropy_model == 'OM':
            r_ani = a_ani * self._r_eff
            kwargs_anisotropy = {'r_ani': r_ani}
        elif self._anisotropy_model == 'GOM':
            r_ani = a_ani * self._r_eff
            kwargs_anisotropy = {'r_ani': r_ani, 'beta_inf': beta_inf}
        elif self._anisotropy_model == 'const':
            kwargs_anisotropy = {'beta': a_ani}
        else:
            raise ValueError('anisotropy model %s not supported.' % self._anisotropy_model)
        return kwargs_anisotropy
