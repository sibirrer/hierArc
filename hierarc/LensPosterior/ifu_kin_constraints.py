import numpy as np
from hierarc.LensPosterior.imaging_constraints import ImageModelPosterior


class IFUKin(ImageModelPosterior):
    """
    class that manages constraints from Integral Field Unit spectral observations.
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
        self._z_lens, self._z_source = z_lens, z_source
        kwargs_model = {'lens_model_list': ['SPP'], 'lens_light_model_list': lens_light_model_list}
        self._sigma_v, self._sigma_v_error = sigma_v, sigma_v_error
        #self._td_cosmo = TDCosmography(z_lens, z_source, kwargs_model, cosmo_fiducial=None,
        #                         lens_model_kinematics_bool=None, light_model_kinematics_bool=None)
        #self._td_cosmo.kinematic_observation_settings(kwargs_aperture, kwargs_seeing)
        if kwargs_lens_light is None and anisotropy_model == 'OsipkovMerritt':
            analytic_kinematics = True
        else:
            analytic_kinematics = False
        #self._td_cosmo.kinematics_modeling_settings(anisotropy_model, kwargs_numerics_galkin,
        #                                            analytic_kinematics=analytic_kinematics,
        #                                            Hernquist_approx=hernquist_approx, MGE_light=MGE_light,
        #                                            MGE_mass=False, kwargs_mge_light=kwargs_mge_light)
        self._kwargs_lens_light = kwargs_lens_light
        self._anisotropy_model = anisotropy_model
        ImageModelPosterior.__init__(self, theta_E, theta_E_error, gamma, gamma_error, r_eff, r_eff_error)

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
        j_mean_list = None
        cov_error = None

        # configuration keyword arguments for the hierarchical sampling
        kwargs_likelihood = {'z_lens': self._z_lens, 'z_source': self._z_source, 'likelihood_type': 'IFUKinCov',
                             'j_mean_list': j_mean_list,  'error_covariance': cov_error,
                             'ani_param_array': ani_param_array, 'ani_scaling_array_list': ani_scaling_array_list}
        return kwargs_likelihood


# compute covariance matrix in J_0 calculation in the bins
