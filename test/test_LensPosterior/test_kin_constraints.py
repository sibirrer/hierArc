from hierarc.LensPosterior.kin_constraints import KinConstraints
from lenstronomy.Analysis.kinematics_api import KinematicsAPI
from hierarc.Likelihood.hierarchy_likelihood import LensLikelihood
import numpy.testing as npt
import pytest
import unittest


class TestIFUKinPosterior(object):

    def setup(self):
        pass

    def test_likelihoodconfiguration_om(self):
        anisotropy_model = 'OM'
        kwargs_aperture = {'aperture_type': 'shell', 'r_in': 0, 'r_out': 3 / 2., 'center_ra': 0.0, 'center_dec': 0}
        kwargs_seeing = {'psf_type': 'GAUSSIAN', 'fwhm': 1.4}

        # numerical settings (not needed if power-law profiles with Hernquist light distribution is computed)
        kwargs_numerics_galkin = {'interpol_grid_num': 1000,  # numerical interpolation, should converge -> infinity
                                  'log_integration': True,
                                  # log or linear interpolation of surface brightness and mass models
                                  'max_integrate': 100,
                                  'min_integrate': 0.001}  # lower/upper bound of numerical integrals

        # redshift
        z_lens = 0.5
        z_source = 1.5

        # lens model
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        theta_E = 1.
        r_eff = 1
        gamma = 2.1

        # kwargs_model
        lens_light_model_list = ['HERNQUIST']
        lens_model_list = ['SPP']
        kwargs_model = {'lens_model_list': lens_model_list, 'lens_light_model_list': lens_light_model_list}

        # settings for kinematics calculation with KinematicsAPI of lenstronomy
        kwargs_kin_api_settings = {'multi_observations': False, 'kwargs_numerics_galkin': kwargs_numerics_galkin,
                                   'MGE_light': False, 'kwargs_mge_light': None, 'sampling_number': 1000,
                 'num_kin_sampling': 1000, 'num_psf_sampling': 100}

        kin_api = KinematicsAPI(z_lens, z_source, kwargs_model, kwargs_aperture, kwargs_seeing, anisotropy_model,
                                cosmo=cosmo, **kwargs_kin_api_settings)

        # compute kinematics with fiducial cosmology
        kwargs_lens = [{'theta_E': theta_E, 'gamma': gamma, 'center_x': 0, 'center_y': 0}]
        kwargs_lens_light = [{'Rs': r_eff * 0.551, 'amp': 1.}]
        kwargs_anisotropy = {'r_ani': r_eff}
        sigma_v = kin_api.velocity_dispersion(kwargs_lens, kwargs_lens_light, kwargs_anisotropy, r_eff=r_eff,
                                                      theta_E=theta_E, gamma=gamma, kappa_ext=0)

        # compute likelihood
        kin_constraints = KinConstraints(z_lens=z_lens, z_source=z_source, theta_E=theta_E, theta_E_error=0.01,
                                         gamma=gamma, gamma_error=0.02, r_eff=r_eff, r_eff_error=0.05,
                                         sigma_v_measured=[sigma_v],
                                         sigma_v_error_independent=[10], sigma_v_error_cov_matrix=[[100]],
                                         sigma_v_error_covariant=0,
                                         kwargs_aperture=kwargs_aperture, kwargs_seeing=kwargs_seeing,
                                         anisotropy_model=anisotropy_model, **kwargs_kin_api_settings)

        kwargs_likelihood = kin_constraints.hierarchy_configuration(num_sample_model=5)
        kwargs_likelihood['normalized'] = False
        ln_class = LensLikelihood(**kwargs_likelihood)
        kwargs_kin = {'a_ani': 1}
        ln_likelihood = ln_class.lens_log_likelihood(cosmo, kwargs_lens={}, kwargs_kin=kwargs_kin)
        npt.assert_almost_equal(ln_likelihood, 0, decimal=1)


    def test_likelihoodconfiguration_gom(self):
        anisotropy_model = 'GOM'
        kwargs_aperture = {'aperture_type': 'shell', 'r_in': 0, 'r_out': 3 / 2., 'center_ra': 0.0, 'center_dec': 0}
        kwargs_seeing = {'psf_type': 'GAUSSIAN', 'fwhm': 1.4}

        # numerical settings (not needed if power-law profiles with Hernquist light distribution is computed)
        kwargs_numerics_galkin = {'interpol_grid_num': 1000,  # numerical interpolation, should converge -> infinity
                                  'log_integration': True,
                                  # log or linear interpolation of surface brightness and mass models
                                  'max_integrate': 100,
                                  'min_integrate': 0.001}  # lower/upper bound of numerical integrals

        # redshift
        z_lens = 0.5
        z_source = 1.5

        # lens model
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        theta_E = 1.
        r_eff = 1
        gamma = 2.1

        # kwargs_model
        lens_light_model_list = ['HERNQUIST']
        lens_model_list = ['SPP']
        kwargs_model = {'lens_model_list': lens_model_list, 'lens_light_model_list': lens_light_model_list}

        # settings for kinematics calculation with KinematicsAPI of lenstronomy
        kwargs_kin_api_settings = {'multi_observations': False, 'kwargs_numerics_galkin': kwargs_numerics_galkin,
                                   'MGE_light': False, 'kwargs_mge_light': None, 'sampling_number': 1000,
                 'num_kin_sampling': 1000, 'num_psf_sampling': 100}

        kin_api = KinematicsAPI(z_lens, z_source, kwargs_model, kwargs_aperture, kwargs_seeing, anisotropy_model,
                                cosmo=cosmo, **kwargs_kin_api_settings)

        # compute kinematics with fiducial cosmology
        kwargs_lens = [{'theta_E': theta_E, 'gamma': gamma, 'center_x': 0, 'center_y': 0}]
        kwargs_lens_light = [{'Rs': r_eff * 0.551, 'amp': 1.}]
        beta_inf = 0.5
        kwargs_anisotropy = {'r_ani': r_eff, 'beta_inf': beta_inf}
        sigma_v = kin_api.velocity_dispersion(kwargs_lens, kwargs_lens_light, kwargs_anisotropy, r_eff=r_eff,
                                                      theta_E=theta_E, gamma=gamma, kappa_ext=0)

        # compute likelihood
        kin_constraints = KinConstraints(z_lens=z_lens, z_source=z_source, theta_E=theta_E, theta_E_error=0.01,
                                         gamma=gamma, gamma_error=0.02, r_eff=r_eff, r_eff_error=0.05, sigma_v_measured=[sigma_v],
                                         sigma_v_error_independent=[10], sigma_v_error_covariant=0,
                                         kwargs_aperture=kwargs_aperture, kwargs_seeing=kwargs_seeing,
                                         anisotropy_model=anisotropy_model, **kwargs_kin_api_settings)

        kwargs_likelihood = kin_constraints.hierarchy_configuration(num_sample_model=5)
        kwargs_likelihood['normalized'] = False
        ln_class = LensLikelihood(**kwargs_likelihood)
        kwargs_kin = {'a_ani': 1, 'beta_inf': beta_inf}
        ln_likelihood = ln_class.lens_log_likelihood(cosmo, kwargs_lens={}, kwargs_kin=kwargs_kin)
        npt.assert_almost_equal(ln_likelihood, 0, decimal=1)


class TestRaise(unittest.TestCase):

    def test_raise(self):

        with self.assertRaises(ValueError):
            anisotropy_model = 'GOM'
            kwargs_aperture = {'aperture_type': 'shell', 'r_in': 0, 'r_out': 3 / 2., 'center_ra': 0.0, 'center_dec': 0}
            kwargs_seeing = {'psf_type': 'GAUSSIAN', 'fwhm': 1.4}

            # numerical settings (not needed if power-law profiles with Hernquist light distribution is computed)
            kwargs_numerics_galkin = {'interpol_grid_num': 1000,  # numerical interpolation, should converge -> infinity
                                      'log_integration': True,
                                      # log or linear interpolation of surface brightness and mass models
                                      'max_integrate': 100,
                                      'min_integrate': 0.001}  # lower/upper bound of numerical integrals

            # redshift
            z_lens = 0.5
            z_source = 1.5
            theta_E = 1.
            r_eff = 1
            gamma = 2.1

            # settings for kinematics calculation with KinematicsAPI of lenstronomy
            kwargs_kin_api_settings = {'multi_observations': False, 'kwargs_numerics_galkin': kwargs_numerics_galkin,
                                       'MGE_light': False, 'kwargs_mge_light': None, 'sampling_number': 1000,
                                       'num_kin_sampling': 1000, 'num_psf_sampling': 100}
            kin_constraints = KinConstraints(z_lens=z_lens, z_source=z_source, theta_E=theta_E, theta_E_error=0.01,
                                             gamma=gamma, gamma_error=0.02, r_eff=r_eff, r_eff_error=0.05,
                                             sigma_v_measured=[200],
                                             sigma_v_error_independent=[10], sigma_v_error_covariant=0,
                                             kwargs_aperture=kwargs_aperture, kwargs_seeing=kwargs_seeing,
                                             anisotropy_model=anisotropy_model, **kwargs_kin_api_settings)
            kin_constraints._anisotropy_model = 'BAD'
            kin_constraints._anisotropy_scaling_relative(j_ani_0=1)


if __name__ == '__main__':
    pytest.main()
