from hierarc.LensPosterior.kin_constraints_composite import KinConstraintsComposite
from lenstronomy.Analysis.kinematics_api import KinematicsAPI
from hierarc.Likelihood.hierarchy_likelihood import LensLikelihood
import numpy.testing as npt
import numpy as np
import pytest
import unittest
from astropy.cosmology import FlatLambdaCDM


class TestKinConstraintsComposite(object):
    def setup(self):
        pass

    def test_likelihoodconfiguration_om(self):
        anisotropy_model = "OM"
        kwargs_aperture = {
            "aperture_type": "shell",
            "r_in": 0,
            "r_out": 3 / 2.0,
            "center_ra": 0.0,
            "center_dec": 0,
        }
        kwargs_seeing = {"psf_type": "GAUSSIAN", "fwhm": 1.4}

        # numerical settings (not needed if power-law profiles with Hernquist light distribution is computed)
        kwargs_numerics_galkin = {
            "interpol_grid_num": 50,  # numerical interpolation, should converge -> infinity
            "log_integration": True,
            # log or linear interpolation of surface brightness and mass models
            "max_integrate": 100,
            "min_integrate": 0.001,
        }  # lower/upper bound of numerical integrals

        # redshift
        z_lens = 0.5
        z_source = 1.5

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        theta_E = 1.0
        r_eff = 1
        gamma = 2.1

        kwargs_mge_light = {
            "grid_spacing": 0.1,
            "grid_num": 10,
            "n_comp": 20,
            "center_x": 0,
            "center_y": 0,
        }

        kwargs_kin_api_settings = {
            "multi_observations": False,
            "kwargs_numerics_galkin": kwargs_numerics_galkin,
            "kwargs_mge_light": kwargs_mge_light,
            "sampling_number": 50,
            "num_kin_sampling": 50,
            "num_psf_sampling": 50,
        }

        kwargs_lens_light = [
            {
                "R_sersic": 2,
                "amp": 1,
                "n_sersic": 2,
                "center_x": 0,
                "center_y": 0,
            }
        ]
        lens_light_model_list = ["SERSIC"]

        gamma_in_array = np.linspace(0.1, 2.9, 5)
        log_m2l_array = np.linspace(0.1, 1, 5)

        rho0_array = 10 ** np.random.normal(8, 0, 100) / 1e6
        r_s_array = np.random.normal(0.1, 0, 100)

        # compute likelihood
        kin_constraints = KinConstraintsComposite(
            z_lens=z_lens,
            z_source=z_source,
            gamma_in_array=gamma_in_array,
            log_m2l_array=log_m2l_array,
            kappa_s_array=rho0_array,
            r_s_angle_array=r_s_array,
            theta_E=theta_E,
            theta_E_error=0.01,
            gamma=gamma,
            gamma_error=0.02,
            r_eff=r_eff,
            r_eff_error=0.05,
            sigma_v_measured=[200],
            sigma_v_error_independent=[10],
            sigma_v_error_covariant=0,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_seeing,
            anisotropy_model=anisotropy_model,
            kwargs_lens_light=kwargs_lens_light,
            lens_light_model_list=lens_light_model_list,
            **kwargs_kin_api_settings
        )

        kin_constraints.draw_lens(no_error=True)

        kwargs_likelihood = kin_constraints.hierarchy_configuration(num_sample_model=5)
        kwargs_likelihood["normalized"] = False
        ln_class = LensLikelihood(**kwargs_likelihood)
        kwargs_kin = {"a_ani": 1}
        ln_class.lens_log_likelihood(cosmo, kwargs_lens={}, kwargs_kin=kwargs_kin)

        kwargs_lens_light_test = [{"amp": [1, 1], "sigma": [1, 2]}]
        lens_light_model_list_test = ["MULTI_GAUSSIAN"]

        kin_constraints_test = KinConstraintsComposite(
            z_lens=z_lens,
            z_source=z_source,
            gamma_in_array=gamma_in_array,
            log_m2l_array=log_m2l_array,
            kappa_s_array=rho0_array,
            r_s_angle_array=r_s_array,
            theta_E=theta_E,
            theta_E_error=0.01,
            gamma=gamma,
            gamma_error=0.02,
            r_eff=r_eff,
            r_eff_error=0.05,
            sigma_v_measured=[200],
            sigma_v_error_independent=[10],
            sigma_v_error_covariant=0,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_seeing,
            anisotropy_model=anisotropy_model,
            kwargs_lens_light=kwargs_lens_light_test,
            lens_light_model_list=lens_light_model_list_test,
            **kwargs_kin_api_settings
        )

        kappa_s_array = 10 ** np.random.normal(8, 0, 100) / 1e6
        r_s_angle_array = np.random.normal(0.1, 0, 100)

        kin_constraints_kappa = KinConstraintsComposite(
            z_lens=z_lens,
            z_source=z_source,
            gamma_in_array=gamma_in_array,
            log_m2l_array=log_m2l_array,
            kappa_s_array=[],
            r_s_angle_array=[],
            theta_E=theta_E,
            theta_E_error=0.01,
            gamma=gamma,
            gamma_error=0.02,
            r_eff=r_eff,
            r_eff_error=0.05,
            sigma_v_measured=[200],
            sigma_v_error_independent=[10],
            sigma_v_error_covariant=0,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_seeing,
            anisotropy_model=anisotropy_model,
            kwargs_lens_light=kwargs_lens_light,
            lens_light_model_list=lens_light_model_list,
            rho0_array=kappa_s_array,
            r_s_array=r_s_angle_array,
            **kwargs_kin_api_settings
        )

    def test_likelihoodconfiguration_gom(self):
        anisotropy_model = "GOM"
        kwargs_aperture = {
            "aperture_type": "shell",
            "r_in": 0,
            "r_out": 3 / 2.0,
            "center_ra": 0.0,
            "center_dec": 0,
        }
        kwargs_seeing = {"psf_type": "GAUSSIAN", "fwhm": 1.4}

        # numerical settings (not needed if power-law profiles with Hernquist light distribution is computed)
        kwargs_numerics_galkin = {
            "interpol_grid_num": 50,  # numerical interpolation, should converge -> infinity
            "log_integration": True,
            # log or linear interpolation of surface brightness and mass models
            "max_integrate": 100,
            "min_integrate": 0.001,
        }  # lower/upper bound of numerical integrals

        # redshift
        z_lens = 0.5
        z_source = 1.5

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        theta_E = 1.0
        r_eff = 1
        gamma = 2.1

        kwargs_mge_light = {
            "grid_spacing": 0.1,
            "grid_num": 10,
            "n_comp": 20,
            "center_x": 0,
            "center_y": 0,
        }

        kwargs_kin_api_settings = {
            "multi_observations": False,
            "kwargs_numerics_galkin": kwargs_numerics_galkin,
            "kwargs_mge_light": kwargs_mge_light,
            "sampling_number": 10,
            "num_kin_sampling": 10,
            "num_psf_sampling": 10,
        }

        kwargs_lens_light = [
            {
                "R_sersic": 2,
                "amp": 1,
                "n_sersic": 2,
                "center_x": 0,
                "center_y": 0,
            }
        ]
        lens_light_model_list = ["SERSIC"]

        gamma_in_array = np.linspace(0.1, 2.9, 5)
        log_m2l_array = np.linspace(0.1, 1, 5)

        rho0_array = 10 ** np.random.normal(8, 0, 100) / 1e6
        r_s_array = np.random.normal(0.1, 0, 100)

        # compute likelihood
        kin_constraints = KinConstraintsComposite(
            z_lens=z_lens,
            z_source=z_source,
            gamma_in_array=gamma_in_array,
            log_m2l_array=log_m2l_array,
            kappa_s_array=rho0_array,
            r_s_angle_array=r_s_array,
            theta_E=theta_E,
            theta_E_error=0.01,
            gamma=gamma,
            gamma_error=0.02,
            r_eff=r_eff,
            r_eff_error=0.05,
            sigma_v_measured=[200],
            sigma_v_error_independent=[10],
            sigma_v_error_covariant=0,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_seeing,
            anisotropy_model=anisotropy_model,
            kwargs_lens_light=kwargs_lens_light,
            lens_light_model_list=lens_light_model_list,
            **kwargs_kin_api_settings
        )

        kwargs_likelihood = kin_constraints.hierarchy_configuration(num_sample_model=5)
        kwargs_likelihood["normalized"] = False
        ln_class = LensLikelihood(**kwargs_likelihood)
        kwargs_kin = {"a_ani": 1, "beta_inf": 0.5}
        ln_class.lens_log_likelihood(cosmo, kwargs_lens={}, kwargs_kin=kwargs_kin)


class TestKinConstraintsCompositeM2l(object):
    def setup(self):
        pass

    def test_likelihoodconfiguration_om(self):
        anisotropy_model = "OM"
        kwargs_aperture = {
            "aperture_type": "shell",
            "r_in": 0,
            "r_out": 3 / 2.0,
            "center_ra": 0.0,
            "center_dec": 0,
        }
        kwargs_seeing = {"psf_type": "GAUSSIAN", "fwhm": 1.4}

        # numerical settings (not needed if power-law profiles with Hernquist light distribution is computed)
        kwargs_numerics_galkin = {
            "interpol_grid_num": 50,  # numerical interpolation, should converge -> infinity
            "log_integration": True,
            # log or linear interpolation of surface brightness and mass models
            "max_integrate": 100,
            "min_integrate": 0.001,
        }  # lower/upper bound of numerical integrals

        # redshift
        z_lens = 0.5
        z_source = 1.5

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        theta_E = 1.0
        r_eff = 1
        gamma = 2.1

        kwargs_mge_light = {
            "grid_spacing": 0.1,
            "grid_num": 10,
            "n_comp": 20,
            "center_x": 0,
            "center_y": 0,
        }

        kwargs_kin_api_settings = {
            "multi_observations": False,
            "kwargs_numerics_galkin": kwargs_numerics_galkin,
            "kwargs_mge_light": kwargs_mge_light,
            "sampling_number": 50,
            "num_kin_sampling": 50,
            "num_psf_sampling": 50,
        }

        kwargs_lens_light = [
            {
                "R_sersic": 2,
                "amp": 1,
                "n_sersic": 2,
                "center_x": 0,
                "center_y": 0,
            }
        ]
        lens_light_model_list = ["SERSIC"]

        gamma_in_array = np.linspace(0.1, 2.9, 5)

        log_m2l_array = np.random.uniform(0.1, 1, 100)
        rho0_array = 10 ** np.random.normal(8, 0, 100) / 1e6
        r_s_array = np.random.normal(0.1, 0, 100)

        # compute likelihood
        kin_constraints = KinConstraintsComposite(
            z_lens=z_lens,
            z_source=z_source,
            gamma_in_array=gamma_in_array,
            log_m2l_array=log_m2l_array,
            kappa_s_array=rho0_array,
            r_s_angle_array=r_s_array,
            theta_E=theta_E,
            theta_E_error=0.01,
            gamma=gamma,
            gamma_error=0.02,
            r_eff=r_eff,
            r_eff_error=0.05,
            sigma_v_measured=[200],
            sigma_v_error_independent=[10],
            sigma_v_error_covariant=0,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_seeing,
            anisotropy_model=anisotropy_model,
            kwargs_lens_light=kwargs_lens_light,
            lens_light_model_list=lens_light_model_list,
            is_m2l_population_level=False,
            **kwargs_kin_api_settings
        )

        kin_constraints.draw_lens(no_error=True)

        kwargs_likelihood = kin_constraints.hierarchy_configuration(num_sample_model=5)
        kwargs_likelihood["normalized"] = False
        ln_class = LensLikelihood(**kwargs_likelihood)
        kwargs_kin = {"a_ani": 1}
        ln_class.lens_log_likelihood(cosmo, kwargs_lens={}, kwargs_kin=kwargs_kin)

    def test_likelihoodconfiguration_gom(self):
        anisotropy_model = "GOM"
        kwargs_aperture = {
            "aperture_type": "shell",
            "r_in": 0,
            "r_out": 3 / 2.0,
            "center_ra": 0.0,
            "center_dec": 0,
        }
        kwargs_seeing = {"psf_type": "GAUSSIAN", "fwhm": 1.4}

        # numerical settings (not needed if power-law profiles with Hernquist light distribution is computed)
        kwargs_numerics_galkin = {
            "interpol_grid_num": 50,  # numerical interpolation, should converge -> infinity
            "log_integration": True,
            # log or linear interpolation of surface brightness and mass models
            "max_integrate": 100,
            "min_integrate": 0.001,
        }  # lower/upper bound of numerical integrals

        # redshift
        z_lens = 0.5
        z_source = 1.5

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        theta_E = 1.0
        r_eff = 1
        gamma = 2.1

        kwargs_mge_light = {
            "grid_spacing": 0.1,
            "grid_num": 10,
            "n_comp": 20,
            "center_x": 0,
            "center_y": 0,
        }

        kwargs_kin_api_settings = {
            "multi_observations": False,
            "kwargs_numerics_galkin": kwargs_numerics_galkin,
            "kwargs_mge_light": kwargs_mge_light,
            "sampling_number": 10,
            "num_kin_sampling": 10,
            "num_psf_sampling": 10,
        }

        kwargs_lens_light = [
            {
                "R_sersic": 2,
                "amp": 1,
                "n_sersic": 2,
                "center_x": 0,
                "center_y": 0,
            }
        ]
        lens_light_model_list = ["SERSIC"]

        gamma_in_array = np.linspace(0.1, 2.9, 5)

        log_m2l_array = np.random.uniform(0.1, 1, 100)
        rho0_array = 10 ** np.random.normal(8, 0, 100) / 1e6
        r_s_array = np.random.normal(0.1, 0, 100)

        # compute likelihood
        kin_constraints = KinConstraintsComposite(
            z_lens=z_lens,
            z_source=z_source,
            gamma_in_array=gamma_in_array,
            log_m2l_array=log_m2l_array,
            kappa_s_array=rho0_array,
            r_s_angle_array=r_s_array,
            theta_E=theta_E,
            theta_E_error=0.01,
            gamma=gamma,
            gamma_error=0.02,
            r_eff=r_eff,
            r_eff_error=0.05,
            sigma_v_measured=[200],
            sigma_v_error_independent=[10],
            sigma_v_error_covariant=0,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_seeing,
            anisotropy_model=anisotropy_model,
            kwargs_lens_light=kwargs_lens_light,
            lens_light_model_list=lens_light_model_list,
            is_m2l_population_level=False,
            **kwargs_kin_api_settings
        )

        kwargs_likelihood = kin_constraints.hierarchy_configuration(num_sample_model=5)
        kwargs_likelihood["normalized"] = False
        ln_class = LensLikelihood(**kwargs_likelihood)
        kwargs_kin = {"a_ani": 1, "beta_inf": 0.5}
        ln_class.lens_log_likelihood(cosmo, kwargs_lens={}, kwargs_kin=kwargs_kin)


class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            anisotropy_model = "OM"
            kwargs_aperture = {
                "aperture_type": "shell",
                "r_in": 0,
                "r_out": 3 / 2.0,
                "center_ra": 0.0,
                "center_dec": 0,
            }
            kwargs_seeing = {"psf_type": "GAUSSIAN", "fwhm": 1.4}

            # numerical settings (not needed if power-law profiles with Hernquist light distribution is computed)
            kwargs_numerics_galkin = {
                "interpol_grid_num": 50,  # numerical interpolation, should converge -> infinity
                "log_integration": True,
                # log or linear interpolation of surface brightness and mass models
                "max_integrate": 100,
                "min_integrate": 0.001,
            }  # lower/upper bound of numerical integrals

            # redshift
            z_lens = 0.5
            z_source = 1.5
            theta_E = 1.0
            r_eff = 1
            gamma = 2.1

            kwargs_mge_light = {
                "grid_spacing": 0.1,
                "grid_num": 10,
                "n_comp": 20,
                "center_x": 0,
                "center_y": 0,
            }

            # settings for kinematics calculation with KinematicsAPI of lenstronomy
            kwargs_kin_api_settings = {
                "multi_observations": False,
                "kwargs_numerics_galkin": kwargs_numerics_galkin,
                "kwargs_mge_light": kwargs_mge_light,
                "sampling_number": 10,
                "num_kin_sampling": 10,
                "num_psf_sampling": 10,
            }
            kwargs_lens_light = [{"Rs": r_eff * 0.551, "amp": 1.0}]

            gamma_in_array = np.linspace(0.1, 2.9, 5)
            log_m2l_array = np.linspace(0.1, 1, 5)
            rho0_array = 10 ** np.random.normal(8, 0.2, 100) / 1e6
            r_s_array = np.random.normal(0.1, 0.01, 100)

            kin_constraints = KinConstraintsComposite(
                z_lens=z_lens,
                z_source=z_source,
                gamma_in_array=gamma_in_array,
                log_m2l_array=log_m2l_array,
                kappa_s_array=rho0_array,
                r_s_angle_array=r_s_array,
                theta_E=theta_E,
                theta_E_error=0.01,
                gamma=gamma,
                gamma_error=0.02,
                r_eff=r_eff,
                r_eff_error=0.05,
                sigma_v_measured=[200],
                sigma_v_error_independent=[10],
                sigma_v_error_covariant=0,
                kwargs_aperture=kwargs_aperture,
                kwargs_seeing=kwargs_seeing,
                anisotropy_model=anisotropy_model,
                kwargs_lens_light=kwargs_lens_light,
                **kwargs_kin_api_settings
            )
            kin_constraints._anisotropy_model = "BAD"
            kin_constraints._anisotropy_scaling_relative(j_ani_0=1)

        with self.assertRaises(ValueError):
            anisotropy_model = "OM"
            kwargs_aperture = {
                "aperture_type": "shell",
                "r_in": 0,
                "r_out": 3 / 2.0,
                "center_ra": 0.0,
                "center_dec": 0,
            }
            kwargs_seeing = {"psf_type": "GAUSSIAN", "fwhm": 1.4}

            # numerical settings (not needed if power-law profiles with Hernquist light distribution is computed)
            kwargs_numerics_galkin = {
                "interpol_grid_num": 100,  # numerical interpolation, should converge -> infinity
                "log_integration": True,
                # log or linear interpolation of surface brightness and mass models
                "max_integrate": 100,
                "min_integrate": 0.001,
            }  # lower/upper bound of numerical integrals

            # redshift
            z_lens = 0.5
            z_source = 1.5
            theta_E = 1.0
            r_eff = 1
            gamma = 2.1

            kwargs_mge_light = {
                "grid_spacing": 0.1,
                "grid_num": 10,
                "n_comp": 20,
                "center_x": 0,
                "center_y": 0,
            }

            # settings for kinematics calculation with KinematicsAPI of lenstronomy
            kwargs_kin_api_settings = {
                "multi_observations": False,
                "kwargs_numerics_galkin": kwargs_numerics_galkin,
                "kwargs_mge_light": kwargs_mge_light,
                "sampling_number": 50,
                "num_kin_sampling": 50,
                "num_psf_sampling": 50,
            }

            kwargs_lens_light = [{"Rs": r_eff * 0.551, "amp": 1.0}]
            gamma_in_array = np.linspace(0.1, 2.9, 5)
            log_m2l_array = np.linspace(0.1, 1, 5)

            rho0_array = 10 ** np.random.normal(8, 0.2, 5) / 1e6
            r_s_array = np.random.normal(0.1, 0.01, 6)

            KinConstraintsComposite(
                z_lens=z_lens,
                z_source=z_source,
                gamma_in_array=gamma_in_array,
                log_m2l_array=log_m2l_array,
                kappa_s_array=rho0_array,
                r_s_angle_array=r_s_array,
                theta_E=theta_E,
                theta_E_error=0.01,
                gamma=gamma,
                gamma_error=0.02,
                r_eff=r_eff,
                r_eff_error=0.05,
                sigma_v_measured=[200],
                sigma_v_error_independent=[10],
                sigma_v_error_covariant=0,
                kwargs_aperture=kwargs_aperture,
                kwargs_seeing=kwargs_seeing,
                anisotropy_model=anisotropy_model,
                kwargs_lens_light=kwargs_lens_light,
                **kwargs_kin_api_settings
            )

        with self.assertRaises(ValueError):
            anisotropy_model = "OM"
            kwargs_aperture = {
                "aperture_type": "shell",
                "r_in": 0,
                "r_out": 3 / 2.0,
                "center_ra": 0.0,
                "center_dec": 0,
            }
            kwargs_seeing = {"psf_type": "GAUSSIAN", "fwhm": 1.4}

            # numerical settings (not needed if power-law profiles with Hernquist light distribution is computed)
            kwargs_numerics_galkin = {
                "interpol_grid_num": 50,  # numerical interpolation, should converge -> infinity
                "log_integration": True,
                # log or linear interpolation of surface brightness and mass models
                "max_integrate": 100,
                "min_integrate": 0.001,
            }  # lower/upper bound of numerical integrals

            # redshift
            z_lens = 0.5
            z_source = 1.5
            theta_E = 1.0
            r_eff = 1
            gamma = 2.1

            kwargs_mge_light = {
                "grid_spacing": 0.1,
                "grid_num": 10,
                "n_comp": 20,
                "center_x": 0,
                "center_y": 0,
            }

            # settings for kinematics calculation with KinematicsAPI of lenstronomy
            kwargs_kin_api_settings = {
                "multi_observations": False,
                "kwargs_numerics_galkin": kwargs_numerics_galkin,
                "kwargs_mge_light": kwargs_mge_light,
                "sampling_number": 10,
                "num_kin_sampling": 10,
                "num_psf_sampling": 10,
            }
            kwargs_lens_light = [{"Rs": r_eff * 0.551, "amp": 1.0}]

            gamma_in_array = np.linspace(0.1, 2.9, 5)
            log_m2l_array = np.linspace(0.1, 1, 5)
            kappa_s_array = 10 ** np.random.normal(8, 0.2, 100) / 1e6
            r_s_angle_array = np.random.normal(0.1, 0.01, 101)

            kin_constraints = KinConstraintsComposite(
                z_lens=z_lens,
                z_source=z_source,
                gamma_in_array=gamma_in_array,
                log_m2l_array=log_m2l_array,
                kappa_s_array=[],
                r_s_angle_array=[],
                theta_E=theta_E,
                theta_E_error=0.01,
                gamma=gamma,
                gamma_error=0.02,
                r_eff=r_eff,
                r_eff_error=0.05,
                sigma_v_measured=[200],
                sigma_v_error_independent=[10],
                sigma_v_error_covariant=0,
                kwargs_aperture=kwargs_aperture,
                kwargs_seeing=kwargs_seeing,
                anisotropy_model=anisotropy_model,
                kwargs_lens_light=kwargs_lens_light,
                rho0_array=kappa_s_array,
                r_s_array=r_s_angle_array,
                **kwargs_kin_api_settings
            )

    def test_raise_m2l(self):
        with self.assertRaises(ValueError):
            anisotropy_model = "OM"
            kwargs_aperture = {
                "aperture_type": "shell",
                "r_in": 0,
                "r_out": 3 / 2.0,
                "center_ra": 0.0,
                "center_dec": 0,
            }
            kwargs_seeing = {"psf_type": "GAUSSIAN", "fwhm": 1.4}

            # numerical settings (not needed if power-law profiles with Hernquist light distribution is computed)
            kwargs_numerics_galkin = {
                "interpol_grid_num": 50,  # numerical interpolation, should converge -> infinity
                "log_integration": True,
                # log or linear interpolation of surface brightness and mass models
                "max_integrate": 100,
                "min_integrate": 0.001,
            }  # lower/upper bound of numerical integrals

            # redshift
            z_lens = 0.5
            z_source = 1.5
            theta_E = 1.0
            r_eff = 1
            gamma = 2.1

            kwargs_mge_light = {
                "grid_spacing": 0.1,
                "grid_num": 10,
                "n_comp": 20,
                "center_x": 0,
                "center_y": 0,
            }

            # settings for kinematics calculation with KinematicsAPI of lenstronomy
            kwargs_kin_api_settings = {
                "multi_observations": False,
                "kwargs_numerics_galkin": kwargs_numerics_galkin,
                "kwargs_mge_light": kwargs_mge_light,
                "sampling_number": 10,
                "num_kin_sampling": 10,
                "num_psf_sampling": 10,
            }
            kwargs_lens_light = [{"Rs": r_eff * 0.551, "amp": 1.0}]

            gamma_in_array = np.linspace(0.1, 2.9, 5)
            log_m2l_array = np.random.uniform(0.1, 1, 100)
            rho0_array = 10 ** np.random.normal(8, 0.2, 100) / 1e6
            r_s_array = np.random.normal(0.1, 0.01, 100)

            kin_constraints = KinConstraintsComposite(
                z_lens=z_lens,
                z_source=z_source,
                gamma_in_array=gamma_in_array,
                log_m2l_array=log_m2l_array,
                kappa_s_array=rho0_array,
                r_s_angle_array=r_s_array,
                theta_E=theta_E,
                theta_E_error=0.01,
                gamma=gamma,
                gamma_error=0.02,
                r_eff=r_eff,
                r_eff_error=0.05,
                sigma_v_measured=[200],
                sigma_v_error_independent=[10],
                sigma_v_error_covariant=0,
                kwargs_aperture=kwargs_aperture,
                kwargs_seeing=kwargs_seeing,
                anisotropy_model=anisotropy_model,
                kwargs_lens_light=kwargs_lens_light,
                is_m2l_population_level=False,
                **kwargs_kin_api_settings
            )
            kin_constraints._anisotropy_model = "BAD"
            kin_constraints._anisotropy_scaling_relative(j_ani_0=1)

        with self.assertRaises(ValueError):
            anisotropy_model = "OM"
            kwargs_aperture = {
                "aperture_type": "shell",
                "r_in": 0,
                "r_out": 3 / 2.0,
                "center_ra": 0.0,
                "center_dec": 0,
            }
            kwargs_seeing = {"psf_type": "GAUSSIAN", "fwhm": 1.4}

            # numerical settings (not needed if power-law profiles with Hernquist light distribution is computed)
            kwargs_numerics_galkin = {
                "interpol_grid_num": 100,  # numerical interpolation, should converge -> infinity
                "log_integration": True,
                # log or linear interpolation of surface brightness and mass models
                "max_integrate": 100,
                "min_integrate": 0.001,
            }  # lower/upper bound of numerical integrals

            # redshift
            z_lens = 0.5
            z_source = 1.5
            theta_E = 1.0
            r_eff = 1
            gamma = 2.1

            kwargs_mge_light = {
                "grid_spacing": 0.1,
                "grid_num": 10,
                "n_comp": 20,
                "center_x": 0,
                "center_y": 0,
            }

            # settings for kinematics calculation with KinematicsAPI of lenstronomy
            kwargs_kin_api_settings = {
                "multi_observations": False,
                "kwargs_numerics_galkin": kwargs_numerics_galkin,
                "kwargs_mge_light": kwargs_mge_light,
                "sampling_number": 50,
                "num_kin_sampling": 50,
                "num_psf_sampling": 50,
            }

            kwargs_lens_light = [{"Rs": r_eff * 0.551, "amp": 1.0}]
            gamma_in_array = np.linspace(0.1, 2.9, 5)

            log_m2l_array = np.random.uniform(0.1, 1, 6)
            rho0_array = 10 ** np.random.normal(8, 0.2, 5) / 1e6
            r_s_array = np.random.normal(0.1, 0.01, 5)

            KinConstraintsComposite(
                z_lens=z_lens,
                z_source=z_source,
                gamma_in_array=gamma_in_array,
                log_m2l_array=log_m2l_array,
                kappa_s_array=rho0_array,
                r_s_angle_array=r_s_array,
                theta_E=theta_E,
                theta_E_error=0.01,
                gamma=gamma,
                gamma_error=0.02,
                r_eff=r_eff,
                r_eff_error=0.05,
                sigma_v_measured=[200],
                sigma_v_error_independent=[10],
                sigma_v_error_covariant=0,
                kwargs_aperture=kwargs_aperture,
                kwargs_seeing=kwargs_seeing,
                anisotropy_model=anisotropy_model,
                kwargs_lens_light=kwargs_lens_light,
                is_m2l_population_level=False,
                **kwargs_kin_api_settings
            )

        with self.assertRaises(ValueError):
            anisotropy_model = "FAKE_MODEL"
            kwargs_aperture = {
                "aperture_type": "shell",
                "r_in": 0,
                "r_out": 3 / 2.0,
                "center_ra": 0.0,
                "center_dec": 0,
            }
            kwargs_seeing = {"psf_type": "GAUSSIAN", "fwhm": 1.4}

            # numerical settings (not needed if power-law profiles with Hernquist light distribution is computed)
            kwargs_numerics_galkin = {
                "interpol_grid_num": 100,  # numerical interpolation, should converge -> infinity
                "log_integration": True,
                # log or linear interpolation of surface brightness and mass models
                "max_integrate": 100,
                "min_integrate": 0.001,
            }  # lower/upper bound of numerical integrals

            # redshift
            z_lens = 0.5
            z_source = 1.5
            theta_E = 1.0
            r_eff = 1
            gamma = 2.1

            kwargs_mge_light = {
                "grid_spacing": 0.1,
                "grid_num": 10,
                "n_comp": 20,
                "center_x": 0,
                "center_y": 0,
            }

            # settings for kinematics calculation with KinematicsAPI of lenstronomy
            kwargs_kin_api_settings = {
                "multi_observations": False,
                "kwargs_numerics_galkin": kwargs_numerics_galkin,
                "kwargs_mge_light": kwargs_mge_light,
                "sampling_number": 50,
                "num_kin_sampling": 50,
                "num_psf_sampling": 50,
            }

            kwargs_lens_light = [{"Rs": r_eff * 0.551, "amp": 1.0}]
            gamma_in_array = np.linspace(0.1, 2.9, 5)

            log_m2l_array = np.random.uniform(0.1, 1, 5)
            rho0_array = 10 ** np.random.normal(8, 0.2, 5) / 1e6
            r_s_array = np.random.normal(0.1, 0.01, 5)

            KinConstraintsComposite(
                z_lens=z_lens,
                z_source=z_source,
                gamma_in_array=gamma_in_array,
                log_m2l_array=log_m2l_array,
                kappa_s_array=rho0_array,
                r_s_angle_array=r_s_array,
                theta_E=theta_E,
                theta_E_error=0.01,
                gamma=gamma,
                gamma_error=0.02,
                r_eff=r_eff,
                r_eff_error=0.05,
                sigma_v_measured=[200],
                sigma_v_error_independent=[10],
                sigma_v_error_covariant=0,
                kwargs_aperture=kwargs_aperture,
                kwargs_seeing=kwargs_seeing,
                anisotropy_model=anisotropy_model,
                kwargs_lens_light=kwargs_lens_light,
                is_m2l_population_level=False,
                **kwargs_kin_api_settings
            )


if __name__ == "__main__":
    pytest.main()
