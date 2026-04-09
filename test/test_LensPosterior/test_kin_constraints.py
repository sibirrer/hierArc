from hierarc.LensPosterior.kin_constraints import KinConstraints
from hierarc.Likelihood.hierarchy_likelihood import LensLikelihood
from lenstronomy.Analysis.kinematics_api import KinematicsAPI
from lenstronomy.Util.param_util import phi_q2_ellipticity
from astropy.cosmology import FlatLambdaCDM
import numpy.testing as npt
import numpy as np
import pytest


class TestKinConstraints(object):
    def setup_method(self):
        pass

    def test_likelihoodconfiguration_om(self):
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
            "interpol_grid_num": 1000,  # numerical interpolation, should converge -> infinity
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

        # kwargs_model
        lens_light_model_list = ["HERNQUIST"]
        lens_model_list = ["SPP"]
        kwargs_model = {
            "lens_model_list": lens_model_list,
            "lens_light_model_list": lens_light_model_list,
        }

        # settings for kinematics calculation with KinematicsAPI of lenstronomy
        kwargs_kin_api_settings = {
            "multi_observations": False,
            "kwargs_numerics_galkin": kwargs_numerics_galkin,
            "MGE_light": False,
            "kwargs_mge_light": None,
            "sampling_number": 1000,
            "num_kin_sampling": 1000,
            "num_psf_sampling": 100,
        }

        kin_api = KinematicsAPI(
            z_lens,
            z_source,
            kwargs_model,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_seeing,
            anisotropy_model=anisotropy_model,
            cosmo=cosmo,
            kinematics_backend="galkin",
            **kwargs_kin_api_settings
        )

        # compute kinematics with fiducial cosmology
        kwargs_lens = [
            {"theta_E": theta_E, "gamma": gamma, "center_x": 0, "center_y": 0}
        ]
        beta_inf = 0.9
        kwargs_lens_light = [{"Rs": r_eff * 0.551, "amp": 1.0}]
        kwargs_anisotropy = {"r_ani": r_eff, "beta_inf": beta_inf}
        sigma_v = kin_api.velocity_dispersion(
            kwargs_lens,
            kwargs_lens_light,
            kwargs_anisotropy,
            r_eff=r_eff,
            theta_E=theta_E,
            gamma=gamma,
            kappa_ext=0,
        )

        # compute likelihood
        kin_constraints = KinConstraints(
            z_lens=z_lens,
            z_source=z_source,
            theta_E=theta_E,
            theta_E_error=0.01,
            gamma=gamma,
            gamma_error=0.02,
            r_eff=r_eff,
            r_eff_error=0.05,
            sigma_v_measured=[sigma_v],
            sigma_v_error_independent=[10],
            sigma_v_error_cov_matrix=[[100]],
            sigma_v_error_covariant=0,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_seeing,
            anisotropy_model=anisotropy_model,
            gamma_pl_scaling=np.linspace(1.8, 2.2, 5),
            kinematics_backend="galkin",
            axial_symmetry="spherical",
            **kwargs_kin_api_settings
        )

        kwargs_likelihood = kin_constraints.hierarchy_configuration(num_sample_model=5)
        kwargs_likelihood["normalized"] = False
        ln_class = LensLikelihood(gamma_pl_index=0, **kwargs_likelihood)
        kwargs_kin = {"a_ani": 1, "beta_inf": beta_inf}
        kwargs_lens = {"gamma_pl_list": [gamma]}
        ln_likelihood = ln_class.lens_log_likelihood(
            cosmo, kwargs_lens=kwargs_lens, kwargs_kin=kwargs_kin
        )
        npt.assert_almost_equal(ln_likelihood, 0, decimal=1)

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
            "interpol_grid_num": 1000,  # numerical interpolation, should converge -> infinity
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

        # kwargs_model
        lens_light_model_list = ["HERNQUIST"]
        lens_model_list = ["SPP"]
        kwargs_model = {
            "lens_model_list": lens_model_list,
            "lens_light_model_list": lens_light_model_list,
        }

        # settings for kinematics calculation with KinematicsAPI of lenstronomy
        kwargs_kin_api_settings = {
            "multi_observations": False,
            "kwargs_numerics_galkin": kwargs_numerics_galkin,
            "MGE_light": False,
            "kwargs_mge_light": None,
            "sampling_number": 1000,
            "num_kin_sampling": 1000,
            "num_psf_sampling": 100,
        }

        kin_api = KinematicsAPI(
            z_lens,
            z_source,
            kwargs_model,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_seeing,
            anisotropy_model=anisotropy_model,
            cosmo=cosmo,
            kinematics_backend="galkin",
            **kwargs_kin_api_settings
        )

        # compute kinematics with fiducial cosmology
        kwargs_lens = [
            {"theta_E": theta_E, "gamma": gamma, "center_x": 0, "center_y": 0}
        ]
        kwargs_lens_light = [{"Rs": r_eff * 0.551, "amp": 1.0}]
        beta_inf = 0.5
        kwargs_anisotropy = {"r_ani": r_eff, "beta_inf": beta_inf}
        sigma_v = kin_api.velocity_dispersion(
            kwargs_lens,
            kwargs_lens_light,
            kwargs_anisotropy,
            r_eff=r_eff,
            theta_E=theta_E,
            gamma=gamma,
            kappa_ext=0,
        )

        # compute likelihood
        kin_constraints = KinConstraints(
            z_lens=z_lens,
            z_source=z_source,
            theta_E=theta_E,
            theta_E_error=0.01,
            gamma=gamma,
            gamma_error=0.02,
            r_eff=r_eff,
            r_eff_error=0.05,
            sigma_v_measured=[sigma_v],
            sigma_v_error_independent=[10],
            sigma_v_error_covariant=0,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_seeing,
            anisotropy_model=anisotropy_model,
            kinematics_backend="galkin",
            axial_symmetry="spherical",
            **kwargs_kin_api_settings
        )

        kwargs_likelihood = kin_constraints.hierarchy_configuration(num_sample_model=5)
        kwargs_likelihood["normalized"] = False
        ln_class = LensLikelihood(**kwargs_likelihood)
        kwargs_kin = {"a_ani": 1, "beta_inf": beta_inf}
        ln_likelihood = ln_class.lens_log_likelihood(
            cosmo, kwargs_lens={}, kwargs_kin=kwargs_kin
        )
        npt.assert_almost_equal(ln_likelihood, 0, decimal=1)

    def test_likelihoodconfiguration_sersic(self):
        # sersic must be with jampy backend
        anisotropy_model = "const"
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
            "interpol_grid_num": 1000,  # numerical interpolation, should converge -> infinity
            "log_integration": True,
            # log or linear interpolation of surface brightness and mass models
            "max_integrate": 100,
            "min_integrate": 0.001,
        }  # lower/upper bound of numerical integrals

        # redshift
        z_lens = 0.5
        z_source = 1.5

        # lens model
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        theta_E = 1.0
        r_eff = 1
        gamma = 2.1

        # kwargs_model
        lens_light_model_list = ["SERSIC_ELLIPSE"]
        lens_model_list = ["EPL"]
        kwargs_model = {
            "lens_model_list": lens_model_list,
            "lens_light_model_list": lens_light_model_list,
        }

        kin_api = KinematicsAPI(
            z_lens,
            z_source,
            kwargs_model,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_seeing,
            anisotropy_model=anisotropy_model,
            cosmo=cosmo,
            kinematics_backend="jampy",
        )

        # compute kinematics with fiducial cosmology
        kwargs_lens = [
            {"theta_E": theta_E, "gamma": gamma, "center_x": 0, "center_y": 0}
        ]
        kwargs_lens_light = [
            {"R_sersic": r_eff, "n_sersic": 4.0, "amp": 1.0, "e1": 0, "e2": 0}
        ]
        kwargs_anisotropy = {"beta": 0.0}
        sigma_v = kin_api.velocity_dispersion(
            kwargs_lens,
            kwargs_lens_light,
            kwargs_anisotropy,
            r_eff=r_eff,
            theta_E=theta_E,
            gamma=gamma,
            kappa_ext=0,
        )

        # compute likelihood
        kin_constraints = KinConstraints(
            z_lens=z_lens,
            z_source=z_source,
            theta_E=theta_E,
            theta_E_error=0.01,
            gamma=gamma,
            gamma_error=0.02,
            r_eff=r_eff,
            r_eff_error=0.05,
            sigma_v_measured=[sigma_v],
            sigma_v_error_independent=[10],
            sigma_v_error_covariant=0,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_seeing,
            anisotropy_model=anisotropy_model,
            kinematics_backend="jampy",
            axial_symmetry="spherical",
            lens_model_list=lens_model_list,
            lens_light_model_list=lens_light_model_list,
            kwargs_lens_light=kwargs_lens_light,
        )

        kwargs_likelihood = kin_constraints.hierarchy_configuration(num_sample_model=5)
        kwargs_likelihood["normalized"] = False
        ln_class = LensLikelihood(**kwargs_likelihood)
        kwargs_kin = {"a_ani": 0.0}
        ln_likelihood = ln_class.lens_log_likelihood(
            cosmo, kwargs_lens={}, kwargs_kin=kwargs_kin
        )
        npt.assert_almost_equal(ln_likelihood, 0, decimal=1)

    def test_likelihoodconfiguration_const_axisymmetric(self):
        anisotropy_model = "GOM"
        kwargs_aperture = {
            "aperture_type": "shell",
            "r_in": 0,
            "r_out": 3 / 2.0,
            "center_ra": 0.0,
            "center_dec": 0,
        }
        kwargs_seeing = {"psf_type": "GAUSSIAN", "fwhm": 1.4}

        # redshift
        z_lens = 0.5
        z_source = 1.5

        # lens model
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        theta_E = 1.0
        r_eff = 1
        gamma = 2.1

        # anisotropy
        beta_inf = 0.5
        kwargs_anisotropy = {"r_ani": r_eff, "beta_inf": beta_inf}

        # axial symmetry
        axial_symmetry = "axi_sph"
        q_intrinsic = 0.80
        q_observed = 0.86
        cos_i_squared = (q_observed**2 - q_intrinsic**2) / (1 - q_intrinsic**2)
        cos_i_squared = np.clip(cos_i_squared, 0, 1)
        inclination = np.rad2deg(np.arccos(np.sqrt(cos_i_squared)))
        e1, e2 = phi_q2_ellipticity(0, q_observed)

        # kwargs_model
        lens_light_model_list = ["HERNQUIST"]
        lens_model_list = ["SPP"]
        kwargs_model = {
            "lens_model_list": lens_model_list,
            "lens_light_model_list": lens_light_model_list,
        }

        kin_api = KinematicsAPI(
            z_lens,
            z_source,
            kwargs_model,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_seeing,
            anisotropy_model=anisotropy_model,
            axial_symmetry=axial_symmetry,
            kinematics_backend="jampy",
            cosmo=cosmo,
        )

        # compute kinematics with fiducial cosmology
        kwargs_lens = [
            {
                "theta_E": theta_E,
                "gamma": gamma,
                "center_x": 0,
                "center_y": 0,
                "e1": e1,
                "e2": e2,
            }
        ]
        kwargs_lens_light = [
            {
                "Rs": r_eff * 0.551,
                "amp": 1.0,
                "center_x": 0,
                "center_y": 0,
                "e1": e1,
                "e2": e2,
            }
        ]
        sigma_v = kin_api.velocity_dispersion(
            kwargs_lens,
            kwargs_lens_light,
            kwargs_anisotropy,
            inclination=inclination,
            r_eff=r_eff,
            theta_E=theta_E,
            gamma=gamma,
            kappa_ext=0,
        )

        # compute likelihood
        kin_constraints = KinConstraints(
            z_lens=z_lens,
            z_source=z_source,
            theta_E=theta_E,
            theta_E_error=0.01,
            gamma=gamma,
            gamma_error=0.02,
            r_eff=r_eff,
            r_eff_error=0.05,
            sigma_v_measured=[sigma_v],
            sigma_v_error_independent=[10],
            sigma_v_error_cov_matrix=[[100]],
            sigma_v_error_covariant=0,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_seeing,
            anisotropy_model=anisotropy_model,
            gamma_pl_scaling=np.linspace(2.0, 2.2, 3),
            # axisymmetric JAM modeling
            axial_symmetry=axial_symmetry,
            kinematics_backend="jampy",
            q_intrinsic_scaling=np.linspace(0.6, 1.0, 5),
            kwargs_lens_light=kwargs_lens_light,
        )
        kwargs_likelihood = kin_constraints.hierarchy_configuration(num_sample_model=5)
        kwargs_likelihood["normalized"] = False
        ln_class = LensLikelihood(
            gamma_pl_index=0,
            q_intrinsic_sampling=True,
            q_intrinsic_distribution="NONE",  # fixed q_intrinsic for likelihood evaluation
            **kwargs_likelihood
        )
        kwargs_kin = {"a_ani": 1, "beta_inf": beta_inf}
        kwargs_lens = {"gamma_pl_list": [gamma]}
        kwargs_deprojection = {"q_intrinsic": q_intrinsic}
        ln_likelihood = ln_class.lens_log_likelihood(
            cosmo,
            kwargs_lens=kwargs_lens,
            kwargs_kin=kwargs_kin,
            kwargs_deprojection=kwargs_deprojection,
        )
        npt.assert_almost_equal(ln_likelihood, 0, decimal=1)

    def test_likelihoodconfiguration_voronoi_bins(self):
        anisotropy_model = "const"
        x = y = np.linspace(-5, 5, 20)
        x_grid, y_grid = np.meshgrid(x, y)
        voronoi_bins = np.ones_like(x_grid) * -1
        voronoi_bins[3:-2, 3:-2] = np.kron(
            np.arange(25).reshape(5, 5), np.ones((3, 3))
        ).astype(int)
        kwargs_aperture = {
            "aperture_type": "IFU_binned",
            "x_grid": x_grid,
            "y_grid": y_grid,
            "bins": voronoi_bins,
        }
        kwargs_seeing = {"psf_type": "GAUSSIAN", "fwhm": 1.4}

        # numerical settings (not needed if power-law profiles with Hernquist light distribution is computed)
        kwargs_numerics_galkin = {
            "interpol_grid_num": 1000,  # numerical interpolation, should converge -> infinity
            "log_integration": True,
            # log or linear interpolation of surface brightness and mass models
            "max_integrate": 100,
            "min_integrate": 0.001,
        }  # lower/upper bound of numerical integrals

        # redshift
        z_lens = 0.5
        z_source = 1.5

        # lens model
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        theta_E = 1.0
        r_eff = 1
        gamma = 2.1

        # kwargs_model
        lens_light_model_list = ["HERNQUIST"]
        lens_model_list = ["SPP"]
        kwargs_model = {
            "lens_model_list": lens_model_list,
            "lens_light_model_list": lens_light_model_list,
        }

        # settings for kinematics calculation with KinematicsAPI of lenstronomy
        kwargs_kin_api_settings = {
            "multi_observations": False,
            "kwargs_numerics_galkin": kwargs_numerics_galkin,
            "MGE_light": False,
            "kwargs_mge_light": None,
            "sampling_number": 1000,
            "num_kin_sampling": 1000,
            "num_psf_sampling": 100,
        }

        kin_api = KinematicsAPI(
            z_lens,
            z_source,
            kwargs_model,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_seeing,
            anisotropy_model=anisotropy_model,
            cosmo=cosmo,
            kinematics_backend="galkin",
            **kwargs_kin_api_settings
        )

        # compute kinematics with fiducial cosmology
        kwargs_lens = [
            {"theta_E": theta_E, "gamma": gamma, "center_x": 0, "center_y": 0}
        ]
        beta_inf = 0.9
        kwargs_lens_light = [{"Rs": r_eff * 0.551, "amp": 1.0}]
        kwargs_anisotropy = {"beta": 0.0}
        sigma_v_map_voronoi = kin_api.velocity_dispersion_map(
            kwargs_lens,
            kwargs_lens_light,
            kwargs_anisotropy,
            r_eff=r_eff,
            theta_E=theta_E,
            gamma=gamma,
            kappa_ext=0,
        )

        # compute likelihood
        kin_constraints = KinConstraints(
            z_lens=z_lens,
            z_source=z_source,
            theta_E=theta_E,
            theta_E_error=0.01,
            gamma=gamma,
            gamma_error=0.02,
            r_eff=r_eff,
            r_eff_error=0.05,
            sigma_v_measured=sigma_v_map_voronoi,
            sigma_v_error_independent=np.ones(sigma_v_map_voronoi.size) * 10,
            sigma_v_error_cov_matrix=np.eye(sigma_v_map_voronoi.size) * 100,
            sigma_v_error_covariant=0,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_seeing,
            anisotropy_model=anisotropy_model,
            gamma_pl_scaling=np.linspace(1.8, 2.2, 5),
            kinematics_backend="galkin",
            axial_symmetry="spherical",
            **kwargs_kin_api_settings
        )

        kwargs_likelihood = kin_constraints.hierarchy_configuration(num_sample_model=5)
        assert len(kwargs_likelihood["j_kin_scaling_grid_list"]) == len(
            np.unique(voronoi_bins[voronoi_bins > -1])
        )

    def test_likelihoodconfiguration_multiobs(self):
        anisotropy_model = "const"
        kwargs_aperture = {
            "aperture_type": "shell",
            "r_in": 0,
            "r_out": 3 / 2.0,
            "center_ra": 0.0,
            "center_dec": 0,
        }
        kwargs_seeing = {"psf_type": "GAUSSIAN", "fwhm": 1.4}

        # redshift
        z_lens = 0.5
        z_source = 1.5

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        theta_E = 1.0
        r_eff = 1
        gamma = 2.1

        # kwargs_model
        lens_light_model_list = ["HERNQUIST"]
        lens_model_list = ["SPP"]
        kwargs_model = {
            "lens_model_list": lens_model_list,
            "lens_light_model_list": lens_light_model_list,
        }

        # settings for kinematics calculation with KinematicsAPI of lenstronomy
        kwargs_kin_api_settings = {
            "multi_observations": True,
        }

        kin_api = KinematicsAPI(
            z_lens,
            z_source,
            kwargs_model,
            kwargs_aperture=[kwargs_aperture, kwargs_aperture],
            kwargs_seeing=[kwargs_seeing, kwargs_seeing],
            anisotropy_model=anisotropy_model,
            cosmo=cosmo,
            kinematics_backend="jampy",
            **kwargs_kin_api_settings
        )

        # compute kinematics with fiducial cosmology
        kwargs_lens = [
            {"theta_E": theta_E, "gamma": gamma, "center_x": 0, "center_y": 0}
        ]
        kwargs_lens_light = [
            {"Rs": r_eff * 0.551, "amp": 1.0},
        ]
        kwargs_anisotropy = {"beta": 0.0}
        sigma_v = kin_api.velocity_dispersion(
            kwargs_lens,
            kwargs_lens_light,
            kwargs_anisotropy,
            r_eff=r_eff,
            theta_E=theta_E,
            gamma=gamma,
            kappa_ext=0,
        )

        # compute likelihood
        kin_constraints = KinConstraints(
            z_lens=z_lens,
            z_source=z_source,
            theta_E=theta_E,
            theta_E_error=0.01,
            gamma=gamma,
            gamma_error=0.02,
            r_eff=r_eff,
            r_eff_error=0.05,
            sigma_v_measured=[sigma_v, sigma_v],
            sigma_v_error_independent=[10, 10],
            sigma_v_error_cov_matrix=[[100, 0], [0, 100]],
            sigma_v_error_covariant=0,
            kwargs_aperture=[kwargs_aperture, kwargs_aperture],
            kwargs_seeing=[kwargs_seeing, kwargs_seeing],
            anisotropy_model=anisotropy_model,
            kinematics_backend="jampy",
            axial_symmetry="spherical",
            **kwargs_kin_api_settings
        )

        kwargs_likelihood = kin_constraints.hierarchy_configuration(num_sample_model=5)
        kwargs_likelihood["normalized"] = False
        ln_class = LensLikelihood(gamma_pl_index=0, **kwargs_likelihood)
        kwargs_kin = {"a_ani": 0.0}
        kwargs_lens = {"gamma_pl_list": [gamma]}
        ln_likelihood = ln_class.lens_log_likelihood(
            cosmo, kwargs_lens=kwargs_lens, kwargs_kin=kwargs_kin
        )
        npt.assert_almost_equal(ln_likelihood, 0, decimal=1)


class TestRaise(object):
    def test_raise(self):
        with pytest.raises(ValueError):
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
                "interpol_grid_num": 1000,  # numerical interpolation, should converge -> infinity
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

            # settings for kinematics calculation with KinematicsAPI of lenstronomy
            kwargs_kin_api_settings = {
                "multi_observations": False,
                "kwargs_numerics_galkin": kwargs_numerics_galkin,
                "MGE_light": False,
                "kwargs_mge_light": None,
                "sampling_number": 1000,
                "num_kin_sampling": 1000,
                "num_psf_sampling": 100,
            }
            kin_constraints = KinConstraints(
                z_lens=z_lens,
                z_source=z_source,
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
                kinematics_backend="galkin",
                **kwargs_kin_api_settings
            )
            kin_constraints._anisotropy_model = "BAD"
            kin_constraints._anisotropy_scaling_relative(j_ani_0=1)

    def test_kwargs_lens_light(self):
        with pytest.raises(ValueError, match="kwargs_lens_light"):
            # redshift
            z_lens = 0.5
            z_source = 1.5
            theta_E = 1.0
            r_eff = 1
            gamma = 2.1
            KinConstraints(
                z_lens=z_lens,
                z_source=z_source,
                theta_E=theta_E,
                theta_E_error=0.01,
                gamma=gamma,
                gamma_error=0.02,
                r_eff=r_eff,
                r_eff_error=0.05,
                sigma_v_measured=[200],
                sigma_v_error_independent=[10],
                sigma_v_error_covariant=0,
                kwargs_aperture={},
                kwargs_seeing={},
                anisotropy_model="const",
                kinematics_backend="jampy",
                axial_symmetry="axi_sph",
                kwargs_lens_light=None,  # this should be provided
            )

    def test_kwargs_lens_ellip(self):
        with pytest.raises(ValueError, match="ellipticities"):
            # redshift
            z_lens = 0.5
            z_source = 1.5
            theta_E = 1.0
            r_eff = 1
            gamma = 2.1
            KinConstraints(
                z_lens=z_lens,
                z_source=z_source,
                theta_E=theta_E,
                theta_E_error=0.01,
                gamma=gamma,
                gamma_error=0.02,
                r_eff=r_eff,
                r_eff_error=0.05,
                sigma_v_measured=[200],
                sigma_v_error_independent=[10],
                sigma_v_error_covariant=0,
                kwargs_aperture={},
                kwargs_seeing={},
                anisotropy_model="const",
                kinematics_backend="jampy",
                axial_symmetry="axi_sph",
                kwargs_lens_light=[{}],  # does not have e1, e2
            )


if __name__ == "__main__":
    pytest.main()
