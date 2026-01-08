from hierarc.JAMLensPosterior.jam_kin_constraints import JAMKinConstraints as KinConstraints
from hierarc.JAM.jam_kinematics_api import JAMKinematicsAPI
from hierarc.JAMLikelihood.jam_hierarchy_likelihood import JAMLensLikelihood
from lenstronomy.Util.param_util import phi_q2_ellipticity
import numpy.testing as npt
import numpy as np
import pytest


class TestKinConstraints(object):
    def setup_method(self):
        pass

    def test_likelihoodconfiguration_om_axisymmetric(self):
        anisotropy_model = "const"
        kwargs_aperture = {
            "aperture_type": "shell",
            "r_in": 0,
            "r_out": 3 / 2.0,
            "center_ra": 0.0,
            "center_dec": 0,
        }
        kwargs_seeing = {"psf_type": "GAUSSIAN", "fwhm": 1.4}

        kwargs_numeric_jam = {
            "mge_n_gauss_light": 20,
            "mge_n_gauss_mass": 20,
        }

        # redshift
        z_lens = 0.5
        z_source = 1.5

        # lens model
        from astropy.cosmology import FlatLambdaCDM

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        theta_E = 1.0
        r_eff = 1
        gamma = 2.1

        # anisotropy
        anisotropy_beta = 0.1

        # axial symmetry
        axial_symmetry = "axi_sph"
        q_intrinsic = 0.85
        q_observed = 0.86
        e1, e2 = phi_q2_ellipticity(0, q_observed)

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
            "kwargs_numerics_jam": kwargs_numeric_jam,
            "MGE_light": False,
            "kwargs_mge_light": None,
        }

        kin_api = JAMKinematicsAPI(
            z_lens,
            z_source,
            kwargs_model,
            kwargs_aperture,
            kwargs_seeing,
            anisotropy_model,
            axial_symmetry="axi_sph",
            cosmo=cosmo,
            **kwargs_kin_api_settings
        )

        # compute kinematics with fiducial cosmology
        kwargs_lens = [
            {"theta_E": theta_E, "gamma": gamma, "center_x": 0, "center_y": 0, "e1": e1, "e2": e2}
        ]
        kwargs_lens_light = [{"Rs": r_eff * 0.551, "amp": 1.0, "center_x": 0, "center_y": 0, "e1": e1, "e2": e2}]
        kwargs_anisotropy = {"beta": anisotropy_beta}
        sigma_v = kin_api.velocity_dispersion(
            kwargs_lens,
            kwargs_lens_light,
            kwargs_anisotropy,
            q_intrinsic=q_intrinsic,
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
            # axisymmetric JAM modeling
            axial_symmetry="axi_sph",
            gamma_pl_scaling=np.linspace(1.8, 2.2, 2),
            q_intrinsic_scaling=np.linspace(0.4, 0.8, 3),
            kwargs_lens_light=kwargs_lens_light,
            **kwargs_kin_api_settings
        )
        kwargs_likelihood = kin_constraints.hierarchy_configuration(num_sample_model=5)
        kwargs_likelihood["normalized"] = False
        ln_class = JAMLensLikelihood(
            gamma_pl_index=0,
            q_intrinsic_global_sampling=True,
            q_intrinsic_global_distribution="GAUSSIAN",
            deprojection_parameterization="q_intrinsic",
            **kwargs_likelihood
        )
        kwargs_kin = {"a_ani": anisotropy_beta}
        kwargs_lens = {"gamma_pl_list": [gamma]}
        kwargs_deprojection = {"q_intrinsic": 0.6, "q_intrinsic_sigma": 0.1}
        ln_likelihood = ln_class.lens_log_likelihood(
            cosmo, kwargs_lens=kwargs_lens, kwargs_kin=kwargs_kin, kwargs_deprojection=kwargs_deprojection
        )
        npt.assert_almost_equal(ln_likelihood, 0, decimal=1)


if __name__ == "__main__":
    pytest.main()
