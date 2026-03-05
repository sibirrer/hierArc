__author__ = "sibirrer,furcelay"

import numpy.testing as npt
import numpy as np
import pytest
import unittest

from hierarc.JAM.jam_kinematics_api import JAMKinematicsAPI
import lenstronomy.Util.param_util as param_util
from astropy.cosmology import FlatLambdaCDM


class TestJAMKinematicsAPI(object):
    def setup_method(self):
        pass

    def test_velocity_dispersion(self):
        z_lens = 0.5
        z_source = 1.5
        kwargs_model = {
            "lens_model_list": ["SPEP", "SHEAR", "SIS", "SIS", "SIS"],
            "lens_light_model_list": ["SERSIC_ELLIPSE", "SERSIC"],
        }

        theta_E = 1.5
        gamma = 1.8
        kwargs_lens = [
            {
                "theta_E": theta_E,
                "e1": 0,
                "center_x": -0.044798916793300093,
                "center_y": 0.0054408937891703788,
                "e2": 0,
                "gamma": gamma,
            },
            {"e1": -0.050871696555354479, "e2": -0.0061601733920590464},
            {
                "center_y": 2.79985456,
                "center_x": -2.32019894,
                "theta_E": 0.28165274714097904,
            },
            {
                "center_y": 3.83985426,
                "center_x": -2.32019933,
                "theta_E": 0.0038110812674654873,
            },
            {
                "center_y": 4.31985428,
                "center_x": -1.68019931,
                "theta_E": 0.45552039839735037,
            },
        ]

        phi, q = -0.52624727893702705, 0.79703498156919605
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_lens_light = [
            {
                "n_sersic": 1.1212528655709217,
                "center_x": -0.019674496231393473,
                "e1": e1,
                "e2": e2,
                "amp": 1.1091367792010356,
                "center_y": 0.076914975081560991,
                "R_sersic": 0.42691611878867058,
            },
            {
                "R_sersic": 0.03025682660635394,
                "amp": 139.96763298885992,
                "n_sersic": 1.90000008624093865,
                "center_x": -0.019674496231393473,
                "center_y": 0.076914975081560991,
            },
        ]
        r_ani = 0.62
        kwargs_anisotropy = {"r_ani": r_ani}
        R_slit = 3.8
        dR_slit = 1.0
        aperture_type = "slit"
        kwargs_aperture = {
            "aperture_type": aperture_type,
            "center_ra": 0,
            "width": dR_slit,
            "length": R_slit,
            "angle": 0,
            "center_dec": 0,
        }

        psf_fwhm = 0.7
        kwargs_psf = {"psf_type": "GAUSSIAN", "fwhm": psf_fwhm}
        anisotropy_model = "OM"
        kwargs_mge = {"n_comp": 20}
        r_eff = 0.211919902322
        kinematicAPI = JAMKinematicsAPI(
            z_lens,
            z_source,
            kwargs_model,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_psf,
            lens_model_kinematics_bool=[True, False, False, False, False],
            anisotropy_model=anisotropy_model,
            kwargs_mge_light=kwargs_mge,
            kwargs_mge_mass=kwargs_mge,
            sampling_number=1000,
            MGE_light=True,
        )

        v_sigma = kinematicAPI.velocity_dispersion(
            kwargs_lens, kwargs_lens_light, kwargs_anisotropy, r_eff=r_eff
        )

        kinematicAPI = JAMKinematicsAPI(
            z_lens,
            z_source,
            kwargs_model,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_psf,
            lens_model_kinematics_bool=[True, False, False, False, False],
            anisotropy_model=anisotropy_model,
            kwargs_mge_light=kwargs_mge,
            kwargs_mge_mass=kwargs_mge,
            sampling_number=1000,
            MGE_light=True,
            MGE_mass=True,
        )
        v_sigma_mge_lens = kinematicAPI.velocity_dispersion(
            kwargs_lens,
            kwargs_lens_light,
            kwargs_anisotropy,
            r_eff=r_eff,
            theta_E=theta_E,
        )
        # v_sigma_mge_lens = kinematicAPI.velocity_dispersion_numerical(kwargs_lens, kwargs_lens_light, kwargs_anisotropy, kwargs_aperture,
        #                                                          kwargs_psf, anisotropy_model, MGE_light=True, MGE_mass=True, theta_E=theta_E,
        #                                                          kwargs_mge_light=kwargs_mge, kwargs_mge_mass=kwargs_mge,
        #                                                          r_eff=r_eff)
        kinematicAPI = JAMKinematicsAPI(
            z_lens,
            z_source,
            kwargs_model,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_psf,
            lens_model_kinematics_bool=[True, False, False, False, False],
            anisotropy_model=anisotropy_model,
            kwargs_mge_light=kwargs_mge,
            kwargs_mge_mass=kwargs_mge,
            sampling_number=1000,
            MGE_light=False,
            MGE_mass=False,
            Hernquist_approx=True,
        )
        v_sigma_hernquist = kinematicAPI.velocity_dispersion(
            kwargs_lens,
            kwargs_lens_light,
            kwargs_anisotropy,
            r_eff=r_eff,
            theta_E=theta_E,
        )
        # v_sigma_hernquist = kinematicAPI.velocity_dispersion_numerical(kwargs_lens, kwargs_lens_light, kwargs_anisotropy,
        #                                                          kwargs_aperture, kwargs_psf, anisotropy_model,
        #                                                          MGE_light=False, MGE_mass=False,
        #                                                          r_eff=r_eff, Hernquist_approx=True)

        vel_disp_temp = kinematicAPI.velocity_dispersion_analytical(
            theta_E, gamma, r_ani=r_ani, r_eff=r_eff
        )
        # assert 1 == 0
        npt.assert_almost_equal(v_sigma / vel_disp_temp, 1, decimal=1)
        npt.assert_almost_equal(v_sigma_mge_lens / v_sigma, 1, decimal=1)
        npt.assert_almost_equal(v_sigma / v_sigma_hernquist, 1, decimal=1)

    def test_jam_settings(self):
        z_lens = 0.5
        z_source = 1.5
        kwargs_model = {
            "lens_model_list": ["SIS"],
            "lens_light_model_list": ["HERNQUIST"],
        }

        kwargs_lens = [{"theta_E": 1, "center_x": 0, "center_y": 0}]
        kwargs_lens_light = [{"amp": 1, "Rs": 1, "center_x": 0, "center_y": 0}]
        r_ani = 0.62
        kwargs_anisotropy = {"r_ani": r_ani}
        R_slit = 3.8
        dR_slit = 1.0
        aperture_type = "slit"
        kwargs_aperture = {
            "aperture_type": aperture_type,
            "center_ra": 0,
            "width": dR_slit,
            "length": R_slit,
            "angle": 0,
            "center_dec": 0,
        }

        psf_fwhm = 0.7
        kwargs_psf = {"psf_type": "GAUSSIAN", "fwhm": psf_fwhm}
        anisotropy_model = "OM"
        kwargs_mge = {"n_comp": 20}
        kinematicAPI = JAMKinematicsAPI(
            z_lens,
            z_source,
            kwargs_model,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_psf,
            analytic_kinematics=True,
            anisotropy_model=anisotropy_model,
            kwargs_mge_light=kwargs_mge,
            kwargs_mge_mass=kwargs_mge,
            sampling_number=1000,
        )
        galkin, kwargs_profile, kwargs_light = kinematicAPI.jam_settings(
            kwargs_lens, kwargs_lens_light, r_eff=None, theta_E=None, gamma=None
        )
        npt.assert_almost_equal(kwargs_profile["gamma"], 2, decimal=2)

        kinematicAPI = JAMKinematicsAPI(
            z_lens,
            z_source,
            kwargs_model,
            kwargs_aperture=[kwargs_aperture],
            kwargs_seeing=[kwargs_psf],
            analytic_kinematics=True,
            anisotropy_model=anisotropy_model,
            multi_observations=True,
            kwargs_mge_light=kwargs_mge,
            kwargs_mge_mass=kwargs_mge,
            sampling_number=1000,
        )
        galkin, kwargs_profile, kwargs_light = kinematicAPI.jam_settings(
            kwargs_lens, kwargs_lens_light, r_eff=None, theta_E=None, gamma=None
        )
        npt.assert_almost_equal(kwargs_profile["gamma"], 2, decimal=2)

        kinematicAPI = JAMKinematicsAPI(
            z_lens,
            z_source,
            kwargs_model,
            kwargs_aperture=[kwargs_aperture],
            kwargs_seeing=[kwargs_psf],
            analytic_kinematics=False,
            anisotropy_model=anisotropy_model,
            multi_observations=True,
            multi_light_profile=True,
            kwargs_mge_light=kwargs_mge,
            kwargs_mge_mass=kwargs_mge,
            sampling_number=1000,
        )
        galkin, kwargs_profile, kwargs_light = kinematicAPI.jam_settings(
            kwargs_lens, [kwargs_lens_light], r_eff=None, theta_E=None, gamma=None
        )
        npt.assert_almost_equal(kwargs_light[0][0]["Rs"], 1, decimal=2)

    def test_kinematic_light_profile(self):
        z_lens = 0.5
        z_source = 1.5
        kwargs_options = {"lens_light_model_list": ["HERNQUIST_ELLIPSE", "SERSIC"]}
        kwargs_mge = {"n_comp": 20}
        kinematicAPI = JAMKinematicsAPI(
            z_lens,
            z_source,
            kwargs_options,
            kwargs_seeing={},
            kwargs_aperture={"aperture_type": "slit"},
            anisotropy_model="OM",
        )
        r_eff = 0.2
        kwargs_lens_light = [
            {
                "amp": 1,
                "Rs": r_eff * 0.551,
                "e1": 0.0,
                "e2": 0,
                "center_x": 0,
                "center_y": 0,
            },
            {"amp": 1, "R_sersic": 1, "n_sersic": 2, "center_x": -10, "center_y": -10},
        ]
        light_profile_list, kwargs_light = kinematicAPI.kinematic_light_profile(
            kwargs_lens_light,
            MGE_fit=True,
            r_eff=r_eff,
            model_kinematics_bool=[True, False],
            kwargs_mge=kwargs_mge,
        )
        assert light_profile_list[0] == "MULTI_GAUSSIAN"

        light_profile_list, kwargs_light = kinematicAPI.kinematic_light_profile(
            kwargs_lens_light,
            MGE_fit=False,
            r_eff=r_eff,
            model_kinematics_bool=[True, False],
        )
        assert light_profile_list[0] == "HERNQUIST_ELLIPSE"

        light_profile_list, kwargs_light = kinematicAPI.kinematic_light_profile(
            kwargs_lens_light,
            MGE_fit=False,
            Hernquist_approx=True,
            r_eff=r_eff,
            model_kinematics_bool=[True, False],
        )
        assert light_profile_list[0] == "HERNQUIST"
        npt.assert_almost_equal(
            kwargs_light[0]["Rs"] / kwargs_lens_light[0]["Rs"], 1, decimal=2
        )

    def test_kinematic_lens_profiles(self):
        z_lens = 0.5
        z_source = 1.5
        kwargs_options = {"lens_model_list": ["SPEP", "SHEAR"]}
        kin_api = JAMKinematicsAPI(
            z_lens,
            z_source,
            kwargs_options,
            kwargs_aperture={"aperture_type": "slit"},
            kwargs_seeing={},
            anisotropy_model="OM",
        )
        kwargs_lens = [
            {
                "theta_E": 1.4272358196260446,
                "e1": 0,
                "center_x": -0.044798916793300093,
                "center_y": 0.0054408937891703788,
                "e2": 0,
                "gamma": 1.8,
            },
            {"e1": -0.050871696555354479, "e2": -0.0061601733920590464},
        ]

        kwargs_mge = {"n_comp": 20}
        mass_profile_list, kwargs_profile = kin_api.kinematic_lens_profiles(
            kwargs_lens,
            MGE_fit=True,
            kwargs_mge=kwargs_mge,
            theta_E=1.4,
            model_kinematics_bool=[True, False],
        )
        assert mass_profile_list[0] == "MULTI_GAUSSIAN"

        mass_profile_list, kwargs_profile = kin_api.kinematic_lens_profiles(
            kwargs_lens, MGE_fit=False, model_kinematics_bool=[True, False]
        )
        assert mass_profile_list[0] == "SPEP"

    def test_model_dispersion(self):
        np.random.seed(42)
        z_lens = 0.5
        z_source = 1.5
        r_eff = 1.0
        theta_E = 1.0
        kwargs_model = {
            "lens_model_list": ["SIS"],
            "lens_light_model_list": ["HERNQUIST"],
        }
        kwargs_lens = [{"theta_E": theta_E, "center_x": 0, "center_y": 0}]
        kwargs_lens_light = [
            {"amp": 1, "Rs": r_eff * 0.551, "center_x": 0, "center_y": 0}
        ]
        kwargs_anisotropy = {"r_ani": 1}
        # settings

        R_slit = 3.8
        dR_slit = 1.0
        aperture_type = "slit"
        kwargs_aperture = {
            "aperture_type": aperture_type,
            "center_ra": 0,
            "width": dR_slit,
            "length": R_slit,
            "angle": 0,
            "center_dec": 0,
        }
        psf_fwhm = 0.7
        kwargs_seeing = {"psf_type": "GAUSSIAN", "fwhm": psf_fwhm}
        anisotropy_model = "OM"
        kin_api = JAMKinematicsAPI(
            z_lens,
            z_source,
            kwargs_model,
            kwargs_aperture,
            kwargs_seeing,
            anisotropy_model=anisotropy_model,
        )

        kwargs_numerics_galkin = {
            "interpol_grid_num": 2000,
            "log_integration": True,
            "max_integrate": 1000,
            "min_integrate": 0.0001,
        }
        kin_api.kinematics_modeling_settings(
            anisotropy_model,
            kwargs_numerics_galkin,
            analytic_kinematics=True,
            Hernquist_approx=False,
            MGE_light=False,
            MGE_mass=False,
        )
        vel_disp_analytic = kin_api.velocity_dispersion(
            kwargs_lens,
            kwargs_lens_light,
            kwargs_anisotropy,
            r_eff=r_eff,
            theta_E=theta_E,
            gamma=2,
        )

        kin_api.kinematics_modeling_settings(
            anisotropy_model,
            kwargs_numerics_galkin,
            analytic_kinematics=False,
            Hernquist_approx=False,
            MGE_light=False,
            MGE_mass=False,
        )
        vel_disp_numerical = kin_api.velocity_dispersion(
            kwargs_lens, kwargs_lens_light, kwargs_anisotropy
        )  # ,
        # r_eff=r_eff, theta_E=theta_E, gamma=2)
        npt.assert_almost_equal(vel_disp_numerical / vel_disp_analytic, 1, decimal=2)

        kin_api.kinematics_modeling_settings(
            anisotropy_model,
            kwargs_numerics_galkin,
            analytic_kinematics=False,
            Hernquist_approx=False,
            MGE_light=False,
            MGE_mass=False,
            kwargs_mge_light={"n_comp": 10},
            kwargs_mge_mass={"n_comp": 5},
        )
        assert kin_api._kwargs_mge_mass["n_comp"] == 5
        assert kin_api._kwargs_mge_light["n_comp"] == 10

    def test_velocity_dispersion_map_direct_convolved_against_jampy(self):
        """Test the computed velocity dispersion map through the Kinematics_API with PSF
        convolution against `jampy` computed values.

        The `jampy` values are computed using the same model, grid, and PSF used for
        Galkin using the code below:

        .. code-block:: python

            import numpy as np
            from astropy.cosmology import FlatLambdaCDM
            from lenstronomy.LightModel.light_model import LightModel
            from lenstronomy.LensModel.lens_model import LensModel
            import jampy as jam
            from mgefit.mge_fit_1d import mge_fit_1d

            z_l = 0.3
            z_s = 0.7

            pixel_size = 0.1457
            x_grid, y_grid = np.meshgrid(
                np.arange(-3.0597, 3.1597, pixel_size),
                np.arange(-3.0597, 3.1597, pixel_size),
            )
            psf_fwhm = 0.7

            light_model = LightModel(["SERSIC"])
            kwargs_lens_light = [
                {
                    "amp": 0.09,
                    "R_sersic": 1.2,
                    "n_sersic": 2.9,
                    "center_x": 0.0,
                    "center_y": 0.0,
                }
            ]

            rs = np.logspace(-2.5, 2, 300)
            flux_r = light_model.surface_brightness(rs, 0 * rs, kwargs_lens_light)

            mge_fit = mge_fit_1d(rs, flux_r, ngauss=20, quiet=True)
            sigma_lum = mge_fit.sol[1]
            surf_lum = mge_fit.sol[0] / (np.sqrt(2 * np.pi) * sigma_lum)
            qobs_lum = np.ones_like(sigma_lum)


            lens_model = LensModel(["EPL"])
            kwargs_lens = [
                {
                    "theta_E": 1.63,
                    "gamma": 2.02,
                    "e1": 0.0,
                    "e2": 0.0,
                    "center_x": 0.0,
                    "center_y": 0.0,
                }
            ]

            mass_r = lens_model.kappa(rs, rs * 0, kwargs_lens)
            mass_mge = mge_fit_1d(rs, mass_r, ngauss=20, quiet=True, plot=False)

            cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
            D_d = cosmo.angular_diameter_distance(z_l).value
            D_s = cosmo.angular_diameter_distance(z_s).value
            D_ds = cosmo.angular_diameter_distance_z1z2(z_l, z_s).value
            c2_4piG = 1.6624541593797972e6
            sigma_crit = c2_4piG * D_s / D_ds / D_d

            sigma_pot = mass_mge.sol[1]
            surf_pot = mass_mge.sol[0] / (np.sqrt(2 * np.pi) * sigma_pot) * sigma_crit
            qobs_pot = np.ones_like(sigma_pot)

            bs = np.ones_like(surf_lum) * 0.25

            jam = jam.axi.proj(
                surf_lum,
                sigma_lum,
                qobs_lum,
                surf_pot,
                sigma_pot,
                qobs_pot,
                inc=90,
                mbh=0,
                distance=D_d,
                xbin=x_grid.flatten(),
                ybin=y_grid.flatten(),
                plot=False,
                pixsize=pixel_size,
                pixang=0,
                quiet=1,
                sigmapsf=psf_fwhm / 2.355,
                normpsf=1,
                moment="zz",
                align="sph",
                beta=bs,
                ml=1,
            ).model

            jampy_vel_dis = jam.reshape(x_grid.shape)[14:28, 14:28]
        """
        z_l = 0.3
        z_s = 0.7

        anisotropy_type = "const"

        kwargs_model = {
            "lens_model_list": ["EPL"],
            "lens_light_model_list": ["SERSIC"],
        }

        pixel_size = 0.1457
        x_grid, y_grid = np.meshgrid(
            np.arange(-3.0597, 3.1597, pixel_size),  # x-axis points to negative RA
            np.arange(-3.0597, 3.1597, pixel_size),
        )
        psf_fwhm = 0.7

        kwargs_aperture = {
            "aperture_type": "IFU_grid",
            "x_grid": x_grid,
            "y_grid": y_grid,
        }
        kwargs_seeing = {
            "psf_type": "GAUSSIAN",
            "fwhm": psf_fwhm,
        }

        kwargs_galkin_numerics = None

        light_model_bool = [True]
        lens_model_bool = [True]

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)

        kinematics_api = JAMKinematicsAPI(
            z_lens=z_l,
            z_source=z_s,
            kwargs_model=kwargs_model,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_seeing,
            anisotropy_model=anisotropy_type,
            cosmo=cosmo,
            lens_model_kinematics_bool=lens_model_bool,
            light_model_kinematics_bool=light_model_bool,
            multi_observations=False,
            kwargs_numerics_jam=kwargs_galkin_numerics,
            analytic_kinematics=False,
            Hernquist_approx=False,
            MGE_light=False,
            MGE_mass=False,
            kwargs_mge_light=None,
            kwargs_mge_mass=None,
            sampling_number=1000,
            num_kin_sampling=2000,
            num_psf_sampling=500,
        )

        beta = 0.25

        kwargs_lens = [
            {
                "theta_E": 1.63,
                "gamma": 2.02,
                "e1": 0.0,
                "e2": 0.0,
                "center_x": 0.0,
                "center_y": 0.0,
            }
        ]

        kwargs_lens_light = [
            {
                "amp": 0.09,
                "R_sersic": 1.2,
                "n_sersic": 2.9,
                "center_x": 0.0,
                "center_y": 0.0,
            },
        ]

        kwargs_anisotropy = {"beta": beta}

        vel_dis = kinematics_api.velocity_dispersion_map(
            kwargs_lens,
            kwargs_lens_light,
            kwargs_anisotropy,
            r_eff=kwargs_lens_light[0]["R_sersic"],
            theta_E=kwargs_lens[0]["theta_E"],
            gamma=kwargs_lens[0]["gamma"],
            kappa_ext=0,
            voronoi_bins=None,
        )

        jampy_vel_dis = np.array(
            [
                [
                    257.78155172,
                    261.12579438,
                    264.56084969,
                    267.94853335,
                    271.06981289,
                    273.63412877,
                    275.3336265,
                    275.93007038,
                    275.3336265,
                    273.63412877,
                    271.06981289,
                    267.94853335,
                    264.56084969,
                    261.12579438,
                ],
                [
                    261.12579438,
                    265.27733488,
                    269.70911632,
                    274.2569744,
                    278.59389873,
                    282.24444519,
                    284.69702581,
                    285.56258357,
                    284.69702581,
                    282.24444519,
                    278.59389873,
                    274.2569744,
                    269.70911632,
                    265.27733488,
                ],
                [
                    264.56084969,
                    269.70911632,
                    275.42487086,
                    281.4940479,
                    287.40303998,
                    292.40786532,
                    295.76156343,
                    296.94067873,
                    295.76156343,
                    292.40786532,
                    287.40303998,
                    281.4940479,
                    275.42487086,
                    269.70911632,
                ],
                [
                    267.94853335,
                    274.2569744,
                    281.4940479,
                    289.32766168,
                    296.95384828,
                    303.32194835,
                    307.5144872,
                    308.97174889,
                    307.5144872,
                    303.32194835,
                    296.95384828,
                    289.32766168,
                    281.4940479,
                    274.2569744,
                ],
                [
                    271.06981289,
                    278.59389873,
                    287.40303998,
                    296.95384828,
                    306.0907097,
                    313.51829858,
                    318.28953379,
                    319.92457097,
                    318.28953379,
                    313.51829858,
                    306.0907097,
                    296.95384828,
                    287.40303998,
                    278.59389873,
                ],
                [
                    273.63412877,
                    282.24444519,
                    292.40786532,
                    303.32194835,
                    313.51829858,
                    321.58061025,
                    326.64377058,
                    328.35763268,
                    326.64377058,
                    321.58061025,
                    313.51829858,
                    303.32194835,
                    292.40786532,
                    282.24444519,
                ],
                [
                    275.3336265,
                    284.69702581,
                    295.76156343,
                    307.5144872,
                    318.28953379,
                    326.64377058,
                    331.81191179,
                    333.54753621,
                    331.81191179,
                    326.64377058,
                    318.28953379,
                    307.5144872,
                    295.76156343,
                    284.69702581,
                ],
                [
                    275.93007038,
                    285.56258357,
                    296.94067873,
                    308.97174889,
                    319.92457097,
                    328.35763268,
                    333.54753621,
                    335.28580444,
                    333.54753621,
                    328.35763268,
                    319.92457097,
                    308.97174889,
                    296.94067873,
                    285.56258357,
                ],
                [
                    275.3336265,
                    284.69702581,
                    295.76156343,
                    307.5144872,
                    318.28953379,
                    326.64377058,
                    331.81191179,
                    333.54753621,
                    331.81191179,
                    326.64377058,
                    318.28953379,
                    307.5144872,
                    295.76156343,
                    284.69702581,
                ],
                [
                    273.63412877,
                    282.24444519,
                    292.40786532,
                    303.32194835,
                    313.51829858,
                    321.58061025,
                    326.64377058,
                    328.35763268,
                    326.64377058,
                    321.58061025,
                    313.51829858,
                    303.32194835,
                    292.40786532,
                    282.24444519,
                ],
                [
                    271.06981289,
                    278.59389873,
                    287.40303998,
                    296.95384828,
                    306.0907097,
                    313.51829858,
                    318.28953379,
                    319.92457097,
                    318.28953379,
                    313.51829858,
                    306.0907097,
                    296.95384828,
                    287.40303998,
                    278.59389873,
                ],
                [
                    267.94853335,
                    274.2569744,
                    281.4940479,
                    289.32766168,
                    296.95384828,
                    303.32194835,
                    307.5144872,
                    308.97174889,
                    307.5144872,
                    303.32194835,
                    296.95384828,
                    289.32766168,
                    281.4940479,
                    274.2569744,
                ],
                [
                    264.56084969,
                    269.70911632,
                    275.42487086,
                    281.4940479,
                    287.40303998,
                    292.40786532,
                    295.76156343,
                    296.94067873,
                    295.76156343,
                    292.40786532,
                    287.40303998,
                    281.4940479,
                    275.42487086,
                    269.70911632,
                ],
                [
                    261.12579438,
                    265.27733488,
                    269.70911632,
                    274.2569744,
                    278.59389873,
                    282.24444519,
                    284.69702581,
                    285.56258357,
                    284.69702581,
                    282.24444519,
                    278.59389873,
                    274.2569744,
                    269.70911632,
                    265.27733488,
                ],
            ]
        )
        n = int(np.sqrt(len(vel_dis)))
        vel_dis = vel_dis.reshape(int(n), int(n))
        assert np.max(np.abs(jampy_vel_dis / vel_dis[14:28, 14:28] - 1)) < 0.009

    def test_velocity_dispersion_map(self):
        np.random.seed(42)
        z_lens = 0.5
        z_source = 1.5
        kwargs_options = {
            "lens_model_list": ["SIS"],
            "lens_light_model_list": ["HERNQUIST"],
        }
        r_eff = 1.0
        theta_E = 1
        kwargs_lens = [{"theta_E": theta_E, "center_x": 0, "center_y": 0}]
        kwargs_lens_light = [
            {"amp": 1, "Rs": r_eff * 0.551, "center_x": 0, "center_y": 0}
        ]
        kwargs_anisotropy = {"r_ani": 1}

        r_bins = np.array([0, 0.5, 1])
        aperture_type = "IFU_shells"
        kwargs_aperture = {
            "aperture_type": aperture_type,
            "center_ra": 0,
            "r_bins": r_bins,
            "center_dec": 0,
        }
        psf_fwhm = 0.7
        kwargs_seeing = {"psf_type": "GAUSSIAN", "fwhm": psf_fwhm}
        anisotropy_model = "OM"
        kin_api = JAMKinematicsAPI(
            z_lens,
            z_source,
            kwargs_options,
            kwargs_aperture=kwargs_aperture,
            kwargs_seeing=kwargs_seeing,
            anisotropy_model=anisotropy_model,
        )

        kwargs_numerics_galkin = {
            "interpol_grid_num": 500,
            "log_integration": True,
            "max_integrate": 10,
            "min_integrate": 0.001,
        }
        kin_api.kinematics_modeling_settings(
            anisotropy_model,
            kwargs_numerics_galkin,
            analytic_kinematics=True,
            Hernquist_approx=False,
            MGE_light=False,
            MGE_mass=False,
            num_kin_sampling=1000,
            num_psf_sampling=100,
        )
        vel_disp_analytic = kin_api.velocity_dispersion_map(
            kwargs_lens,
            kwargs_lens_light,
            kwargs_anisotropy,
            r_eff=r_eff,
            theta_E=theta_E,
            gamma=2,
        )

        kin_api.kinematics_modeling_settings(
            anisotropy_model,
            kwargs_numerics_galkin,
            analytic_kinematics=False,
            Hernquist_approx=False,
            MGE_light=False,
            MGE_mass=False,
            num_kin_sampling=1000,
            num_psf_sampling=100,
        )
        vel_disp_numerical = kin_api.velocity_dispersion_map(
            kwargs_lens,
            kwargs_lens_light,
            kwargs_anisotropy,
            r_eff=r_eff,
            theta_E=theta_E,
            gamma=2,
        )
        npt.assert_almost_equal(vel_disp_numerical, vel_disp_analytic, decimal=-1)

        z_lens = 0.5
        z_source = 1.5
        kwargs_model = {
            "lens_model_list": ["SIS"],
            "lens_light_model_list": ["HERNQUIST"],
        }

        xs, ys = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        kwargs_aperture = {
            "aperture_type": "IFU_grid",
            "x_grid": xs,
            "y_grid": ys,
        }
        kwargs_seeing = {"psf_type": "GAUSSIAN", "fwhm": psf_fwhm}
        kin_api = JAMKinematicsAPI(
            z_lens,
            z_source,
            kwargs_model,
            kwargs_seeing=kwargs_seeing,
            kwargs_aperture=kwargs_aperture,
            anisotropy_model="OM",
        )
        kin_api.velocity_dispersion_map(
            [{"theta_E": 1, "center_x": 0, "center_y": 0}],
            [{"Rs": 1, "amp": 1, "center_x": 0, "center_y": 0}],
            {"r_ani": 1},
        )

        kin_api.velocity_dispersion_map(
            [{"theta_E": 1, "center_x": 0, "center_y": 0}],
            [{"Rs": 1, "amp": 1, "center_x": 0, "center_y": 0}],
            {"r_ani": 1},
        )

    def test_interpolated_sersic(self):
        from lenstronomy.Analysis.light2mass import light2mass_interpol

        kwargs_light = [
            {
                "n_sersic": 2,
                "R_sersic": 0.5,
                "amp": 1,
                "center_x": 0.01,
                "center_y": 0.01,
            }
        ]
        kwargs_lens = [
            {
                "n_sersic": 2,
                "R_sersic": 0.5,
                "k_eff": 1,
                "center_x": 0.01,
                "center_y": 0.01,
            }
        ]
        deltaPix = 0.1
        numPix = 100

        kwargs_interp = light2mass_interpol(
            ["SERSIC"],
            kwargs_lens_light=kwargs_light,
            numPix=numPix,
            deltaPix=deltaPix,
            subgrid_res=5,
        )
        kwargs_lens_interp = [kwargs_interp]

        z_lens = 0.5
        z_source = 1.5
        r_ani = 0.62
        kwargs_anisotropy = {"r_ani": r_ani}
        R_slit = 3.8
        dR_slit = 1.0
        aperture_type = "slit"
        kwargs_aperture = {
            "center_ra": 0,
            "width": dR_slit,
            "length": R_slit,
            "angle": 0,
            "center_dec": 0,
            "aperture_type": aperture_type,
        }
        psf_fwhm = 0.7
        kwargs_psf = {"psf_type": "GAUSSIAN", "fwhm": psf_fwhm}
        anisotropy_model = "OM"
        r_eff = 0.5
        kwargs_model = {
            "lens_model_list": ["SERSIC"],
            "lens_light_model_list": ["SERSIC"],
        }
        kwargs_mge = {"n_comp": 20}
        kinematic_api = JAMKinematicsAPI(
            z_lens,
            z_source,
            kwargs_model,
            kwargs_aperture,
            kwargs_seeing=kwargs_psf,
            anisotropy_model=anisotropy_model,
            MGE_light=True,
            MGE_mass=True,
            kwargs_mge_mass=kwargs_mge,
            kwargs_mge_light=kwargs_mge,
        )

        v_sigma = kinematic_api.velocity_dispersion(
            kwargs_lens, kwargs_light, kwargs_anisotropy, r_eff=r_eff, theta_E=1
        )
        kwargs_model_interp = {
            "lens_model_list": ["INTERPOL"],
            "lens_light_model_list": ["SERSIC"],
        }
        kinematic_api_interp = JAMKinematicsAPI(
            z_lens,
            z_source,
            kwargs_model_interp,
            kwargs_aperture,
            kwargs_seeing=kwargs_psf,
            anisotropy_model=anisotropy_model,
            MGE_light=True,
            MGE_mass=True,
            kwargs_mge_mass=kwargs_mge,
            kwargs_mge_light=kwargs_mge,
        )
        v_sigma_interp = kinematic_api_interp.velocity_dispersion(
            kwargs_lens_interp,
            kwargs_light,
            kwargs_anisotropy,
            theta_E=1.0,
            r_eff=r_eff,
        )
        npt.assert_almost_equal(v_sigma / v_sigma_interp, 1, 1)
        # use as kinematic constraints
        # compare with MGE Sersic kinematic estimate

    def test_copy_centers(self):
        z_lens = 0.5
        z_source = 1.5
        kwargs_model = {
            "lens_model_list": ["SIS"],
            "lens_light_model_list": ["HERNQUIST"],
        }
        kwargs_aperture = {
            "aperture_type": "slit",
            "center_ra": 0,
            "width": 1,
            "length": 1,
            "angle": 0,
            "center_dec": 0,
        }
        psf_fwhm = 0.7
        kwargs_seeing = {"psf_type": "GAUSSIAN", "fwhm": psf_fwhm}
        anisotropy_model = "OM"
        kinematicAPI = JAMKinematicsAPI(
            z_lens,
            z_source,
            kwargs_model,
            kwargs_aperture,
            kwargs_seeing,
            anisotropy_model=anisotropy_model,
            analytic_kinematics=True,
        )
        kwargs_lens = [{"theta_E": 1, "center_x": 0, "center_y": 0}]
        kwargs_lens_light = [{"Rs": 1, "amp": 1, "center_x": 0, "center_y": 0}]
        kwargs_anisotropy = {"r_ani": 1}

        assert kinematicAPI._copy_centers({}, kwargs_lens) == {
            "center_x": 0,
            "center_y": 0,
        }

        kinematicAPI._analytic_kinematics = False
        assert kinematicAPI._copy_centers([{}], kwargs_lens_light) == [
            {"center_x": 0, "center_y": 0}
        ]


class TestRaise(unittest.TestCase):
    def test_raise(self):

        with self.assertRaises(ValueError):
            z_lens = 0.5
            z_source = 1.5
            kwargs_model = {"lens_light_model_list": ["HERNQUIST"]}
            kinematicAPI = JAMKinematicsAPI(
                z_lens,
                z_source,
                kwargs_model,
                kwargs_seeing={},
                kwargs_aperture={"aperture_type": "slit"},
                anisotropy_model="OM",
            )
            kwargs_light = [{"Rs": 1, "amp": 1, "center_x": 0, "center_y": 0}]
            kinematicAPI.kinematic_light_profile(
                kwargs_light,
                MGE_fit=False,
                Hernquist_approx=True,
                r_eff=None,
                model_kinematics_bool=[True],
            )
        with self.assertRaises(ValueError):
            z_lens = 0.5
            z_source = 1.5
            kwargs_model = {"lens_light_model_list": ["HERNQUIST"]}
            kinematicAPI = JAMKinematicsAPI(
                z_lens,
                z_source,
                kwargs_model,
                kwargs_seeing={},
                kwargs_aperture={"aperture_type": "slit"},
                anisotropy_model="OM",
            )
            kwargs_light = [{"Rs": 1, "amp": 1, "center_x": 0, "center_y": 0}]
            kinematicAPI.kinematic_light_profile(
                kwargs_light,
                MGE_fit=False,
                Hernquist_approx=False,
                r_eff=None,
                analytic_kinematics=True,
            )
        with self.assertRaises(ValueError):
            z_lens = 0.5
            z_source = 1.5
            kwargs_model = {
                "lens_light_model_list": ["HERNQUIST"],
                "lens_model_list": [],
            }
            kinematicAPI = JAMKinematicsAPI(
                z_lens,
                z_source,
                kwargs_model,
                kwargs_seeing={},
                kwargs_aperture={"aperture_type": "slit"},
                anisotropy_model="OM",
            )
            kwargs_light = [{"Rs": 1, "amp": 1, "center_x": 0, "center_y": 0}]
            kinematicAPI.kinematic_lens_profiles(
                kwargs_light, MGE_fit=True, model_kinematics_bool=[True]
            )
        with self.assertRaises(ValueError):
            z_lens = 0.5
            z_source = 1.5
            kwargs_model = {
                "lens_light_model_list": ["HERNQUIST"],
                "lens_model_list": [],
            }
            kinematicAPI = JAMKinematicsAPI(
                z_lens,
                z_source,
                kwargs_model,
                kwargs_seeing={},
                kwargs_aperture={"aperture_type": "slit"},
                anisotropy_model="OM",
            )
            kinematicAPI.kinematic_lens_profiles(
                kwargs_lens=None, analytic_kinematics=True
            )

        with self.assertRaises(ValueError):
            z_lens = 0.5
            z_source = 1.5
            kwargs_model = {
                "lens_light_model_list": ["HERNQUIST"],
                "lens_model_list": [],
            }
            kinematicAPI = JAMKinematicsAPI(
                z_lens,
                z_source,
                kwargs_model,
                kwargs_seeing={},
                kwargs_aperture={"aperture_type": "slit"},
                anisotropy_model="OM",
            )
            kwargs_lens_light = [{"Rs": 1, "center_x": 0, "center_y": 0}]
            kinematicAPI.kinematic_light_profile(
                kwargs_lens_light,
                r_eff=None,
                MGE_fit=True,
                model_kinematics_bool=None,
                Hernquist_approx=False,
                kwargs_mge=None,
            )
        with self.assertRaises(ValueError):
            z_lens = 0.5
            z_source = 1.5
            kwargs_model = {
                "lens_light_model_list": ["HERNQUIST"],
                "lens_model_list": ["SIS"],
            }
            kwargs_lens = [{"theta_E": 1, "center_x": 0, "center_y": 0}]
            kinematicAPI = JAMKinematicsAPI(
                z_lens,
                z_source,
                kwargs_model,
                kwargs_seeing={},
                kwargs_aperture={"aperture_type": "slit"},
                anisotropy_model="OM",
            )
            kinematicAPI.kinematic_lens_profiles(
                kwargs_lens,
                MGE_fit=True,
                model_kinematics_bool=None,
                theta_E=None,
                kwargs_mge={},
            )

    def test_dispersion_map_grid_convolved_numeric_vs_analytical(self):
        """Test numerical vs analytical computation of IFU_grid velocity dispersion."""
        r_eff = 1.85
        theta_e = 1.63
        gamma = 2
        a_ani = 1

        def get_v_rms(
            theta_e, gamma, r_eff, a_ani=1, z_d=0.295, z_s=0.657, analytic=False
        ):
            """Compute v_rms for power-law mass and Hernquist light using Galkin's
            numerical approach.

            :param hernquist_mass: if mass in M_sun provided, uses Hernquist mass
                  profile. For debugging purpose.
            :param do_mge: True will use lenstronomy's own MGE implementation
            """
            cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

            D_d = cosmo.angular_diameter_distance(z_d).value
            D_s = cosmo.angular_diameter_distance(z_s).value
            D_ds = cosmo.angular_diameter_distance_z1z2(0.5, 2.0).value

            kwargs_cosmo = {"d_d": D_d, "d_s": D_s, "d_ds": D_ds}

            xs, ys = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))

            kwargs_aperture = {
                "aperture_type": "IFU_grid",
                "x_grid": xs,
                "y_grid": ys,
            }

            kwargs_seeing = {
                "psf_type": "GAUSSIAN",
                "fwhm": 0.7,
            }

            kwargs_galkin_numerics = {  # 'sampling_number': 1000,
                "interpol_grid_num": 2000,
                "log_integration": True,
                "max_integrate": 100,
                "min_integrate": 0.001,
            }

            kwargs_model = {
                "lens_model_list": ["EPL"],
                "lens_light_model_list": ["HERNQUIST"],
            }

            kinematics_api = JAMKinematicsAPI(
                z_lens=z_d,
                z_source=z_s,
                kwargs_model=kwargs_model,
                kwargs_aperture=kwargs_aperture,
                kwargs_seeing=kwargs_seeing,
                anisotropy_model="OM",
                cosmo=cosmo,
                multi_observations=False,
                # kwargs_numerics_galkin=kwargs_galkin_numerics,
                analytic_kinematics=analytic,
                Hernquist_approx=False,
                MGE_light=False,
                MGE_mass=False,  # self._cgd,
                kwargs_mge_light=None,
                kwargs_mge_mass=None,
                sampling_number=1000,
                num_kin_sampling=2000,
                num_psf_sampling=500,
            )

            kwargs_mass = [
                {
                    "theta_E": theta_e,
                    "gamma": gamma,
                    "center_x": 0,
                    "center_y": 0,
                    "e1": 0,
                    "e2": 0,
                }
            ]

            kwargs_light = [
                {"Rs": 0.551 * r_eff, "amp": 1.0, "center_x": 0, "center_y": 0}
            ]

            kwargs_anisotropy = {"r_ani": a_ani * r_eff}

            vel_dis = kinematics_api.velocity_dispersion_map(
                kwargs_mass,
                kwargs_light,
                kwargs_anisotropy,
                r_eff=r_eff,
                theta_E=theta_e,
                gamma=gamma,
                kappa_ext=0,
                supersampling_factor=5,
                voronoi_bins=None,
            )

            return vel_dis

        analytic_sigma = get_v_rms(theta_e, gamma, r_eff, analytic=True)
        numeric_sigma = get_v_rms(theta_e, gamma, r_eff, analytic=False)

        # check if values match within 1%
        npt.assert_array_less(
            (analytic_sigma - numeric_sigma) / analytic_sigma,
            0.01 * np.ones_like(analytic_sigma),
        )


if __name__ == "__main__":
    pytest.main()
