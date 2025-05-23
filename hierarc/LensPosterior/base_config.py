from lenstronomy.Analysis.td_cosmography import TDCosmography
from hierarc.LensPosterior.imaging_constraints import ImageModelPosterior
from hierarc.LensPosterior.kin_scaling_config import KinScalingConfig


class BaseLensConfig(TDCosmography, ImageModelPosterior, KinScalingConfig):
    """This class contains and manages the base configurations of the lens posteriors
    and makes sure that they are universally applied consistently through the different
    likelihood definitions."""

    def __init__(
        self,
        z_lens,
        z_source,
        theta_E,
        theta_E_error,
        gamma,
        gamma_error,
        r_eff,
        r_eff_error,
        kwargs_aperture,
        kwargs_seeing,
        kwargs_numerics_galkin,
        anisotropy_model,
        lens_model_list=None,
        kwargs_lens_light=None,
        lens_light_model_list=["HERNQUIST"],
        MGE_light=False,
        kwargs_mge_light=None,
        hernquist_approx=True,
        sampling_number=1000,
        num_psf_sampling=100,
        num_kin_sampling=1000,
        multi_observations=False,
        multi_light_profile=False,
        cosmo_fiducial=None,
        gamma_in_scaling=None,
        log_m2l_scaling=None,
        gamma_pl_scaling=None,
    ):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param theta_E: Einstein radius (in arc seconds)
        :param theta_E_error: 1-sigma error on Einstein radius
        :param gamma: power-law slope (2 = isothermal) estimated from imaging data
        :param gamma_error: 1-sigma uncertainty on power-law slope
        :param r_eff: half-light radius of the deflector (arc seconds)
        :param r_eff_error: uncertainty on half-light radius
        :param kwargs_aperture: spectroscopic aperture keyword arguments, see
        lenstronomy.Galkin.aperture for options
        :param kwargs_seeing: seeing condition of spectroscopic observation, corresponds
            to kwargs_psf in the GalKin module specified in lenstronomy.GalKin.psf
        :param kwargs_numerics_galkin: numerical settings for the integrated
            line-of-sight velocity dispersion
        :param anisotropy_model: type of stellar anisotropy model. See details in
            MamonLokasAnisotropy() class of lenstronomy.GalKin.anisotropy
        :param multi_observations: bool, if True, interprets kwargs_aperture and
            kwargs_seeing as lists of multiple observations
        :param multi_light_profile: bool, if True (and if multi_observation=True) then treats the light profile input
         as a list for each individual observation condition.
        :param lens_model_list: keyword argument list of lens model (optional)
        :param kwargs_lens_light: keyword argument list of lens light model (optional)
        :param kwargs_mge_light: keyword arguments that go into the MGE decomposition
            routine
        :param hernquist_approx: bool, if True, uses the Hernquist approximation for the
            light profile
        :param cosmo_fiducial: astropy.cosmology instance, if None,
            uses astropy's default cosmology
        :param gamma_in_scaling: array of gamma_in parameter to be interpolated (optional, otherwise None)
        :param log_m2l_scaling: array of log_m2l parameter to be interpolated (optional, otherwise None)
        :param gamma_pl_scaling: array of power-law density profile slopes to be interpolated (optional, otherwise None)
        """
        self._z_lens, self._z_source = z_lens, z_source

        if lens_model_list is None:
            kwargs_model = {
                "lens_model_list": ["SPP"],
                "lens_light_model_list": lens_light_model_list,
            }
        else:
            kwargs_model = {
                "lens_model_list": lens_model_list,
                "lens_light_model_list": lens_light_model_list,
            }
        TDCosmography.__init__(
            self,
            z_lens,
            z_source,
            kwargs_model,
            cosmo_fiducial=cosmo_fiducial,
            lens_model_kinematics_bool=None,
            light_model_kinematics_bool=None,
            kwargs_seeing=kwargs_seeing,
            kwargs_aperture=kwargs_aperture,
            multi_observations=multi_observations,
            multi_light_profile=multi_light_profile,
        )

        analytic_kinematics = False
        self.kinematics_modeling_settings(
            anisotropy_model,
            kwargs_numerics_galkin,
            analytic_kinematics=analytic_kinematics,
            Hernquist_approx=hernquist_approx,
            MGE_light=MGE_light,
            MGE_mass=False,
            kwargs_mge_light=kwargs_mge_light,
            sampling_number=sampling_number,
            num_psf_sampling=num_psf_sampling,
            num_kin_sampling=num_kin_sampling,
        )
        self._kwargs_lens_light = kwargs_lens_light
        ImageModelPosterior.__init__(
            self, theta_E, theta_E_error, gamma, gamma_error, r_eff, r_eff_error
        )
        KinScalingConfig.__init__(
            self,
            anisotropy_model,
            r_eff,
            gamma_in_scaling=gamma_in_scaling,
            log_m2l_scaling=log_m2l_scaling,
            gamma_pl_scaling=gamma_pl_scaling,
            gamma_pl_mean=gamma,
        )
