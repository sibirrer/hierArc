__author__ = "sibirrer", "furcelay"

import numpy as np
import copy
from hierarc.JAM.jam_wrapper import JAMWrapper
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Util import class_creator
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
from lenstronomy.Analysis.light_profile import LightProfileAnalysis
import warnings

__all__ = ["JAMKinematicsAPI"]


class JAMKinematicsAPI(object):
    """This class contains routines to compute time delays, magnification ratios, line
    of sight velocity dispersions etc for a given lens model."""

    def __init__(
        self,
        z_lens,
        z_source,
        kwargs_model,
        kwargs_aperture,
        kwargs_seeing,
        anisotropy_model,
        axial_symmetry="axi_sph",
        cosmo=None,
        lens_model_kinematics_bool=None,
        light_model_kinematics_bool=None,
        multi_observations=False,
        multi_light_profile=False,
        kwargs_numerics_jam=None,
    ):
        """Initialize the class with the lens model and cosmology.

        :param z_lens: redshift of lens
        :param z_source: redshift of source
        :param kwargs_model: model keyword arguments, needs 'lens_model_list',
            'lens_light_model_list'
        :param kwargs_aperture: spectroscopic aperture keyword arguments, see
            lenstronomy.Galkin.aperture for options
        :param kwargs_seeing: seeing condition of spectroscopic observation, corresponds
            to kwargs_psf in the GalKin module specified in lenstronomy.GalKin.psf
        :param cosmo: ~astropy.cosmology instance, if None then will be set to the
            default cosmology
        :param lens_model_kinematics_bool: bool list of length of the lens model. Only
            takes a subset of all the models as part of the kinematics computation ( can
            be used to ignore substructure, shear etc. that do not describe the main
            deflector potential
        :param light_model_kinematics_bool: bool list of length of the light model. Only
            takes a subset of all the models as part of the kinematics computation (can
            be used to ignore light components that do not describe the main deflector)
        :param multi_observations: bool, if True uses multi-observation to predict a set
            of different observations with the GalkinMultiObservation() class.
            kwargs_aperture and kwargs_seeing require to be lists of the individual
            observations.
        :param multi_light_profile: bool, if True (and if multi_observation=True) then
            treats the light profile input as a list for each individual observation
            condition.
        :param anisotropy_model: type of stellar anisotropy model. See details in
            MamonLokasAnisotropy() class of lenstronomy.GalKin.anisotropy
        :param axial_symmetry: string, symmetry assumption for JAM modeling. Options are
            spherical, axi_sph and axi_cyl.
        """
        self.z_d = z_lens
        self.z_s = z_source
        # put it in list of apertures and seeing conditions
        if not multi_observations:
            self._kwargs_aperture_kin = [kwargs_aperture]
            self._kwargs_psf_kin = [kwargs_seeing]
        else:
            self._kwargs_aperture_kin = kwargs_aperture
            self._kwargs_psf_kin = kwargs_seeing
        self.lensCosmo = LensCosmo(z_lens, z_source, cosmo=cosmo)
        (
            self.LensModel,
            self.SourceModel,
            self.LensLightModel,
            self.PointSource,
            extinction_class,
        ) = class_creator.create_class_instances(all_models=True, **kwargs_model)
        self._lensLightProfile = LightProfileAnalysis(light_model=self.LensLightModel)
        self._lensMassProfile = LensProfileAnalysis(lens_model=self.LensModel)
        self._lens_light_model_list = self.LensLightModel.profile_type_list
        self._lens_model_list = self.LensModel.lens_model_list
        self._kwargs_cosmo = {
            "d_d": self.lensCosmo.dd,
            "d_s": self.lensCosmo.ds,
            "d_ds": self.lensCosmo.dds,
        }
        self._lens_model_kinematics_bool = lens_model_kinematics_bool
        self._light_model_kinematics_bool = light_model_kinematics_bool

        self._kwargs_numerics_kin = kwargs_numerics_jam
        self._anisotropy_model = anisotropy_model
        self._axial_symmetry = axial_symmetry
        self._multi_observations = multi_observations
        self._multi_light_profile = multi_light_profile

    def velocity_dispersion(
        self,
        kwargs_lens,
        kwargs_lens_light,
        kwargs_anisotropy,
        r_eff=None,
        theta_E=None,
        gamma=None,
        kappa_ext=0,
        q_intrinsic=1.0,
        voronoi_bins=None,
    ):
        """For any aperture (single or IFU), uses JamPy with axisymmetric JAM modeling.

        API for numerical JAM to compute the velocity dispersion [km/s]

        :param kwargs_lens: lens model keyword arguments
        :param kwargs_lens_light: lens light model keyword arguments
        :param kwargs_anisotropy: stellar anisotropy keyword arguments
        :param r_eff: projected half-light radius of the stellar light associated with
            the deflector galaxy, optional, if set to None will be computed in this
            function with default settings that may not be accurate.
        :param theta_E: Einstein radius (optional)
        :param gamma: power-law slope (optional)
        :param kappa_ext: external convergence (optional)
        :return: velocity dispersion [km/s]
        """
        jam, kwargs_profile, kwargs_light = self.jam_settings(
            kwargs_lens,
            kwargs_lens_light,
            r_eff=r_eff,
        )

        sigma_v = []
        for i in range(len(self._kwargs_aperture_kin)):
            if self._multi_light_profile:
                kwargs_light_ = kwargs_light[i]
            else:
                kwargs_light_ = kwargs_light

            # JAMWrapper (axisymmetric, with inclination)
            sigma_v_ = jam[i].dispersion(
                kwargs_profile,
                kwargs_light_,
                kwargs_anisotropy,
                q_intrinsic=q_intrinsic,
                voronoi_bins=voronoi_bins,
            )
            sigma_v = np.append(sigma_v, sigma_v_)
        sigma_v = self.transform_kappa_ext(sigma_v, kappa_ext=kappa_ext)
        return sigma_v

    def velocity_dispersion_map(
        self,
        kwargs_lens,
        kwargs_lens_light,
        kwargs_anisotropy,
        r_eff=None,
        theta_E=None,
        gamma=None,
        kappa_ext=0,
        q_intrinsic=1.0,
        supersampling_factor=1,
        voronoi_bins=None,
    ):
        """For a IFU measurements (regular or binned grid, or shells), it uses JamPy
        with  axisymmetric JAM modeling. Note that this function does the same as
        velocity_dispersion, but both are kept for compatibility.

        API for numerical JAM to compute the velocity dispersion
        map with IFU data or multiple apertures [km/s]

        :param kwargs_lens: lens model keyword arguments
        :param kwargs_lens_light: lens light model keyword arguments
        :param kwargs_anisotropy: stellar anisotropy keyword arguments
        :param r_eff: projected half-light radius of the stellar light associated with
            the deflector galaxy, optional, if set to None will be computed in this
            function with default settings that may not be accurate.
        :param theta_E: circularized Einstein radius, optional, if not provided will
            either be computed in this function with default settings or not required
        :param gamma: power-law slope at the Einstein radius, optional
        :param kappa_ext: external convergence
        :param q_intrinsic: intrinsic axis ratio of the light profile to compute the inclination angle
        :param supersampling_factor: supersampling factor for 2D integration grid
            NOTE: this parameter is ignored as JamPy does its own internal supersampling
        :param voronoi_bins: mapping of the voronoi bins, -1 values for pixels not
            binned
        :return: velocity dispersion map in specified bins or grid in `kwargs_aperture`,
            in [km/s] unit
        """
        if supersampling_factor > 1:
            warnings.warn(
                "supersampling_factor is ignored, JamPy does its own internal supersampling",
                UserWarning,
            )
        return self.velocity_dispersion(
            kwargs_lens,
            kwargs_lens_light,
            kwargs_anisotropy,
            r_eff=r_eff,
            theta_E=theta_E,
            gamma=gamma,
            kappa_ext=kappa_ext,
            q_intrinsic=q_intrinsic,
            voronoi_bins=voronoi_bins,
        )

    def jam_settings(
        self,
        kwargs_lens,
        kwargs_lens_light,
        r_eff=None,
    ):
        """

        :param kwargs_lens: lens model keyword argument list
        :param kwargs_lens_light: deflector light keyword argument list
        :param r_eff: half-light radius (optional)
        :return: JAMWrapper() instance and mass and light profiles configured for JamPy
        """
        if r_eff is None:
            if self._multi_light_profile is True:
                kwargs_lens_light_ = kwargs_lens_light[0]
            else:
                kwargs_lens_light_ = kwargs_lens_light
            r_eff = self._lensLightProfile.half_light_radius(
                kwargs_lens_light_,
                grid_spacing=0.05,
                grid_num=200,
                center_x=None,
                center_y=None,
                model_bool_list=self._light_model_kinematics_bool,
            )

        mass_profile_list, kwargs_profile = self.kinematic_lens_profiles(
            kwargs_lens,
            model_kinematics_bool=self._lens_model_kinematics_bool,
        )
        light_profile_list, kwargs_light = self.kinematic_light_profile(
            kwargs_lens_light,
            r_eff=r_eff,
            model_kinematics_bool=self._light_model_kinematics_bool,
        )

        jam_models = []

        for i in range(len(self._kwargs_aperture_kin)):
            kwargs_model = {
                "mass_profile_list": mass_profile_list,
                "light_profile_list": light_profile_list,
                "anisotropy_model": self._anisotropy_model,
                "symmetry": self._axial_symmetry,
            }
            model_i = JAMWrapper(
                kwargs_model=kwargs_model,
                kwargs_aperture=self._kwargs_aperture_kin[i],
                kwargs_psf=self._kwargs_psf_kin[i],
                kwargs_cosmo=self._kwargs_cosmo,
                kwargs_numerics=self._kwargs_numerics_kin,
            )
            jam_models.append(model_i)

        return jam_models, kwargs_profile, kwargs_light

    def _copy_centers(self, kwargs_1, kwargs_2):
        """Fills the centers of the kwargs_1 with the centers of kwargs_2.

        :param kwargs_1: target
        :param kwargs_2: source
        :return: kwargs_1 with filled centers
        """
        if "center_x" in kwargs_2[0] and "center_y" in kwargs_2[0]:
            kwargs_1[0]["center_x"] = kwargs_2[0]["center_x"]
            kwargs_1[0]["center_y"] = kwargs_2[0]["center_y"]
        return kwargs_1

    def kinematic_lens_profiles(
        self,
        kwargs_lens,
        model_kinematics_bool=None,
    ):
        """Translates the lenstronomy lens and mass profiles into a (sub) set of
        profiles that are compatible with the JAMWrapper module to compute the
        kinematics thereof. The requirement is that the profiles are centered at (0, 0)
        and that for all profile types there exists a 3d de-projected analytical
        representation.

        :param kwargs_lens: lens model parameters
        :param model_kinematics_bool: bool list of length of the lens model. Only takes
            a subset of all the models as part of the kinematics computation (can be
            used to ignore substructure, shear etc that do not describe the main
            deflector potential
        :return: mass_profile_list, keyword argument list
        """
        mass_profile_list = []
        kwargs_profile = []
        if model_kinematics_bool is None:
            model_kinematics_bool = [True] * len(kwargs_lens)
        for i, lens_model in enumerate(self._lens_model_list):
            if model_kinematics_bool[i] is True:
                mass_profile_list.append(lens_model)
                if lens_model in ["INTERPOL", "INTERPOL_SCLAED"]:
                    center_x_i, center_y_i = self._lensMassProfile.convergence_peak(
                        kwargs_lens,
                        model_bool_list=i,
                        grid_num=200,
                        grid_spacing=0.01,
                        center_x_init=0,
                        center_y_init=0,
                    )
                    kwargs_lens_i = copy.deepcopy(kwargs_lens[i])
                    kwargs_lens_i["grid_interp_x"] -= center_x_i
                    kwargs_lens_i["grid_interp_y"] -= center_y_i
                else:
                    kwargs_lens_i = {
                        k: v
                        for k, v in kwargs_lens[i].items()
                        if not k in ["center_x", "center_y"]
                    }
                kwargs_profile.append(kwargs_lens_i)

        kwargs_profile = self._copy_centers(kwargs_profile, kwargs_lens)

        return mass_profile_list, kwargs_profile

    def kinematic_light_profile(
        self,
        kwargs_lens_light,
        r_eff=None,
        model_kinematics_bool=None,
    ):
        """Setting up of the light profile to compute the kinematics in the JAMWrapper
        module. The requirement is that the profiles are centered at (0, 0) and that for
        all profile types there exists a 3d de-projected analytical representation.

        :param kwargs_lens_light: deflector light model keyword argument list
        :param r_eff: (optional float, else=None) Pre-calculated projected half-light
            radius of the deflector profile. If not provided, numerical calculation is
            done in this routine if required.
        :param model_kinematics_bool: list of booleans to indicate a subset of light
            profiles to be part of the physical deflector light.
        """
        light_profile_list = []
        if model_kinematics_bool is None:
            model_kinematics_bool = [True] * len(self._lens_light_model_list)

        if self._multi_light_profile is True:
            kwargs_light = []
            for i in range(len(kwargs_lens_light)):
                kwargs_lens_light_ = kwargs_lens_light[i]
                light_profile_list, kwargs_light_ = self._setup_light_parameters(
                    kwargs_lens_light_,
                    model_kinematics_bool,
                )
                kwargs_light.append(kwargs_light_)
        else:
            light_profile_list, kwargs_light = self._setup_light_parameters(
                kwargs_lens_light, model_kinematics_bool
            )

        return light_profile_list, kwargs_light

    def kinematics_modeling_settings(
        self,
        anisotropy_model,
        axial_symmetry="axi_sph",
        kwargs_numerics_jam=None,
    ):
        """Return the settings for the kinematic modeling.

        :param anisotropy_model: type of stellar anisotropy model. See details in
            MamonLokasAnisotropy() class of lenstronomy.GalKin.anisotropy
        :param axial_symmetry: for axisymmetric JAM modeling :kwargs_numerics_jam:
            kwargs for JamWrapper MGE decomposition
        :return: updated settings
        """
        self._kwargs_numerics_kin = kwargs_numerics_jam
        self._anisotropy_model = anisotropy_model
        self._axial_symmetry = axial_symmetry

    @staticmethod
    def transform_kappa_ext(sigma_v, kappa_ext=0):
        """

        :param sigma_v: velocity dispersion estimate of the lensing deflector without
            considering external convergence
        :param kappa_ext: external convergence to be used in the mass-sheet degeneracy
        :return: transformed velocity dispersion
        """
        sigma_v_mst = sigma_v * np.sqrt(1 - kappa_ext)
        return sigma_v_mst

    def _setup_light_parameters(self, kwargs_lens_light, model_kinematics_bool):
        light_profile_list = []
        kwargs_light = []

        for i, light_model in enumerate(self._lens_light_model_list):
            if model_kinematics_bool[i] is True:
                light_profile_list.append(light_model)
                kwargs_light_i = kwargs_lens_light[i].copy()
                kwargs_light.append(kwargs_light_i)

        return light_profile_list, kwargs_light
