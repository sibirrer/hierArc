from lenstronomy.Analysis.td_cosmography import TDCosmography
from hierarc.JAM.jam_td_cosmography import JAMTDCosmography
import warnings


class KinematicsBackend:
    """
    class wrapper to select the kinematics calculation backend: JamPy or Lenstronomy/Galkin
    replaces the previous TDCosmography/JAMTDCosmography initialisation
    """

    def __init__(
        self,
        z_lens,
        z_source,
        kwargs_model,
        cosmo_fiducial=None,
        lens_model_kinematics_bool=None,
        light_model_kinematics_bool=None,
        kwargs_seeing=None,
        kwargs_aperture=None,
        anisotropy_model=None,
        analytic_kinematics=False,
        axial_symmetry="axi_sph",
        backend=None,
        kwargs_numerics_jam=None,
        kwargs_numerics_galkin=None,
        **kwargs_kin_api
    ):

        if axial_symmetry != "spherical":
            if backend is None:
                backend = "jampy"
            else:
                if backend != "jampy":
                    raise ValueError(
                        "Only the JamPy backend is currently supported for axisymmetric JAM models."
                    )
            if analytic_kinematics:
                raise ValueError(
                    "Analytic kinematics not supported for axisymmetric JAM models with JamPy backend."
                )
        else:
            if backend is None:
                backend = "galkin"

        if kwargs_numerics_galkin is not None:
            warnings.warn(
                "`kwargs_numerics_galkin` is deprecated, please use `kwargs_numerics_backend` instead.",
                DeprecationWarning
            )
            if kwargs_numerics_jam is not None:
                warnings.warn(
                    "Both `kwargs_numerics_backend` and `kwargs_numerics_galkin` are provided. "
                    "only `kwargs_numerics_backend` will be used.",
                    UserWarning
                )
            else:
                kwargs_numerics_jam = kwargs_numerics_galkin

        if backend == "jampy":
            kinematics_backend = JAMTDCosmography(
                z_lens,
                z_source,
                kwargs_model,
                cosmo_fiducial=cosmo_fiducial,
                lens_model_kinematics_bool=lens_model_kinematics_bool,
                light_model_kinematics_bool=light_model_kinematics_bool,
                kwargs_seeing=kwargs_seeing,
                kwargs_aperture=kwargs_aperture,
                anisotropy_model=anisotropy_model,
                axial_symmetry=axial_symmetry,
                kwargs_numerics_jam=kwargs_numerics_jam,
                **kwargs_kin_api
            )
        elif backend == "galkin":
            kinematics_backend = TDCosmography(
                z_lens,
                z_source,
                kwargs_model,
                cosmo_fiducial=cosmo_fiducial,
                lens_model_kinematics_bool=lens_model_kinematics_bool,
                light_model_kinematics_bool=light_model_kinematics_bool,
                kwargs_seeing=kwargs_seeing,
                kwargs_aperture=kwargs_aperture,
                anisotropy_model=anisotropy_model,
                analytic_kinematics=analytic_kinematics,
                kwargs_numerics_galkin=kwargs_numerics_jam,
                **kwargs_kin_api
            )
        else:
            raise ValueError("Kinematics backend %s not recognized." % backend)

        self.kinematics_backend = kinematics_backend
        self.backend = backend
        self.axial_symmetry = axial_symmetry

    def time_delays(
        self, kwargs_lens, kwargs_ps, kappa_ext=0, original_ps_position=False
    ):
            """Predicts the time delays of the image positions given the fiducial cosmology
            relative to a straight line without lensing. Negative values correspond to
            images arriving earlier, and positive signs correspond to images arriving later.

            :param kwargs_lens: lens model parameters
            :param kwargs_ps: point source parameters
            :param kappa_ext: external convergence (optional)
            :param original_ps_position: boolean (only applies when first point source model
                is of type 'LENSED_POSITION'), uses the image positions in the model
                parameters and does not re-compute images (which might be differently
                ordered) in case of the lens equation solver
            :return: time delays at image positions for the fixed cosmology in units of days
            """
            return self.kinematics_backend.time_delays(
                kwargs_lens,
                kwargs_ps,
                kappa_ext=kappa_ext,
                original_ps_position=original_ps_position,
            )

    def fermat_potential(self, kwargs_lens, kwargs_ps, original_ps_position=False):
        """Fermat potential (negative sign means earlier arrival time)

        :param kwargs_lens: lens model keyword argument list
        :param kwargs_ps: point source keyword argument list
        :param original_ps_position: boolean (only applies when first point source model
            is of type 'LENSED_POSITION'), uses the image positions in the model
            parameters and does not re-compute images (which might be differently
            ordered) in case of the lens equation solver
        :return: Fermat potential of all the image positions in the first point source
            list entry
        """
        return self.kinematics_backend.fermat_potential(
            kwargs_lens,
            kwargs_ps,
            original_ps_position=original_ps_position,
        )

    def velocity_dispersion_dimension_less(
        self,
        kwargs_lens,
        kwargs_lens_light,
        kwargs_anisotropy,
        q_intrinsic=1.0,
        r_eff=None,
        theta_E=None,
        gamma=None,
    ):
        """Sigma**2 = Dd/Dds * c**2 * J(kwargs_lens, kwargs_light, anisotropy) (Equation
        4.11 in Birrer et al. 2016 or Equation 6 in Birrer et al. 2019) J() is a
        dimensionless and cosmological independent quantity only depending on angular
        units. This function returns J given the lens and light parameters and the
        anisotropy choice without an external mass sheet correction.

        :param kwargs_lens: lens model keyword arguments
        :param kwargs_lens_light: lens light model keyword arguments
        :param kwargs_anisotropy: stellar anisotropy keyword arguments
        :param q_intrinsic: intrinsic axis ratio of the light profile to compute the inclination angle
        :param r_eff: projected half-light radius of the stellar light associated with
            the deflector galaxy, optional, if set to None will be computed in this
            function with default settings that may not be accurate.
        :param theta_E: pre-computed Einstein radius (optional)
        :param gamma: pre-computed power-law slope of mass profile
        :return: dimensionless velocity dispersion (see e.g. Birrer et al. 2016, 2019)
        """
        if self.backend == 'galkin':
            return self.kinematics_backend.velocity_dispersion_dimension_less(
                kwargs_lens,
                kwargs_lens_light,
                kwargs_anisotropy,
                r_eff=r_eff,
                theta_E=theta_E,
                gamma=gamma,
            )
        else:
            return self.kinematics_backend.velocity_dispersion_dimension_less(
                kwargs_lens,
                kwargs_lens_light,
                kwargs_anisotropy,
                q_intrinsic=q_intrinsic,
                r_eff=r_eff,
                theta_E=theta_E,
                gamma=gamma,
            )

    def velocity_dispersion_map_dimension_less(
        self,
        kwargs_lens,
        kwargs_lens_light,
        kwargs_anisotropy,
        q_intrinsic=1.0,
        r_eff=None,
        theta_E=None,
        gamma=None,
        supersampling_factor=1,
        voronoi_bins=None,
    ):
        """Sigma**2 = Dd/Dds * c**2 * J(kwargs_lens, kwargs_light, anisotropy) (Equation
        4.11 in Birrer et al. 2016 or Equation 6 in Birrer et al. 2019) J() is a
        dimensionless and cosmological independent quantity only depending on angular
        units. This function returns J given the lens and light parameters and the
        anisotropy choice without an external mass sheet correction. This routine
        computes the IFU map of the kinematic quantities.

        :param kwargs_lens: lens model keyword arguments
        :param kwargs_lens_light: lens light model keyword arguments
        :param kwargs_anisotropy: stellar anisotropy keyword arguments
        :param q_intrinsic: intrinsic axis ratio of the light profile to compute the inclination angle
        :param r_eff: projected half-light radius of the stellar light associated
            with the deflector galaxy, optional, if set to None will be computed in this
            function with default settings that may not be accurate.
        :param supersampling_factor: supersampling factor for 2D integration grid
        :param voronoi_bins: mapping of the voronoi bins, -1 values for  pixels not
            binned
        :return: dimensionless velocity dispersion (see e.g. Birrer et al. 2016, 2019)
        """
        if self.backend == 'galkin':
            return self.kinematics_backend.velocity_dispersion_map_dimension_less(
                kwargs_lens,
                kwargs_lens_light,
                kwargs_anisotropy,
                r_eff=r_eff,
                theta_E=theta_E,
                gamma=gamma,
                supersampling_factor=supersampling_factor,
                voronoi_bins=voronoi_bins,
            )
        else:
            return self.kinematics_backend.velocity_dispersion_map_dimension_less(
                kwargs_lens,
                kwargs_lens_light,
                kwargs_anisotropy,
                q_intrinsic=q_intrinsic,
                r_eff=r_eff,
                theta_E=theta_E,
                gamma=gamma,
                supersampling_factor=supersampling_factor,
                voronoi_bins=voronoi_bins,
            )

    def ddt_from_time_delay(
        self, d_fermat_model, dt_measured, kappa_s=0, kappa_ds=0, kappa_d=0
    ):
        """Time-delay distance in units of Mpc from the modeled Fermat potential and
        measured time delay from an image pair.

        :param d_fermat_model: relative Fermat potential between two images from the
            same source in units arcsec^2
        :param dt_measured: measured time delay between the same image pair in units of
            days
        :param kappa_s: external convergence from observer to source
        :param kappa_ds: external convergence from lens to source
        :param kappa_d: external convergence form observer to lens
        :return: D_dt, time-delay distance
        """
        return self.kinematics_backend.ddt_from_time_delay(
            d_fermat_model, dt_measured, kappa_s=kappa_s, kappa_ds=kappa_ds, kappa_d=kappa_d
        )

    def ds_dds_from_kinematics(self, sigma_v, J, kappa_s=0, kappa_ds=0):
        """Computes the estimate of the ratio of angular diameter distances Ds/Dds from
        the kinematic estimate of the lens and the measured dispersion.

        :param sigma_v: velocity dispersion [km/s]
        :param J: dimensionless kinematic constraint (see Birrer et al. 2016, 2019)
        :return: Ds/Dds
        """
        return self.kinematics_backend.ds_dds_from_kinematics(
            sigma_v, J, kappa_s=kappa_s, kappa_ds=kappa_ds
        )

    def ddt_dd_from_time_delay_and_kinematics(
        self,
        d_fermat_model,
        dt_measured,
        sigma_v_measured,
        J,
        kappa_s=0,
        kappa_ds=0,
        kappa_d=0,
    ):
        """

        :param d_fermat_model: relative Fermat potential in units arcsec^2
        :param dt_measured: measured relative time delay [days]
        :param sigma_v_measured: 1-sigma Gaussian uncertainty in the measured velocity dispersion
        :param J: modeled dimensionless kinematic estimate
        :param kappa_s: LOS convergence from observer to source
        :param kappa_ds: LOS convergence from deflector to source
        :param kappa_d: LOS convergence from observer to deflector
        :return: D_dt, D_d
        """
        return self.kinematics_backend.ddt_dd_from_time_delay_and_kinematics(
            d_fermat_model,
            dt_measured,
            sigma_v_measured,
            J,
            kappa_s=kappa_s,
            kappa_ds=kappa_ds,
            kappa_d=kappa_d,
        )

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
        """
        For any aperture (single or IFU), uses JamPy with axisymmetric JAM modeling

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
        if self.backend == 'galkin':
            return self.kinematics_backend.velocity_dispersion(
                kwargs_lens,
                kwargs_lens_light,
                kwargs_anisotropy,
                r_eff=r_eff,
                theta_E=theta_E,
                gamma=gamma,
                kappa_ext=kappa_ext,
                voronoi_bins=voronoi_bins,
            )
        else:
            return self.kinematics_backend.velocity_dispersion(
                kwargs_lens,
                kwargs_lens_light,
                kwargs_anisotropy,
                q_intrinsic=q_intrinsic,
                r_eff=r_eff,
                theta_E=theta_E,
                gamma=gamma,
                kappa_ext=kappa_ext,
                voronoi_bins=voronoi_bins,
            )

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
        """
        For a IFU measurements (regular or binned grid, or shells),
        it uses JamPy with  axisymmetric JAM modeling

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
        if self.backend == 'galkin':
            return self.kinematics_backend.velocity_dispersion_map(
                kwargs_lens,
                kwargs_lens_light,
                kwargs_anisotropy,
                r_eff=r_eff,
                theta_E=theta_E,
                gamma=gamma,
                kappa_ext=kappa_ext,
                supersampling_factor=supersampling_factor,
                voronoi_bins=voronoi_bins,
            )
        else:
            return self.kinematics_backend.velocity_dispersion_map(
                kwargs_lens,
                kwargs_lens_light,
                kwargs_anisotropy,
                q_intrinsic=q_intrinsic,
                r_eff=r_eff,
                theta_E=theta_E,
                gamma=gamma,
                kappa_ext=kappa_ext,
                supersampling_factor=supersampling_factor,
                voronoi_bins=voronoi_bins,
            )

    def velocity_dispersion_analytical(self, theta_E, gamma, r_eff, r_ani, kappa_ext=0):
        """Computes the LOS velocity dispersion of the lens within a slit of size R_slit
        x dR_slit and seeing psf_fwhm. The assumptions are a Hernquist light profile and
        the spherical power-law lens model at the first position and an Osipkov and
        Merritt ('OM') stellar anisotropy distribution.

        Further information can be found in the AnalyticKinematics() class.

        :param theta_E: Einstein radius
        :param gamma: power-low slope of the mass profile (=2 corresponds to isothermal)
        :param r_ani: anisotropy radius in units of angles
        :param r_eff: projected half-light radius
        :param kappa_ext: external convergence not accounted in the lens models
        :return: velocity dispersion in units [km/s]
        """
        if self.backend != 'galkin':
            raise ValueError(
                "Analytical velocity dispersion calculation is only supported with the Galkin backend."
            )
        return self.kinematics_backend.velocity_dispersion_analytical(
            theta_E, gamma, r_eff, r_ani, kappa_ext=kappa_ext
        )

    def kinematics_modeling_settings(
        self,
        anisotropy_model,
        kwargs_numerics_backend=None,
        kwargs_numerics_galkin=None,  # deprecated
        analytic_kinematics=False,
        Hernquist_approx=False,
        MGE_light=False,
        MGE_mass=False,
        kwargs_mge_light=None,
        kwargs_mge_mass=None,
        sampling_number=1000,
        num_kin_sampling=1000,
        num_psf_sampling=100,
    ):
        """Return the settings for the kinematic modeling.

        :param anisotropy_model: type of stellar anisotropy model. See details in
            MamonLokasAnisotropy() class of lenstronomy.GalKin.anisotropy
        :param kwargs_numerics_backend: numerical settings for the integrated
            line-of-sight velocity dispersion
        :param kwargs_numerics_galkin: numerical settings for the integrated
            line-of-sight velocity dispersion (deprecated, use kwargs_numerics_backend)
        :param analytic_kinematics: boolean, if True, used the analytic JAM modeling for
            a power-law profile on top of a Hernquist light profile
            ATTENTION: This may not be accurate for your specific problem!
        :param Hernquist_approx: bool, if True, uses a Hernquist light profile matched
            to the half light radius of the deflector light profile to compute the kinematics
        :param MGE_light: bool, if true performs the MGE for the light distribution
        :param MGE_mass: bool, if true performs the MGE for the mass distribution
        :param kwargs_mge_mass: keyword arguments that go into the MGE decomposition
            routine
        :param kwargs_mge_light: keyword arguments that go into the MGE decomposition
            routine
        :param sampling_number: number of spectral rendering on a single slit
        :param num_kin_sampling: number of kinematic renderings on a total IFU
        :param num_psf_sampling: number of PSF displacements for each kinematic
            rendering on the IFU
        :return: updated settings
        """
        if kwargs_numerics_galkin is not None:
            warnings.warn(
                "`kwargs_numerics_galkin` is deprecated, please use `kwargs_numerics_backend` instead.",
                DeprecationWarning
            )
            if kwargs_numerics_backend is not None:
                warnings.warn(
                    "Both `kwargs_numerics_backend` and `kwargs_numerics_galkin` are provided. "
                    "only `kwargs_numerics_backend` will be used.",
                    UserWarning
                )
            else:
                kwargs_numerics_backend = kwargs_numerics_galkin

        if self.backend == 'galkin':
            self.kinematics_backend.kinematics_modeling_settings(
                anisotropy_model,
                analytic_kinematics=analytic_kinematics,
                Hernquist_approx=Hernquist_approx,
                MGE_light=MGE_light,
                MGE_mass=MGE_mass,
                kwargs_numerics_galkin=kwargs_numerics_backend,
                kwargs_mge_light=kwargs_mge_light,
                sampling_number=sampling_number,
                num_kin_sampling=num_kin_sampling,
                num_psf_sampling=num_psf_sampling,
            )

        else:
            if analytic_kinematics:
                raise ValueError(
                    "Analytic kinematics not supported for axisymmetric JAM models with JamPy backend."
                )
            self.kinematics_backend.kinematics_modeling_settings(
                anisotropy_model,
                kwargs_numerics_jam=kwargs_numerics_backend,
                MGE_light=MGE_light,
                kwargs_mge_light=kwargs_mge_light,
                MGE_mass=MGE_mass,
                kwargs_mge_mass=kwargs_mge_mass,
            )
