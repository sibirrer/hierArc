from lenstronomy.Analysis.td_cosmography import TDCosmography
from hierarc.JAM.jam_td_cosmography import JAMTDCosmography


def get_kinematics_backend(
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
    kwargs_numerics_backend=None,
    **kwargs_kin_api
):
    """
    class initializer to select the kinematics calculation backend: JamPy or Lenstronomy/Galkin
    """
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
            kwargs_numerics_jam=kwargs_numerics_backend,
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
            kwargs_numerics_galkin=kwargs_numerics_backend,
            **kwargs_kin_api
        )
    else:
        raise ValueError("Kinematics backend %s not recognized." % backend)
    return kinematics_backend
