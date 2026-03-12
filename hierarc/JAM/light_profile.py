from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Analysis.light_profile import LightProfileAnalysis
import numpy as np
import mgefit as mge
from copy import deepcopy


class LightProfile:
    """Computes radial surface brightness for a list of lenstronomy mass profiles.

    Assumes that all the profiles share the same geometry.
    """

    def __init__(self, profile_list):
        # we only need the radial profile, so no ellipticity is considered
        self.profile_list = profile_list
        self.light_model = LightModel(profile_list)
        self.light_analysis = LightProfileAnalysis(self.light_model)

    def radial_surface_brightness(self, r, kwargs_list):
        kwargs_list = self._parse_kwargs(kwargs_list)
        center_x = kwargs_list[0].get("center_x", 0.0)
        center_y = kwargs_list[0].get("center_y", 0.0)
        surf = self.light_analysis.radial_light_profile(
            r, kwargs_list, center_x, center_y
        )
        return np.asarray(surf)

    def effective_radius(self, kwargs_list):
        """Half-light radius of the light profile, used to scale the radial range where
        the MGE is fitted."""
        if len(self.profile_list) == 1:
            if self.profile_list[0] == "SERSIC":
                return kwargs_list[0]["R_sersic"]
            elif self.profile_list[0] == "HERNQUIST":
                return 1.8153 * kwargs_list[0]["Rs"]
        kwargs_list = self._parse_kwargs(kwargs_list)
        center_x = kwargs_list[0].get("center_x", 0.0)
        center_y = kwargs_list[0].get("center_y", 0.0)
        return self.light_analysis.half_light_radius(
            kwargs_light=kwargs_list,
            center_x=center_x,
            center_y=center_y,
            grid_spacing=0.02,
            grid_num=200,
        )

    def mge_lum_tracer(
        self, r_mge, kwargs_list, n_gauss, linear_solver=True, mge_kwargs=None
    ):
        # TODO: cache the MGE fit for repeated calls with same kwargs_list
        if (len(self.profile_list) == 1) and (
            self.profile_list[0] in ["MULTI_GAUSSIAN", "MULTI_GAUSSIAN_ELLIPSE"]
        ):
            sigma_lum = np.asarray(kwargs_list[0]["sigma"])
            surf_lum = np.asarray(kwargs_list[0]["amp"]) / (2 * np.pi * sigma_lum**2)
            # clean zero amplitudes as Jampy doesn't like them
            zero_surf = surf_lum == 0
            surf_lum = surf_lum[~zero_surf]
            sigma_lum = sigma_lum[~zero_surf]
        else:
            if mge_kwargs is None:
                mge_kwargs = {}
            r_eff = self.effective_radius(kwargs_list)
            light_1d = self.radial_surface_brightness(r_mge * r_eff, kwargs_list)
            mge_lum = mge.fit_1d(
                r_mge * r_eff,
                light_1d,
                ngauss=n_gauss,
                linear=linear_solver,
                plot=False,
                quiet=True,
                **mge_kwargs,
            )
            sigma_lum = mge_lum.sol[1]  # in arcsec
            # convert to surface brightness
            surf_lum = mge_lum.sol[0] / (np.sqrt(2 * np.pi) * sigma_lum)
        return surf_lum, sigma_lum

    def _parse_kwargs(self, kwargs_list):
        """Removes e1 and e2 kwargs if not present in the profile :param kwargs_list:
        list of keyword arguments of light profiles (see LightModule) :return: parsed
        arguments."""
        kwargs_list_copy = deepcopy(kwargs_list)
        kwargs_list_new = []
        profiles = self.light_model.func_list
        for kwargs, profile in zip(kwargs_list_copy, profiles):
            if ("e1" in kwargs) and ("e1" not in profile.param_names):
                kwargs.pop("e1")
            elif ("e1" not in kwargs) and ("e1" in profile.param_names):
                kwargs["e1"] = 0.0
            if ("e2" in kwargs) and ("e2" not in profile.param_names):
                kwargs.pop("e2")
            elif ("e2" not in kwargs) and ("e2" in profile.param_names):
                kwargs["e2"] = 0.0
            kwargs_list_new.append(kwargs)
        return kwargs_list_new
