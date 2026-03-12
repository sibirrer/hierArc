from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
import numpy as np
import mgefit as mge
from copy import deepcopy


class MassProfile:
    """Computes radial surface density for a list of lenstronomy mass profiles.
    """

    def __init__(self, profile_list):
        self.profile_list = profile_list
        self.mass_model = LensModel(profile_list)
        self.lens_analysis = LensProfileAnalysis(self.mass_model)

    def radial_convergence(self, r, kwargs_list):
        """
        convergence radial profile
        :param r: projected radius in angular units
        :param kwargs_list: list of keyword arguments of lens model parameters matching the
        lens model classes
        :return: surface mass density at radius r (in angular units, modulo epsilon_crit)
        """
        kwargs_list = self._parse_kwargs(kwargs_list)
        center_x = kwargs_list[0].get("center_x", 0.0)
        center_y = kwargs_list[0].get("center_y", 0.0)
        kappa = self.lens_analysis.radial_lens_profile(
            r, kwargs_list, center_x, center_y
        )
        return np.asarray(kappa)

    def einstein_radius(self, kwargs_list):
        """Einstein radius of the mass profile, used to scale the radial range where the
        MGE is fitted."""
        if (len(self.profile_list) == 1) and ("theta_E" in kwargs_list[0]):
            return kwargs_list[0]["theta_E"]
        else:
            kwargs_list = self._parse_kwargs(kwargs_list)
            if "center_x" not in kwargs_list[0]:
                kwargs_list[0]["center_x"] = 0.0
                kwargs_list[0]["center_y"] = 0.0
            return self.lens_analysis.effective_einstein_radius(kwargs_list)

    def mge_mass(self, r_mge, kwargs_list, n_gauss, linear_solver=True, mge_kwargs=None):
        # TODO: cache the MGE fit for repeated calls with same kwargs_list
        if (len(self.profile_list) == 1) and (
            self.profile_list[0] in ["MULTI_GAUSSIAN", "MULTI_GAUSSIAN_ELLIPSE_KAPPA"]
        ):
            sigma_mass = np.asarray(kwargs_list[0]["sigma"])
            surf_mass = np.asarray(kwargs_list[0]["amp"]) / (2 * np.pi * sigma_mass**2)
            # clean zero amplitudes as Jampy doesn't like them
            zero_surf = surf_mass == 0
            surf_mass = surf_mass[~zero_surf]
            sigma_mass = sigma_mass[~zero_surf]
        else:
            if mge_kwargs is None:
                mge_kwargs = {}
            theta_E = self.einstein_radius(kwargs_list)
            radial_density = self.radial_convergence(
                r_mge * theta_E,
                kwargs_list
            )
            mge_mass = mge.fit_1d(
                r_mge * theta_E,
                radial_density,
                ngauss=n_gauss,
                linear=linear_solver,
                plot=False,
                quiet=True,
                **mge_kwargs,
            )
            sigma_mass = mge_mass.sol[1]  # in arcsec
            surf_mass = mge_mass.sol[0] / (np.sqrt(2 * np.pi) * sigma_mass)
        return surf_mass, sigma_mass

    def _parse_kwargs(self, kwargs_list):
        """
        removes e1 and e2 kwargs if not present in the profile
        :param kwargs_list: list of keyword arguments of light profiles (see LightModule)
        :return: parsed arguments
        """
        kwargs_list_copy = deepcopy(kwargs_list)
        kwargs_list_new = []
        profiles = self.mass_model.lens_model.func_list
        for kwargs, profile in zip(kwargs_list_copy, profiles):
            if ("e1" in kwargs) and ("e1" not in profile.param_names):
                kwargs.pop("e1")
            elif ("e1" not in kwargs) and ("e1" in profile.param_names):
                kwargs['e1'] = 0.
            if ("e2" in kwargs) and ("e2" not in profile.param_names):
                kwargs.pop("e2")
            elif ("e2" not in kwargs) and ("e2" in profile.param_names):
                kwargs['e2'] = 0.
            kwargs_list_new.append(kwargs)
        return kwargs_list_new
