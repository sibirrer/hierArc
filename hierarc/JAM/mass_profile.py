import numpy as np
from lenstronomy.LensModel.single_plane import SinglePlane
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
from copy import deepcopy


class MassProfile:
    """
    Computes radial 3D density for a list of lenstronomy mass profiles.
    Assumes that all the profiles share the same geometry.
    """

    def __init__(
        self,
        profile_list
    ):
        # exclude convergence and shear profiles as they do not have 3D density or Einstein radius
        profile_list = [p for p in profile_list if p not in ['CONVERGENCE', 'SHEAR']]
        self.profile_list = profile_list
        self.mass_model = SinglePlane(profile_list)

    def radial_density(self, r, kwargs_list):
        """
        3D density at radius r
        :param r: 3D radius in angular units
        :param kwargs_list: list of keyword arguments of lens model parameters matching the
            lens model classes
        :return: mass density at radius r (in angular units, modulo epsilon_crit)
        """
        kwargs_list = self._circularize_kwargs(kwargs_list)
        density = np.zeros_like(r)
        for i, func in enumerate(self.mass_model.func_list):
            kwargs_i = {
                k: v
                for k, v in kwargs_list[i].items()
                if k not in ["center_x", "center_y"]
            }
            try:
                density_i = func.density_lens(r, **kwargs_i)
            except ValueError:
                density_i = func.density(r, **kwargs_i)
            density += density_i
        return density

    def einstein_radius(self, kwargs_list):
        if (len(self.profile_list) == 1) and ('theta_E' in kwargs_list[0]):
            return kwargs_list[0]['theta_E']
        else:
            analysis = LensProfileAnalysis(LensModel(self.profile_list))
            return analysis.effective_einstein_radius(kwargs_list)

    def _circularize_kwargs(self, kwargs_list):
        """
        :param kwargs_list: list of keyword arguments of light profiles (see LightModule)
        :return: circularized arguments
        """
        kwargs_list_copy = deepcopy(kwargs_list)
        kwargs_list_new = []
        for kwargs, profile in zip(kwargs_list_copy, self.mass_model.func_list):
            if "e1" in kwargs:
                if "e1" in profile.param_names:
                    kwargs["e1"] = 0.0
                else:
                    kwargs.pop("e1")
            if "e2" in kwargs:
                if "e2" in profile.param_names:
                    kwargs["e2"] = 0.0
                else:
                    kwargs.pop("e2")
            kwargs_list_new.append(
                {
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["center_x", "center_y"]
                }
            )
        return kwargs_list_new

    def __repr__(self):
        return f"MassProfile[{','.join(self.profile_list)}]"
