from lenstronomy.LensModel.single_plane import SinglePlane
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
        return self.mass_model.density(r, kwargs_list)

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
