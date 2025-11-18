from lenstronomy.LightModel.light_model import LightModel
from copy import deepcopy


class LightProfile:
    """
    Computes radial surface brightness for a list of lenstronomy mass profiles.
    Assumes that all the profiles share the same geometry.
    """

    def __init__(
        self,
        profile_list
    ):
        self.profile_list = profile_list
        self.light_model = LightModel(profile_list)

    def radial_surface_brightness(self, r, kwargs_list):
        kwargs_list = self._circularize_kwargs(kwargs_list)
        return self.light_model.surface_brightness(
            x=r,
            y=0,
            kwargs_list=kwargs_list
        )

    def _circularize_kwargs(self, kwargs_list):
        """
        :param kwargs_list: list of keyword arguments of light profiles (see LightModule)
        :return: circularized arguments
        """
        kwargs_list_copy = deepcopy(kwargs_list)
        kwargs_list_new = []
        for kwargs, profile in zip(kwargs_list_copy, self.light_model.func_list):
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
