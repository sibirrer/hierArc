from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Analysis.light_profile import LightProfileAnalysis
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

    def effective_radius(self, kwargs_list):
        if len(self.profile_list) == 1:
            if self.profile_list[0] in ['SERSIC', 'SERSIC_ELLIPSE']:
                return kwargs_list[0]['R_sersic']
            elif self.profile_list[0] in ['HERNQUIST']:
                return 1.8153 * kwargs_list[0]['Rs']
        light_analysis = LightProfileAnalysis(self.light_model)
        center_x = self.profile_list[0]['center_x']
        center_y = self.profile_list[0]['center_y']
        kwargs_list = self._circularize_kwargs(kwargs_list)
        return light_analysis.half_light_radius(
            kwargs_light=kwargs_list,
            center_x=center_x,
            center_y=center_y,
            grid_spacing=0.02,
            grid_num=200,
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
