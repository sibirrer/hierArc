from lenstronomy.LensModel.Profiles.spp import SPP as SPP
from lenstronomy.LensModel.Profiles.epl import EPL as EPL
from lenstronomy.LensModel.single_plane import SinglePlane
from copy import deepcopy
import numpy as np


class MassProfile:
    """
    Computes radial convergence for a list of lenstronomy mass profiles.
    Assumes that all the profiles share the same geometry.
    """

    def __init__(
        self,
        profile_list
    ):
        self.mass_model = SinglePlane(profile_list)

    def radial_convergence(self, r, kwargs_list):
        kwargs_list = self._circularize_kwargs(kwargs_list)
        r = np.array(r, dtype=float)
        k_r = np.zeros_like(r)
        for i, func in enumerate(self.mass_model.func_list):
            k_r += self._component_convergence(r, func, kwargs_list[i])
        return k_r

    def _component_convergence(self, r, component, component_kwargs):
        if isinstance(component, SPP) or isinstance(component, EPL):
            return self.power_law_convergence(r, **component_kwargs)
        else:
            # not optimal but generic way to get the radial convergence
            f_xx, f_xy, f_yx, f_yy = component.hessian(x=r, y=0, **component_kwargs)
            return (f_xx + f_yy) / 2

    @staticmethod
    def power_law_convergence(r, theta_E, gamma, center_x=0, center_y=0):
        """
        :param r: projected radius [arcsec]
        :param theta_E: Einstein radius [arcsec]
        :param gamma: power-law slope
        :return: convergence at radius r
        """
        return (3 - gamma) / 2 * (theta_E / r) ** (gamma - 1)

    @staticmethod
    def _circularize_kwargs(kwargs_list):
        """
        :param kwargs_list: list of keyword arguments of light profiles (see LightModule)
        :return: circularized arguments
        """
        # TODO make sure averaging is done azimuthally
        kwargs_list_copy = deepcopy(kwargs_list)
        kwargs_list_new = []
        for kwargs in kwargs_list_copy:
            if "e1" in kwargs:
                kwargs["e1"] = 0
            if "e2" in kwargs:
                kwargs["e2"] = 0
            kwargs_list_new.append(
                {
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["center_x", "center_y"]
                }
            )
        return kwargs_list_new
