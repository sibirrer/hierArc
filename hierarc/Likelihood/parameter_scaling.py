__author__ = "sibirrer", "ajshajib"

from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator
import numpy as np


class ParameterScalingSingleAperture(object):
    """Class to manage anisotropy scaling for single slit observation."""

    def __init__(self, param_arrays, scaling_grid):
        """

        :param param_arrays: list of arrays of interpolated parameter values
        :param scaling_grid: array with the scaling of J() for single slit
        """
        self._evalute_scaling = False
        # check if param arrays is 1d list or 2d list
        if param_arrays is not None and scaling_grid is not None:
            if isinstance(param_arrays, list):
                self._dim_scaling = len(param_arrays)
            else:
                self._dim_scaling = 1
            print(param_arrays, len(param_arrays))
            print(self._dim_scaling)
            if self._dim_scaling == 1:
                print(param_arrays)
                self._f_ani = interp1d(param_arrays, scaling_grid, kind="linear")
            elif self._dim_scaling == 2:
                self._f_ani = interp2d(param_arrays[0], param_arrays[1], scaling_grid.T)
            else:
                self._f_ani = RegularGridInterpolator(
                    (arr for arr in param_arrays),
                    scaling_grid,
                )
            self._evalute_scaling = True

    def param_scaling(self, param_array):
        """

        :param aniso_param_array: anisotropy parameter array
        :return: scaling J(a_ani) for single slit
        """
        if self._evalute_scaling is not True or param_array is None:
            return 1
        if self._dim_scaling == 1:
            return self._f_ani(param_array[0])
        elif self._dim_scaling == 2:
            return self._f_ani(param_array[0], param_array[1])[0]
        else:
            return self._f_ani(param_array)


class ParameterScalingIFU(object):
    """Class to manage model parameter and anisotropy scalings for IFU data."""

    def __init__(
        self, anisotropy_model="NONE", param_arrays=None, scaling_grid_list=None
    ):
        """

        :param anisotropy_model: string, either 'NONE', 'OM' or 'GOM'
        :param param_array: array of parameter values
        :param scaling_grid_list: list of array with the scalings of J() for each IFU
        :param gamma_in_array: array of gamma_in values
        :param m2l_array: array of M/L values
        """
        self._anisotropy_model = anisotropy_model
        self._evalute_ani = False
        if (
            param_arrays is not None
            and scaling_grid_list is not None
            and self._anisotropy_model != "NONE"
        ):
            self._evalute_ani = True
            self._anisotropy_scaling_list = []
            self._f_ani_list = []
            for scaling_grid in scaling_grid_list:
                self._anisotropy_scaling_list.append(
                    ParameterScalingSingleAperture(param_arrays, scaling_grid)
                )

            if isinstance(param_arrays, list):
                self._dim_scaling = len(param_arrays)
            else:
                self._dim_scaling = 1

            if anisotropy_model in ["OM", "const"]:
                self._ani_param_min = np.min(param_arrays)
                self._ani_param_max = np.max(param_arrays)
            elif anisotropy_model == "GOM":
                self._ani_param_min = [min(param_arrays[0]), min(param_arrays[1])]
                self._ani_param_max = [max(param_arrays[0]), max(param_arrays[1])]
            else:
                raise ValueError(
                    "anisotropy scaling with dimension %s does not match anisotropy model %s"
                    % (self._dim_scaling, self._anisotropy_model)
                )

    def param_scaling(self, param_array):
        """

        :param param_array: parameter array for scaling
        :return: scaling J(a_ani) for the IFU's
        """
        if self._evalute_ani is not True or param_array is None:
            return [1]
        scaling_list = []
        for scaling_class in self._anisotropy_scaling_list:
            scaling = scaling_class.param_scaling(param_array)
            scaling_list.append(scaling)
        return np.array(scaling_list)

    def draw_anisotropy(
        self, a_ani=None, a_ani_sigma=0, beta_inf=None, beta_inf_sigma=0
    ):
        """Draw Gaussian distribution and re-sample if outside bounds.

        :param a_ani: mean of the distribution
        :param a_ani_sigma: std of the distribution
        :param beta_inf: anisotropy at infinity (relevant for GOM model)
        :param beta_inf_sigma: std of beta_inf distribution
        :return: random draw from the distribution
        """
        if self._anisotropy_model in ["OM", "const"]:
            if a_ani < self._ani_param_min or a_ani > self._ani_param_max:
                raise ValueError(
                    "anisotropy parameter is out of bounds of the interpolated range!"
                )
            # we draw a linear gaussian for 'const' anisotropy and a scaled proportional one for 'OM
            if self._anisotropy_model == "OM":
                a_ani_draw = np.random.normal(a_ani, a_ani_sigma * a_ani)
            else:
                a_ani_draw = np.random.normal(a_ani, a_ani_sigma)
            if a_ani_draw < self._ani_param_min or a_ani_draw > self._ani_param_max:
                return self.draw_anisotropy(a_ani, a_ani_sigma)
            return np.array([a_ani_draw])
        elif self._anisotropy_model in ["GOM"]:
            if (
                a_ani < self._ani_param_min[0]
                or a_ani > self._ani_param_max[0]
                or beta_inf < self._ani_param_min[1]
                or beta_inf > self._ani_param_max[1]
            ):
                raise ValueError(
                    "anisotropy parameter is out of bounds of the interpolated range!"
                )
            a_ani_draw = np.random.normal(a_ani, a_ani_sigma * a_ani)
            beta_inf_draw = np.random.normal(beta_inf, beta_inf_sigma)
            if (
                a_ani_draw < self._ani_param_min[0]
                or a_ani_draw > self._ani_param_max[0]
                or beta_inf_draw < self._ani_param_min[1]
                or beta_inf_draw > self._ani_param_max[1]
            ):
                return self.draw_anisotropy(
                    a_ani, a_ani_sigma, beta_inf, beta_inf_sigma
                )
            return np.array([a_ani_draw, beta_inf_draw])
        return None

    def ani_scaling(self, arr):
        return self.param_scaling(arr)
