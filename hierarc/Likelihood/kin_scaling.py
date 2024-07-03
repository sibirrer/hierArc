__author__ = "sibirrer", "ajshajib"

from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator
import numpy as np


class KinScalingParamManager(object):
    """Class to handle the sorting of parameters in the kinematics scaling."""

    def __init__(self, j_kin_scaling_param_name_list):
        """

        :param j_kin_scaling_param_name_list: list of strings for the parameters as they are interpolated in the same
         order as j_kin_scaling_grid
        """
        if j_kin_scaling_param_name_list is None:
            self._param_list = []
        else:
            self._param_list = j_kin_scaling_param_name_list
        self._num_param = len(self._param_list)

    @property
    def num_scaling_dim(self):
        """Number of parameter dimensions for kinematic scaling.

        :return: number of scaling dimensions
        :rtype: int
        """
        return self._num_param

    def kwargs2param_array(self, kwargs):
        """Converts dictionary to sorted array in same order as interpolation grid.

        :param kwargs: dictionary of all model components, must include the one that are
            interpolated
        :return: sorted list of parameters to interpolate
        """
        param_array = []
        for param in self._param_list:
            if param not in kwargs:
                raise ValueError(
                    "key %s not in parameters and hence kinematic scaling not possible"
                    % param
                )
            param_array.append(kwargs.get(param))
        return param_array

    def param_array2kwargs(self, param_array):
        """Inverse function of kwargs2param_array for a given param_array returns the
        dictionary split in anisotropy and lens models.

        :param param_array:
        :return: kwargs_anisotropy, kwargs_lens
        """
        kwargs_anisotropy, kwargs_lens = {}, {}
        for i, param in enumerate(self._param_list):
            if param in ["gamma_in", "gamma_pl", "log_m2l"]:
                kwargs_lens[param] = param_array[i]
            else:
                kwargs_anisotropy[param] = param_array[i]
        return kwargs_anisotropy, kwargs_lens


class ParameterScalingSingleMeasurement(object):
    """Class to manage anisotropy scaling for single slit observation."""

    def __init__(self, param_grid_axes, j_kin_scaling_grid):
        """

        :param param_grid_axes: list of arrays of interpolated parameter values
        :param j_kin_scaling_grid: array with the scaling of J() for single measurement bin in same dimensions as the
         param_arrays
        """
        self._evalute_scaling = False
        # check if param arrays is 1d list or 2d list
        if param_grid_axes is not None and j_kin_scaling_grid is not None:
            if isinstance(param_grid_axes, list):
                self._dim_scaling = len(param_grid_axes)
            else:
                self._dim_scaling = 1
                param_grid_axes = [param_grid_axes]

            if self._dim_scaling == 1:
                self._f_ani = interp1d(
                    param_grid_axes[0], j_kin_scaling_grid, kind="linear"
                )
            elif self._dim_scaling == 2:
                self._f_ani = interp2d(
                    param_grid_axes[0], param_grid_axes[1], j_kin_scaling_grid.T
                )
            else:
                self._f_ani = RegularGridInterpolator(
                    tuple(param_grid_axes),
                    j_kin_scaling_grid,
                )
            self._evalute_scaling = True

    def j_scaling(self, param_array):
        """

        :param param_array: sorted list of parameters for the interpolation function
        :return: scaling J(a_ani) for single slit
        """

        if self._evalute_scaling is not True or len(param_array) == 0:
            return 1
        if self._dim_scaling == 1:
            return self._f_ani(param_array[0])
        elif self._dim_scaling == 2:
            return self._f_ani(param_array[0], param_array[1])[0]
        else:
            return self._f_ani(param_array)[0]


class KinScaling(KinScalingParamManager):
    """Class to manage model parameter and anisotropy scalings for IFU data."""

    def __init__(
        self,
        j_kin_scaling_param_axes=None,
        j_kin_scaling_grid_list=None,
        j_kin_scaling_param_name_list=None,
    ):
        """

        :param j_kin_scaling_param_axes: array of parameter values for each axes of j_kin_scaling_grid
        :param j_kin_scaling_grid_list: list of array with the scalings of J() for each IFU
        :param j_kin_scaling_param_name_list: list of strings for the parameters as they are interpolated in the same
         order as j_kin_scaling_grid
        """

        self._param_arrays = j_kin_scaling_param_axes
        if (
            not isinstance(j_kin_scaling_param_axes, list)
            and j_kin_scaling_param_name_list is not None
        ):
            self._param_arrays = [j_kin_scaling_param_axes]
        self._evaluate_scaling = False
        self._is_log_m2l_population_level = False
        if (
            j_kin_scaling_param_axes is not None
            and j_kin_scaling_grid_list is not None
            and j_kin_scaling_param_name_list is not None
        ):
            self._evaluate_scaling = True
            self._j_scaling_ifu = []
            self._f_ani_list = []
            for scaling_grid in j_kin_scaling_grid_list:
                self._j_scaling_ifu.append(
                    ParameterScalingSingleMeasurement(
                        j_kin_scaling_param_axes, scaling_grid
                    )
                )

        if isinstance(j_kin_scaling_param_axes, list):
            self._dim_scaling = len(j_kin_scaling_param_axes)
        else:
            self._dim_scaling = 1
        KinScalingParamManager.__init__(
            self, j_kin_scaling_param_name_list=j_kin_scaling_param_name_list
        )

    def param_bounds_interpol(self):
        """Minimum and maximum bounds of parameters that are being used to call
        interpolation function.

        :return: dictionaries of minimum and maximum bounds
        """
        kwargs_min, kwargs_max = {}, {}
        if self._evaluate_scaling is True:
            for i, key in enumerate(self._param_list):
                kwargs_min[key] = min(self._param_arrays[i])
                kwargs_max[key] = max(self._param_arrays[i])
        return kwargs_min, kwargs_max

    def kin_scaling(self, kwargs_param):
        """

        :param kwargs_param: dictionary of parameters for scaling the kinematics
        :return: scaling J(a_ani) for the IFU's
        """
        if kwargs_param is None:
            return np.ones(self._dim_scaling)
        param_array = self.kwargs2param_array(kwargs_param)
        if self._evaluate_scaling is not True or len(param_array) == 0:
            return np.ones(self._dim_scaling)
        scaling_list = []
        for scaling_class in self._j_scaling_ifu:
            scaling = scaling_class.j_scaling(param_array)
            scaling_list.append(scaling)
        return np.array(scaling_list)
