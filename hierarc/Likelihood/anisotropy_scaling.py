__author__ = 'sibirrer'
from scipy.interpolate import interp1d, RectBivariateSpline
import numpy as np


class AnisotropyScaling(object):
    """
    class to manage anisotropy scaling for single slit observation
    """

    def __init__(self, ani_param_array, ani_scaling_array):
        """

        :param ani_param_array: array of anisotropy parameter value
        :param ani_scaling_array: array with the scaling of J() for single slit
        """
        self._evalute_ani = False
        if ani_param_array is not None and ani_scaling_array is not None:
            self._dim_scaling = ani_param_array.ndim
            if self._dim_scaling == 1:
                self._f_ani = interp1d(ani_param_array, ani_scaling_array, kind='cubic')
            elif self._dim_scaling == 2:
                self._f_ani = RectBivariateSpline(ani_param_array[0], ani_param_array[1], ani_scaling_array)
            else:
                raise ValueError('anisotropy scaling with dimension %s not supported.' % self._dim_scaling)
            self._ani_param_min = np.min(ani_param_array, axis=0)
            self._ani_param_max = np.max(ani_param_array, axis=0)
            if self._dim_scaling > 1:
                assert len(self._ani_param_min) == self._dim_scaling
            self._evalute_ani = True

    def ani_scaling(self, aniso_param_array):
        """

        :param aniso_param_array: anisotropy parameter array
        :return: scaling J(a_ani) for single slit
        """
        if not self._evalute_ani is True or aniso_param_array is None:
            return 1
        if self._dim_scaling == 1:
            return self._f_ani(aniso_param_array[0])
        elif self._dim_scaling == 2:
            return self._f_ani(aniso_param_array[0], aniso_param_array[1])


class AnisotropyScalingIFU(object):
    """
    class to manage anisotropy scalings for IFU data
    """

    def __init__(self, ani_param_array, ani_scaling_array_list):
        """

        :param ani_param_array: array of anisotropy parameter value
        :param ani_scaling_array_list: list of array with the scalings of J() for each IFU
        """
        self._evalute_ani = False
        if ani_param_array is not None and ani_scaling_array_list is not None:
            self._f_ani_list = []
            for ani_scaling_array in ani_scaling_array_list:
                self._f_ani_list.append(interp1d(ani_param_array, ani_scaling_array, kind='cubic'))
            self._ani_param_min = np.min(ani_param_array)
            self._ani_param_max = np.max(ani_param_array)
            self._evalute_ani = True

    def ani_scaling(self, a_ani):
        """

        :param a_ani: anisotropy parameter
        :return: scaling J(a_ani) for the IFU's
        """
        if not self._evalute_ani is True or a_ani is None:
            return 1
        scaling_list = []
        for f_ani in self._f_ani_list:
            scaling = f_ani(a_ani)
            scaling_list.append(scaling)
        return np.array(scaling_list)
