__author__ = 'sibirrer'
from scipy.interpolate import interp1d
import numpy as np


class AnisotropyScaling(object):
    """
    class to manage anisotropy scaling for single slit observation
    """

    def __init__(self, ani_param_array, ani_scaling_array):
        """

        :param ani_param_array: array of anisotropy parameter value
        :param ani_scaling_array: array with the scalings of J() for single slit
        """
        self._evalute_ani = False
        if ani_param_array is not None and ani_scaling_array is not None:
            self._f_ani = interp1d(ani_param_array, ani_scaling_array, kind='cubic')
            self._ani_param_min = np.min(ani_param_array)
            self._ani_param_max = np.max(ani_param_array)
            self._evalute_ani = True

    def ani_scaling(self, a_ani):
        """

        :param a_ani: anisotropy parameter
        :return: scaling J(a_ani) for single slit
        """
        if not self._evalute_ani is True or a_ani is None:
            return 1
        return self._f_ani(a_ani)


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
