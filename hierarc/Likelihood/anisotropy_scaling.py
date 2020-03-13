__author__ = 'sibirrer'
from scipy.interpolate import interp1d, interp2d
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
            if isinstance(ani_param_array, list):
                self._dim_scaling = len(ani_param_array)
            else:
                self._dim_scaling = 1
           # self._dim_scaling = ani_param_array.ndim
            if self._dim_scaling == 1:
                self._f_ani = interp1d(ani_param_array, ani_scaling_array, kind='linear')
                #self._ani_param_min = np.min(ani_param_array)
                #self._ani_param_max = np.max(ani_param_array)
            elif self._dim_scaling == 2:
                self._f_ani = interp2d(ani_param_array[0], ani_param_array[1], ani_scaling_array.T)
                #self._ani_param_min = [min(ani_param_array[0]), min(ani_param_array[1])]
                #self._ani_param_max = [max(ani_param_array[0]), max(ani_param_array[1])]
            else:
                raise ValueError('anisotropy scaling with dimension %s not supported.' % self._dim_scaling)
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
            return self._f_ani(aniso_param_array[0], aniso_param_array[1])[0]


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
            self._evalute_ani = True
            self._anisotropy_scaling_list = []
            self._f_ani_list = []
            for ani_scaling_array in ani_scaling_array_list:
                self._anisotropy_scaling_list.append(AnisotropyScaling(ani_param_array=ani_param_array,
                                                                       ani_scaling_array=ani_scaling_array))

    def ani_scaling(self, aniso_param_array):
        """

        :param aniso_param_array: anisotropy parameter array
        :return: scaling J(a_ani) for the IFU's
        """
        if not self._evalute_ani is True or aniso_param_array is None:
            return 1
        scaling_list = []
        for scaling_class in self._anisotropy_scaling_list:
            scaling = scaling_class.ani_scaling(aniso_param_array)
            scaling_list.append(scaling)
        return np.array(scaling_list)
