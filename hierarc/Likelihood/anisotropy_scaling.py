__author__ = 'sibirrer'
from scipy.interpolate import interp1d, interp2d
import numpy as np


class AnisotropyScalingSingleAperture(object):
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
            if self._dim_scaling == 1:
                self._f_ani = interp1d(ani_param_array, ani_scaling_array, kind='linear')
            elif self._dim_scaling == 2:
                self._f_ani = interp2d(ani_param_array[0], ani_param_array[1], ani_scaling_array.T)
            else:
                raise ValueError('anisotropy scaling with dimension %s not supported.' % self._dim_scaling)
            self._evalute_ani = True

    def ani_scaling(self, aniso_param_array):
        """

        :param aniso_param_array: anisotropy parameter array
        :return: scaling J(a_ani) for single slit
        """
        if self._evalute_ani is not True or aniso_param_array is None:
            return 1
        if self._dim_scaling == 1:
            return self._f_ani(aniso_param_array[0])
        elif self._dim_scaling == 2:
            return self._f_ani(aniso_param_array[0], aniso_param_array[1])[0]


class AnisotropyScalingIFU(object):
    """
    class to manage anisotropy scalings for IFU data
    """

    def __init__(self, anisotropy_model='NONE', ani_param_array=None, ani_scaling_array_list=None):
        """

        :param anisotropy_model: string, either 'NONE', 'OM' or 'GOM'
        :param ani_param_array: array of anisotropy parameter value (1d for 'OM' model, 2d for 'GOM' model)
        :param ani_scaling_array_list: list of array with the scalings of J() for each IFU
        """
        self._anisotropy_model = anisotropy_model
        self._evalute_ani = False
        if ani_param_array is not None and ani_scaling_array_list is not None and self._anisotropy_model != 'NONE':
            self._evalute_ani = True
            self._anisotropy_scaling_list = []
            self._f_ani_list = []
            for ani_scaling_array in ani_scaling_array_list:
                self._anisotropy_scaling_list.append(AnisotropyScalingSingleAperture(ani_param_array=ani_param_array,
                                                                                     ani_scaling_array=ani_scaling_array))

            if isinstance(ani_param_array, list):
                self._dim_scaling = len(ani_param_array)
            else:
                self._dim_scaling = 1
            if self._dim_scaling == 1 and anisotropy_model in ['OM', 'const']:
                self._ani_param_min = np.min(ani_param_array)
                self._ani_param_max = np.max(ani_param_array)
            elif self._dim_scaling == 2 and anisotropy_model == 'GOM':
                self._ani_param_min = [min(ani_param_array[0]), min(ani_param_array[1])]
                self._ani_param_max = [max(ani_param_array[0]), max(ani_param_array[1])]
            else:
                raise ValueError('anisotropy scaling with dimension %s does not match anisotropy model %s'
                                 % (self._dim_scaling, self._anisotropy_model))

    def ani_scaling(self, aniso_param_array):
        """

        :param aniso_param_array: anisotropy parameter array
        :return: scaling J(a_ani) for the IFU's
        """
        if self._evalute_ani is not True or aniso_param_array is None:
            return [1]
        scaling_list = []
        for scaling_class in self._anisotropy_scaling_list:
            scaling = scaling_class.ani_scaling(aniso_param_array)
            scaling_list.append(scaling)
        return np.array(scaling_list)

    def draw_anisotropy(self, a_ani=None, a_ani_sigma=0, beta_inf=None, beta_inf_sigma=0):
        """
        draw Gaussian distribution and re-sample if outside bounds

        :param a_ani: mean of the distribution
        :param a_ani_sigma: std of the distribution
        :param beta_inf: anisotropy at infinity (relevant for GOM model)
        :param beta_inf_sigma: std of beta_inf distribution
        :return: random draw from the distribution
        """
        if self._anisotropy_model in ['OM', 'const']:
            if a_ani < self._ani_param_min or a_ani > self._ani_param_max:
                raise ValueError('anisotropy parameter is out of bounds of the interpolated range!')
            # we draw a linear gaussian for 'const' anisotropy and a scaled proportional one for 'OM
            if self._anisotropy_model == 'OM':
                a_ani_draw = np.random.normal(a_ani, a_ani_sigma*a_ani)
            else:
                a_ani_draw = np.random.normal(a_ani, a_ani_sigma)
            if a_ani_draw < self._ani_param_min or a_ani_draw > self._ani_param_max:
                return self.draw_anisotropy(a_ani, a_ani_sigma)
            return np.array([a_ani_draw])
        elif self._anisotropy_model in ['GOM']:
            if a_ani < self._ani_param_min[0] or a_ani > self._ani_param_max[0] or beta_inf < self._ani_param_min[1] or beta_inf > self._ani_param_max[1]:
                raise ValueError('anisotropy parameter is out of bounds of the interpolated range!')
            a_ani_draw = np.random.normal(a_ani, a_ani_sigma*a_ani)
            beta_inf_draw = np.random.normal(beta_inf, beta_inf_sigma)
            if a_ani_draw < self._ani_param_min[0] or a_ani_draw > self._ani_param_max[0] or beta_inf_draw < self._ani_param_min[1] or beta_inf_draw > self._ani_param_max[1]:
                return self.draw_anisotropy(a_ani, a_ani_sigma, beta_inf, beta_inf_sigma)
            return np.array([a_ani_draw, beta_inf_draw])
        return None
