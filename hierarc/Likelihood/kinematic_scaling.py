__author__ = 'sibirrer'
from scipy.interpolate import interp1d, interp2d
import numpy as np


class KinematicScalingSingleAperture(object):
    """
    class to manage anisotropy scaling for single slit observation
    """

    def __init__(self, j_kin_param_array, j_kin_scaling_array):
        """

        :param j_kin_param_array: array of J() kinematics parameter values (anisotropy, power-law slope)
        :param j_kin_scaling_array: array with the scaling of J() for single slit
        """
        self._evalute_ani = False
        if j_kin_param_array is not None and j_kin_scaling_array is not None:
            if isinstance(j_kin_param_array, list):
                self._dim_scaling = len(j_kin_param_array)
            else:
                self._dim_scaling = 1
            if self._dim_scaling == 1:
                self._f_scaling = interp1d(j_kin_param_array, j_kin_scaling_array, kind='linear')
            elif self._dim_scaling == 2:
                self._f_scaling = interp2d(j_kin_param_array[0], j_kin_param_array[1], j_kin_scaling_array.T)
            else:
                raise ValueError('anisotropy scaling with dimension %s not supported.' % self._dim_scaling)
            self._evalute_ani = True

    def j_kin_scaling(self, j_kin_param_array):
        """

        :param j_kin_param_array: J() kinematic parameter array
        :return: scaling J(a_ani) for single slit
        """
        if self._evalute_ani is not True or j_kin_param_array is None:
            return 1
        if self._dim_scaling == 1:
            return self._f_scaling(j_kin_param_array[0])
        elif self._dim_scaling == 2:
            return self._f_scaling(j_kin_param_array[0], j_kin_param_array[1])[0]


class KinematicScalingIFU(object):
    """
    class to manage anisotropy scalings for IFU data
    """

    def __init__(self, anisotropy_model='NONE', power_law_scaling=False, scaling_param_array=None,
                 scaling_array_list=None):
        """

        :param anisotropy_model: string, either 'NONE', 'OM' or 'GOM'
        :param power_law_scaling: if True, includes a scaling parameter with the power-law
        :type power_law_scaling: bool
        :param scaling_param_array: array of kinematic scaling parameter value
         (1d for 'OM' model, 2d for 'GOM' model, adding one for power_law_scaling=True)
        :param scaling_array_list: list of array with the scalings of J() for each IFU

        """
        if anisotropy_model not in ['OM', 'const', 'GOM', 'NONE']:
            raise ValueError('anisotropy_model % s not supported.' % anisotropy_model)
        self._anisotropy_model = anisotropy_model
        self._evaluate_ani = False
        self._power_law_scaling = power_law_scaling
        if scaling_param_array is not None and scaling_array_list is not None and \
            (self._anisotropy_model != 'NONE' or self._power_law_scaling is True):
            self._evaluate_ani = True
            self._j_kin_scaling_list = []
            self._f_ani_list = []
            for ani_scaling_array in scaling_array_list:
                self._j_kin_scaling_list.append(KinematicScalingSingleAperture(j_kin_param_array=scaling_param_array,
                                                                               j_kin_scaling_array=ani_scaling_array))

            if isinstance(scaling_param_array, list):
                self._dim_scaling = len(scaling_param_array)
            else:
                self._dim_scaling = 1
            if self._dim_scaling == 1:
                self._scaling_param_min = np.min(scaling_param_array)
                self._scaling_param_max = np.max(scaling_param_array)
            elif self._dim_scaling == 2:
                self._scaling_param_min = [min(scaling_param_array[0]), min(scaling_param_array[1])]
                self._scaling_param_max = [max(scaling_param_array[0]), max(scaling_param_array[1])]
            else:
                raise ValueError('j_kin scaling with dimension %s does not match anisotropy model %s and '
                                 'power-law scaling % s'
                                 % (self._dim_scaling, self._anisotropy_model, self._power_law_scaling))

    def j_kin_scaling(self, j_kin_param_array):
        """

        :param j_kin_param_array: J() kinematics parameter array
        :return: scaling J(a_ani, gamma) for the IFU's
        """
        if self._evaluate_ani is not True or j_kin_param_array is None:
            return [1]
        scaling_list = []
        for scaling_class in self._j_kin_scaling_list:
            scaling = scaling_class.j_kin_scaling(j_kin_param_array)
            scaling_list.append(scaling)
        return np.array(scaling_list)

    def draw_j_kin(self, a_ani=None, a_ani_sigma=0, beta_inf=None, beta_inf_sigma=0, gamma_pl=None, gamma_pl_sigma=0):
        """
        draw Gaussian distribution and re-sample if outside bounds

        :param a_ani: mean of the distribution
        :param a_ani_sigma: std of the distribution
        :param beta_inf: anisotropy at infinity (relevant for GOM model)
        :param beta_inf_sigma: std of beta_inf distribution
        :param gamma_pl: power-law slope (not distribution
        :param gamma_pl_sigma: std of the power-law distribution
        :return: random draw from the distribution
        """
        if self._anisotropy_model in ['OM', 'const'] and not self._power_law_scaling:
            if a_ani < self._scaling_param_min or a_ani > self._scaling_param_max:
                raise ValueError('anisotropy parameter is out of bounds of the interpolated range!')
            # we draw a linear gaussian for 'const' anisotropy and a scaled proportional one for 'OM
            if self._anisotropy_model == 'OM':
                a_ani_draw = np.random.normal(a_ani, a_ani_sigma*a_ani)
            else:
                a_ani_draw = np.random.normal(a_ani, a_ani_sigma)
            if a_ani_draw < self._scaling_param_min or a_ani_draw > self._scaling_param_max:
                return self.draw_j_kin(a_ani, a_ani_sigma)
            return np.array([a_ani_draw])

        elif self._anisotropy_model in ['OM', 'const'] and self._power_law_scaling:
            if a_ani < self._scaling_param_min[0] or a_ani > self._scaling_param_max[0]:
                raise ValueError('anisotropy parameter is out of bounds of the interpolated range!')
            a_ani_draw = np.random.normal(a_ani, a_ani_sigma*a_ani)
            if a_ani_draw < self._scaling_param_min[0] or a_ani_draw > self._scaling_param_max[0]:
                return self.draw_j_kin(a_ani, a_ani_sigma, beta_inf, beta_inf_sigma, gamma_pl=gamma_pl,
                                       gamma_pl_sigma=gamma_pl_sigma)
            gamma_pl_draw = np.random.normal(gamma_pl, gamma_pl_sigma)
            return np.array([a_ani_draw, gamma_pl_draw])

        elif self._anisotropy_model in ['GOM'] and not self._power_law_scaling:
            if a_ani < self._scaling_param_min[0] or a_ani > self._scaling_param_max[0] or beta_inf < self._scaling_param_min[1] or beta_inf > self._scaling_param_max[1]:
                raise ValueError('anisotropy parameter is out of bounds of the interpolated range!')
            a_ani_draw = np.random.normal(a_ani, a_ani_sigma*a_ani)
            beta_inf_draw = np.random.normal(beta_inf, beta_inf_sigma)
            if a_ani_draw < self._scaling_param_min[0] or a_ani_draw > self._scaling_param_max[0] or beta_inf_draw < self._scaling_param_min[1] or beta_inf_draw > self._scaling_param_max[1]:
                return self.draw_j_kin(a_ani, a_ani_sigma, beta_inf, beta_inf_sigma)
            return np.array([a_ani_draw, beta_inf_draw])
        return None
