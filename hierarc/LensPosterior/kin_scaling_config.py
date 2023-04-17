import numpy as np


class KinScalingConfig(object):
    """
    class to manage the anisotropy model and parameters for the Posterior processing
    """
    def __init__(self, anisotropy_model, r_eff, gamma0=2, density_slope_scaling=False):
        """

        :param anisotropy_model: type of stellar anisotropy model. Supported are 'OM' and 'GOM' or 'const', see details in lenstronomy.Galkin module
        :param r_eff: half-light radius of the deflector galaxy
        :param gamma0: power-law density slope mean
        :param density_slope_scaling: scaling of kinematics with power-law density slope of deflector
        :type density_slope_scaling: bool
        """
        self._r_eff = r_eff
        self._anisotropy_model = anisotropy_model
        self._density_scaling = density_slope_scaling

        if self._density_scaling:
            gamma_scaling = np.linspace(start=gamma0-0.2, stop=gamma0+0.2, num=5)
            if self._anisotropy_model == 'OM':
                self._scaling_param_array = [np.array([0.1, 0.2, 0.5, 1, 2, 5]), gamma_scaling]
            elif self._anisotropy_model == 'const':
                self._scaling_param_array = [np.linspace(-0.49, 1, 7), gamma_scaling]
            else:
                raise ValueError('Anisotropy model %s does not support power-law slope scaling.'
                                 % self._anisotropy_model)

        else:
            if self._anisotropy_model == 'OM':
                self._scaling_param_array = np.array(
                    [0.1, 0.2, 0.5, 1, 2, 5])  # used for r_ani OsipkovMerritt anisotropy description
            elif self._anisotropy_model == 'GOM':
                self._scaling_param_array = [np.array([0.1, 0.2, 0.5, 1, 2, 5]), np.array([0, 0.5, 0.8, 1])]
            elif self._anisotropy_model == 'const':
                self._scaling_param_array = np.linspace(-0.49, 1, 7)  # used for constant anisotropy description
            else:
                raise ValueError('anisotropy model %s not supported.' % self._anisotropy_model)

    @property
    def kwargs_anisotropy_base(self):
        """

        :return: keyword arguments of base anisotropy model configuration
        """
        if self._anisotropy_model == 'OM':
            a_ani_0 = 1
            r_ani = a_ani_0 * self._r_eff
            kwargs_anisotropy_0 = {'r_ani': r_ani}
        elif self._anisotropy_model == 'GOM':
            a_ani_0 = 1
            r_ani = a_ani_0 * self._r_eff
            beta_inf_0 = 1
            kwargs_anisotropy_0 = {'r_ani': r_ani, 'beta_inf': beta_inf_0}
        elif self._anisotropy_model == 'const':
            a_ani_0 = 0.1
            kwargs_anisotropy_0 = {'beta': a_ani_0}
        else:
            raise ValueError('anisotropy model %s not supported.' % self._anisotropy_model)
        return kwargs_anisotropy_0

    @property
    def scaling_param_array(self):
        """

        :return: numpy array of anisotropy and power-law parameter values to be explored with which the kinematics are
         being scaled.
        """
        return self._scaling_param_array

    def anisotropy_kwargs(self, a_ani, beta_inf=None):
        """

        :param a_ani: anisotropy parameter
        :param beta_inf: anisotropy at infinity (only used for 'GOM' model)
        :return: list of anisotropy keyword arguments, value of anisotropy parameter list
        """

        if self._anisotropy_model == 'OM':
            r_ani = a_ani * self._r_eff
            kwargs_anisotropy = {'r_ani': r_ani}
        elif self._anisotropy_model == 'GOM':
            r_ani = a_ani * self._r_eff
            kwargs_anisotropy = {'r_ani': r_ani, 'beta_inf': beta_inf}
        elif self._anisotropy_model == 'const':
            kwargs_anisotropy = {'beta': a_ani}
        else:
            raise ValueError('anisotropy model %s not supported.' % self._anisotropy_model)
        return kwargs_anisotropy
