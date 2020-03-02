import numpy as np


class BaseLensConfig(object):
    """
    this class contains and manages the base configurations of the lens posteriors and makes sure that they
    are universally applied consistently through the different likelihood definitions
    """
    def __init__(self, z_lens, z_source, anisotropy_model, r_eff):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param r_eff: half light radius
        :param anisotropy_model: type of stellar anisotropy model. See details in MamonLokasAnisotropy() class of lenstronomy.GalKin.anisotropy
        """
        self._z_lens, self._z_source = z_lens, z_source
        self._anisotropy_model = anisotropy_model
        self._r_eff = r_eff
        if self._anisotropy_model == 'OsipkovMerritt':
            self._ani_param_array = np.linspace(0.1, 5, 20)  # used for r_ani OsipkovMerritt anisotropy discription
        elif self._anisotropy_model == 'const':
            self._ani_param_array = np.linspace(0, 1, 10)  # used for constant anisotropy description
        else:
            raise ValueError('anisotropy model %s not supported.' % self._anisotropy_model)

    @property
    def kwargs_anisotropy_base(self):
        """

        :return: keyword arguments of base anisotropy model configuration
        """
        if self._anisotropy_model == 'OsipkovMerritt':
            a_ani_0 = 1
            r_ani = a_ani_0 * self._r_eff
            kwargs_anisotropy_0 = {'r_ani': r_ani}
        elif self._anisotropy_model == 'const':
            a_ani_0 = 0.1
            kwargs_anisotropy_0 = {'beta': a_ani_0}
        else:
            raise ValueError('anisotropy model %s not supported.' % self._anisotropy_model)
        return kwargs_anisotropy_0

    @property
    def ani_param_array(self):
        """

        :return: numpy array of anisotropy parameter values to be explored
        """
        return self._ani_param_array

    def anisotropy_kwargs(self, a_ani):
        """

        :param a_ani: anisotropy parameter
        :return: list of anisotropy keyword arguments, value of anisotropy parameter list
        """

        if self._anisotropy_model == 'OsipkovMerritt':
            r_ani = a_ani * self._r_eff
            kwargs_anisotropy_0 = {'r_ani': r_ani}
        elif self._anisotropy_model == 'const':
            kwargs_anisotropy_0 = {'beta': a_ani}
        else:
            raise ValueError('anisotropy model %s not supported.' % self._anisotropy_model)
        return kwargs_anisotropy_0
