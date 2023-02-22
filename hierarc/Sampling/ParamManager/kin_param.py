import numpy as np


class KinParam(object):
    """
    manager for the kinematics anisotropy parameters
    """
    def __init__(self, anisotropy_sampling=False, anisotropy_model='OM', distribution_function='NONE',
                 sigma_v_systematics=False, log_scatter=False, kwargs_fixed=None):
        """

        :param anisotropy_sampling: bool, if True, makes use of this module, else ignores it's functionalities
        :param anisotropy_model: string, name of anisotropy model to consider
        :param distribution_function: string, 'NONE', 'GAUSSIAN', description of the distribution function of the
        anisotropy model parameters
        :param sigma_v_systematics: bool, if True samples parameters relative to systematics in the velocity dispersion
         measurement
        :param log_scatter: boolean, if True, samples the Gaussian scatter amplitude in log space (and thus flat prior in log)
        :param kwargs_fixed: keyword arguments of the fixed parameters
        """
        assert anisotropy_model in ['NONE', 'GOM', 'OM', 'const']
        self._anisotropy_sampling = anisotropy_sampling
        self._anisotropy_model = anisotropy_model
        self._distribution_function = distribution_function
        self._sigma_v_systematics = sigma_v_systematics
        if kwargs_fixed is None:
            kwargs_fixed = {}
        self._kwargs_fixed = kwargs_fixed
        self._log_scatter = log_scatter

    def param_list(self, latex_style=False):
        """

        :param latex_style: bool, if True returns strings in latex symbols, else in the convention of the sampler
        :param i: int, index of the parameter to start with
        :return: list of the free parameters being sampled in the same order as the sampling
        """
        list = []
        if self._anisotropy_sampling is True:
            if self._anisotropy_model in ['OM', 'GOM', 'const']:
                if 'a_ani' not in self._kwargs_fixed:
                    if latex_style is True:
                        list.append(r'$\langle a_{\rm ani}\rangle$')
                    else:
                        list.append('a_ani')
                if self._distribution_function in ['GAUSSIAN']:
                    if 'a_ani_sigma' not in self._kwargs_fixed:
                        if latex_style is True:
                            if self._log_scatter is True:
                                list.append(r'$\log_{10}\sigma(a_{\rm ani})$')
                            else:
                                list.append(r'$\sigma(a_{\rm ani})$')
                        else:
                            list.append('a_ani_sigma')
            if self._anisotropy_model in ['GOM']:
                if 'beta_inf' not in self._kwargs_fixed:
                    if latex_style is True:
                        list.append(r'$\beta_{\infty}$')
                    else:
                        list.append('beta_inf')
                if self._distribution_function in ['GAUSSIAN']:
                    if 'beta_inf_sigma' not in self._kwargs_fixed:
                        if latex_style is True:
                            if self._log_scatter is True:
                                list.append(r'$\log_{10}\sigma(\beta_{\infty})$')
                            else:
                                list.append(r'$\sigma(\beta_{\infty})$')
                        else:
                            list.append('beta_inf_sigma')
        if self._sigma_v_systematics is True:
            if 'sigma_v_sys_error' not in self._kwargs_fixed:
                if latex_style is True:
                    if self._log_scatter is True:
                        list.append(r'$\log_{10}\sigma_{\rm sys}(\sigma_v)$')
                    else:
                        list.append(r'$\sigma_{\rm sys}(\sigma_v)$')
                else:
                    list.append('sigma_v_sys_error')
        return list

    def args2kwargs(self, args, i=0):
        """

        :param args: sampling argument list
        :param i: integer, index to start reading out the argument list
        :return: keyword argument list with parameter names
        """
        kwargs = {}
        if self._anisotropy_sampling is True:
            if self._anisotropy_model in ['OM', 'GOM', 'const']:
                if 'a_ani' in self._kwargs_fixed:
                    kwargs['a_ani'] = self._kwargs_fixed['a_ani']
                else:
                    kwargs['a_ani'] = args[i]
                    i += 1
                if self._distribution_function in ['GAUSSIAN']:
                    if 'a_ani_sigma' in self._kwargs_fixed:
                        kwargs['a_ani_sigma'] = self._kwargs_fixed['a_ani_sigma']
                    else:
                        if self._log_scatter is True:
                            kwargs['a_ani_sigma'] = 10**(args[i])
                        else:
                            kwargs['a_ani_sigma'] = args[i]
                        i += 1
            if self._anisotropy_model in ['GOM']:
                if 'beta_inf' in self._kwargs_fixed:
                    kwargs['beta_inf'] = self._kwargs_fixed['beta_inf']
                else:
                    kwargs['beta_inf'] = args[i]
                    i += 1
                if self._distribution_function in ['GAUSSIAN']:
                    if 'beta_inf_sigma' in self._kwargs_fixed:
                        kwargs['beta_inf_sigma'] = self._kwargs_fixed['beta_inf_sigma']
                    else:
                        if self._log_scatter is True:
                            kwargs['beta_inf_sigma'] = 10**(args[i])
                        else:
                            kwargs['beta_inf_sigma'] = args[i]
                        i += 1
        if self._sigma_v_systematics is True:
            if 'sigma_v_sys_error' in self._kwargs_fixed:
                kwargs['sigma_v_sys_error'] = self._kwargs_fixed['sigma_v_sys_error']
            else:
                if self._log_scatter is True:
                    kwargs['sigma_v_sys_error'] = 10**(args[i])
                else:
                    kwargs['sigma_v_sys_error'] = args[i]
                i += 1
        return kwargs, i

    def kwargs2args(self, kwargs):
        """

        :param kwargs: keyword argument list of parameters
        :return: sampling argument list in specified order
        """
        args = []
        if self._anisotropy_sampling is True:
            if self._anisotropy_model in ['OM', 'GOM', 'const']:
                if 'a_ani' not in self._kwargs_fixed:
                    args.append(kwargs['a_ani'])
                if self._distribution_function in ['GAUSSIAN']:
                    if 'a_ani_sigma' not in self._kwargs_fixed:
                        if self._log_scatter is True:
                            args.append(np.log10(kwargs['a_ani_sigma']))
                        else:
                            args.append(kwargs['a_ani_sigma'])
            if self._anisotropy_model in ['GOM']:
                if 'beta_inf' not in self._kwargs_fixed:
                    args.append(kwargs['beta_inf'])
                if self._distribution_function in ['GAUSSIAN']:
                    if 'beta_inf_sigma' not in self._kwargs_fixed:
                        if self._log_scatter is True:
                            args.append(np.log10(kwargs['beta_inf_sigma']))
                        else:
                            args.append(kwargs['beta_inf_sigma'])
        if self._sigma_v_systematics is True:
            if 'sigma_v_sys_error' not in self._kwargs_fixed:
                if self._log_scatter is True:
                    args.append(np.log10(kwargs['sigma_v_sys_error']))
                else:
                    args.append(kwargs['sigma_v_sys_error'])
        return args
