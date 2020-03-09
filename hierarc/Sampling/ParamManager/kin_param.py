class KinParam(object):
    """
    manager for the kinematics anisotropy parameters
    """
    def __init__(self, anisotropy_sampling=False, anisotropy_model='GM', distribution_function='NONE',
                 kwargs_fixed={}):
        """

        :param anisotropy_sampling: bool, if True, makes use of this module, else ignores it's functionalities
        :param anisotropy_model: string, name of anisotropy model to consider
        :param distribution_function: string, 'NONE', 'GAUSSIAN', description of the distribution function of the
        anisotropy model parameters
        :param kwargs_fixed: keyword arguments of the fixed parameters
        """
        self._anisotropy_sampling = anisotropy_sampling
        self._anisotropy_model = anisotropy_model
        self._distribution_function = distribution_function
        self._kwargs_fixed = kwargs_fixed

    def param_list(self, latex_style=False):
        """

        :param latex_style: bool, if True returns strings in latex symbols, else in the convention of the sampler
        :param i: int, index of the parameter to start with
        :return: list of the free parameters being sampled in the same order as the sampling
        """
        list = []
        if self._anisotropy_sampling is True:
            if self._anisotropy_model in ['OM', 'GOM']:
                if 'a_ani' not in self._kwargs_fixed:
                    if latex_style is True:
                        list.append(r'$a_{\rm ani}$')
                    else:
                        list.append('a_ani')
                if self._distribution_function in ['GAUSSIAN']:
                    if 'a_ani_sigma' not in self._kwargs_fixed:
                        if latex_style is True:
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
                            list.append(r'$\sigma(\beta_{\infty})$')
                        else:
                            list.append('beta_inf_sigma')
        return list

    def args2kwargs(self, args, i=0):
        """

        :param args: sampling argument list
        :return: keyword argument list with parameter names
        """
        kwargs = {}
        if self._anisotropy_sampling is True:
            if self._anisotropy_model in ['OM', 'GOM']:
                if 'a_ani' in self._kwargs_fixed:
                    kwargs['a_ani'] = self._kwargs_fixed['a_ani']
                else:
                    kwargs['a_ani'] = args[i]
                    i += 1
                if self._distribution_function in ['GAUSSIAN']:
                    if 'a_ani_sigma' in self._kwargs_fixed:
                        kwargs['a_ani_sigma'] = self._kwargs_fixed['a_ani_sigma']
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
                        kwargs['beta_inf_sigma'] = args[i]
                        i += 1
        return kwargs, i

    def kwargs2args(self, kwargs):
        """

        :param kwargs: keyword argument list of parameters
        :return: sampling argument list in specified order
        """
        args = []
        if self._anisotropy_sampling is True:
            if self._anisotropy_model in ['OM', 'GOM']:
                if 'a_ani' not in self._kwargs_fixed:
                    args.append(kwargs['a_ani'])
                if self._distribution_function in ['GAUSSIAN']:
                    if 'a_ani_sigma' not in self._kwargs_fixed:
                        args.append(kwargs['a_ani_sigma'])
            if self._anisotropy_model in ['GOM']:
                if 'beta_inf' not in self._kwargs_fixed:
                    args.append(kwargs['beta_inf'])
                if self._distribution_function in ['GAUSSIAN']:
                    if 'beta_inf_sigma' not in self._kwargs_fixed:
                        args.append(kwargs['beta_inf_sigma'])
        return args
