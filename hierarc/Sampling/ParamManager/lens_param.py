class LensParam(object):
    """
    manages the lens model covariant parameters
    """
    def __init__(self, lambda_mst_sampling=False, lambda_mst_distribution='NONE', kappa_ext_sampling=False,
                 kappa_ext_distribution='NONE', kwargs_fixed={}):
        """

        :param lambda_mst_sampling: bool, if True adds a global mass-sheet transform parameter in the sampling
        :param lambda_mst_distribution: string, distribution function of the MST transform
        :param kappa_ext_sampling: bool, if True samples a global external convergence parameter
        :param kappa_ext_distribution: string, distribution function of the kappa_ext parameter
        :param kwargs_fixed: keyword arguments that are held fixed through the sampling
        """
        self._lambda_mst_sampling = lambda_mst_sampling
        self._lambda_mst_distribution = lambda_mst_distribution
        self._kappa_ext_sampling = kappa_ext_sampling
        self._kappa_ext_distribution = kappa_ext_distribution
        self._kwargs_fixed = kwargs_fixed

    def param_list(self, latex_style=False):
        """

        :param latex_style: bool, if True returns strings in latex symbols, else in the convention of the sampler
        :return: list of the free parameters being sampled in the same order as the sampling
        """
        list = []
        if self._lambda_mst_sampling is True:
            if 'lambda_mst' not in self._kwargs_fixed:
                if latex_style is True:
                    list.append(r'$\lambda_{\rm mst}$')
                else:
                    list.append('lambda_mst')
            if self._lambda_mst_distribution == 'GAUSSIAN':
                if 'lambda_mst_sigma' not in self._kwargs_fixed:
                    if latex_style is True:
                        list.append(r'$\sigma(\lambda_{\rm mst})$')
                    else:
                        list.append('lambda_mst_sigma')
        if self._kappa_ext_sampling is True:
            if 'kappa_ext' not in self._kwargs_fixed:
                if latex_style is True:
                    list.append(r'$\kappa_{\rm ext}$')
                else:
                    list.append('kappa_ext')
            if self._kappa_ext_distribution == 'GAUSSIAN':
                if 'kappa_ext_sigma' not in self._kwargs_fixed:
                    if latex_style is True:
                        list.append(r'$\sigma(\kappa_{\rm ext})$')
                    else:
                        list.append('kappa_ext_sigma')
        return list

    def args2kwargs(self, args, i=0):
        """

        :param args: sampling argument list
        :return: keyword argument list with parameter names
        """
        kwargs = {}
        if self._lambda_mst_sampling is True:
            if 'lambda_mst' in self._kwargs_fixed:
                kwargs['lambda_mst'] = self._kwargs_fixed['lambda_mst']
            else:
                kwargs['lambda_mst'] = args[i]
                i += 1
            if self._lambda_mst_distribution == 'GAUSSIAN':
                if 'lambda_mst_sigma' in self._kwargs_fixed:
                    kwargs['lambda_mst_sigma'] = self._kwargs_fixed['lambda_mst_sigma']
                else:
                    kwargs['lambda_mst_sigma'] = args[i]
                    i += 1
        if self._kappa_ext_sampling is True:
            if 'kappa_ext' in self._kwargs_fixed:
                kwargs['kappa_ext'] = self._kwargs_fixed['kappa_ext']
            else:
                kwargs['kappa_ext'] = args[i]
                i += 1
            if self._kappa_ext_distribution == 'GAUSSIAN':
                if 'kappa_ext_sigma' in self._kwargs_fixed:
                    kwargs['kappa_ext_sigma'] = self._kwargs_fixed['kappa_ext_sigma']
                else:
                    kwargs['kappa_ext_sigma'] = args[i]
                    i += 1
        return kwargs, i

    def kwargs2args(self, kwargs):
        """

        :param kwargs: keyword argument list of parameters
        :return: sampling argument list in specified order
        """
        args = []
        if self._lambda_mst_sampling is True:
            if 'lambda_mst' not in self._kwargs_fixed:
                args.append(kwargs['lambda_mst'])
            if self._lambda_mst_distribution == 'GAUSSIAN':
                if 'lambda_mst_sigma' not in self._kwargs_fixed:
                    args.append(kwargs['lambda_mst_sigma'])
        if self._kappa_ext_sampling is True:
            if 'kappa_ext' not in self._kwargs_fixed:
                args.append(kwargs['kappa_ext'])
            if self._kappa_ext_distribution == 'GAUSSIAN':
                if 'kappa_ext_sigma' not in self._kwargs_fixed:
                    args.append(kwargs['kappa_ext_sigma'])
        return args
