import numpy as np


class LensParam(object):
    """
    manages the lens model covariant parameters
    """
    def __init__(self, lambda_mst_sampling=False, lambda_mst_distribution='NONE', kappa_ext_sampling=False,
                 kappa_ext_distribution='NONE', lambda_ifu_sampling=False, lambda_ifu_distribution='NONE',
                 alpha_lambda_sampling=False, beta_lambda_sampling=False, kwargs_fixed=None, log_scatter=False):
        """

        :param lambda_mst_sampling: bool, if True adds a global mass-sheet transform parameter in the sampling
        :param lambda_mst_distribution: string, distribution function of the MST transform
        :param kappa_ext_sampling: bool, if True samples a global external convergence parameter
        :param kappa_ext_distribution: string, distribution function of the kappa_ext parameter
        :param lambda_ifu_sampling: bool, if True samples a separate lambda_mst for a second (e.g. IFU) data set
        independently
        :param lambda_ifu_distribution: string, distribution function of the lambda_ifu parameter
        :param alpha_lambda_sampling: bool, if True samples a parameter alpha_lambda, which scales lambda_mst linearly
         according to a predefined quantity of the lens
        :param beta_lambda_sampling: bool, if True samples a parameter beta_lambda, which scales lambda_mst linearly
         according to a predefined quantity of the lens
        :param log_scatter: boolean, if True, samples the Gaussian scatter amplitude in log space (and thus flat prior in log)
        :param kwargs_fixed: keyword arguments that are held fixed through the sampling
        """
        self._lambda_mst_sampling = lambda_mst_sampling
        self._lambda_mst_distribution = lambda_mst_distribution
        self._lambda_ifu_sampling = lambda_ifu_sampling
        self._lambda_ifu_distribution = lambda_ifu_distribution
        self._kappa_ext_sampling = kappa_ext_sampling
        self._kappa_ext_distribution = kappa_ext_distribution
        self._alpha_lambda_sampling = alpha_lambda_sampling
        self._beta_lambda_sampling = beta_lambda_sampling
        self._log_scatter = log_scatter
        if kwargs_fixed is None:
            kwargs_fixed = {}
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
                    list.append(r'$\overline{\lambda}_{\rm int}$')
                else:
                    list.append('lambda_mst')
            if self._lambda_mst_distribution == 'GAUSSIAN':
                if 'lambda_mst_sigma' not in self._kwargs_fixed:
                    if latex_style is True:
                        if self._log_scatter is True:
                            list.append(r'$\log_{10}\sigma(\lambda_{\rm int})$')
                        else:
                            list.append(r'$\sigma(\lambda_{\rm int})$')
                    else:
                        list.append('lambda_mst_sigma')
        if self._lambda_ifu_sampling is True:
            if 'lambda_ifu' not in self._kwargs_fixed:
                if latex_style is True:
                    list.append(r'$\lambda_{\rm ifu}$')
                else:
                    list.append('lambda_ifu')
            if self._lambda_ifu_distribution == 'GAUSSIAN':
                if 'lambda_ifu_sigma' not in self._kwargs_fixed:
                    if latex_style is True:
                        if self._log_scatter is True:
                            list.append(r'$\log_{10}\sigma(\lambda_{\rm ifu})$')
                        else:
                            list.append(r'$\sigma(\lambda_{\rm ifu})$')
                    else:
                        list.append('lambda_ifu_sigma')
        if self._kappa_ext_sampling is True:
            if 'kappa_ext' not in self._kwargs_fixed:
                if latex_style is True:
                    list.append(r'$\overline{\kappa}_{\rm ext}$')
                else:
                    list.append('kappa_ext')
            if self._kappa_ext_distribution == 'GAUSSIAN':
                if 'kappa_ext_sigma' not in self._kwargs_fixed:
                    if latex_style is True:
                        list.append(r'$\sigma(\kappa_{\rm ext})$')
                    else:
                        list.append('kappa_ext_sigma')
        if self._alpha_lambda_sampling is True:
            if 'alpha_lambda' not in self._kwargs_fixed:
                if latex_style is True:
                    list.append(r'$\alpha_{\lambda}$')
                else:
                    list.append('alpha_lambda')
        if self._beta_lambda_sampling is True:
            if 'beta_lambda' not in self._kwargs_fixed:
                if latex_style is True:
                    list.append(r'$\beta_{\lambda}$')
                else:
                    list.append('beta_lambda')
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
                    if self._log_scatter is True:
                        kwargs['lambda_mst_sigma'] = 10**(args[i])
                    else:
                        kwargs['lambda_mst_sigma'] = args[i]
                    i += 1
        if self._lambda_ifu_sampling is True:
            if 'lambda_ifu' in self._kwargs_fixed:
                kwargs['lambda_ifu'] = self._kwargs_fixed['lambda_ifu']
            else:
                kwargs['lambda_ifu'] = args[i]
                i += 1
            if self._lambda_ifu_distribution == 'GAUSSIAN':
                if 'lambda_ifu_sigma' in self._kwargs_fixed:
                    kwargs['lambda_ifu_sigma'] = self._kwargs_fixed['lambda_ifu_sigma']
                else:
                    if self._log_scatter is True:
                        kwargs['lambda_ifu_sigma'] = 10**(args[i])
                    else:
                        kwargs['lambda_ifu_sigma'] = args[i]
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
        if self._alpha_lambda_sampling is True:
            if 'alpha_lambda' in self._kwargs_fixed:
                kwargs['alpha_lambda'] = self._kwargs_fixed['alpha_lambda']
            else:
                kwargs['alpha_lambda'] = args[i]
                i += 1
        if self._beta_lambda_sampling is True:
            if 'beta_lambda' in self._kwargs_fixed:
                kwargs['beta_lambda'] = self._kwargs_fixed['beta_lambda']
            else:
                kwargs['beta_lambda'] = args[i]
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
                    if self._log_scatter is True:
                        args.append(np.log10(kwargs['lambda_mst_sigma']))
                    else:
                        args.append(kwargs['lambda_mst_sigma'])
        if self._lambda_ifu_sampling is True:
            if 'lambda_ifu' not in self._kwargs_fixed:
                args.append(kwargs['lambda_ifu'])
            if self._lambda_ifu_distribution == 'GAUSSIAN':
                if 'lambda_ifu_sigma' not in self._kwargs_fixed:
                    if self._log_scatter is True:
                        args.append(np.log10(kwargs['lambda_ifu_sigma']))
                    else:
                        args.append(kwargs['lambda_ifu_sigma'])
        if self._kappa_ext_sampling is True:
            if 'kappa_ext' not in self._kwargs_fixed:
                args.append(kwargs['kappa_ext'])
            if self._kappa_ext_distribution == 'GAUSSIAN':
                if 'kappa_ext_sigma' not in self._kwargs_fixed:
                    args.append(kwargs['kappa_ext_sigma'])
        if self._alpha_lambda_sampling is True:
            if 'alpha_lambda' not in self._kwargs_fixed:
                args.append(kwargs['alpha_lambda'])
        if self._beta_lambda_sampling is True:
            if 'beta_lambda' not in self._kwargs_fixed:
                args.append(kwargs['beta_lambda'])
        return args
