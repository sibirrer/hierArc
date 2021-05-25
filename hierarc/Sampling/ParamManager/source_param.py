
_SNE_DISTRIBUTIONS = ['GAUSSIAN', 'NONE']


class SourceParam(object):
    """
    manager for the source property parameters (currently particularly source magnitudes for SNe)
    """
    def __init__(self, sne_apparent_m_sampling=False, sne_distribution='GAUSSIAN', kwargs_fixed=None,
                 z_apparent_m_anchor=0.1):
        """

        :param sne_apparent_m_sampling: boolean, if True, samples/queries SNe unlensed magnitude distribution
        (not intrinsic magnitudes but apparent!)
        (only adviced when evaluating supernova datasets that can constrain this parameter, or when providing priors)
        :param sne_distribution: string, apparent non-lensed brightness distribution (in linear space).
         Currently supports:
         'GAUSSIAN': Gaussian distribution
        :param z_apparent_m_anchor: redshift of pivot/anchor at which the apparent SNe brightness is defined relative to
        :param kwargs_fixed: keyword arguments of fixed parameters (and values)
        """
        self._sne_apparent_m_sampling = sne_apparent_m_sampling
        if sne_distribution not in _SNE_DISTRIBUTIONS:
            raise ValueError('SNE distribution %s not supported. Please chose among %s.' % (sne_distribution, _SNE_DISTRIBUTIONS))
        self._sne_distribution = sne_distribution
        if kwargs_fixed is None:
            kwargs_fixed = {}
        self._kwargs_fixed = kwargs_fixed
        self._z_apparent_m_anchor = z_apparent_m_anchor

    def param_list(self, latex_style=False):
        """

        :param latex_style: bool, if True returns strings in latex symbols, else in the convention of the sampler
        :param i: int, index of the parameter to start with
        :return: list of the free parameters being sampled in the same order as the sampling
        """
        name_list = []
        if self._sne_apparent_m_sampling is True:
            if 'mu_sne' not in self._kwargs_fixed:
                if latex_style is True:
                    name_list.append(r'$\overline{m}_{\rm p}$')
                else:
                    name_list.append('mu_sne')
            if self._sne_distribution in ['GAUSSIAN']:
                if 'sigma_sne' not in self._kwargs_fixed:
                    if latex_style is True:
                        name_list.append(r'$\sigma(m_{\rm p})$')
                    else:
                        name_list.append('sigma_sne')
        return name_list

    def args2kwargs(self, args, i=0):
        """

        :param args: sampling argument list
        :param i: index of argument list to start reading out
        :return: keyword argument list with parameter names
        """
        kwargs = {}
        if self._sne_apparent_m_sampling is True:
            if 'mu_sne' in self._kwargs_fixed:
                kwargs['mu_sne'] = self._kwargs_fixed['mu_sne']
            else:
                kwargs['mu_sne'] = args[i]
                i += 1
            if self._sne_distribution in ['GAUSSIAN']:
                if 'sigma_sne' in self._kwargs_fixed:
                    kwargs['sigma_sne'] = self._kwargs_fixed['sigma_sne']
                else:
                    kwargs['sigma_sne'] = args[i]
                    i += 1
        kwargs['z_apparent_m_anchor'] = self._z_apparent_m_anchor
        return kwargs, i

    def kwargs2args(self, kwargs):
        """

        :param kwargs: keyword argument list of parameters
        :return: sampling argument list in specified order
        """
        args = []
        if self._sne_apparent_m_sampling is True:
            if 'mu_sne' not in self._kwargs_fixed:
                args.append(kwargs['mu_sne'])
            if self._sne_distribution in ['GAUSSIAN']:
                if 'sigma_sne' not in self._kwargs_fixed:
                    args.append(kwargs['sigma_sne'])
        return args
