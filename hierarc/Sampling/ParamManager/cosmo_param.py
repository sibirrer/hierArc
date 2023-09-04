from astropy.cosmology import FlatLambdaCDM, FlatwCDM, LambdaCDM, w0waCDM


class CosmoParam(object):
    """
    manages the cosmological parameters in the sampling
    """
    def __init__(self, cosmology, ppn_sampling=False, kwargs_fixed=None):
        """

        :param cosmology: string describing cosmological model
        :param ppn_sampling: post-newtonian parameter sampling
        :param kwargs_fixed: keyword arguments of fixed parameters during sampling
        """
        self._cosmology = cosmology
        if kwargs_fixed is None:
            kwargs_fixed = {}
        self._kwargs_fixed = kwargs_fixed
        self._ppn_sampling = ppn_sampling
        self._supported_cosmologies = ["FLCDM", "FwCDM", "w0waCDM", "oLCDM", "NONE"]
        if cosmology not in self._supported_cosmologies:
            raise ValueError(
                'cosmology %s not supported!. Please chose among %s ' % (cosmology, self._supported_cosmologies))

    def param_list(self, latex_style=False):
        """

        :param latex_style: bool, if True returns strings in latex symbols, else in the convention of the sampler
        :return: list of the free parameters being sampled in the same order as the sampling
        """
        list = []
        if self._cosmology not in ["NONE"]:
            if 'h0' not in self._kwargs_fixed:
                if latex_style is True:
                    list.append(r'$H_0$')
                else:
                    list.append('h0')
            if self._cosmology in ["FLCDM", "FwCDM", "w0waCDM", "oLCDM"]:
                if 'om' not in self._kwargs_fixed:
                    if latex_style is True:
                        list.append(r'$\Omega_{\rm m}$')
                    else:
                        list.append('om')
            if self._cosmology in ["FwCDM"]:
                if 'w' not in self._kwargs_fixed:
                    if latex_style is True:
                        list.append(r'$w$')
                    else:
                        list.append('w')
            if self._cosmology in ["w0waCDM"]:
                if 'w0' not in self._kwargs_fixed:
                    if latex_style is True:
                        list.append(r'$w_0$')
                    else:
                        list.append('w0')
                if 'wa' not in self._kwargs_fixed:
                    if latex_style is True:
                        list.append(r'$w_{\rm a}$')
                    else:
                        list.append('wa')
            if self._cosmology in ["oLCDM"]:
                if 'ok' not in self._kwargs_fixed:
                    if latex_style is True:
                        list.append(r'$\Omega_{\rm k}$')
                    else:
                        list.append('ok')
        if self._ppn_sampling is True:
            if 'gamma_ppn' not in self._kwargs_fixed:
                if latex_style is True:
                    list.append(r'$\gamma_{\rm ppn}$')
                else:
                    list.append('gamma_ppn')
        return list

    def args2kwargs(self, args, i=0):
        """

        :param args: sampling argument list
        :return: keyword argument list with parameter names
        """
        kwargs = {}
        if self._cosmology not in ["NONE"]:
            if 'h0' in self._kwargs_fixed:
                kwargs['h0'] = self._kwargs_fixed['h0']
            else:
                kwargs['h0'] = args[i]
                i += 1
            if self._cosmology in ["FLCDM", "FwCDM", "w0waCDM", "oLCDM"]:
                if 'om' in self._kwargs_fixed:
                    kwargs['om'] = self._kwargs_fixed['om']
                else:
                    kwargs['om'] = args[i]
                    i += 1
            if self._cosmology in ["FwCDM"]:
                if 'w' in self._kwargs_fixed:
                    kwargs['w'] = self._kwargs_fixed['w']
                else:
                    kwargs['w'] = args[i]
                    i += 1
            if self._cosmology in ["w0waCDM"]:
                if 'w0' in self._kwargs_fixed:
                    kwargs['w0'] = self._kwargs_fixed['w0']
                else:
                    kwargs['w0'] = args[i]
                    i += 1
                if 'wa' in self._kwargs_fixed:
                    kwargs['wa'] = self._kwargs_fixed['wa']
                else:
                    kwargs['wa'] = args[i]
                    i += 1
            if self._cosmology in ["oLCDM"]:
                if 'ok' in self._kwargs_fixed:
                    kwargs['ok'] = self._kwargs_fixed['ok']
                else:
                    kwargs['ok'] = args[i]
                    i += 1
        if self._ppn_sampling is True:
            if 'gamma_ppn' in self._kwargs_fixed:
                kwargs['gamma_ppn'] = self._kwargs_fixed['gamma_ppn']
            else:
                kwargs['gamma_ppn'] = args[i]
                i += 1
        return kwargs, i

    def kwargs2args(self, kwargs):
        """

        :param kwargs: keyword argument list of parameters
        :return: sampling argument list in specified order
        """
        args = []
        if self._cosmology not in ["NONE"]:
            if 'h0' not in self._kwargs_fixed:
                args.append(kwargs['h0'])
            if self._cosmology in ["FLCDM", "FwCDM", "w0waCDM", "oLCDM"]:
                if 'om' not in self._kwargs_fixed:
                    args.append(kwargs['om'])
            if self._cosmology in ["FwCDM"]:
                if 'w' not in self._kwargs_fixed:
                    args.append(kwargs['w'])
            if self._cosmology in ["w0waCDM"]:
                if 'w0' not in self._kwargs_fixed:
                    args.append(kwargs['w0'])
                if 'wa' not in self._kwargs_fixed:
                    args.append(kwargs['wa'])
            if self._cosmology in ["oLCDM"]:
                if 'ok' not in self._kwargs_fixed:
                    args.append(kwargs['ok'])
        if self._ppn_sampling is True:
            if 'gamma_ppn' not in self._kwargs_fixed:
                args.append(kwargs['gamma_ppn'])
        return args

    def cosmo(self, kwargs):
        """

        :param kwargs: keyword arguments of parameters (can include others not used for the cosmology)
        :return: astropy.cosmology instance
        """
        if self._cosmology == "FLCDM":
            cosmo = FlatLambdaCDM(H0=kwargs['h0'], Om0=kwargs['om'])
        elif self._cosmology == "FwCDM":
            cosmo = FlatwCDM(H0=kwargs['h0'], Om0=kwargs['om'], w0=kwargs['w'])
        elif self._cosmology == "w0waCDM":
            cosmo = w0waCDM(H0=kwargs['h0'], Om0=kwargs['om'], Ode0=1.0 - kwargs['om'], w0=kwargs['w0'], wa=kwargs['wa'])
        elif self._cosmology == "oLCDM":
            cosmo = LambdaCDM(H0=kwargs['h0'], Om0=kwargs['om'], Ode0=1.0 - kwargs['om'] - kwargs['ok'])
        elif self._cosmology == "NONE":
            cosmo = None
        else:
            raise ValueError("Cosmology %s is not supported" % self._cosmology)
        return cosmo
