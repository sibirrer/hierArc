import matplotlib.pyplot as plt
import numpy as np
from hierarc.Likelihood.lens_sample_likelihood import LensSampleLikelihood


class GoodnessOfFit(object):
    """
    class to manage goodness of fit diagnostics
    """
    def __init__(self, kwargs_likelihood_list):
        """

        :param kwargs_likelihood_list: list of likelihood kwargs of individual lenses consistent with the
        LensLikelihood module
        """
        self._kwargs_likelihood_list = kwargs_likelihood_list
        self._sample_likelihood = LensSampleLikelihood(kwargs_likelihood_list)

    def plot_fit(self, cosmo, kwargs_lens, kwargs_kin):
        """
        plots the prediction and the uncorrelated error bars on the individual lenses

        :param cosmo: astropy.cosmology instance
        :param kwargs_lens: lens model parameter keyword arguments
        :param kwargs_kin: kinematics model keyword arguments
        :return: fig, axes of matplotlib instance
        """

        # list of values for 'KinGaussian' likelihood
        ds_dds_model_list = []
        ds_dds_name_list = []
        ds_dds_data_list = []
        ds_dds_sigma_list = []

        # list of values for 'TDKinGaussian' likelihood
        ddt_model_list = []
        ddt_name_list = []
        ddt_data_list = []
        ddt_sigma_list = []

        dd_model_list = []
        dd_name_list = []
        dd_data_list = []
        dd_sigma_list = []


        for i, kwargs_likelihood in enumerate(self._kwargs_likelihood_list):
            name = kwargs_likelihood['name']
            likelihood = self._sample_likelihood._lens_list[i]
            ddt, dd = likelihood.angular_diameter_distances(cosmo)
            ddt_, dd_ = likelihood.displace_prediction(ddt, dd, **kwargs_lens)
            aniso_param_array = likelihood.draw_anisotropy(**kwargs_kin)
            if likelihood.likelihood_type == 'TDKinGaussian':
                dd_ = dd_ * likelihood.ani_scaling(aniso_param_array)
                ddt_model_list.append(ddt_)
                ddt_name_list.append(name)
                ddt_data_list.append(kwargs_likelihood['ddt_mean'])
                ddt_sigma_list.append(kwargs_likelihood['ddt_sigma'])

                dd_model_list.append(dd_)
                dd_name_list.append(name)
                dd_data_list.append(kwargs_likelihood['dd_mean'])
                dd_sigma_list.append(kwargs_likelihood['dd_sigma'])

            elif likelihood.likelihood_type == 'KinGaussian':
                dd_ = dd_ * likelihood.ani_scaling(aniso_param_array)
                ds_dds_model_list.append(ddt / dd_ / (1 + likelihood._z_lens))
                ds_dds_data_list.append(kwargs_likelihood['ds_dds_mean'])
                ds_dds_sigma_list.append(kwargs_likelihood['ds_dds_sigma'])
                ds_dds_name_list.append(name)
            elif likelihood.likelihood_type == 'IFUKinCov':
                pass

        f, axes = plt.subplots(1, 3, figsize=(12, 10))
        axes[0].errorbar(np.arange(len(ddt_name_list)), ddt_data_list, yerr=ddt_sigma_list, xerr=None, fmt='o', ecolor=None, elinewidth=None,
                     capsize=None, barsabove=False, lolims=False, uplims=False,
                     xlolims=False, xuplims=False, errorevery=1, capthick=None, data=None)
        axes[0].plot(np.arange(len(ddt_name_list)), ddt_model_list, 'ok')
        axes[0].xticks(np.arange(len(ddt_name_list)), ddt_name_list, rotation='vertical')

        axes[1].errorbar(np.arange(len(dd_name_list)), dd_data_list, yerr=dd_sigma_list, xerr=None, fmt='o',
                         ecolor=None, elinewidth=None,
                         capsize=None, barsabove=False, lolims=False, uplims=False,
                         xlolims=False, xuplims=False, errorevery=1, capthick=None, data=None)
        axes[1].plot(np.arange(len(dd_name_list)), dd_model_list, 'ok')
        axes[1].xticks(np.arange(len(dd_name_list)), dd_name_list, rotation='vertical')


        axes[2].errorbar(np.arange(len(ds_dds_name_list)), ds_dds_data_list, yerr=ds_dds_sigma_list, xerr=None, fmt='o',
                         ecolor=None, elinewidth=None,
                         capsize=None, barsabove=False, lolims=False, uplims=False,
                         xlolims=False, xuplims=False, errorevery=1, capthick=None, data=None)
        axes[2].plot(np.arange(len(ds_dds_name_list)), ds_dds_model_list, 'ok')
        axes[2].xticks(np.arange(len(ds_dds_name_list)), ds_dds_name_list, rotation='vertical')

        # separate panel for
        # - Ddt vs Dd
        # - Ds/Dds
        # IFU fit
        return f, axes
