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
        self._sample_likelihood = LensSampleLikelihood(kwargs_likelihood_list, normalized=False)

    def reduced_chi2(self, cosmo, kwargs_lens, kwargs_kin):
        """
        reduced chi^2 of fit

        """
        logL = self._sample_likelihood.log_likelihood(cosmo, kwargs_lens, kwargs_kin)
        num_data = self._sample_likelihood.num_data()
        return -logL * 2 / num_data

    def plot_ddt_fit(self, cosmo, kwargs_lens, kwargs_kin, color_measurement=None, color_prediction=None,
                     redshift_trend=False):
        """
        plots the prediction and the uncorrelated error bars on the individual lenses
        currently works for likelihood classes 'TDKinGaussian', 'KinGaussian'

        :param cosmo: astropy.cosmology instance
        :param kwargs_lens: lens model parameter keyword arguments
        :param kwargs_kin: kinematics model keyword arguments
        :param color_measurement: color of measurement
        :param color_prediction: color of model prediction
        :param redshift_trend: boolean, if True, plots as a function of redshift
        :return: fig, axes of matplotlib instance
        """
        red_chi2 = self.reduced_chi2(cosmo, kwargs_lens, kwargs_kin)
        print(red_chi2, 'reduced chi2')

        ddt_name_list = []
        ddt_model_mean_list = []
        ddt_model_sigma_list = []

        ddt_data_mean_list = []
        ddt_data_sigma_list = []
        z_list = []

        for i, kwargs_likelihood in enumerate(self._kwargs_likelihood_list):
            name = kwargs_likelihood.get('name', 'lens ' + str(i))
            likelihood = self._sample_likelihood._lens_list[i]
            ddt_mean_measurement, ddt_sigma_measurement = likelihood.ddt_measurement()
            if ddt_mean_measurement is not None:
                ddt_model_mean, ddt_model_sigma, dd_model_mean, dd_model_sigma = likelihood.ddt_dd_model_prediction(
                    cosmo, kwargs_lens=kwargs_lens)

                ddt_name_list.append(name)
                ddt_model_mean_list.append(ddt_model_mean)
                ddt_model_sigma_list.append(ddt_model_sigma)
                ddt_data_mean_list.append(ddt_mean_measurement)
                ddt_data_sigma_list.append(ddt_sigma_measurement)
                z_list.append(likelihood.z_lens)
        if redshift_trend:
            x = z_list
        else:
            x = np.arange(len(ddt_name_list))

        f, ax = plt.subplots(1, 1, figsize=(len(ddt_name_list)/1.5, 4))
        ax.errorbar(x, ddt_data_mean_list, yerr=ddt_data_sigma_list,
                    color=color_measurement,
                    xerr=None, fmt='o', ecolor=None, elinewidth=None,
                    capsize=None, barsabove=False, lolims=False, uplims=False,
                    xlolims=False, xuplims=False, errorevery=1, capthick=None, data=None, label='measurement')

        ax.errorbar(x, ddt_model_mean_list, yerr=ddt_model_sigma_list,
                    color=color_prediction,
                    xerr=None, fmt='o', ecolor=None, elinewidth=None,
                    capsize=None, barsabove=False, lolims=False, uplims=False,
                    xlolims=False, xuplims=False, errorevery=1, capthick=None, data=None, label='prediction')

        if redshift_trend:
            # put redshift ticks on top of plot
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            # ax2.set_xticks(z_list)
            # ax2.set_xticklabels(labels=ddt_name_list, rotation='vertical')
            ax2.set_xlabel('lens redshift', fontsize=15)

        ax.set_xticks(ticks=x)
        ax.set_xticklabels(labels=ddt_name_list, rotation='vertical')
        ax.set_ylabel(r'$D_{\Delta t}$ [Mpc]', fontsize=15)
        ax.legend()
        return f, ax

    def kin_fit(self, cosmo, kwargs_lens, kwargs_kin):
        """
        plots the prediction and the uncorrelated error bars on the individual lenses
        currently works for likelihood classes 'TDKinGaussian', 'KinGaussian'

        :param cosmo: astropy.cosmology instance
        :param kwargs_lens: lens model parameter keyword arguments
        :param kwargs_kin: kinematics model keyword arguments
        :return: list of name, measurement, measurement errors, model prediction, model prediction error
        """

        sigma_v_name_list = []
        sigma_v_measurement_list = []
        sigma_v_measurement_error_list = []
        sigma_v_model_list = []
        sigma_v_model_error_list = []

        for i, kwargs_likelihood in enumerate(self._kwargs_likelihood_list):
            name = kwargs_likelihood.get('name', 'lens ' + str(i))
            likelihood = self._sample_likelihood._lens_list[i]
            sigma_v_measurement, cov_error_measurement, sigma_v_predict_mean, cov_error_predict = likelihood.sigma_v_measured_vs_predict(
                cosmo, kwargs_lens=kwargs_lens, kwargs_kin=kwargs_kin)

            if sigma_v_measurement is not None:
                num = len(sigma_v_measurement)
                for k in range(num):
                    sigma_v = sigma_v_measurement[k]
                    sigma_v_sigma = np.sqrt(cov_error_measurement[k, k])
                    sigma_v_predict = sigma_v_predict_mean[k]
                    sigma_v_sigma_model = np.sqrt(cov_error_predict[k, k])
                    sigma_v_name_list.append(name)
                    sigma_v_measurement_list.append(sigma_v)
                    sigma_v_measurement_error_list.append(sigma_v_sigma)
                    sigma_v_model_list.append(sigma_v_predict)
                    sigma_v_model_error_list.append(sigma_v_sigma_model)
        return sigma_v_name_list, sigma_v_measurement_list, sigma_v_measurement_error_list, sigma_v_model_list, sigma_v_model_error_list

    def plot_kin_fit(self, cosmo, kwargs_lens, kwargs_kin, color_measurement=None, color_prediction=None):
        """
        plots the prediction and the uncorrelated error bars on the individual lenses
        currently works for likelihood classes 'TDKinGaussian', 'KinGaussian'

        :param cosmo: astropy.cosmology instance
        :param kwargs_lens: lens model parameter keyword arguments
        :param kwargs_kin: kinematics model keyword arguments
        :param color_measurement: color of measurement
        :param color_prediction: color of model prediction
        :return: fig, axes of matplotlib instance
        """
        logL = self._sample_likelihood.log_likelihood(cosmo, kwargs_lens, kwargs_kin)
        print(logL, 'log likelihood')
        sigma_v_name_list, sigma_v_measurement_list, sigma_v_measurement_error_list, sigma_v_model_list, sigma_v_model_error_list = self.kin_fit(cosmo, kwargs_lens, kwargs_kin)

        f, ax = plt.subplots(1, 1, figsize=(int(len(sigma_v_name_list)/2), 4))
        ax.errorbar(np.arange(len(sigma_v_name_list)), sigma_v_measurement_list,
                    yerr=sigma_v_measurement_error_list, xerr=None, fmt='o',
                    ecolor=None, elinewidth=None, color=color_measurement,
                    capsize=5, barsabove=False, lolims=False, uplims=False,
                    xlolims=False, xuplims=False, errorevery=1, capthick=None, data=None, label='measurement')
        ax.errorbar(np.arange(len(sigma_v_name_list)), sigma_v_model_list, color=color_prediction,
                    yerr=sigma_v_model_error_list, xerr=None, fmt='o',
                    ecolor=None, elinewidth=None, label='prediction', capsize=5)
        ax.set_xticks(ticks=np.arange(len(sigma_v_name_list)))
        ax.set_xticklabels(labels=sigma_v_name_list, rotation='vertical')
        ax.set_ylabel(r'$\sigma^{\rm P}$ [km/s]', fontsize=15)
        ax.legend()
        return f, ax

    def plot_ifu_fit(self, ax, cosmo, kwargs_lens, kwargs_kin, lens_index, bin_edges, show_legend=True,
                     color_measurement=None, color_prediction=None):
        """
        plot an individual IFU data goodness of fit

        :param ax: matplotlib axes instance
        :param cosmo: astropy.cosmology instance
        :param kwargs_lens: lens model parameter keyword arguments
        :param kwargs_kin: kinematics model keyword arguments
        :param lens_index: int, index in kwargs_lens to be plotted (needs to be of type 'IFUKinCov')
        :param bin_edges: radial bin edges in arc seconds. If number, then uniform bin_edges sampled from 0
        :type bin_edges: numpy array or float.
        :param show_legend: bool, to show legend
        :param color_measurement: color of measurement
        :param color_prediction: color of model prediction
        :return: figure as axes instance
        """
        kwargs_likelihood = self._kwargs_likelihood_list[lens_index]
        name = kwargs_likelihood.get('name', 'lens ' + str(lens_index))
        likelihood = self._sample_likelihood._lens_list[lens_index]

        if likelihood.likelihood_type not in ['IFUKinCov', 'DdtHistKin', 'DdtGaussKin']:
            raise ValueError('likelihood type of lens %s is %s. Must be "IFUKinCov", "DdtHistKin", "DdtGaussKin" '
                             % (name, likelihood.likelihood_type))
        sigma_v_measurement, cov_error_measurement, sigma_v_predict_mean, cov_error_predict = likelihood.sigma_v_measured_vs_predict(
            cosmo, kwargs_lens=kwargs_lens, kwargs_kin=kwargs_kin)

        if len(np.atleast_1d(bin_edges)) < 2:
            bin_edges = np.arange(len(sigma_v_measurement)+1) * bin_edges
        r_bins = (bin_edges[:-1] + np.diff(bin_edges) / 2)

        ax.errorbar(x=r_bins, y=sigma_v_measurement, xerr=np.diff(bin_edges) / 2.,
                    yerr=np.sqrt(np.diag(cov_error_measurement)),
                    fmt='o', label='data', capsize=5, color=color_measurement)
        ax.errorbar(r_bins, sigma_v_predict_mean, yerr=np.sqrt(np.diag(cov_error_predict)), xerr=np.diff(bin_edges) / 2.,
                    fmt='o', label='model', capsize=5, color=color_prediction)
        if show_legend is True:
            ax.legend(fontsize=20)
        ax.set_title(name, fontsize=20)
        ax.set_ylabel(r'$\sigma^{\rm P}$[km/s]', fontsize=20)
        ax.set_xlabel('radial bin [arcsec]', fontsize=20)
        return ax
