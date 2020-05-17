import matplotlib.pyplot as plt
import numpy as np
from hierarc.Likelihood.lens_sample_likelihood import LensSampleLikelihood
from lenstronomy.Util import constants as const
from hierarc.Util.distribution_util import PDFSampling


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

    def plot_ddt_fit(self, cosmo, kwargs_lens, kwargs_kin):
        """
        plots the prediction and the uncorrelated error bars on the individual lenses
        currently works for likelihood classes 'TDKinGaussian', 'KinGaussian'

        :param cosmo: astropy.cosmology instance
        :param kwargs_lens: lens model parameter keyword arguments
        :param kwargs_kin: kinematics model keyword arguments
        :return: fig, axes of matplotlib instance
        """
        logL = self._sample_likelihood.log_likelihood(cosmo, kwargs_lens, kwargs_kin)
        print(logL, 'log likelihood')
        num_data = self._sample_likelihood.num_data()
        print(-logL * 2 / num_data, 'reduced chi2')

        ddt_model_list = []
        ddt_name_list = []
        ddt_data_list = []
        ddt_sigma_list = []

        for i, kwargs_likelihood in enumerate(self._kwargs_likelihood_list):
            name = kwargs_likelihood.get('name', 'lens ' + str(i))
            likelihood = self._sample_likelihood._lens_list[i]
            ddt, dd = likelihood.angular_diameter_distances(cosmo)
            #TODO kappa_ext can be either as a parameter or a distribution. Currently this is inconsistent if chosen as a distribution
            ddt_, dd_ = likelihood.displace_prediction(ddt, dd, kappa_ext=kwargs_lens.get('kappa_ext', 0),
                                                       lambda_mst=kwargs_lens.get('lambda_mst', 1),
                                                       gamma_ppn=kwargs_lens.get('gamma_ppn', 1))
            if likelihood.likelihood_type in ['DdtGaussKin', 'DdtDdGaussian', 'DdtGaussKin']:
                ddt_model_list.append(ddt_)
                ddt_name_list.append(name)
                ddt_data_list.append(kwargs_likelihood['ddt_mean'])
                ddt_sigma_list.append(kwargs_likelihood['ddt_sigma'])
            if likelihood.likelihood_type in ['DdtHist', 'DdtHistKin']:
                ddt_model_list.append(ddt_)
                ddt_name_list.append(name)
                ddt_mean = np.mean(kwargs_likelihood['ddt_samples'])
                ddt_sigma = np.std(kwargs_likelihood['ddt_samples'])
                if 'kappa_pdf' in kwargs_likelihood and 'kappa_bin_edges' in kwargs_likelihood:
                    pdf = PDFSampling(pdf_array=kwargs_likelihood['kappa_pdf'], bin_edges=kwargs_likelihood['kappa_bin_edges'])
                    sample = pdf.draw(n=20000)
                    kappa_mean = np.mean(sample)
                    kappa_sigma = np.std(sample)
                else:
                    kappa_mean = 0
                    kappa_sigma = 0
                ddt_mean /= (1 - kappa_mean)
                ddt_sigma = np.sqrt((ddt_sigma/ddt_mean)**2 + kappa_sigma**2) * ddt_mean
                ddt_data_list.append(ddt_mean)
                ddt_sigma_list.append(ddt_sigma)

        f, ax = plt.subplots(1, 1, figsize=(len(ddt_name_list), 4))
        ax.errorbar(np.arange(len(ddt_name_list)), ddt_data_list, yerr=ddt_sigma_list, xerr=None, fmt='o', ecolor=None, elinewidth=None,
                     capsize=None, barsabove=False, lolims=False, uplims=False,
                     xlolims=False, xuplims=False, errorevery=1, capthick=None, data=None, label='measurement')
        ax.plot(np.arange(len(ddt_name_list)), ddt_model_list, 'ok', label='prediction')
        ax.set_xticks(ticks=np.arange(len(ddt_name_list)))
        ax.set_xticklabels(labels=ddt_name_list, rotation='vertical')
        ax.set_ylabel(r'$D_{\Delta t}$ [Mpc]', fontsize=15)
        ax.legend()
        return f, ax

    def plot_kin_fit(self, cosmo, kwargs_lens, kwargs_kin):
        """
        plots the prediction and the uncorrelated error bars on the individual lenses
        currently works for likelihood classes 'TDKinGaussian', 'KinGaussian'

        :param cosmo: astropy.cosmology instance
        :param kwargs_lens: lens model parameter keyword arguments
        :param kwargs_kin: kinematics model keyword arguments
        :return: fig, axes of matplotlib instance
        """
        logL = self._sample_likelihood.log_likelihood(cosmo, kwargs_lens, kwargs_kin)
        print(logL, 'log likelihood')


        sigma_v_name_list = []
        sigma_v_measurement_list = []
        sigma_v_measurement_error_list = []
        sigma_v_model_list = []
        sigma_v_model_error_list = []

        for i, kwargs_likelihood in enumerate(self._kwargs_likelihood_list):
            name = kwargs_likelihood.get('name', 'lens '+str(i))
            likelihood = self._sample_likelihood._lens_list[i]
            ddt, dd = likelihood.angular_diameter_distances(cosmo)
            ddt_, dd_ = likelihood.displace_prediction(ddt, dd, kappa_ext=kwargs_lens.get('kappa_ext', 0),
                                                       lambda_mst=kwargs_lens.get('lambda_mst', 1),
                                                       gamma_ppn=kwargs_lens.get('gamma_ppn', 1))
            aniso_param_array = likelihood.draw_anisotropy(**kwargs_kin)

            if likelihood.likelihood_type in ['IFUKinCov', 'DdtHistKin', 'DdtGaussKin']:
                ds_dds = ddt_ / dd_ / (1 + likelihood._z_lens)
                j_model = kwargs_likelihood['j_model'][0] * likelihood.ani_scaling(aniso_param_array)[0]
                J_error = np.atleast_2d(kwargs_likelihood['error_cov_j_sqrt'])[0, 0]
                sigma_v = kwargs_likelihood['sigma_v_measurement'][0]
                sigma_v_sigma = np.sqrt(np.array(kwargs_likelihood['error_cov_measurement'])[0, 0])
                sigma_v_predict = np.sqrt(j_model) * const.c * np.sqrt(ds_dds) / 1000
                #sigma_v_sigma_tot = np.sqrt(sigma_v_sigma ** 2 + const.c ** 2 * ds_dds * J_error / 1000 ** 2)
                sigma_v_sigma_model = np.sqrt(const.c ** 2 * ds_dds * J_error / 1000 ** 2)
                sigma_v_name_list.append(name)
                sigma_v_measurement_list.append(sigma_v)
                sigma_v_measurement_error_list.append(sigma_v_sigma)
                sigma_v_model_list.append(sigma_v_predict)
                sigma_v_model_error_list.append(sigma_v_sigma_model)

        f, ax = plt.subplots(1, 1, figsize=(int(len(sigma_v_name_list)/2), 4))
        ax.errorbar(np.arange(len(sigma_v_name_list)), sigma_v_measurement_list,
                         yerr=sigma_v_measurement_error_list, xerr=None, fmt='o',
                         ecolor=None, elinewidth=None,
                         capsize=None, barsabove=False, lolims=False, uplims=False,
                         xlolims=False, xuplims=False, errorevery=1, capthick=None, data=None, label='measurement')
        ax.errorbar(np.arange(len(sigma_v_name_list)), sigma_v_model_list,
                         yerr=sigma_v_model_error_list, xerr=None, fmt='o',
                         ecolor=None, elinewidth=None, label='prediction')
        ax.set_xticks(ticks=np.arange(len(sigma_v_name_list)))
        ax.set_xticklabels(labels=sigma_v_name_list, rotation='vertical')
        ax.set_ylabel(r'$\sigma^{\rm P}$ [km/s]', fontsize=15)
        ax.legend()
        return f, ax

    def plot_ifu_fit(self, ax, cosmo, kwargs_lens, kwargs_kin, lens_index, show_legend=True):
        """
        plot an individual IFU data goodness of fit

        :param ax: matplotlib axes instance
        :param cosmo: astropy.cosmology instance
        :param kwargs_lens: lens model parameter keyword arguments
        :param kwargs_kin: kinematics model keyword arguments
        :param lens_index: int, index in kwargs_lens to be plotted (needs to by of type 'IFUKinCov')
        :param show_legend: bool, to show legend
        :return: figure as axes instance
        """
        kwargs_likelihood = self._kwargs_likelihood_list[lens_index]
        name = kwargs_likelihood.get('name', 'lens ' + str(lens_index))
        likelihood = self._sample_likelihood._lens_list[lens_index]
        if not likelihood.likelihood_type == 'IFUKinCov':
            raise ValueError('likelihood type of lens %s is %s. Must be "IFUKinCov"' %(name, likelihood.likelihood_type))
        ddt, dd = likelihood.angular_diameter_distances(cosmo)
        ddt_, dd_ = likelihood.displace_prediction(ddt, dd, **kwargs_lens)
        aniso_param_array = likelihood.draw_anisotropy(**kwargs_kin)
        aniso_scaling = likelihood.ani_scaling(aniso_param_array)
        ifu_likelihood = likelihood._lens_type
        ds_dds = ddt_ / dd_ / (1 + likelihood._z_lens)
        sigma_v_model = ifu_likelihood.sigma_v_model(ds_dds, aniso_scaling=aniso_scaling)
        cov_error_model = ifu_likelihood.cov_error_model(ds_dds, scaling_ifu=aniso_scaling)

        sigma_v_data = ifu_likelihood._sigma_v_measured
        cov_error_data = ifu_likelihood._error_cov_measurement

        ax.errorbar(np.arange(len(sigma_v_data)), sigma_v_data, yerr=np.sqrt(np.diag(cov_error_data)), xerr=None,
                      fmt='o', label='data')
        ax.errorbar(np.arange(len(sigma_v_model)), sigma_v_model, yerr=np.sqrt(np.diag(cov_error_model)), xerr=None,
                      fmt='o', label='model')
        if show_legend is True:
            ax.legend(fontsize=20)
        ax.set_title(name, fontsize=20)
        ax.set_ylabel(r'$\sigma^{\rm P}$[km/s]', fontsize=20)
        return ax
