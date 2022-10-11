import pytest
import unittest
import numpy as np
from hierarc.Diagnostics.goodness_of_fit import GoodnessOfFit
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from astropy.cosmology import FlatLambdaCDM

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class TestGoodnessOfFit(object):

    def setup(self):
        np.random.seed(seed=41)
        z_lens = 0.8
        z_source = 3.0
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.0)
        lensCosmo = LensCosmo(z_lens, z_source, cosmo=self.cosmo)
        dd = lensCosmo.dd
        ddt = lensCosmo.ddt
        dd_sigma = 0.1 * dd
        ddt_sigma = 0.1 * ddt

        ds_dds = lensCosmo.ds / lensCosmo.dds
        ds_dds_sigma = 0.1 * ds_dds

        num_samples = 10000
        ddt_samples = np.random.normal(ddt, ddt_sigma, num_samples)
        dd_samples = np.random.normal(dd, dd_sigma, num_samples)

        kappa_posterior = np.random.normal(loc=0, scale=0.03, size=100000)
        kappa_pdf, kappa_bin_edges = np.histogram(kappa_posterior, density=True)

        likelihood_type_list = ['DdtGaussian',
                                     'DdtDdKDE',
                                     'DdtDdGaussian',
                                     'DsDdsGaussian',
                                     'DdtLogNorm',
                                     'IFUKinCov',
                                     'DdtHist',
                                     'DdtHistKDE',
                                     'DdtHistKin',
                                     'DdtGaussKin']
        self.ifu_index = 5

        self.kwargs_likelihood_list = [{'ddt_mean': ddt, 'ddt_sigma': ddt_sigma},
                                       {'dd_samples': dd_samples, 'ddt_samples': ddt_samples,
                                        'kde_type': 'scipy_gaussian', 'bandwidth': 1},
                                       {'ddt_mean': ddt, 'ddt_sigma': ddt_sigma, 'dd_mean': dd, 'dd_sigma': dd_sigma},
                                       {'ds_dds_mean': ds_dds, 'ds_dds_sigma': ds_dds_sigma},
                                       {'ddt_mu': 1, 'ddt_sigma': 0.1},
                                       {'sigma_v_measurement': [1], 'j_model': [1], 'error_cov_measurement': np.array([[1]]),
                                        'error_cov_j_sqrt': [[1]]},
                                       {'ddt_samples': ddt_samples, 'kappa_pdf': kappa_pdf, 'kappa_bin_edges': kappa_bin_edges},
                                       {'ddt_samples': ddt_samples},
                                       {'ddt_samples': ddt_samples, 'sigma_v_measurement': [1], 'j_model': [1],
                                        'error_cov_measurement': [[1]], 'error_cov_j_sqrt': [[1]]},
                                       {'ddt_mean': 1, 'ddt_sigma': 0.1, 'sigma_v_measurement': [1], 'j_model': [1],
                                        'error_cov_measurement': [[1]], 'error_cov_j_sqrt': [[1]]},
                                       ]
        for i, likelihood_type in enumerate(likelihood_type_list):
            self.kwargs_likelihood_list[i]['z_lens'] = z_lens
            self.kwargs_likelihood_list[i]['z_source'] = z_source
            self.kwargs_likelihood_list[i]['likelihood_type'] = likelihood_type

        self.goodnessofFit = GoodnessOfFit(kwargs_likelihood_list=self.kwargs_likelihood_list)

    def test_plot_ddt_fit(self):
        kwargs_lens = {'lambda_mst': 1}
        kwargs_kin = {}
        f, ax = self.goodnessofFit.plot_ddt_fit(self.cosmo, kwargs_lens, kwargs_kin, redshift_trend=False)
        plt.close()
        f, ax = self.goodnessofFit.plot_ddt_fit(self.cosmo, kwargs_lens, kwargs_kin, redshift_trend=True)
        plt.close()

    def test_plot_kin_fit(self):
        kwargs_lens = {'lambda_mst': 1}
        kwargs_kin = {}
        f, ax = self.goodnessofFit.plot_kin_fit(self.cosmo, kwargs_lens, kwargs_kin)
        plt.close()

    def test_plot_ifu_fit(self):
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        kwargs_lens = {'lambda_mst': 1}
        kwargs_kin = {}
        self.goodnessofFit.plot_ifu_fit(ax, self.cosmo, kwargs_lens, kwargs_kin, lens_index=self.ifu_index,
                                        bin_edges=1., show_legend=True)
        plt.close()


class TestRaise(unittest.TestCase):

    def test_raise(self):

        with self.assertRaises(ValueError):
            kwargs_likelihood_list = [{'ddt_mean': 1, 'ddt_sigma': 0.1, 'z_lens': 0.5, 'z_source': 1.5,
                                       'likelihood_type': 'DdtGaussian'}]
            goodness_of_fit = GoodnessOfFit(kwargs_likelihood_list=kwargs_likelihood_list)
            f, ax = plt.subplots(1, 1, figsize=(4, 4))
            kwargs_lens = {'lambda_mst': 1}
            kwargs_kin = {}
            cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.0)
            goodness_of_fit.plot_ifu_fit(ax, cosmo, kwargs_lens, kwargs_kin, lens_index=0, bin_edges=1,
                                         show_legend=True)
            plt.close()


if __name__ == '__main__':
    pytest.main()
