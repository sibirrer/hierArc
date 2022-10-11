import copy
import unittest

import numpy as np
import pytest

from hierarc.Likelihood.KDELikelihood.chain import import_Planck_chain, rescale_vector_to_unity, \
    rescale_vector_from_unity
from hierarc.Likelihood.KDELikelihood.kde_likelihood import _PATH_2_PLANCKDATA, KDELikelihood


class TestKDELikelihood(object):

    def setup(self):
        self.cosmo_params = ['h0', 'om']
        self.cosmology = 'FLCDM'
        self.chain = import_Planck_chain(_PATH_2_PLANCKDATA, 'base', 'plikHM_TTTEEE_lowl_lowE', self.cosmo_params,
                                         self.cosmology, rescale=True)
        print(self.chain)

    def test_full_likelihood(self):
        kdelikelihood = KDELikelihood(self.chain, likelihood_type="kde_full")
        samples = np.asarray([70, 0.3])
        log_l = kdelikelihood.kdelikelihood()  # function
        log_L = log_l(samples.reshape(-1, 1).T)
        log_L2 = kdelikelihood.kdelikelihood_samples(samples.reshape(-1, 1).T)[0]
        assert log_L2 == log_L

        kdelikelihood = KDELikelihood(self.chain, likelihood_type="kde_hist_nd")
        log_L3 = kdelikelihood.kdelikelihood_samples(samples.reshape(-1, 1).T)[0]

        assert log_L3 == pytest.approx(log_L2, rel=0.001)  # better than 0.1%

    def test_chains_methods(self):
        cosmo_params = self.chain.list_params()
        weights = self.chain.list_weights()
        assert cosmo_params == self.cosmo_params

        self.chain.rescale_from_unity(verbose=True)
        self.chain.rescale_to_unity(verbose=True)

        samples = copy.deepcopy(self.chain.params)
        samples = np.asarray([samples[k] for k in self.cosmo_params])
        samples_copy = copy.deepcopy(samples)

        samples = rescale_vector_from_unity(samples, self.chain.rescale_dic, self.cosmo_params)
        samples = rescale_vector_to_unity(samples, self.chain.rescale_dic, self.cosmo_params)

        assert samples == pytest.approx(samples_copy, rel=1e-6)

        self.chain.create_param('w0')
        self.chain.fill_default('w0', -1.0, verbose=True)

        wa = np.ones_like(self.chain.params['h0'])
        self.chain.fill_default_array('wa', wa, verbose=True)


class TestRaise(unittest.TestCase):

    def test_raise(self):
        cosmo_params = ['h0', 'om']
        cosmology = 'FLCDM'
        chain = import_Planck_chain(_PATH_2_PLANCKDATA, 'base', 'plikHM_TTTEEE_lowl_lowE', cosmo_params,
                                         cosmology, rescale=True)

        with self.assertRaises(RuntimeError):
            chain.rescale_to_unity()

        with self.assertRaises(RuntimeError):
            chain.rescale_from_unity()
            chain.rescale_from_unity()

        with self.assertRaises(NameError):
            kdelikelihood = KDELikelihood(chain, likelihood_type="BAD")

if __name__ == '__main__':
    pytest.main()
