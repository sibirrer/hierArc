import numpy as np
import pytest
import emcee
import os

from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from hierarc.Sampling.mcmc_sampling import MCMCSampler
from astropy.cosmology import FlatLambdaCDM


class TestMCMCSampling(object):

    def setup(self):
        np.random.seed(seed=41)
        self.z_L = 0.8
        self.z_S = 3.0

        self.H0_true = 70
        self.omega_m_true = 0.3
        self.cosmo = FlatLambdaCDM(H0=self.H0_true, Om0=self.omega_m_true, Ob0=0.05)
        lensCosmo = LensCosmo(self.z_L, self.z_S, cosmo=self.cosmo)
        self.Dd_true = lensCosmo.dd
        self.D_dt_true = lensCosmo.ddt

        self.sigma_Dd = 100
        self.sigma_Ddt = 100
        num_samples = 10000
        self.D_dt_samples = np.random.normal(self.D_dt_true, self.sigma_Ddt, num_samples)
        self.D_d_samples = np.random.normal(self.Dd_true, self.sigma_Dd, num_samples)

    def test_mcmc_emcee(self):
        n_walkers = 6
        n_run = 2
        n_burn = 2
        kwargs_mean_start = {'kwargs_cosmo': {'h0': self.H0_true}}
        kwargs_fixed = {'om': self.omega_m_true}
        kwargs_sigma_start = {'kwargs_cosmo': {'h0': 5}}
        kwargs_lower = {'h0': 10}
        kwargs_upper = {'h0': 200}
        kwargs_likelihood_list = [{'z_lens': self.z_L, 'z_source': self.z_S, 'likelihood_type': 'DdtDdKDE',
                             'dd_samples': self.D_d_samples, 'ddt_samples': self.D_dt_samples,
                             'kde_type': 'scipy_gaussian', 'bandwidth': 1}]
        cosmology = 'FLCDM'
        kwargs_bounds = {'kwargs_fixed_cosmo': kwargs_fixed, 'kwargs_lower_cosmo': kwargs_lower,
                         'kwargs_upper_cosmo': kwargs_upper}

        path = os.getcwd()
        backup_filename = 'test_emcee.h5'
        try:
            os.remove(os.path.join(path, backup_filename))  # just remove the backup file created above
        except:
            pass

        backend = emcee.backends.HDFBackend(backup_filename)
        kwargs_emcee = {'backend': backend}

        mcmc_sampler = MCMCSampler(kwargs_likelihood_list=kwargs_likelihood_list, cosmology=cosmology,
                                   kwargs_bounds=kwargs_bounds, ppn_sampling=False,
                 lambda_mst_sampling=False, lambda_mst_distribution='delta', anisotropy_sampling=False,
                 anisotropy_model='OM', custom_prior=None, interpolate_cosmo=True, num_redshift_interp=100,
                 cosmo_fixed=None)
        samples, log_prob = mcmc_sampler.mcmc_emcee(n_walkers, n_burn, n_run, kwargs_mean_start, kwargs_sigma_start, **kwargs_emcee)
        assert len(samples) == n_walkers*n_run
        assert len(log_prob) == n_walkers*n_run

        n_burn_2 = 0
        n_run_2 = 2
        samples, log_prob = mcmc_sampler.mcmc_emcee(n_walkers, n_burn_2, n_run_2, kwargs_mean_start, kwargs_sigma_start,
                                                    continue_from_backend=True, **kwargs_emcee)
        assert len(samples) == n_walkers * (n_run + n_run_2 + n_burn)
        assert len(log_prob) == n_walkers * (n_run + n_run_2 + n_burn)

        name_list = mcmc_sampler.param_names(latex_style=False)
        assert len(name_list) == 1

        os.remove(os.path.join(path,backup_filename))  # just remove the backup file created above


if __name__ == '__main__':
    pytest.main()
