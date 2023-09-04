import numpy as np
import numpy.testing as npt
import pytest
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Cosmo.lens_cosmo import LensCosmo

from hierarc.Likelihood.KDELikelihood.chain import import_Planck_chain
from hierarc.Likelihood.KDELikelihood.kde_likelihood import _PATH_2_PLANCKDATA
from hierarc.Likelihood.cosmo_likelihood import CosmoLikelihood


class TestCosmoLikelihood(object):

    def setup(self):
        np.random.seed(seed=41)
        self.z_L = 0.8
        self.z_S = 3.0

        self.H0_true = 70
        self.omega_m_true = 0.3
        self.cosmo = FlatLambdaCDM(H0=self.H0_true, Om0=self.omega_m_true, Ob0=0.0)
        lensCosmo = LensCosmo(self.z_L, self.z_S, cosmo=self.cosmo)
        self.Dd_true = lensCosmo.dd
        self.D_dt_true = lensCosmo.ddt
        self.sigma_Dd = 0.9 * self.Dd_true
        self.sigma_Ddt = 0.01 * self.D_dt_true
        num_samples = 20000
        self.ddt_samples = np.random.normal(self.D_dt_true, self.sigma_Ddt, num_samples)
        self.dd_samples = np.random.normal(self.Dd_true, self.sigma_Dd, num_samples)
        self.kwargs_likelihood_list = [{'z_lens': self.z_L, 'z_source': self.z_S, 'likelihood_type': 'DdtDdGaussian',
                                        'ddt_mean': self.D_dt_true, 'ddt_sigma': self.sigma_Ddt,
                                        'dd_mean': self.Dd_true, 'dd_sigma': self.sigma_Dd}]
        kwargs_lower_lens = {'gamma_ppn': 0, 'lambda_mst': 0}
        kwargs_upper_lens = {'gamma_ppn': 2, 'lambda_mst': 2}
        kwargs_fixed_lens = {}
        kwargs_lower_cosmo = {'h0': 10, 'om': 0., 'ok': -0.8, 'w': -2, 'wa': -1, 'w0': -2}
        kwargs_upper_cosmo = {'h0': 200, 'om': 1, 'ok': 0.8, 'w': 0, 'wa': 1, 'w0': 1}
        self.cosmology = 'oLCDM'
        self.kwargs_bounds = {'kwargs_lower_lens': kwargs_lower_lens, 'kwargs_upper_lens': kwargs_upper_lens,
                              'kwargs_fixed_lens': kwargs_fixed_lens,
                              'kwargs_lower_cosmo': kwargs_lower_cosmo, 'kwargs_upper_cosmo': kwargs_upper_cosmo}

        # self.kwargs_likelihood_list = [{'z_lens': self.z_L, 'z_source': self.z_S, 'likelihood_type': 'TDKinKDE',
        #                          'dd_sample': self.dd_samples, 'ddt_sample': self.ddt_samples,
        #                          'kde_type': 'scipy_gaussian', 'bandwidth': 10}]

    def test_log_likelihood(self):
        cosmoL = CosmoLikelihood(self.kwargs_likelihood_list, self.cosmology, self.kwargs_bounds, ppn_sampling=False,
                                 lambda_mst_sampling=False, lambda_mst_distribution='delta', anisotropy_sampling=False,
                                 anisotropy_model='OM', custom_prior=None, interpolate_cosmo=True,
                                 num_redshift_interp=100,
                                 cosmo_fixed=None)

        def custom_prior(kwargs_cosmo, kwargs_lens, kwargs_kin, kwargs_source):
            return -1

        cosmoL_prior = CosmoLikelihood(self.kwargs_likelihood_list, self.cosmology, self.kwargs_bounds,
                                       ppn_sampling=False,
                                       lambda_mst_sampling=False, lambda_mst_distribution='delta',
                                       anisotropy_sampling=False,
                                       anisotropy_model='OM', custom_prior=custom_prior, interpolate_cosmo=True,
                                       num_redshift_interp=100,
                                       cosmo_fixed=None)

        kwargs_cosmo = {'h0': self.H0_true, 'om': self.omega_m_true, 'ok': 0}
        args = cosmoL.param.kwargs2args(kwargs_cosmo=kwargs_cosmo)
        logl = cosmoL.likelihood(args=args)
        logl_prior = cosmoL_prior.likelihood(args=args)
        npt.assert_almost_equal(logl - logl_prior, 1, decimal=8)

        kwargs = {'h0': self.H0_true * 0.99, 'om': self.omega_m_true, 'ok': 0}
        args = cosmoL.param.kwargs2args(kwargs_cosmo=kwargs)
        logl_sigma = cosmoL.likelihood(args=args)
        npt.assert_almost_equal(logl - logl_sigma, 0.5, decimal=2)

        kwargs = {'h0': 100, 'om': 1., 'ok': 0.1}
        args = cosmoL.param.kwargs2args(kwargs)
        logl = cosmoL.likelihood(args=args)
        assert logl == -np.inf

        kwargs = {'h0': 100, 'om': .1, 'ok': -0.6}
        args = cosmoL.param.kwargs2args(kwargs)
        logl = cosmoL.likelihood(args=args)
        assert logl == -np.inf

        # outside the prior limit
        kwargs = {'h0': 1000, 'om': .3, 'ok': -0.1}
        args = cosmoL.param.kwargs2args(kwargs)
        logl = cosmoL.likelihood(args=args)
        assert logl == -np.inf

    def test_cosmo_instance(self):
        kwargs_cosmo = {'h0': 100, 'om': .1, 'ok': -0.1}
        cosmoL = CosmoLikelihood(self.kwargs_likelihood_list, self.cosmology, self.kwargs_bounds,
                                 interpolate_cosmo=False, cosmo_fixed=None)
        cosmo_astropy = cosmoL.cosmo_instance(kwargs_cosmo)

        cosmoL = CosmoLikelihood(self.kwargs_likelihood_list, self.cosmology, self.kwargs_bounds,
                                 interpolate_cosmo=True,
                                 num_redshift_interp=100, cosmo_fixed=None)
        cosmo_interp = cosmoL.cosmo_instance(kwargs_cosmo)

        cosmoL = CosmoLikelihood(self.kwargs_likelihood_list, self.cosmology, self.kwargs_bounds,
                                 interpolate_cosmo=True,
                                 num_redshift_interp=100, cosmo_fixed=cosmo_astropy)
        kwargs_cosmo_wrong = {'h0': 10, 'om': .3, 'ok': 0}
        cosmo_fixed_interp = cosmoL.cosmo_instance(kwargs_cosmo_wrong)

        cosmoL = CosmoLikelihood(self.kwargs_likelihood_list, self.cosmology, self.kwargs_bounds,
                                 interpolate_cosmo=False,
                                 num_redshift_interp=100, cosmo_fixed=cosmo_astropy)
        kwargs_cosmo_wrong = {'h0': 10, 'om': .3, 'ok': 0}
        cosmo_fixed = cosmoL.cosmo_instance(kwargs_cosmo_wrong)

        # and here we inject pre-computed angular diameter distances as a function of redshift
        redshifts = np.linspace(start=0, stop=5, num=100)
        ang_diameter_distances = cosmo_astropy.angular_diameter_distance(redshifts)
        Ok0 = kwargs_cosmo['ok']
        K = cosmo_interp.k  # compute the curvature from the interpolation class (as easily available)
        kwargs_cosmo_interp = {'ang_diameter_distances': ang_diameter_distances,
                               'redshifts': redshifts, 'ok': Ok0, 'K': K}

        cosmo_interp_input = cosmoL.cosmo_instance(kwargs_cosmo_interp)

        z = 1
        dd_astropy = cosmo_astropy.angular_diameter_distance(z=z).value
        dd_interp = cosmo_interp.angular_diameter_distance(z=z).value
        dd_interp_input = cosmo_interp_input.angular_diameter_distance(z=z).value
        dd_fixed = cosmo_fixed.angular_diameter_distance(z=z).value
        dd_fixed_interp = cosmo_fixed_interp.angular_diameter_distance(z=z).value

        npt.assert_almost_equal(dd_astropy, dd_interp, decimal=1)
        npt.assert_almost_equal(dd_astropy, dd_interp_input, decimal=1)
        npt.assert_almost_equal(dd_astropy, dd_fixed, decimal=1)
        npt.assert_almost_equal(dd_astropy, dd_fixed_interp, decimal=1)

    def test_oLCDM_init(self):
        kwargs_cosmo = {'h0': 100, 'om': .1, 'ok': -0.1}
        kwargs_likelihood_list = [{'likelihood_type': 'DSPL', 'z_lens': 0.3, 'z_source_1': 0.5, 'z_source_2':1.5,
                                   'beta_dspl': 1.2, 'sigma_beta_dspl': 0.01}]
        cosmoL = CosmoLikelihood(kwargs_likelihood_list, self.cosmology, self.kwargs_bounds,
                                 interpolate_cosmo=False, cosmo_fixed=None)

    def test_sne_likelihood_integration(self):
        cosmoL = CosmoLikelihood([], self.cosmology, self.kwargs_bounds, sne_likelihood='Pantheon_binned',
                                 interpolate_cosmo=True, num_redshift_interp=100, cosmo_fixed=None)
        kwargs_cosmo = {'h0': self.H0_true, 'om': self.omega_m_true, 'ok': 0}
        args = cosmoL.param.kwargs2args(kwargs_cosmo=kwargs_cosmo)
        logl = cosmoL.likelihood(args=args)
        assert logl < 0

    def test_kde_likelihood_integration(self):
        chain = import_Planck_chain(_PATH_2_PLANCKDATA, 'base', 'plikHM_TTTEEE_lowl_lowE', ['h0', 'om'],
                                    self.cosmology, rescale=True)
        cosmoL = CosmoLikelihood([], 'FLCDM', self.kwargs_bounds, KDE_likelihood_chain=chain)
        kwargs_cosmo = {'h0': self.H0_true, 'om': self.omega_m_true, 'ok': 0}
        args = cosmoL.param.kwargs2args(kwargs_cosmo=kwargs_cosmo)
        logl = cosmoL.likelihood(args=args)
        assert logl < 0


if __name__ == '__main__':
    pytest.main()
