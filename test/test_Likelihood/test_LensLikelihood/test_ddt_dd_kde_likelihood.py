from hierarc.Likelihood.LensLikelihood.ddt_dd_kde_likelihood import DdtDdKDELikelihood
import numpy as np
import numpy.testing as npt
import pytest
import copy
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Cosmo.lens_cosmo import LensCosmo


class TestDdtDdKDELikelihood(object):

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
        self.sigma_Ddt = 500
        num_samples = 10000
        self.D_dt_samples = np.random.normal(self.D_dt_true, self.sigma_Ddt, num_samples)
        self.D_d_samples = np.random.normal(self.Dd_true, self.sigma_Dd, num_samples)

        self.kwargs_lens = {'z_lens': self.z_L, 'z_source': self.z_S,
                                  'dd_samples': self.D_d_samples, 'ddt_samples': self.D_dt_samples,
                                  'kde_type': 'scipy_gaussian', 'bandwidth': 1}

    def test_log_likelihood(self):
        tdkin = DdtDdKDELikelihood(**self.kwargs_lens)
        kwargs_lens = copy.deepcopy(self.kwargs_lens)
        kwargs_lens['interpol'] = True
        tdkin_interp = DdtDdKDELikelihood(**kwargs_lens)

        delta_ddt = 1
        delta_dd = 200
        logl = tdkin.log_likelihood(ddt=self.D_dt_true+delta_ddt, dd=self.Dd_true+delta_dd)
        logl_interp = tdkin_interp.log_likelihood(ddt=self.D_dt_true+delta_ddt, dd=self.Dd_true+delta_dd)
        print(logl, logl_interp)
        npt.assert_almost_equal(logl, logl_interp, decimal=3)
        logl_interp = tdkin_interp.log_likelihood(ddt=self.D_dt_true + 100000, dd=self.Dd_true+100000)
        #assert logl_interp == -np.inf


if __name__ == '__main__':
    pytest.main()
