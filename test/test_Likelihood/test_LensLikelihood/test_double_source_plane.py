from astropy.cosmology import FlatLambdaCDM
import numpy as np
import numpy.testing as npt
from hierarc.Likelihood.LensLikelihood.double_source_plane import DSPLikelihood, beta_double_source_plane


class TestDSPLikelihood(object):
    """

    """
    def setup(self):
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        self.zl = 0.5
        self.zs1 = 1
        self.zs2 = 2
        self.beta = beta_double_source_plane(z_lens=self.zl, z_source_1=self.zs1, z_source_2=self.zs2, cosmo=self.cosmo)
        self.sigma_beta = 0.1

    def test_log_likelihood(self):
        """

        :return:
        """
        # make cosmo instance
        # compute beta
        dspl_likelihood = DSPLikelihood(z_lens=self.zl, z_source_1=self.zs1, z_source_2=self.zs2,
                                        beta_dspl=self.beta, sigma_beta_dspl=self.sigma_beta, normalized=False)
        log_l = dspl_likelihood.lens_log_likelihood(cosmo=self.cosmo)
        npt.assert_almost_equal(log_l, 0, decimal=5)

        dspl_likelihood = DSPLikelihood(z_lens=self.zl, z_source_1=self.zs1, z_source_2=self.zs2,
                                        beta_dspl=self.beta - self.sigma_beta, sigma_beta_dspl=self.sigma_beta,
                                        normalized=False)
        log_l = dspl_likelihood.lens_log_likelihood(cosmo=self.cosmo)
        npt.assert_almost_equal(log_l, -0.5, decimal=5)

        dspl_likelihood = DSPLikelihood(z_lens=self.zl, z_source_1=self.zs1, z_source_2=self.zs2,
                                        beta_dspl=self.beta, sigma_beta_dspl=self.sigma_beta, normalized=True)
        log_l = dspl_likelihood.lens_log_likelihood(cosmo=self.cosmo)
        npt.assert_almost_equal(log_l, np.log(1/np.sqrt(2*np.pi) / self.sigma_beta), decimal=5)

    def test_num_data(self):
        dspl_likelihood = DSPLikelihood(z_lens=self.zl, z_source_1=self.zs1, z_source_2=self.zs2,
                                        beta_dspl=self.beta, sigma_beta_dspl=self.sigma_beta, normalized=False)
        num_data = dspl_likelihood.num_data()
        assert num_data == 1
