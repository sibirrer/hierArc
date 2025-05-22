from hierarc.Likelihood.BAOLikelihood.bao_likelihood import BAOLikelihood
import pytest
import numpy as np
import numpy.testing as npt


class TestBAO(object):
    def setup_method(self):
        np.random.seed(42)
        # define redshifts
        num = 8  # number of BAO measurements
        z = np.linspace(start=0.1, stop=0.8, num=num)

        # define cosmology
        from astropy.cosmology import FlatLambdaCDM

        om_mean, om_sigma = 0.3, 0.01
        rd = 150 # comoving sound horizon at the drag epoch
        cosmo_true = FlatLambdaCDM(H0=70, Om0=om_mean)

        # compute BAO distances (DM only)
        dist_true = cosmo_true.comoving_transverse_distance(z).value / rd
        dist_type = ["DM_over_rs"] * num

        # draw from scatter
        sigma_d = np.sqrt(4e-02)
        cov = np.diag(np.ones(num) * sigma_d**2)

        dist_measured = np.random.multivariate_normal(dist_true, cov)
        kwargs_bao_likelihood = {
            "z": z,
            "d": dist_measured,
            "distance_type": dist_type,
            "cov":cov,
        }

        self.likelihood = BAOLikelihood(sample_name="CUSTOM", **kwargs_bao_likelihood)
        self.dists_true = dist_true
        self.rd_true = rd
        self.sigma_d_true = sigma_d
        self.cosmo_true = cosmo_true

    def test_log_likelihood(self):
        logL = self.likelihood.log_likelihood(
            self.cosmo_true,
            self.rd_true,
        )

        logL_high = self.likelihood.log_likelihood(
            self.cosmo_true,
            self.rd_true + 10,
        )
        assert logL > logL_high

        logL_low = self.likelihood.log_likelihood(
            self.cosmo_true,
            self.rd_true - 10,
        )
        print(logL_low, logL, logL_high)
        assert logL > logL_low

    def test_desi_dr2(self):
        likelihood = BAOLikelihood(sample_name="DESI_DR2")
        om_mean, om_sigma = 0.2975, 0.0086  # from DESI collaboration et al. 2025
        from astropy.cosmology import FlatLambdaCDM

        cosmo_mean = FlatLambdaCDM(H0=67.5, Om0=om_mean)
        logL_mean = likelihood.log_likelihood(cosmo_mean, self.rd_true)

        cosmo_sigma_plus = FlatLambdaCDM(H0=67.5, Om0=om_mean + om_sigma)
        logL_sigma_plus = likelihood.log_likelihood(cosmo_sigma_plus, self.rd_true)

        npt.assert_almost_equal(logL_sigma_plus - logL_mean, -0.8915, decimal=1)
        cosmo_sigma_neg = FlatLambdaCDM(H0=67.5, Om0=om_mean - om_sigma)
        logL_sigma_neg = likelihood.log_likelihood(cosmo_sigma_neg, self.rd_true)

        npt.assert_almost_equal(logL_sigma_neg - logL_mean, -6.03403, decimal=1)

    def test_raise(self):
        with pytest.raises(ValueError):
            BAOLikelihood(sample_name="UNKNOWN")

        with pytest.raises(NotImplementedError):
            self.likelihood.log_likelihood(self.cosmo_true)

if __name__ == "__main__":
    pytest.main()
