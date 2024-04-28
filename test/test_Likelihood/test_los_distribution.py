from hierarc.Likelihood.los_distributions import LOSDistribution
from scipy.stats import genextreme
import numpy as np
import numpy.testing as npt


class TestLOSDistribution(object):

    def setup_method(self):
        pass

    def test_gev(self):

        xi = -0.1
        mean_gev = 0.02
        sigma_gev = np.exp(-5.46)

        mean_gauss = 0.1
        sigma_gauss = 0.2

        kappa_ext_draw = genextreme.rvs(c=xi, loc=mean_gev, scale=sigma_gev, size=10000)
        npt.assert_almost_equal(np.mean(kappa_ext_draw), mean_gev, decimal=2)
        npt.assert_almost_equal(np.std(kappa_ext_draw), sigma_gev, decimal=2)

        kappa_pdf, kappa_bin_edges = np.histogram(kappa_ext_draw, bins=100)
        kappa_pdf = np.array(kappa_pdf, dtype=float) / np.sum(kappa_pdf)

        los_distribution = ["GAUSSIAN", "GEV"]

        kwargs_los = [
            {"mean": mean_gauss, "sigma": sigma_gauss},
            {"mean": mean_gev, "sigma": sigma_gev, "xi": xi},
        ]

        # here we draw from the scipy function
        dist_gev = LOSDistribution(
            kappa_pdf=kappa_pdf,
            kappa_bin_edges=kappa_bin_edges,
            global_los_distribution=1,
            los_distributions=los_distribution,
        )

        kappa_dist_drawn = dist_gev.draw_los(kwargs_los, size=10000)
        npt.assert_almost_equal(np.mean(kappa_dist_drawn), mean_gev, decimal=2)
        npt.assert_almost_equal(np.std(kappa_dist_drawn), sigma_gev, decimal=2)

        # here we draw from the distribution
        dist_gev = LOSDistribution(
            kappa_pdf=kappa_pdf,
            kappa_bin_edges=kappa_bin_edges,
            global_los_distribution=False,
            los_distributions=los_distribution,
        )

        kappa_dist_drawn = dist_gev.draw_los(kwargs_los, size=10000)
        npt.assert_almost_equal(np.mean(kappa_dist_drawn), mean_gev, decimal=2)
        npt.assert_almost_equal(np.std(kappa_dist_drawn), sigma_gev, decimal=2)

        # draw from Gaussian
        dist_gev = LOSDistribution(
            kappa_pdf=kappa_pdf,
            kappa_bin_edges=kappa_bin_edges,
            global_los_distribution=0,
            los_distributions=los_distribution,
        )

        kappa_dist_drawn = dist_gev.draw_los(kwargs_los, size=10000)
        npt.assert_almost_equal(np.mean(kappa_dist_drawn), mean_gauss, decimal=2)
        npt.assert_almost_equal(np.std(kappa_dist_drawn), sigma_gauss, decimal=2)

    def test_draw_bool(self):
        xi = -0.1
        mean_gev = 0.02
        sigma_gev = np.exp(-5.46)

        mean_gauss = 0.1
        sigma_gauss = 0

        kappa_ext_draw = genextreme.rvs(c=xi, loc=mean_gev, scale=sigma_gev, size=10000)
        npt.assert_almost_equal(np.mean(kappa_ext_draw), mean_gev, decimal=2)
        npt.assert_almost_equal(np.std(kappa_ext_draw), sigma_gev, decimal=2)

        kappa_pdf, kappa_bin_edges = np.histogram(kappa_ext_draw, bins=100)
        kappa_pdf = np.array(kappa_pdf, dtype=float) / np.sum(kappa_pdf)

        los_distribution = ["GAUSSIAN", "GEV"]

        kwargs_los = [
            {"mean": mean_gauss, "sigma": sigma_gauss},
            {"mean": mean_gev, "sigma": sigma_gev, "xi": xi},
        ]

        dist = LOSDistribution(
            kappa_pdf=kappa_pdf,
            kappa_bin_edges=kappa_bin_edges,
            global_los_distribution=1,
            los_distributions=los_distribution,
        )
        bool_draw = dist.draw_bool(kwargs_los)
        assert bool_draw is True

        dist = LOSDistribution(
            kappa_pdf=kappa_pdf,
            kappa_bin_edges=kappa_bin_edges,
            global_los_distribution=0,
            los_distributions=los_distribution,
        )
        bool_draw = dist.draw_bool(kwargs_los)
        assert bool_draw is False

        dist = LOSDistribution(
            kappa_pdf=kappa_pdf,
            kappa_bin_edges=kappa_bin_edges,
            global_los_distribution=False,
            los_distributions=los_distribution,
        )
        bool_draw = dist.draw_bool(kwargs_los)
        assert bool_draw is True
