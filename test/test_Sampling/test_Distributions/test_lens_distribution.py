import copy
import numpy.testing as npt

from hierarc.Sampling.Distributions.lens_distribution import LensDistribution


class TestLensDistribution(object):

    def setup_method(self):
        self.kwargs_sampling = {
            "lambda_mst_sampling": True,
            "lambda_mst_distribution": "GAUSSIAN",
            "gamma_in_sampling": True,
            "gamma_in_distribution": "GAUSSIAN",
            "log_m2l_sampling": True,
            "log_m2l_distribution": "GAUSSIAN",
            "alpha_lambda_sampling": True,
            "beta_lambda_sampling": True,
            "alpha_gamma_in_sampling": True,
            "alpha_log_m2l_sampling": True,
            "log_scatter": False,  # change for different tests
            "mst_ifu": False,  # change for different tests
            "lambda_scaling_property": 0.1,
            "lambda_scaling_property_beta": 0.2,
            "kwargs_min": {"gamma_in": 1, "log_m2l": 0},
            "kwargs_max": {"gamma_in": 2, "log_m2l": 1},
        }

        self.kwargs_lens = {
            "lambda_mst": 1.1,
            "lambda_mst_sigma": 0.1,
            "gamma_ppn": 0.9,
            "lambda_ifu": 0.5,
            "lambda_ifu_sigma": 0.2,
            "alpha_lambda": -0.2,
            "beta_lambda": 0.3,
            "gamma_in": 1.5,
            "gamma_in_sigma": 1,
            "alpha_gamma_in": 0.2,
            "log_m2l": 0.6,
            "log_m2l_sigma": 1,
            "alpha_log_m2l": -0.1,
        }

    def test_draw_lens(self):
        lens_dist = LensDistribution(**self.kwargs_sampling)
        kwargs_return = lens_dist.draw_lens(**self.kwargs_lens)

        assert "lambda_mst" in kwargs_return

        kwargs_sampling = copy.deepcopy(self.kwargs_sampling)
        kwargs_sampling["log_scatter"] = True
        kwargs_sampling["lambda_ifu"] = True
        lens_dist = LensDistribution(kwargs_sampling)
        for i in range(100):
            kwargs_return = lens_dist.draw_lens(**self.kwargs_lens)

            assert "lambda_mst" in kwargs_return

    def test_raises(self):

        with npt.assert_raises(ValueError):
            lens_dist = LensDistribution(**self.kwargs_sampling)
            kwargs_lens = copy.deepcopy(self.kwargs_lens)
            kwargs_lens["gamma_in"] = -10
            kwargs_return = lens_dist.draw_lens(**kwargs_lens)

        with npt.assert_raises(ValueError):
            lens_dist = LensDistribution(**self.kwargs_sampling)
            kwargs_lens = copy.deepcopy(self.kwargs_lens)
            kwargs_lens["log_m2l"] = -100
            kwargs_return = lens_dist.draw_lens(**kwargs_lens)
