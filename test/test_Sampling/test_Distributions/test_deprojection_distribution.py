import numpy.testing as npt
from hierarc.Sampling.Distributions.deprojection_distributions import (
    DeprojectionDistribution,
)
import pytest


class TestDeprojectionDistribution(object):

    def setup_method(self):

        self._q_intrinsic_gaussian = DeprojectionDistribution(
            deprojection_sampling=True,
            distribution_function="GAUSSIAN",
            kwargs_deprojection_min={"q_intrinsic": 0.2},
            kwargs_deprojection_max={"q_intrinsic": 1.0},
            parameterization="q_intrinsic",
        )

        self._q_intrinsic_gaussian_scaled = DeprojectionDistribution(
            deprojection_sampling=True,
            distribution_function="GAUSSIAN_SCALED",
            kwargs_deprojection_min={"q_intrinsic": 0.2},
            kwargs_deprojection_max={"q_intrinsic": 1.0},
            parameterization="q_intrinsic",
        )

        # inclination is not yet supported
        # self._inclination_uniform = DeprojectionDistribution(
        #     deprojection_sampling=True,
        #     distribution_function="UNIFORM",
        #     kwargs_deprojection_min={"inclination": 0},
        #     kwargs_deprojection_max={"inclination": 90},
        #     parameterization="inclination",
        # )

    def test_draw_deprojection(self):
        kwargs_q_intrinsic = {
            "q_intrinsic": 0.6,
            "q_intrinsic_sigma": 0.1,
        }
        kwargs_drawn = self._q_intrinsic_gaussian.draw_deprojection(
            **kwargs_q_intrinsic
        )
        assert "q_intrinsic" in kwargs_drawn
        assert (kwargs_drawn["q_intrinsic"] > 0.2) and (
            kwargs_drawn["q_intrinsic"] <= 1.0
        )

        kwargs_drawn = self._q_intrinsic_gaussian_scaled.draw_deprojection(
            **kwargs_q_intrinsic
        )
        assert "q_intrinsic" in kwargs_drawn
        assert (kwargs_drawn["q_intrinsic"] > 0.2) and (
            kwargs_drawn["q_intrinsic"] <= 1.0
        )

        kwargs_inclination = {
            "q_intrinsic": 45,
            "q_intrinsic_sigma": 0,
            "q_observed": 0.7,
        }
        # kwargs_drawn = self._inclination_uniform.draw_deprojection(**kwargs_inclination)
        # assert "q_intrinsic" in kwargs_drawn
        # assert (kwargs_drawn["q_intrinsic"] > 0.) and (kwargs_drawn["q_intrinsic"] <= 1.0)

        no_dist_no_sampling = DeprojectionDistribution(
            deprojection_sampling=False,
            distribution_function="NONE",
            kwargs_deprojection_min=None,
            kwargs_deprojection_max=None,
        )
        kwargs_drawn = no_dist_no_sampling.draw_deprojection()
        assert "q_intrinsic" not in kwargs_drawn
        kwargs_drawn = no_dist_no_sampling.draw_deprojection(q_intrinsic=0.6)
        assert kwargs_drawn["q_intrinsic"] == 0.6

        no_dist_sampling = DeprojectionDistribution(
            deprojection_sampling=True,
            distribution_function="NONE",
            kwargs_deprojection_min=None,
            kwargs_deprojection_max=None,
        )
        kwargs_drawn = no_dist_sampling.draw_deprojection(q_intrinsic=0.6)
        assert kwargs_drawn["q_intrinsic"] == 0.6

        for i in range(100):
            kwargs_drawn = self._q_intrinsic_gaussian.draw_deprojection(
                **kwargs_q_intrinsic
            )

    def test_raises(self):

        with npt.assert_raises(ValueError):
            kwargs_q_intrinsic_invalid = {
                "q_intrinsic": 2.0,
                "q_intrinsic_sigma": 0.1,
            }
            kwargs_drawn = self._q_intrinsic_gaussian.draw_deprojection(
                **kwargs_q_intrinsic_invalid
            )

        with npt.assert_raises(ValueError):
            DeprojectionDistribution(
                deprojection_sampling=True,
                distribution_function="INVALID_DIST",
                kwargs_deprojection_min={"q_intrinsic": 0.2},
                kwargs_deprojection_max={"q_intrinsic": 1.0},
            )

        with npt.assert_raises(ValueError):
            DeprojectionDistribution(
                deprojection_sampling=True,
                distribution_function="GAUSSIAN",
                kwargs_deprojection_min={"q_intrinsic": 0.2},
                kwargs_deprojection_max={"q_intrinsic": 1.0},
                parameterization="invalid_param",
            )


if __name__ == "__main__":
    pytest.main()
