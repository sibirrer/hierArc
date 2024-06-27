from hierarc.Sampling.Distributions.anisotropy_distributions import (
    AnisotropyDistribution,
)


class TestAnisotropyDistribution(object):

    def setup_method(self):
        anisotropy_model = "GOM"
        distribution_function = "GAUSSIAN"
        kwargs_anisotropy_min = {"a_ani": 0, "beta_inf": 0.1}
        kwargs_anisotropy_max = {"a_ani": 5, "beta_inf": 1}

        self._ani_dist = AnisotropyDistribution(
            anisotropy_model=anisotropy_model,
            anisotropy_sampling=True,
            distribution_function=distribution_function,
            kwargs_anisotropy_min=kwargs_anisotropy_min,
            kwargs_anisotropy_max=kwargs_anisotropy_max,
        )

        ani_dist = AnisotropyDistribution(
            anisotropy_model=anisotropy_model,
            anisotropy_sampling=False,
            distribution_function=distribution_function,
            kwargs_anisotropy_min=kwargs_anisotropy_min,
            kwargs_anisotropy_max=kwargs_anisotropy_max,
        )

    def test_draw_anisotropy(self):
        kwargs_anisotropy = {
            "a_ani": 1,
            "beta_inf": 0.8,
            "a_ani_sigma": 0.1,
            "beta_inf_sigma": 0.2,
        }
        kwargs_drawn = self._ani_dist.draw_anisotropy(**kwargs_anisotropy)
        assert "a_ani" in kwargs_drawn
        assert "beta_inf" in kwargs_drawn

        ani_dist = AnisotropyDistribution(
            anisotropy_model="NONE",
            anisotropy_sampling=False,
            distribution_function="NONE",
            kwargs_anisotropy_min=None,
            kwargs_anisotropy_max=None,
        )
        kwargs_drawn = ani_dist.draw_anisotropy()
        assert "a_ani" not in kwargs_drawn
