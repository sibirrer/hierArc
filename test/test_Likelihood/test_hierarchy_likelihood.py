from hierarc.Likelihood.hierarchy_likelihood import LensLikelihood
from astropy.cosmology import FlatLambdaCDM
import pytest
import numpy as np
import numpy.testing as npt


class TestLensLikelihood(object):
    def setup_method(self):
        z_lens = 0.5
        z_source = 1.5
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        dd = self.cosmo.angular_diameter_distance(z=z_lens).value
        ds = self.cosmo.angular_diameter_distance(z=z_source).value
        dds = self.cosmo.angular_diameter_distance_z1z2(z1=z_lens, z2=z_source).value
        ddt = (1.0 + z_lens) * dd * ds / dds

        ani_param_array = np.linspace(start=0, stop=5, num=10)
        ani_scaling_array = ani_param_array

        kwargs_likelihood = {
            "ddt_mean": ddt,
            "ddt_sigma": ddt / 20,
            "dd_mean": dd,
            "dd_sigma": dd / 10,
        }
        kwargs_model = {
            "anisotropy_model": "OM",
            "anisotropy_sampling": True,
            "anisotroy_distribution_function": "GAUSSIAN",
            "lambda_mst_distribution": "GAUSSIAN",
        }
        # "gamma_in_sampling" = False,
        gamma_in_distribution = ("NONE",)
        log_m2l_sampling = (False,)
        log_m2l_distribution = ("NONE",)

        self.likelihood = LensLikelihood(
            z_lens,
            z_source,
            name="name",
            likelihood_type="DdtDdGaussian",
            kin_scaling_param_list=["a_ani"],
            j_kin_scaling_param_axes=ani_param_array,
            j_kin_scaling_grid_list=[ani_scaling_array],
            num_distribution_draws=200,
            los_distributions=["GAUSSIAN"],
            global_los_distribution=0,
            los_distribution_individual=None,
            kwargs_los_individual=None,
            mst_ifu=True,
            **kwargs_likelihood,
            **kwargs_model
        )
        self.likelihood_single = LensLikelihood(
            z_lens,
            z_source,
            name="name",
            likelihood_type="DdtDdGaussian",
            kin_scaling_param_list=["a_ani"],
            j_kin_scaling_param_axes=ani_param_array,
            j_kin_scaling_grid_list=[ani_scaling_array],
            num_distribution_draws=200,
            los_distributions=["GAUSSIAN"],
            global_los_distribution=0,  # testing previously set to =False
            los_distribution_individual=None,
            kwargs_los_individual=None,
            mst_ifu=False,
            **kwargs_likelihood,
            **kwargs_model
        )
        self.likelihood_zero_dist = LensLikelihood(
            z_lens,
            z_source,
            name="name",
            likelihood_type="DdtDdGaussian",
            kin_scaling_param_list=["a_ani"],
            j_kin_scaling_param_axes=ani_param_array,
            j_kin_scaling_grid_list=[ani_scaling_array],
            num_distribution_draws=0,
            los_distributions=["GAUSSIAN"],
            global_los_distribution=0,
            los_distribution_individual=None,
            kwargs_los_individual=None,
            mst_ifu=True,
            **kwargs_likelihood,
            **kwargs_model
        )

        kappa_posterior = np.random.normal(loc=0, scale=0.03, size=100000)
        kappa_pdf, kappa_bin_edges = np.histogram(kappa_posterior, density=True)
        self.likelihood_kappa_ext = LensLikelihood(
            z_lens,
            z_source,
            name="name",
            likelihood_type="DdtDdGaussian",
            kin_scaling_param_list=["a_ani"],
            j_kin_scaling_param_axes=ani_param_array,
            j_kin_scaling_grid_list=[ani_scaling_array],
            num_distribution_draws=200,
            # los_distributions=["GAUSSIAN"],
            global_los_distribution=False,
            los_distribution_individual="PDF",
            kwargs_los_individual={"bin_edges": kappa_bin_edges, "pdf_array": kappa_pdf},
            mst_ifu=False,
            **kwargs_likelihood,
            **kwargs_model
        )

        gamma_in_array = np.linspace(start=0.1, stop=2.9, num=5)
        log_m2l_array = np.linspace(start=0.1, stop=1, num=10)
        param_scaling_array = np.multiply.outer(
            np.ones_like(ani_param_array),
            np.outer(np.ones_like(gamma_in_array), np.ones_like(log_m2l_array)),
        )
        j_kin_scaling_param_axes = [ani_param_array, gamma_in_array, log_m2l_array]
        self.likelihood_gamma_in_log_m2l_list_ani = LensLikelihood(
            z_lens,
            z_source,
            name="name",
            likelihood_type="DdtDdGaussian",
            kin_scaling_param_list=["a_ani", "gamma_in", "log_m2l"],
            j_kin_scaling_param_axes=j_kin_scaling_param_axes,
            j_kin_scaling_grid_list=[param_scaling_array],
            num_distribution_draws=200,
            mst_ifu=False,
            gamma_in_sampling=False,
            gamma_in_distribution="GAUSSIAN",
            log_m2l_sampling=True,
            log_m2l_distribution="GAUSSIAN",
            **kwargs_likelihood,
            **kwargs_model
        )

        param_scaling_array = np.outer(
            np.ones_like(gamma_in_array), np.ones_like(log_m2l_array)
        )

        j_kin_scaling_param_axes = [gamma_in_array, log_m2l_array]

        self.likelihood_gamma_in_log_m2l = LensLikelihood(
            z_lens,
            z_source,
            name="name",
            likelihood_type="DdtDdGaussian",
            kin_scaling_param_list=["gamma_in", "log_m2l"],
            j_kin_scaling_param_axes=j_kin_scaling_param_axes,
            j_kin_scaling_grid_list=[param_scaling_array],
            num_distribution_draws=200,
            mst_ifu=False,
            gamma_in_sampling=True,
            gamma_in_distribution="GAUSSIAN",
            log_m2l_sampling=True,
            log_m2l_distribution="GAUSSIAN",
            **kwargs_likelihood,
            **kwargs_model  # TODO: remove anisotropy sampling in that scenario?
        )

        self.likelihood_gamma_in_fail_case = LensLikelihood(
            z_lens,
            z_source,
            name="name",
            likelihood_type="DdtDdGaussian",
            kin_scaling_param_list=["a_ani"],
            j_kin_scaling_param_axes=ani_param_array,
            j_kin_scaling_grid_list=[ani_scaling_array],
            num_distribution_draws=200,
            mst_ifu=False,
            lambda_scaling_property=100,
            **kwargs_likelihood,
            **kwargs_model
        )

    def test_lens_log_likelihood(self):
        np.random.seed(42)
        kwargs_lens = {
            "lambda_mst": 1,
            "lambda_mst_sigma": 0.01,
            "lambda_ifu": 1,
            "lambda_ifu_sigma": 0.01,
        }
        kwargs_los = [{"mean": 0, "sigma": 0.03}]

        kwargs_kin = {"a_ani": 1, "a_ani_sigma": 0.1}
        ln_likelihood = self.likelihood.lens_log_likelihood(
            self.cosmo,
            kwargs_lens=kwargs_lens,
            kwargs_kin=kwargs_kin,
            kwargs_los=kwargs_los,
        )
        npt.assert_almost_equal(ln_likelihood, -0.5, decimal=1)

        ln_likelihood_zero = self.likelihood_zero_dist.lens_log_likelihood(
            self.cosmo,
            kwargs_lens=kwargs_lens,
            kwargs_kin=kwargs_kin,
            kwargs_los=kwargs_los,
        )
        assert ln_likelihood_zero == -np.inf

        ln_likelihood_kappa_ext = self.likelihood_kappa_ext.lens_log_likelihood(
            self.cosmo,
            kwargs_lens=kwargs_lens,
            kwargs_kin=kwargs_kin,
            kwargs_los=kwargs_los,
        )
        npt.assert_almost_equal(ln_likelihood, ln_likelihood_kappa_ext, decimal=1)

        kwargs_lens = {
            "lambda_mst": 1000000,
            "lambda_mst_sigma": 0,
            "lambda_ifu": 1,
            "lambda_ifu_sigma": 0,
        }
        kwargs_los = [{"mean": 0, "sigma": 0}]
        kwargs_kin = {"a_ani": 1, "a_ani_sigma": 0}
        ln_inf = self.likelihood_single.lens_log_likelihood(
            self.cosmo,
            kwargs_lens=kwargs_lens,
            kwargs_kin=kwargs_kin,
            kwargs_los=kwargs_los,
        )
        assert ln_inf < -10000000

        ln_inf = self.likelihood_single.lens_log_likelihood(
            self.cosmo, kwargs_lens=None, kwargs_kin=kwargs_kin, kwargs_los=kwargs_los
        )
        npt.assert_almost_equal(ln_inf, 0.0, decimal=1)

        ln_inf = self.likelihood_single.sigma_v_measured_vs_predict(
            self.cosmo, kwargs_lens=None, kwargs_kin=None
        )
        assert np.all([x is None for x in ln_inf])

        kwargs_test = self.likelihood._kwargs_init(kwargs=None)
        assert type(kwargs_test) is dict

        kwargs_lens = {
            "gamma_in": 1,
            "gamma_in_sigma": 0,
            "alpha_gamma_in": 0,
            "log_m2l": 1,
            "log_m2l_sigma": 0,
            "alpha_log_m2l": 0,
        }
        ln_likelihood = self.likelihood_gamma_in_log_m2l.lens_log_likelihood(
            self.cosmo, kwargs_lens=kwargs_lens, kwargs_kin=kwargs_kin
        )
        npt.assert_almost_equal(ln_likelihood, -0.0, decimal=1)

        kwargs_source = self.likelihood_gamma_in_fail_case._kwargs_init(None)
        z_apparent_m_anchor = kwargs_source.get("z_apparent_m_anchor", 0.1)
        delta_lum_dist = self.likelihood_gamma_in_fail_case.luminosity_distance_modulus(
            self.cosmo, z_apparent_m_anchor
        )

        z_lens = 0.5
        z_source = 1.5

        kwargs_lens = {
            "gamma_in": 1,
            "gamma_in_sigma": 0,
            "alpha_gamma_in": 0,
            "log_m2l": 1,
            "log_m2l_sigma": 0,
            "alpha_log_m2l": 1000,
        }

        dd = self.cosmo.angular_diameter_distance(z=z_lens).value
        ds = self.cosmo.angular_diameter_distance(z=z_source).value
        dds = self.cosmo.angular_diameter_distance_z1z2(z1=z_lens, z2=z_source).value
        ddt = (1.0 + z_lens) * dd * ds / dds

        # ln_likelihood = self.likelihood_gamma_in_fail_case.log_likelihood_single(
        #    ddt, dd, delta_lum_dist, kwargs_lens, kwargs_kin, kwargs_source, kwargs_los
        # )

        # assert ln_likelihood < -10000000


if __name__ == "__main__":
    pytest.main()
