import numpy as np
import numpy.testing as npt
import pytest

from hierarc.Sampling.ParamManager.deprojection_param import DeprojectionParam


class TestDeprojectionParam:
    def setup_method(self):
        self.param_off = DeprojectionParam(deprojection_sampling=False)

        self.param_on_none = DeprojectionParam(
            deprojection_sampling=True, distribution_function="NONE", kwargs_fixed={}
        )

        self.param_gauss = DeprojectionParam(
            deprojection_sampling=True,
            distribution_function="GAUSSIAN",
            kwargs_fixed={},
        )

        self.param_gauss_log = DeprojectionParam(
            deprojection_sampling=True,
            distribution_function="GAUSSIAN",
            log_scatter=True,
            kwargs_fixed={},
        )

        self.param_fixed_both = DeprojectionParam(
            deprojection_sampling=True,
            distribution_function="GAUSSIAN",
            kwargs_fixed={"q_intrinsic": 0.9, "q_intrinsic_sigma": 0.05},
        )

        self.param_fixed_q = DeprojectionParam(
            deprojection_sampling=True,
            distribution_function="GAUSSIAN",
            kwargs_fixed={"q_intrinsic": 0.92},
        )

        self.sample_kwargs = {"q_intrinsic": 0.87, "q_intrinsic_sigma": 0.12}

    def test_param_list_lengths_and_latex(self):
        # when sampling disabled -> empty
        assert self.param_off.param_list(latex_style=False) == []
        assert self.param_off.param_list(latex_style=True) == []

        # NONE distribution -> only q_intrinsic
        pl = self.param_on_none.param_list(latex_style=False)
        assert pl == ["q_intrinsic"]
        pl_tex = self.param_on_none.param_list(latex_style=True)
        assert pl_tex == [r"$\langle q_{\rm int} \rangle$"]

        # GAUSSIAN -> q_intrinsic + q_intrinsic_sigma
        plg = self.param_gauss.param_list(latex_style=False)
        assert plg == ["q_intrinsic", "q_intrinsic_sigma"]
        plg_tex = self.param_gauss.param_list(latex_style=True)
        assert plg_tex == [r"$\langle q_{\rm int} \rangle$", r"$\sigma(q_{\rm int})$"]

        # GAUSSIAN with log scatter -> latex shows log sigma label
        plglog_tex = self.param_gauss_log.param_list(latex_style=True)
        assert plglog_tex == [
            r"$\langle q_{\rm int} \rangle$",
            r"$\log_{10}\sigma(q_{\rm int})$",
        ]

        # fixed both -> no free params
        assert self.param_fixed_both.param_list(latex_style=False) == []
        assert self.param_fixed_both.param_list(latex_style=True) == []

        # fixed q_intrinsic -> only sigma remains
        pl_fixed_q = self.param_fixed_q.param_list(latex_style=False)
        assert pl_fixed_q == ["q_intrinsic_sigma"]

    def test_kwargs2args_and_args2kwargs_roundtrip(self):
        # Standard gaussian case (linear scatter)
        args = self.param_gauss.kwargs2args(self.sample_kwargs)
        # should produce [q_intrinsic, q_intrinsic_sigma]
        npt.assert_almost_equal(
            np.array(args),
            np.array(
                [
                    self.sample_kwargs["q_intrinsic"],
                    self.sample_kwargs["q_intrinsic_sigma"],
                ]
            ),
        )

        kwargs_new, i = self.param_gauss.args2kwargs(args, i=0)
        # args2kwargs returns kwargs and next index
        assert i == 2
        # roundtrip
        args_back = self.param_gauss.kwargs2args(kwargs_new)
        npt.assert_almost_equal(np.array(args_back), np.array(args))

    def test_log_scatter_conversion(self):
        # when log_scatter True, kwargs2args should return log10(sigma)
        args = self.param_gauss_log.kwargs2args(self.sample_kwargs)
        expected_args = [
            self.sample_kwargs["q_intrinsic"],
            np.log10(self.sample_kwargs["q_intrinsic_sigma"]),
        ]
        npt.assert_almost_equal(np.array(args), np.array(expected_args))

        # args2kwargs should invert: exponentiate base-10 for sigma
        kwargs_new, next_i = self.param_gauss_log.args2kwargs(args, i=0)
        assert next_i == 2
        npt.assert_almost_equal(
            kwargs_new["q_intrinsic"], self.sample_kwargs["q_intrinsic"]
        )
        npt.assert_almost_equal(kwargs_new["q_intrinsic_sigma"], 10 ** expected_args[1])

    def test_args2kwargs_with_fixed_values(self):
        # If q_intrinsic is fixed, args2kwargs should use the fixed value and not read it from args
        args = [0.111]  # this would correspond to sigma only if q_intrinsic fixed
        kwargs_new, next_i = self.param_fixed_q.args2kwargs(args, i=0)
        # q_intrinsic should come from kwargs_fixed
        assert kwargs_new["q_intrinsic"] == pytest.approx(0.92)
        # sigma should be taken from args (since q_intrinsic_sigma not fixed)
        assert "q_intrinsic_sigma" in kwargs_new
        # index should have advanced by 1 (only sigma consumed)
        assert next_i == 1

        # If both are fixed, args2kwargs should ignore input args and return fixed dict unchanged and same index
        kwargs_new2, next_i2 = self.param_fixed_both.args2kwargs([9.9, 9.9], i=0)
        assert kwargs_new2["q_intrinsic"] == pytest.approx(0.9)
        assert kwargs_new2["q_intrinsic_sigma"] == pytest.approx(0.05)
        assert next_i2 == 0

    def test_kwargs2args_with_fixed_behaviour(self):
        # when q_intrinsic fixed, kwargs2args should only return sigma val
        kw = {"q_intrinsic": 0.92, "q_intrinsic_sigma": 0.07}
        args = self.param_fixed_q.kwargs2args(kw)
        npt.assert_almost_equal(np.array(args), np.array([0.07]))

        # when both fixed -> no args
        args2 = self.param_fixed_both.kwargs2args(kw)
        assert args2 == []


if __name__ == "__main__":
    pytest.main()
