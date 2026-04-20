import numpy as np

from hierarc.LensPosterior.kin_scaling_config import KinScalingConfig
import numpy.testing as npt
import pytest


class TestKinScalingConfig(object):

    def setup_method(self):
        pass

    def test_kwargs_lens_base(self):
        kin_scaling = KinScalingConfig(
            anisotropy_model="GOM",
            r_eff=1,
            gamma_pl_scaling=np.linspace(1.5, 2.5, 5),
            log_m2l_scaling=np.linspace(0, 1, 5),
            gamma_in_scaling=np.linspace(0.5, 1.5, 5),
        )
        kin_scaling.kwargs_lens_base

    def test_init(self):
        kin_scaling = KinScalingConfig(
            anisotropy_model="NONE",
            r_eff=None,
            gamma_in_scaling=None,
            log_m2l_scaling=None,
        )
        kin_scaling._anisotropy_model = "BAD"

        with npt.assert_raises(ValueError):
            kin_scaling.anisotropy_kwargs()

    def test_kwargs_axisymmetric(self):
        kin_scaling = KinScalingConfig(
            anisotropy_model="GOM",
            r_eff=1,
            gamma_pl_scaling=np.linspace(1.5, 2.5, 5),
            q_intrinsic_scaling=np.linspace(0.5, 1.0, 6),
        )
        assert (
            kin_scaling.num_scaling_dim == 4
        )  # gamma_pl, q_intrinsic, a_ani, beta_inf
        assert kin_scaling.kwargs_deprojection_base["q_intrinsic"] == 0.75


if __name__ == "__main__":
    pytest.main()
