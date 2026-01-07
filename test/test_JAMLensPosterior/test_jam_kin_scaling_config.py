import numpy as np

from hierarc.JAMLensPosterior.jam_kin_scaling_config import JAMKinScalingConfig


class TestKinScalingConfig(object):

    def setup_method(self):
        pass

    def test_kwargs_axisymmetric(self):
        kin_scaling = JAMKinScalingConfig(
            anisotropy_model="GOM",
            r_eff=1,
            gamma_pl_scaling=np.linspace(1.5, 2.5, 5),
            q_intrinsic_scaling=np.linspace(0.4, 1.0, 6),
            q_intrinsic_mean=0.8,
        )
        assert kin_scaling.num_scaling_dim == 4  # gamma_pl, q_intrinsic, a_ani, beta_inf

        assert kin_scaling.axisymmetric_jam_base['q_intrinsic'] == 0.8

        kin_scaling = JAMKinScalingConfig(
            anisotropy_model="GOM",
            r_eff=1,
            gamma_pl_scaling=np.linspace(1.5, 2.5, 5),
            q_intrinsic_scaling=np.linspace(0.5, 1.0, 6),
        )
        assert kin_scaling.axisymmetric_jam_base['q_intrinsic'] == 0.75
