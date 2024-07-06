import numpy as np

from hierarc.LensPosterior.kin_scaling_config import KinScalingConfig
import numpy.testing as npt


class TestKinScalingConfig(object):

    def setup_method(self):
        pass

    def test_kwargs_lens_base(self):
        kin_scaling = KinScalingConfig(anisotropy_model="GOM",
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
