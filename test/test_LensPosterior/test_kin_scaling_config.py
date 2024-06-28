from hierarc.LensPosterior.kin_scaling_config import KinScalingConfig
import numpy.testing as npt


class TestKinScalingConfig(object):

    def setup_method(self):
        pass

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
