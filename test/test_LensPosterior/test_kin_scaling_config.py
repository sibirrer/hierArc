from hierarc.LensPosterior.kin_scaling_config import KinScalingConfig

class TestKinScalingConfig(object):

    def setup_method(self):
        pass

    def test_init(self):
        KinScalingConfig(anisotropy_model="NONE", r_eff=None, gamma_in_scaling=None, log_m2l_scaling=None)
