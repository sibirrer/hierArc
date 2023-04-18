import pytest
import numpy as np
import unittest
from hierarc.Likelihood.kinematic_scaling import KinematicScalingSingleAperture, KinematicScalingIFU


class TestAnisotropyScalingSingleAperture(object):

    def setup(self):
        ani_param_array = np.linspace(start=0, stop=1, num=10)
        ani_scaling_array = ani_param_array * 2
        self.scaling = KinematicScalingSingleAperture(ani_param_array, ani_scaling_array)

        ani_param_array = [np.linspace(start=0, stop=1, num=10), np.linspace(start=1, stop=2, num=5)]
        ani_scaling_array = np.outer(ani_param_array[0], ani_param_array[1])
        self.scaling_2d = KinematicScalingSingleAperture(ani_param_array, ani_scaling_array)

    def test_ani_scaling(self):
        scaling = self.scaling.j_kin_scaling(j_kin_param_array=[1])
        assert scaling == 2

        scaling = self.scaling.j_kin_scaling(j_kin_param_array=None)
        assert scaling == 1

        scaling = self.scaling_2d.j_kin_scaling(j_kin_param_array=[1, 2])
        assert scaling == 2


class TestAnisotropyScalingIFU(object):

    def setup(self):
        ani_param_array = np.linspace(start=0, stop=1, num=10)
        ani_scaling_array = ani_param_array * 2
        self.scaling = KinematicScalingIFU(anisotropy_model='OM', scaling_param_array=ani_param_array,
                                           scaling_array_list=[ani_scaling_array])

        ani_param_array = [np.linspace(start=0, stop=1, num=10), np.linspace(start=1, stop=2, num=5)]
        ani_scaling_array = np.outer(ani_param_array[0], ani_param_array[1])
        self.scaling_2d = KinematicScalingIFU(anisotropy_model='GOM', scaling_param_array=ani_param_array,
                                              scaling_array_list=[ani_scaling_array])

    def test_ani_scaling(self):
        scaling = self.scaling.j_kin_scaling(j_kin_param_array=[1])
        assert scaling[0] == 2

        scaling = self.scaling.j_kin_scaling(j_kin_param_array=None)
        assert scaling[0] == 1

        scaling = self.scaling_2d.j_kin_scaling(j_kin_param_array=[1, 2])
        assert scaling[0] == 2

    def test_draw_anisotropy(self):
        a_ani = 1
        beta_inf = 1.5
        param_draw = self.scaling.draw_j_kin(a_ani=1, a_ani_sigma=0, beta_inf=beta_inf, beta_inf_sigma=0)
        assert param_draw[0] == a_ani
        for i in range(100):
            param_draw = self.scaling.draw_j_kin(a_ani=1, a_ani_sigma=1, beta_inf=beta_inf, beta_inf_sigma=1)

        param_draw = self.scaling_2d.draw_j_kin(a_ani=1, a_ani_sigma=0, beta_inf=beta_inf, beta_inf_sigma=0)
        assert param_draw[0] == a_ani
        assert param_draw[1] == beta_inf
        for i in range(100):
            param_draw = self.scaling_2d.draw_j_kin(a_ani=1, a_ani_sigma=1, beta_inf=beta_inf, beta_inf_sigma=1)

        scaling = KinematicScalingIFU(anisotropy_model='NONE')
        param_draw = scaling.draw_j_kin(a_ani=1, a_ani_sigma=0, beta_inf=beta_inf, beta_inf_sigma=0)
        assert param_draw is None


class TestRaise(unittest.TestCase):

    def test_raise(self):

        with self.assertRaises(ValueError):
            ani_param_array = [np.linspace(start=0, stop=1, num=10), np.linspace(start=1, stop=2, num=5), 1]
            ani_scaling_array = np.outer(ani_param_array[0], ani_param_array[1])
            self.scaling_2d = KinematicScalingSingleAperture(ani_param_array, ani_scaling_array)

        with self.assertRaises(ValueError):
            KinematicScalingIFU(anisotropy_model='blabla', scaling_param_array=np.array([0, 1]), scaling_array_list=[np.array([0, 1])])

        with self.assertRaises(ValueError):
            ani_param_array = np.linspace(start=0, stop=1, num=10)
            ani_scaling_array = ani_param_array * 2
            scaling = KinematicScalingIFU(anisotropy_model='OM', scaling_param_array=ani_param_array,
                                          scaling_array_list=[ani_scaling_array])
            scaling.draw_j_kin(a_ani=-1, a_ani_sigma=0, beta_inf=-1, beta_inf_sigma=0)

        with self.assertRaises(ValueError):
            ani_param_array = [np.linspace(start=0, stop=1, num=10), np.linspace(start=1, stop=2, num=5)]
            ani_scaling_array = np.outer(ani_param_array[0], ani_param_array[1])
            scaling = KinematicScalingIFU(anisotropy_model='GOM', scaling_param_array=ani_param_array,
                                          scaling_array_list=[ani_scaling_array])
            scaling.draw_j_kin(a_ani=-1, a_ani_sigma=0, beta_inf=-1, beta_inf_sigma=0)


if __name__ == '__main__':
    pytest.main()

