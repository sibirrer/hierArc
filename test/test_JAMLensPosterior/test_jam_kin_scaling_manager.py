import numpy as np
import pytest

from hierarc.JAMLensPosterior.jam_kin_scaling_manager import (
    JAMKinScaling,
    JAMKinScalingParamManager,
)

class TestParameterScalingIFU(object):
    def setup_method(self):
        ani_param_array = np.linspace(start=-0.6, stop=1.0, num=10)
        gamma_pl_array = np.linspace(start=1.5, stop=2.6, num=8)
        q_intrinsic = np.linspace(start=0.4, stop=1.0, num=10)
        param_arrays = [ani_param_array, gamma_pl_array, q_intrinsic]
        param_scaling_array = np.multiply.outer(
            ani_param_array,
            np.outer(gamma_pl_array, q_intrinsic),
        )
        self.scaling_axisymmetric = JAMKinScaling(
            j_kin_scaling_param_axes=param_arrays,
            j_kin_scaling_grid_list=[param_scaling_array],
            j_kin_scaling_param_name_list=["beta", "gamma_pl", "q_intrinsic"],
        )

    def test_kin_scaling(self):

        kwargs_param = {"beta": 0.5, "gamma_pl": 2.0, "q_intrinsic": 0.8}
        scaling = self.scaling_axisymmetric.kin_scaling(kwargs_param=kwargs_param)
        assert scaling[0] == 0.5 * 2.0 * 0.8


class TestKinScalingParamManager(object):

    def test_(self):
        kin_param_manager = JAMKinScalingParamManager(
            j_kin_scaling_param_name_list=["gamma_pl", "a_ani", "beta_inf", "q_intrinsic"]
        )
        param_array = [1, 2, 3, 4]
        kwargs_anisotropy, kwargs_deflector, kwargs_axisymmetry = kin_param_manager.param_array2kwargs(
            param_array=param_array
        )
        assert kwargs_axisymmetry["q_intrinsic"] == param_array[3]

        param_array_new = kin_param_manager.kwargs2param_array(
            kwargs={**kwargs_anisotropy, **kwargs_deflector, **kwargs_axisymmetry}
        )
        for i, param in enumerate(param_array_new):
            assert param == param_array[i]


if __name__ == "__main__":
    pytest.main()
