import numpy as np
import numpy.testing as npt
import pytest

from hierarc.Likelihood.kin_scaling import KinScaling, ParameterScalingSingleMeasurement


class TestKinScaling(object):

    def test_single_param(self):
        param_arrays = np.linspace(0, 1, 11)
        scaling_grid_list = [param_arrays**2]
        param_list = ["a"]
        kin_scaling = KinScaling(j_kin_scaling_param_axes=param_arrays,
                                 j_kin_scaling_grid_list=scaling_grid_list,
                                 j_kin_scaling_param_name_list=param_list)
        kwargs_param = {"a": 0.5}
        j_scaling = kin_scaling.kin_scaling(kwargs_param=kwargs_param)
        npt.assert_almost_equal(j_scaling, 0.5**2, decimal=2)
        kwargs_min, kwargs_max = kin_scaling.param_bounds_interpol()
        assert kwargs_min["a"] == 0

        param_arrays = np.linspace(0, 1, 11)
        scaling_grid_list = [param_arrays ** 2]
        param_list = ["a"]
        kin_scaling = KinScaling(j_kin_scaling_param_axes=[param_arrays],
                                 j_kin_scaling_grid_list=scaling_grid_list,
                                 j_kin_scaling_param_name_list=param_list)
        kwargs_param = {"a": 0.5}
        j_scaling = kin_scaling.kin_scaling(kwargs_param=kwargs_param)
        npt.assert_almost_equal(j_scaling, 0.5 ** 2, decimal=2)
        kwargs_min, kwargs_max = kin_scaling.param_bounds_interpol()
        assert kwargs_min["a"] == 0

    def test_two_parameters(self):
        param_arrays = [np.linspace(0, 1, 11), np.linspace(0, 2, 21)]
        xy, uv = np.meshgrid(param_arrays[0], param_arrays[1])
        scaling_grid_list = [xy.T * uv.T, xy.T, uv.T]
        shape_scaling = np.shape(scaling_grid_list[0])
        assert shape_scaling[0] == 11
        assert shape_scaling[1] == 21
        # assert 1 == 0
        param_list = ["a", "b"]
        kin_scaling = KinScaling(j_kin_scaling_param_axes=param_arrays,
                                 j_kin_scaling_grid_list=scaling_grid_list,
                                 j_kin_scaling_param_name_list=param_list)
        kwargs_param = {"a": 0.5, "b": 0.3}
        j_scaling = kin_scaling.kin_scaling(kwargs_param=kwargs_param)
        print(j_scaling)
        npt.assert_almost_equal(j_scaling[0], 0.5 * 0.3, decimal=2)
        npt.assert_almost_equal(j_scaling[1], 0.5, decimal=2)
        npt.assert_almost_equal(j_scaling[2], 0.3, decimal=2)
        kwargs_min, kwargs_max = kin_scaling.param_bounds_interpol()
        assert kwargs_min["a"] == 0

    def test__kwargs2param_array(self):
        param_arrays = [np.linspace(0, 1, 11), np.linspace(0, 2, 11)]
        xy, uv = np.meshgrid(param_arrays[0], param_arrays[1])
        scaling_grid_list = [xy * uv, xy, uv]

        param_list = ["a", "b"]
        kin_scaling = KinScaling(j_kin_scaling_param_axes=param_arrays,
                                 j_kin_scaling_grid_list=scaling_grid_list,
                                 j_kin_scaling_param_name_list=param_list)
        kwargs_param = {"a": 0.5, "b": 0.3}
        param_array = kin_scaling._kwargs2param_array(kwargs_param)
        assert param_array[0] == kwargs_param["a"]
        assert param_array[1] == kwargs_param["b"]
        kwargs_min, kwargs_max = kin_scaling.param_bounds_interpol()
        assert kwargs_min["a"] == 0
        assert kwargs_min["b"] == 0
        assert kwargs_max["b"] == 2
        assert kwargs_max["a"] == 1

    def test_empty(self):
        kin_scaling = KinScaling(j_kin_scaling_param_axes=None, j_kin_scaling_grid_list=None, j_kin_scaling_param_name_list=None)
        output = kin_scaling.kin_scaling(kwargs_param=None)
        assert output == 1
        kwargs_min, kwargs_max = kin_scaling.param_bounds_interpol()

class TestParameterScalingSingleAperture(object):
    def setup_method(self):
        ani_param_array = np.linspace(start=0, stop=1, num=10)
        param_scaling_array = ani_param_array * 2
        self.scaling = ParameterScalingSingleMeasurement(
            ani_param_array, param_scaling_array
        )

        ani_param_array = [
            np.linspace(start=0, stop=1, num=10),
            np.linspace(start=1, stop=2, num=5),
        ]
        param_scaling_array = np.outer(ani_param_array[0], ani_param_array[1])
        print(np.shape(param_scaling_array), 'test shape')
        self.scaling_2d = ParameterScalingSingleMeasurement(
            ani_param_array, param_scaling_array
        )

        ani_param_array = np.linspace(start=0, stop=1, num=10)
        gamma_in_array = np.linspace(start=0.1, stop=2.9, num=5)
        log_m2l_array = np.linspace(start=0.1, stop=1, num=10)

        param_arrays = [ani_param_array, gamma_in_array, log_m2l_array]
        param_scaling_array = np.multiply.outer(
            ani_param_array,
            np.outer(gamma_in_array, log_m2l_array),
        )
        self.scaling_nfw = ParameterScalingSingleMeasurement(
            param_arrays, param_scaling_array
        )

        gom_param_array = [
            np.linspace(start=0, stop=1, num=10),
            np.linspace(start=1, stop=2, num=5),
        ]
        param_arrays = [
            gom_param_array[0],
            gom_param_array[1],
            gamma_in_array,
            log_m2l_array,
        ]
        param_scaling_array = np.multiply.outer(
            gom_param_array[0],
            np.multiply.outer(
                gom_param_array[1],
                np.outer(gamma_in_array, log_m2l_array),
            ),
        )

        self.scaling_nfw_2d = ParameterScalingSingleMeasurement(
            param_arrays, param_scaling_array
        )

    def test_param_scaling(self):
        scaling = self.scaling.j_scaling(param_array=[1])
        assert scaling == np.array([2])

        scaling = self.scaling.j_scaling(param_array=[])
        assert scaling == 1

        scaling = self.scaling_2d.j_scaling(param_array=[1, 2])
        assert scaling == 2

        scaling = self.scaling_nfw.j_scaling(param_array=[1, 2.9, 0.5])
        assert scaling == 1 * 2.9 * 0.5

        scaling = self.scaling_nfw_2d.j_scaling(param_array=[1, 2, 2.9, 0.5])
        assert scaling == 1 * 2 * 2.9 * 0.5


class TestParameterScalingIFU(object):
    def setup_method(self):
        ani_param_array = np.linspace(start=0, stop=1, num=10)
        param_scaling_array = ani_param_array * 2
        self.scaling = KinScaling(
            j_kin_scaling_param_axes=ani_param_array, j_kin_scaling_grid_list=[param_scaling_array],
            j_kin_scaling_param_name_list=["a"]
        )

        ani_param_array = [
            np.linspace(start=0, stop=1, num=10),
            np.linspace(start=1, stop=2, num=5),
        ]
        param_scaling_array = np.outer(ani_param_array[0], ani_param_array[1])
        self.scaling_2d = KinScaling(j_kin_scaling_param_axes=ani_param_array,
                                     j_kin_scaling_grid_list=[param_scaling_array],
                                     j_kin_scaling_param_name_list=["a", "b"]
                                     )

        ani_param_array = np.linspace(start=0, stop=1, num=10)
        gamma_in_array = np.linspace(start=0.1, stop=2.9, num=5)
        log_m2l_array = np.linspace(start=0.1, stop=1, num=10)

        param_arrays = [ani_param_array, gamma_in_array, log_m2l_array]
        param_scaling_array = np.multiply.outer(
            ani_param_array,
            np.outer(gamma_in_array, log_m2l_array),
        )
        self.scaling_nfw = KinScaling(j_kin_scaling_param_axes=param_arrays,
                                     j_kin_scaling_grid_list=[param_scaling_array],
                                     j_kin_scaling_param_name_list=["a", "b", "c"]
                                     )


        gom_param_array = [
            np.linspace(start=0, stop=1, num=10),
            np.linspace(start=1, stop=2, num=5),
        ]
        param_arrays = [
            gom_param_array[0],
            gom_param_array[1],
            gamma_in_array,
            log_m2l_array,
        ]
        param_scaling_array = np.multiply.outer(
            gom_param_array[0],
            np.multiply.outer(
                gom_param_array[1],
                np.outer(gamma_in_array, log_m2l_array),
            ),
        )
        self.scaling_nfw_2d = KinScaling(j_kin_scaling_param_axes=param_arrays,
                                     j_kin_scaling_grid_list=[param_scaling_array],
                                     j_kin_scaling_param_name_list=["a", "b", "c", "d"]
                                     )


        param_arrays = [ani_param_array, gamma_in_array]
        param_scaling_array = np.multiply.outer(
            ani_param_array,
            gamma_in_array,
        )
        self.scaling_nfw_no_m2l = KinScaling(j_kin_scaling_param_axes=param_arrays,
                                     j_kin_scaling_grid_list=[param_scaling_array],
                                     j_kin_scaling_param_name_list=["a", "b"]
                                     )

        gom_param_array = [
            np.linspace(start=0, stop=1, num=10),
            np.linspace(start=1, stop=2, num=5),
        ]
        param_arrays = [
            gom_param_array[0],
            gom_param_array[1],
            gamma_in_array,
        ]
        param_scaling_array = np.multiply.outer(
            gom_param_array[0],
            np.multiply.outer(
                gom_param_array[1],
                gamma_in_array,
            ),
        )
        self.scaling_nfw_2d_no_m2l = KinScaling(j_kin_scaling_param_axes=param_arrays,
                                     j_kin_scaling_grid_list=[param_scaling_array],
                                     j_kin_scaling_param_name_list=["a", "b", "c"]
                                     )

    def test_kin_scaling(self):

        scaling = self.scaling.kin_scaling(kwargs_param=None)
        assert scaling[0] == 1

        scaling = self.scaling.kin_scaling(kwargs_param={"a":1})
        assert scaling[0] == 2

        kwargs_param = {"a": 1, "b": 2}
        scaling = self.scaling_2d.kin_scaling(kwargs_param=kwargs_param)
        assert scaling[0] == 2

        kwargs_param = {"a": 1, "b": 2.9, "c": 0.5}
        scaling = self.scaling_nfw.kin_scaling(kwargs_param=kwargs_param)
        assert scaling[0] == 1 * 2.9 * 0.5

        kwargs_param = {"a": 1, "b": 2., "c": 2.9, "d": 0.5}
        scaling = self.scaling_nfw_2d.kin_scaling(kwargs_param=kwargs_param)
        assert scaling[0] == 1 * 2 * 2.9 * 0.5

        kwargs_param = {"a": 1, "b": 2.9}
        scaling = self.scaling_nfw_no_m2l.kin_scaling(kwargs_param=kwargs_param)
        assert scaling[0] == 1 * 2.9

        kwargs_param = {"a": 1, "b": 2, "c": 2.9}
        scaling = self.scaling_nfw_2d_no_m2l.kin_scaling(kwargs_param=kwargs_param)
        assert scaling[0] == 1 * 2 * 2.9


if __name__ == "__main__":
    pytest.main()
