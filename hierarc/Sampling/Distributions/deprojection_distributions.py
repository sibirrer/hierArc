import numpy as np

_SUPPORTED_DISTRIBUTIONS = ["GAUSSIAN", "GAUSSIAN_SCALED", "NONE"]
_PARAMETERIZATION = ["q_intrinsic"]


class DeprojectionDistribution(object):
    """Class to draw axisymmetric deprojection parameters (inclination/q_intrinsic) from
    hyperparameter distributions."""

    def __init__(
        self,
        deprojection_sampling,
        distribution_function,
        kwargs_deprojection_min,
        kwargs_deprojection_max,
        parameterization="q_intrinsic",
    ):
        """

        :param deprojection_sampling: bool, if True adds a global stellar deprojection parameter that alters the single lens
         kinematic prediction
        :param distribution_function: string, 'NONE', 'GAUSSIAN', 'GAUSSIAN_SCALED', "GAUSSIAN_TAN_RAD"
         description of the distribution function of the deprojection model parameters
        :param kwargs_deprojection_min: dictionary of bounds in the parameterization (from the interpolation)
        :param kwargs_deprojection_max: dictionary of bounds in the parameterization (from the interpolation)
        :param parameterization: model of parameterization (currently for constant deprojection), ["beta" or "TAN_RAD"]
        """
        self._deprojection_sampling = deprojection_sampling
        if distribution_function not in _SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                "Anisotropy distribution function %s not supported. Chose among %s."
                % (distribution_function, _SUPPORTED_DISTRIBUTIONS)
            )
        if parameterization not in _PARAMETERIZATION:
            raise ValueError(
                "Deprojection parameterization %s not supported. Chose among %s."
                % (parameterization, _PARAMETERIZATION)
            )
        self._distribution_function = distribution_function
        self._parametrization = parameterization
        if kwargs_deprojection_min is None:
            kwargs_deprojection_min = {}
        if kwargs_deprojection_max is None:
            kwargs_deprojection_max = {}
        self._kwargs_min = kwargs_deprojection_min
        self._kwargs_max = kwargs_deprojection_max
        self._q_min, self._q_max = self._kwargs_min.get(
            "q_intrinsic", 0.0
        ), self._kwargs_max.get("q_intrinsic", 1.0)

    def draw_deprojection(self, q_intrinsic=None, q_intrinsic_sigma=0):
        """Draw Gaussian distribution and re-sample if outside bounds.

        :param q_intrinsic: mean of the distribution
        :param q_intrinsic_sigma: std of the distribution
        :return: random draw from the distribution
        """
        kwargs_return = {}
        q = q_intrinsic
        if not self._deprojection_sampling:
            if q_intrinsic is not None:
                kwargs_return["q_intrinsic"] = q
            return kwargs_return
        if q <= self._q_min or q > self._q_max:
            raise ValueError(
                "deprojection parameter with %s is out of bounds of the interpolated range [%s, %s]!"
                % (q, self._q_min, self._q_max)
            )
        if self._distribution_function in ["GAUSSIAN", "GAUSSIAN_SCALED"]:
            if self._distribution_function in ["GAUSSIAN"]:
                q_draw = np.random.normal(q_intrinsic, q_intrinsic_sigma)
            elif self._distribution_function in ["GAUSSIAN_SCALED"]:
                q_draw = np.random.normal(q_intrinsic, q_intrinsic_sigma * q_intrinsic)
            if q_draw <= self._q_min or q_draw > self._q_max:
                return self.draw_deprojection(q_intrinsic, q_intrinsic_sigma)
            kwargs_return["q_intrinsic"] = q_draw
        else:
            kwargs_return["q_intrinsic"] = q
        return kwargs_return
