from lenstronomy.Cosmo.kde_likelihood import KDELikelihood
import numpy as np
from scipy import interpolate


class DdtDdKDELikelihood(object):
    """
    class for evaluating the 2-d posterior of Ddt vs Dd coming from a lens with time delays and kinematics measurement
    """
    def __init__(self, z_lens, z_source, dd_samples, ddt_samples, kde_type='scipy_gaussian', bandwidth=1,
                 interpol=False, num_interp_grid=100):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param dd_samples: angular diameter to the lens posteriors (in physical Mpc)
        :param ddt_samples: time-delay distance posteriors (in physical Mpc)
        :param kde_type: kernel density estimator type (see KDELikelihood class)
        :param bandwidth: width of kernel (in same units as the angular diameter quantities)
        :param interpol: bool, if True pre-computes an interpolation likelihood in 2d on a grid
        :param num_interp_grid: int, number of interpolations per axis
        """
        self._kde_likelihood = KDELikelihood(dd_samples, ddt_samples, kde_type=kde_type, bandwidth=bandwidth)

        if interpol is True:
            dd_grid = np.linspace(start=max(np.min(dd_samples), 0), stop=min(np.max(dd_samples), 10000), num=num_interp_grid)
            ddt_grid = np.linspace(np.min(ddt_samples), np.max(ddt_samples), num=num_interp_grid)
            z = np.zeros((num_interp_grid, num_interp_grid))
            for i, dd in enumerate(dd_grid):
                for j, ddt in enumerate(ddt_grid):
                    z[j, i] = self._kde_likelihood.logLikelihood(dd, ddt)[0]
            self._interp_log_likelihood = interpolate.interp2d(dd_grid, ddt_grid, z, kind='cubic')
        self._interpol = interpol
        self.num_data = 2

    def log_likelihood(self, ddt, dd, aniso_scaling=None):
        """

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :param aniso_scaling: array of size of the velocity dispersion measurement or None, scaling of the predicted
         dimensionless quantity J (proportional to sigma_v^2) of the anisotropy model in the sampling relative to the
         anisotropy model used to derive the prediction and covariance matrix in the init of this class.
        :return: log likelihood given the single lens analysis
        """
        if aniso_scaling is not None:
            dd_ = dd * aniso_scaling[0]
        else:
            dd_ = dd
        if self._interpol is True:
            return self._interp_log_likelihood(dd_, ddt)[0]
        return self._kde_likelihood.logLikelihood(dd_, ddt)[0]
