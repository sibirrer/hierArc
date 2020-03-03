__author__ = 'sibirrer'

from lenstronomy.Cosmo.kde_likelihood import KDELikelihood
from hierarc.Util import likelihood_util
from hierarc.Likelihood.anisotropy_scaling import AnisotropyScaling
from hierarc.Likelihood.LensLikelihood.ifu_kin_likelihood import IFUKinCov
import numpy as np
from scipy import interpolate


class LensLikelihoodBase(object):
    """
    master class containing the likelihood definitions of different analysis
    """
    def __init__(self, z_lens, z_source, likelihood_type='TDKin', name='name', **kwargs_likelihood):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param name: string (optional) to name the specific lens
        :param likelihood_type: string to specify the likelihood type
        :param kwargs_likelihood: keyword arguments specifying the likelihood function,
        see individual classes for their use
        """
        self._name = name
        self._z_lens = z_lens
        self._z_source = z_source
        if likelihood_type in ['TDKinKDE']:
            self._lens_type = TDKinLikelihoodKDE(z_lens, z_source, **kwargs_likelihood)
        elif likelihood_type == 'TDKinGaussian':
            self._lens_type = TDKinGaussian(z_lens, z_source, **kwargs_likelihood)
        elif likelihood_type in ['KinGaussian']:
            self._lens_type = KinLikelihoodGaussian(z_lens, z_source, **kwargs_likelihood)
        elif likelihood_type == 'TDKinSkewLogNorm':
            self._lens_type = TDKinLikelihoodSklogn(z_lens, z_source, **kwargs_likelihood)
        elif likelihood_type == 'TDSkewLogNorm':
            self._lens_type = TDLikelihoodSklogn(z_lens, z_source, **kwargs_likelihood)
        elif likelihood_type == 'IFUKinCov':
            self._lens_type = IFUKinCov(z_lens, z_source, **kwargs_likelihood)
        else:
            raise ValueError('likelihood_type %s not supported!' % likelihood_type)


class TDKinGaussian(AnisotropyScaling):
    """
    class for joint kinematics and time delay likelihood assuming that they are independent and Gaussian
    """
    def __init__(self, z_lens, z_source, ddt_mean, ddt_sigma, dd_mean, dd_sigma, ani_param_array=None,
                 ani_scaling_array=None):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param ddt_mean: mean of Ddt distance
        :param ddt_sigma: 1-sigma uncertainty in the Ddt distance
        :param dd_mean: mean of Dd distance ratio
        :param dd_sigma: 1-sigma uncertainty in the Dd distance
        """
        self._dd_mean = dd_mean
        self._dd_sigma2 = dd_sigma ** 2
        self._tdLikelihood = TDLikelihoodGaussian(z_lens, z_source, ddt_mean=ddt_mean, ddt_sigma=ddt_sigma)
        AnisotropyScaling.__init__(self, ani_param_array=ani_param_array, ani_scaling_array=ani_scaling_array)

    def log_likelihood(self, ddt, dd, a_ani=None):
        """

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :return: log likelihood given the single lens analysis
        """
        dd_ = dd * self.ani_scaling(a_ani)
        return self._tdLikelihood.log_likelihood(ddt, dd_) - (dd_ - self._dd_mean) ** 2 / self._dd_sigma2 / 2


class TDKinLikelihoodKDE(AnisotropyScaling):
    """
    class for evaluating the 2-d posterior of Ddt vs Dd coming from a lens with time delays and kinematics measurement
    """
    def __init__(self, z_lens, z_source, D_d_sample, D_delta_t_sample, kde_type='scipy_gaussian', bandwidth=1,
                 interpol=False, num_interp_grid=100, ani_param_array=None, ani_scaling_array=None):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param D_d_sample: angular diameter to the lens posteriors (in physical Mpc)
        :param D_delta_t_sample: time-delay distance posteriors (in physical Mpc)
        :param kde_type: kernel density estimator type (see KDELikelihood class)
        :param bandwidth: width of kernel (in same units as the angular diameter quantities)
        :param interpol: bool, if True pre-computes an interpolation likelihood in 2d on a grid
        :param num_interp_grid: int, number of interpolations per axis
        """
        self._kde_likelihood = KDELikelihood(D_d_sample, D_delta_t_sample, kde_type=kde_type, bandwidth=bandwidth)

        if interpol is True:
            dd_grid = np.linspace(start=max(np.min(D_d_sample), 0), stop=min(np.max(D_d_sample), 10000), num=num_interp_grid)
            ddt_grid = np.linspace(np.min(D_delta_t_sample), np.max(D_delta_t_sample), num=num_interp_grid)
            z = np.zeros((num_interp_grid, num_interp_grid))
            for i, dd in enumerate(dd_grid):
                for j, ddt in enumerate(ddt_grid):
                    z[j, i] = self._kde_likelihood.logLikelihood(dd, ddt)[0]
            self._interp_log_likelihood = interpolate.interp2d(dd_grid, ddt_grid, z, kind='cubic')
        self._interpol = interpol
        AnisotropyScaling.__init__(self, ani_param_array=ani_param_array, ani_scaling_array=ani_scaling_array)

    def log_likelihood(self, ddt, dd, a_ani=None):
        """

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :return: log likelihood given the single lens analysis
        """
        dd_ = dd * self.ani_scaling(a_ani)
        if self._interpol is True:
            return self._interp_log_likelihood(dd_, ddt)[0]
        return self._kde_likelihood.logLikelihood(dd_, ddt)[0]


class TDKinLikelihoodSklogn(AnisotropyScaling):
    """
    class for evaluating the 2-d posterior of Ddt vs Dd coming from a lens with time delays and kinematics measurement
    with a provided skewed log normal function for both Ddt and Dd
    The two distributions are asssumed independant and can be combined.
    """
    def __init__(self, z_lens, z_source, mu_ddt, lam_ddt, sigma_ddt, mu_dd, lam_dd, sigma_dd, explim=100,
                 ani_param_array=None, ani_scaling_array=None):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param mu_ddt:
        :param lam_ddt:
        :param sigma_ddt:
        :param mu_dd:
        :param lam_dd:
        :param sigma_dd:
        :param explim: exponent limit for evaluating the likelihood
        """
        self._mu_ddt, self._lam_ddt, self._sigma_ddt = mu_ddt, lam_ddt, sigma_ddt
        self._mu_dd, self._lam_dd, self._sigma_dd = mu_dd, lam_dd, sigma_dd
        self._explim = explim
        AnisotropyScaling.__init__(self, ani_param_array=ani_param_array, ani_scaling_array=ani_scaling_array)

    def log_likelihood(self, ddt, dd, a_ani=None):
        """

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :return: log likelihood given the single lens analysis
        """
        dd_ = dd * self.ani_scaling(a_ani)
        logl_ddt = likelihood_util.sklogn_likelihood(ddt, self._mu_ddt, self._lam_ddt, self._sigma_ddt, explim=self._explim)
        logl_dd = likelihood_util.sklogn_likelihood(dd_, self._mu_dd, self._lam_dd, self._sigma_dd, explim=self._explim)
        return logl_ddt + logl_dd


class TDLikelihoodSklogn(object):
    """
    class for evaluating Ddt from a lens with time delays without kinematics measurement
    with a provided skewed log normal function for Ddt.
    """

    def __init__(self, z_lens, z_source, mu_ddt, lam_ddt, sigma_ddt, explim=100, ani_param_array=None,
                 ani_scaling_array=None):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param mu_ddt:
        :param lam_ddt:
        :param sigma_ddt:
        :param explim: exponent limit for evaluating the likelihood
        """
        self._mu_ddt, self._lam_ddt, self._sigma_ddt = mu_ddt, lam_ddt, sigma_ddt
        self._explim = explim

    def log_likelihood(self, ddt, dd=None, a_ani=None):
        """

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :return: log likelihood given the single lens analysis
        """
        return likelihood_util.sklogn_likelihood(ddt, self._mu_ddt, self._lam_ddt, self._sigma_ddt, explim=self._explim)


class TDLikelihoodGaussian(object):
    """
    class to handle cosmographic likelihood coming from modeling lenses with imaging and kinematic data but no time delays.
    Thus Ddt is not constraint but the kinematics can constrain Ds/Dds

    The current version includes a Gaussian in Ds/Dds but can be extended.
    """
    def __init__(self, z_lens, z_source, ddt_mean, ddt_sigma, ani_param_array=None, ani_scaling_array=None):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param ddt_mean: mean of Ddt distance
        :param ddt_sigma: 1-sigma uncertainty in the Ddt distance
        """
        self._z_lens = z_lens
        self._ddt_mean = ddt_mean
        self._ddt_sigma2 = ddt_sigma ** 2

    def log_likelihood(self, ddt, dd=None, a_ani=None):
        """
        Note: kinematics + imaging data can constrain Ds/Dds. The input of Ddt, Dd is transformed here to match Ds/Dds

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :return: log likelihood given the single lens analysis
        """
        return - (ddt - self._ddt_mean) ** 2 / self._ddt_sigma2 / 2


class KinLikelihoodGaussian(AnisotropyScaling):
    """
    class to handle cosmographic likelihood coming from modeling lenses with imaging and kinematic data but no time delays.
    Thus Ddt is not constraint but the kinematics can constrain Ds/Dds

    The current version includes a Gaussian in Ds/Dds but can be extended.
    """
    def __init__(self, z_lens, z_source, ds_dds_mean, ds_dds_sigma, ani_param_array=None, ani_scaling_array=None):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param ds_dds_mean: mean of Ds/Dds distance ratio
        :param ds_dds_sigma: 1-sigma uncertainty in the Ds/Dds distance ratio
        """
        self._z_lens = z_lens
        self._ds_dds_mean = ds_dds_mean
        self._ds_dds_sigma2 = ds_dds_sigma ** 2
        AnisotropyScaling.__init__(self, ani_param_array=ani_param_array, ani_scaling_array=ani_scaling_array)

    def log_likelihood(self, ddt, dd, a_ani=None):
        """
        Note: kinematics + imaging data can constrain Ds/Dds. The input of Ddt, Dd is transformed here to match Ds/Dds

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :return: log likelihood given the single lens analysis
        """
        ds_dds = ddt / dd / (1 + self._z_lens)
        ds_dds_ = ds_dds / self.ani_scaling(a_ani)
        return - (ds_dds_ - self._ds_dds_mean) ** 2 / self._ds_dds_sigma2 / 2
