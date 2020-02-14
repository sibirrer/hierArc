__author__ = 'sibirrer'

from lenstronomy.Cosmo.kde_likelihood import KDELikelihood
from hierarc.Util import likelihood_util


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
        if likelihood_type == 'TDKinKDE':
            self._lens_type = TDKinLikelihoodKDE(z_lens, z_source, **kwargs_likelihood)
        elif likelihood_type == 'KinGaussian':
            self._lens_type = KinLikelihoodGaussian(z_lens, z_source, **kwargs_likelihood)
        elif likelihood_type == 'TDKinSkewLogNorm':
            self._lens_type = TDKinLikelihoodSklogn(z_lens, z_source, **kwargs_likelihood)
        elif likelihood_type == 'TDSkewLogNorm':
            self._lens_type = TDLikelihoodSklogn(z_lens, z_source, **kwargs_likelihood)
        else:
            raise ValueError('likelihood_type %s not supported!' % likelihood_type)


class TDKinLikelihoodKDE(object):
    """
    class for evaluating the 2-d posterior of Ddt vs Dd coming from a lens with time delays and kinematics measurement
    """
    def __init__(self, z_lens, z_source, D_d_sample, D_delta_t_sample, kde_type='scipy_gaussian', bandwidth=1):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param D_d_sample: angular diameter to the lens posteriors (in physical Mpc)
        :param D_delta_t_sample: time-delay distance posteriors (in physical Mpc)
        :param kde_type: kernel density estimator type (see KDELikelihood class)
        :param bandwidth: width of kernel (in same units as the angular diameter quantities)
        """
        self._kde_likelihood = KDELikelihood(D_d_sample, D_delta_t_sample, kde_type=kde_type, bandwidth=bandwidth)

    def log_likelihood(self, ddt, dd):
        """

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :return: log likelihood given the single lens analysis
        """
        return self._kde_likelihood.logLikelihood(dd, ddt)[0]


class TDKinLikelihoodSklogn(object):
    """
    class for evaluating the 2-d posterior of Ddt vs Dd coming from a lens with time delays and kinematics measurement
    with a provided skewed log normal function for both Ddt and Dd
    The two distributions are asssumed independant and can be combined.
    """
    def __init__(self, z_lens, z_source, mu_ddt, lam_ddt, sigma_ddt, mu_dd, lam_dd, sigma_dd, explim=100):
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

    def log_likelihood(self, ddt, dd):
        """

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :return: log likelihood given the single lens analysis
        """
        logl_ddt = likelihood_util.sklogn_likelihood(ddt, self._mu_ddt, self._lam_ddt, self._sigma_ddt, explim=self._explim)
        logl_dd = likelihood_util.sklogn_likelihood(dd, self._mu_dd, self._lam_dd, self._sigma_dd, explim=self._explim)
        return logl_ddt + logl_dd


class TDLikelihoodSklogn(object):
    """
    class for evaluating Ddt from a lens with time delays without kinematics measurement
    with a provided skewed log normal function for Ddt.
    """

    def __init__(self, z_lens, z_source, mu_ddt, lam_ddt, sigma_ddt, explim=100):
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

    def log_likelihood(self, ddt, dd):
        """

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :return: log likelihood given the single lens analysis
        """
        return likelihood_util.sklogn_likelihood(ddt, self._mu_ddt, self._lam_ddt, self._sigma_ddt, explim=self._explim)


class KinLikelihoodGaussian(object):
    """
    class to handle cosmographic likelihood coming from modeling lenses with imaging and kinematic data but no time delays.
    Thus Ddt is not constraint but the kinematics can constrain Ds/Dds

    The current version includes a Gaussian in Ds/Dds but can be extended.
    """
    def __init__(self, z_lens, z_source, ds_dds_mean, ds_dds_sigma):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param ds_dds_mean: mean of Ds/Dds distance ratio
        :param ds_dds_sigma: 1-sigma uncertainty in the Ds/Dds distance ratio
        """
        self._z_lens = z_lens
        self._ds_dds_mean = ds_dds_mean
        self._ds_dds_sigma2 = ds_dds_sigma ** 2

    def log_likelihood(self, ddt, dd):
        """
        Note: kinematics + imaging data can constrain Ds/Dds. The input of Ddt, Dd is transformed here to match Ds/Dds

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :return: log likelihood given the single lens analysis
        """
        ds_dds = ddt / dd / (1 + self._z_lens)
        return - (ds_dds - self._ds_dds_mean) ** 2 / self._ds_dds_sigma2 / 2
