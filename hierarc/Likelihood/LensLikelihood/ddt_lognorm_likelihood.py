import numpy as np


class DdtLogNormLikelihood(object):
    """
    The cosmographic likelihood coming from modeling lenses with imaging and kinematic data but no time delays, where the form of the likelihood is a lognormal distribution.
    Thus Ddt is not constrained but the kinematics can constrain Ds/Dds

    The current version includes a Gaussian in Ds/Dds but can be extended.
    """
    def __init__(self, z_lens, z_source, ddt_mu, ddt_sigma):
        """
        :param z_lens: lens redshift
        :param z_source: source redshift
        :param ddt_mu: mean of log(Ddt distance)
        :param ddt_sigma: 1-sigma uncertainty in the log(Ddt distance)
        """
        self._z_lens = z_lens
        self._ddt_mu = ddt_mu
        self._ddt_sigma2 = ddt_sigma ** 2
        self.num_data = 1

    def log_likelihood(self, ddt, dd=None):
        """
        Note: kinematics + imaging data can constrain Ds/Dds. The input of Ddt, Dd is transformed here to match Ds/Dds

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :return: log likelihood given the single lens analysis
        """
        return -0.5*(np.log(ddt) - self._ddt_mu)**2/self._ddt_sigma2 - np.log(ddt) - 0.5*np.log(self._ddt_sigma2)
