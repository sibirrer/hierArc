
class DdtGaussianLikelihood(object):
    """
    class to handle cosmographic likelihood coming from modeling lenses with imaging and kinematic data but no time delays.
    Thus Ddt is not constrained but the kinematics can constrain Ds/Dds

    The current version includes a Gaussian in Ds/Dds but can be extended.
    """
    def __init__(self, z_lens, z_source, ddt_mean, ddt_sigma):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param ddt_mean: mean of Ddt distance
        :param ddt_sigma: 1-sigma uncertainty in the Ddt distance
        """
        self._z_lens = z_lens
        self._ddt_mean = ddt_mean
        self._ddt_sigma = ddt_sigma
        self._ddt_sigma2 = ddt_sigma ** 2
        self.num_data = 1

    def log_likelihood(self, ddt, dd=None):
        """

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :return: log likelihood given the single lens analysis
        """
        return - (ddt - self._ddt_mean) ** 2 / self._ddt_sigma2 / 2

    def ddt_measurement(self):
        """

        :return: mean, 1-sigma of the ddt inference/model measurement
        """
        return self._ddt_mean, self._ddt_sigma
