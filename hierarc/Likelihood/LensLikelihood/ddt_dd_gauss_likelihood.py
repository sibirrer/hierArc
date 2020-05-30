from hierarc.Likelihood.LensLikelihood.ddt_gauss_likelihood import DdtGaussianLikelihood


class DdtDdGaussian(object):
    """
    class for joint kinematics and time delay likelihood assuming independent Gaussian likelihoods in Ddt and Dd.
    Attention: Gaussian errors in the velocity dispersion do not translate into Gaussian uncertainties in Dd.
    """
    def __init__(self, z_lens, z_source, ddt_mean, ddt_sigma, dd_mean, dd_sigma):
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
        self._tdLikelihood = DdtGaussianLikelihood(z_lens, z_source, ddt_mean=ddt_mean, ddt_sigma=ddt_sigma)
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
        lnlikelihood = self._tdLikelihood.log_likelihood(ddt, dd_) - (dd_ - self._dd_mean) ** 2 / self._dd_sigma2 / 2
        return lnlikelihood

    def ddt_measurement(self):
        """

        :return: mean, 1-sigma of the ddt inference/model measurement
        """
        return self._tdLikelihood.ddt_measurement()
