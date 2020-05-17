
class DsDdsGaussianLikelihood(object):
    """
    class to handle cosmographic likelihood coming from modeling lenses with imaging and kinematic data but no time delays.
    Thus Ddt is not constrained but the kinematics can constrain Ds/Dds. The likelihood in Ds/Dds is assumed Gaussian.
    Attention: Gaussian uncertainties in velocity dispersion do not translate into Gaussian uncertainties in Ds/Dds.
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
        self.num_data = 1

    def log_likelihood(self, ddt, dd, aniso_scaling=None):
        """
        Note: kinematics + imaging data can constrain Ds/Dds. The input of Ddt, Dd is transformed here to match Ds/Dds

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :param aniso_scaling: array of size of the velocity dispersion measurement or None, scaling of the predicted
         dimensionless quantity J (proportional to sigma_v^2) of the anisotropy model in the sampling relative to the
         anisotropy model used to derive the prediction and covariance matrix in the init of this class.
        :return: log likelihood given the single lens analysis
        """
        ds_dds = ddt / dd / (1 + self._z_lens)
        if aniso_scaling is not None:
            scaling = aniso_scaling[0]
        else:
            scaling = 1
        ds_dds_ = ds_dds / scaling
        return - (ds_dds_ - self._ds_dds_mean) ** 2 / self._ds_dds_sigma2 / 2
