from hierarc.Likelihood.LensLikelihood.kin_likelihood import KinLikelihood
from hierarc.Likelihood.LensLikelihood.ddt_hist_likelihood import DdtHistKDELikelihood


class DdtHistKinLikelihood(object):

    """
    class for joint kinematics and time delay likelihood assuming that they are independent
    Uses KinLikelihood and DdtHistLikelihood combined
    """
    def __init__(self, z_lens, z_source, ddt_samples, ddt_weights, sigma_v_measurement, j_model, error_cov_measurement,
                 error_cov_j_sqrt, kde_kernel=None, bandwidth=20, nbins_hist=200, sigma_sys_error_include=False):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param ddt_samples: numpy array of Ddt values
        :param ddt_weights: optional weights for the samples in Ddt
        :param kde_kernel: string of KDE kernel type
        :param bandwidth: bandwith of kernel
        :param nbins_hist: number of bins in the histogram
        :param sigma_v_measurement: numpy array, velocity dispersion measured
        :param j_model: numpy array of the predicted dimensionless dispersion on the IFU's
        :param error_cov_measurement: covariance matrix of the measured velocity dispersions in the IFU's
        :param error_cov_j_sqrt: covariance matrix of sqrt(J) of the model predicted dimensionless dispersion on the IFU's
        :param sigma_sys_error_include: bool, if True will include a systematic error in the velocity dispersion
         measurement (if sampled from), otherwise this sampled value is ignored.
        """
        self._tdLikelihood = DdtHistKDELikelihood(z_lens, z_source, ddt_samples, ddt_weights=ddt_weights,
                                                  kde_kernel=kde_kernel, bandwidth=bandwidth, nbins_hist=nbins_hist)
        self._kinlikelihood = KinLikelihood(z_lens, z_source, sigma_v_measurement, j_model, error_cov_measurement,
                                            error_cov_j_sqrt, sigma_sys_error_include=sigma_sys_error_include)
        self.num_data = self._tdLikelihood.num_data + self._kinlikelihood.num_data

    def log_likelihood(self, ddt, dd, aniso_scaling=None, sigma_v_sys_error=None):
        """

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :param aniso_scaling: numpy array of anisotropy scaling on prediction of Ds/Dds
        :return: log likelihood given the single lens analysis
        """
        lnlikelihood = self._tdLikelihood.log_likelihood(ddt) + self._kinlikelihood.log_likelihood(ddt, dd, aniso_scaling, sigma_v_sys_error=sigma_v_sys_error)
        return lnlikelihood
