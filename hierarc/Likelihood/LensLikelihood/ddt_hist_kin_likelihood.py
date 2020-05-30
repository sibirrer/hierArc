from hierarc.Likelihood.LensLikelihood.kin_likelihood import KinLikelihood
from hierarc.Likelihood.LensLikelihood.ddt_hist_likelihood import DdtHistKDELikelihood


class DdtHistKinLikelihood(object):

    """
    class for joint kinematics and time delay likelihood assuming that they are independent
    Uses KinLikelihood and DdtHistLikelihood combined
    """
    def __init__(self, z_lens, z_source, ddt_samples, sigma_v_measurement, j_model, error_cov_measurement,
                 error_cov_j_sqrt, ddt_weights=None, kde_kernel='gaussian', bandwidth=20, nbins_hist=200,
                 sigma_sys_error_include=False, normalized=True):
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
        :param normalized: bool, if True, returns the normalized likelihood, if False, separates the constant prefactor
         (in case of a Gaussian 1/(sigma sqrt(2 pi)) ) to compute the reduced chi2 statistics
        """
        self._tdLikelihood = DdtHistKDELikelihood(z_lens, z_source, ddt_samples, ddt_weights=ddt_weights,
                                                  kde_kernel=kde_kernel, bandwidth=bandwidth, nbins_hist=nbins_hist)
        self._kinlikelihood = KinLikelihood(z_lens, z_source, sigma_v_measurement, j_model, error_cov_measurement,
                                            error_cov_j_sqrt, sigma_sys_error_include=sigma_sys_error_include,
                                            normalized=normalized)
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

    def sigma_v_prediction(self, ddt, dd, aniso_scaling=1):
        """
        model prediction mean velocity dispersion vector and model prediction covariance matrix

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :param aniso_scaling: array of size of the velocity dispersion measurement or None, scaling of the predicted
         dimensionless quantity J (proportional to sigma_v^2) of the anisotropy model in the sampling relative to the
         anisotropy model used to derive the prediction and covariance matrix in the init of this class.
        :return: model prediction mean velocity dispersion vector and model prediction covariance matrix
        """
        return self._kinlikelihood.sigma_v_prediction(ddt, dd, aniso_scaling)

    def sigma_v_measurement(self, sigma_v_sys_error=None, sigma_v_sys_offset=None):
        """

        :param sigma_v_sys_error: float (optional) added error on the velocity dispersion measurement in quadrature
        :param sigma_v_sys_offset: float (optional) for a fractional systematic offset in the kinematic measurement
         such that sigma_v = sigma_v_measured * (1 + sigma_v_sys_offset)
        :return: measurement mean (vector), measurement covariance matrix
        """
        return self._kinlikelihood.sigma_v_measurement(sigma_v_sys_error=sigma_v_sys_error,
                                                       sigma_v_sys_offset=sigma_v_sys_offset)

    def ddt_measurement(self):
        """

        :return: mean, 1-sigma of the ddt inference/model measurement
        """
        return self._tdLikelihood.ddt_measurement()
