import numpy as np


class IFUKin(object):
    """
    class that manages constraints from Integral Field Unit spectral observations.
    """
    def __init__(self):
        pass

    def hierarchy_configuration(self, num_sample_model=20, num_kin_measurements=50):
        """
        routine to configure the likelihood to be used in the hierarchical sampling. In particular, a default
        configuration is set to compute the Gaussian approximation of Ds/Dds by sampling the posterior and the estimate
        of the variance of the sample. The anisotropy scaling is then performed. Different anisotropy models are
        supported.

        :param num_sample_model: number of samples drawn from the lens and light model posterior to compute the dimensionless
        kinematic component J()
        :param num_kin_measurements: number of draws from the velocity dispersion measurements to simple sample the
        posterior in Ds/Dds. The total number of posteriors is num_sample_model x num_kin_measurements
        :return: keyword arguments
        """

        # configuration keyword arguments for the hierarchical sampling
        kwargs_likelihood = {'z_lens': self._z_lens, 'z_source': self._z_source, 'likelihood_type': 'IFUKinCov',
                             'ds_dds_mean': ds_dds_mean,  'ds_dds_sigma': ds_dds_sigma,
                             'ani_param_array': ani_param_array, 'ani_scaling_array_list': ani_scaling_array}
        return kwargs_likelihood
# compute covariance matrix in J_0 calculation in the bins
