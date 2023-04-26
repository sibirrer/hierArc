from hierarc.LensPosterior.kin_constraints import KinConstraints
from hierarc.LensPosterior import power_law_marginalization
from hierarc.Util.distribution_util import weighted_avg_and_std
import numpy as np


class DdtKinConstraints(KinConstraints):
    """
    class for sampling Ds/Dds posteriors from imaging data and kinematic constraints with additional constraints on the
    time-delay distance Ddt
    """

    def __init__(self, z_lens, z_source, ddt_samples, ddt_weights, theta_E, theta_E_error, gamma, gamma_error, r_eff,
                 r_eff_error, sigma_v_measured, kwargs_aperture, kwargs_seeing, kwargs_numerics_galkin,
                 anisotropy_model, sigma_v_error_independent=None, sigma_v_error_covariant=None,
                 sigma_v_error_cov_matrix=None, density_slope_marginalization=True, deconvolve_ddt_gamma=False,
                 kwargs_lens_light=None, lens_light_model_list=['HERNQUIST'], MGE_light=False, kwargs_mge_light=None,
                 hernquist_approx=False, kappa_ext=0, kappa_ext_sigma=0, sampling_number=1000, num_psf_sampling=100,
                 num_kin_sampling=1000, multi_observations=False):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param ddt_samples: numpy array with a sample reflecting the likelihood density in Ddt given imaging data and
         time delays.
        :param ddt_weights: None or length of ddt_samples, weights of individual ddt_samples
        :param sigma_v_measured: numpy array of IFU velocity dispersion of the main deflector in km/s
        :param sigma_v_error_independent: numpy array of 1-sigma uncertainty in velocity dispersion of the IFU
         observation independent of each other
        :param sigma_v_error_covariant: covariant error in the measured kinematics shared among all IFU measurements
        :param sigma_v_error_cov_matrix: error covariance matrix in the sigma_v measurements (km/s)^2
        :type sigma_v_error_cov_matrix: nxn matrix with n the length of the sigma_v_measured array
        :param kwargs_aperture: spectroscopic aperture keyword arguments, see lenstronomy.Galkin.aperture for options
        :param kwargs_seeing: seeing condition of spectroscopic observation, corresponds to kwargs_psf in the GalKin module specified in lenstronomy.GalKin.psf
        :param theta_E: Einstein radius (in arc seconds)
        :param theta_E_error: 1-sigma error on Einstein radius
        :param gamma: power-law slope (2 = isothermal) estimated from imaging data
        :param gamma_error: 1-sigma uncertainty on power-law slope
        :param r_eff: half-light radius of the deflector (arc seconds)
        :param r_eff_error: uncertainty on half-light radius
        :param density_slope_marginalization: if True, marginalization over power-law slope uncertainty,
         otherwise scaling with power-law slope explicitly in output
        :param deconvolve_ddt_gamma: bool, if True, performs a de-convolution of the Ddt posterior for the
         power-law slope de-marginalization (meaning assumes the Ddt posteriors were marginalized over gamma)
        :param kwargs_numerics_galkin: numerical settings for the integrated line-of-sight velocity dispersion
        :param anisotropy_model: type of stellar anisotropy model. See details in MamonLokasAnisotropy() class of lenstronomy.GalKin.anisotropy
        :param kwargs_lens_light: keyword argument list of lens light model (optional)
        :param kwargs_mge_light: keyword arguments that go into the MGE decomposition routine
        :param hernquist_approx: bool, if True, uses the Hernquist approximation for the light profile
        :param kappa_ext: mean of the external convergence from which the ddt constraints are coming from
        :param kappa_ext_sigma: 1-sigma distribution uncertainty from which the ddt constraints are coming from
        :param multi_observations: bool, if True, interprets kwargs_aperture and kwargs_seeing as lists of multiple
         observations
        """
        self._ddt_sample, self._ddt_weights = ddt_samples, ddt_weights
        self._kappa_ext_mean, self._kappa_ext_sigma = kappa_ext, kappa_ext_sigma
        self._deconvolve_ddt_gamma = deconvolve_ddt_gamma
        if not density_slope_marginalization and self._deconvolve_ddt_gamma:
            self._likelihood_type = 'DdtGaussKin'  # here we (need) to assume a gaussian distribution to do the de-convolution
        else:
            self._likelihood_type = 'DdtHistKin'
        super(DdtKinConstraints, self).__init__(z_lens=z_lens, z_source=z_source, theta_E=theta_E,
                                                theta_E_error=theta_E_error, gamma=gamma, gamma_error=gamma_error,
                                                r_eff=r_eff, r_eff_error=r_eff_error, sigma_v_measured=sigma_v_measured,
                                                kwargs_aperture=kwargs_aperture, kwargs_seeing=kwargs_seeing,
                                                kwargs_numerics_galkin=kwargs_numerics_galkin,
                                                anisotropy_model=anisotropy_model,
                                                sigma_v_error_independent=sigma_v_error_independent,
                                                sigma_v_error_covariant=sigma_v_error_covariant,
                                                sigma_v_error_cov_matrix=sigma_v_error_cov_matrix,
                                                kwargs_lens_light=kwargs_lens_light,
                                                lens_light_model_list=lens_light_model_list, MGE_light=MGE_light,
                                                kwargs_mge_light=kwargs_mge_light, hernquist_approx=hernquist_approx,
                                                sampling_number=sampling_number, num_psf_sampling=num_psf_sampling,
                                                num_kin_sampling=num_kin_sampling,
                                                multi_observations=multi_observations,
                                                density_slope_marginalization=density_slope_marginalization)

    def _kwargs_ddt(self):
        """

        :return: dictionary of Ddt likelihood settings
        """
        if self._likelihood_type == 'DdtHistKin':
            kwargs_ddt = {'ddt_samples': self._ddt_sample, 'ddt_weights': self._ddt_weights}
        elif self._likelihood_type == 'DdtGaussKin':
            mean, sigma = weighted_avg_and_std(values=self._ddt_sample, weights=self._ddt_weights)
            if self._deconvolve_ddt_gamma:
                sigma_ddt_gamma = power_law_marginalization.ddt_uncertainty(self._gamma, self._gamma_error)
                sigma_ddt_gamma *= mean
                if sigma_ddt_gamma > sigma:
                    sigma = 0.00001
                else:
                    sigma = np.sqrt(sigma**2 - sigma_ddt_gamma**2)
            kwargs_ddt = {'ddt_mean': mean, 'ddt_sigma': sigma}
        else:
            raise ValueError('likelihood type %s not supported.' % self._likelihood_type)
        return kwargs_ddt

    def hierarchy_configuration(self, num_sample_model=20):
        """
        routine to configure the likelihood to be used in the hierarchical sampling. In particular, a default
        configuration is set to compute the Gaussian approximation of Ds/Dds by sampling the posterior and the estimate
        of the variance of the sample. The anisotropy scaling is then performed. Different anisotropy models are
        supported.

        :param num_sample_model: number of samples drawn from the lens and light model posterior to compute the dimensionless
         kinematic component J()
        :return: keyword arguments
        """
        j_model_list, error_cov_j_sqrt = self.model_marginalization(num_sample_model)
        scaling_array_list = self.j_kin_scaling()
        error_cov_measurement = self.error_cov_measurement
        # configuration keyword arguments for the hierarchical sampling
        kwargs_likelihood = {'z_lens': self._z_lens, 'z_source': self._z_source,
                             'likelihood_type': self._likelihood_type,
                             'sigma_v_measurement': self._sigma_v_measured, 'anisotropy_model': self._anisotropy_model,
                             'j_model': j_model_list,  'error_cov_measurement': error_cov_measurement,
                             'error_cov_j_sqrt': error_cov_j_sqrt,
                             'j_kin_param_array': self.scaling_param_array,
                             'j_kin_scaling_array_list': scaling_array_list,
                             'gamma_pl': self._gamma, 'gamma_pl_error': self._gamma_error,
                             'gamma_pl_scaling': self._density_slope_scaling
                             }
        kwargs_ddt = self._kwargs_ddt()
        kwargs_likelihood.update(kwargs_ddt)
        return kwargs_likelihood
