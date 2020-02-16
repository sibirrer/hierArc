from hierarc.LensPosterior.kin_constraints import DsDdsConstraints
import numpy as np


class DdtKinConstraints(DsDdsConstraints):
    """
    class for sampling Ds/Dds posteriors from imaging data and kinematic constraints with additional constraints on the
    time-delay distance Ddt
    """

    def __init__(self, z_lens, z_source, ddt_mean, ddt_sigma, theta_E, theta_E_error, gamma, gamma_error, r_eff, r_eff_error, sigma_v,
                 sigma_v_error, kwargs_aperture, kwargs_seeing, kwargs_numerics_galkin, anisotropy_model,
                 kwargs_lens_light=None, lens_light_model_list=['HERNQUIST'], MGE_light=False, kwargs_mge_light=None,
                 hernquist_approx=True, kappa_ext=0, kappa_ext_sigma=0):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param ddt_mean: mean of the time-delay distance inference
        :param ddt_sigma: 1-sigma uncertainty in the time-delay distance inference
        :param theta_E: Einstein radius (in arc seconds)
        :param theta_E_error: 1-sigma error on Einstein radius
        :param gamma: power-law slope (2 = isothermal) estimated from imaging data
        :param gamma_error: 1-sigma uncertainty on power-law slope
        :param r_eff: half-light radius of the deflector (arc seconds)
        :param r_eff_error: uncertainty on half-light radius
        :param sigma_v: velocity dispersion of the main deflector in km/s
        :param sigma_v_error: 1-sigma uncertainty in velocity dispersion
        :param kwargs_aperture: spectroscopic aperture keyword arguments, see lenstronomy.Galkin.aperture for options
        :param kwargs_seeing: seeing condition of spectroscopic observation, corresponds to kwargs_psf in the GalKin module specified in lenstronomy.GalKin.psf
        :param kwargs_numerics_galkin: numerical settings for the integrated line-of-sight velocity dispersion
        :param anisotropy_model: type of stellar anisotropy model. See details in MamonLokasAnisotropy() class of lenstronomy.GalKin.anisotropy
        :param kwargs_lens_light: keyword argument list of lens light model (optional)
        :param kwargs_mge_light: keyword arguments that go into the MGE decomposition routine
        :param hernquist_approx: bool, if True, uses the Hernquist approximation for the light profile
        :param kappa_ext: mean of the external convergence from which the ddt constraints are coming from
        :param kappa_ext_sigma: 1-sigma distribution uncertainty from which the ddt constraints are coming from
        """
        self._ddt_mean, self._ddt_sigma = ddt_mean, ddt_sigma
        self._kappa_ext_mean, self._kappa_ext_sigma = kappa_ext, kappa_ext_sigma
        super(DdtKinConstraints, self).__init__(z_lens, z_source, theta_E, theta_E_error, gamma, gamma_error, r_eff,
                                                r_eff_error, sigma_v, sigma_v_error, kwargs_aperture, kwargs_seeing,
                                                kwargs_numerics_galkin, anisotropy_model,
                                                kwargs_lens_light=kwargs_lens_light,
                                                lens_light_model_list=lens_light_model_list, MGE_light=MGE_light,
                                                kwargs_mge_light=kwargs_mge_light, hernquist_approx=hernquist_approx)

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
        ds_dds_mean, ds_dds_sigma, ani_param_array, ani_scaling_array = self.kin_constraints(num_sample_model,
                                                                                             num_kin_measurements)
        ds_dds_mean /= (1 - self._kappa_ext_mean)  # perform the mean correction of the Ds/Dds constraints when the Ddt posteriors are corrected
        # Gaussian error propagation of ddt and ds_dds uncertainties into dd uncertainties
        dd_mean = self._ddt_mean / ds_dds_mean / (1 + self._z_lens)
        dd_sigma2 = self._ddt_sigma**2 * (1 / ds_dds_mean / (1 + self._z_lens))**2 + ds_dds_sigma**2 * (self._ddt_mean / (1 + self._z_lens) / ds_dds_mean**2)
        # subtract the component in the uncertainty in ddt that does not impact the uncertainty on dd
        dd_sigma = np.sqrt(dd_sigma2 - (self._kappa_ext_sigma * dd_mean)**2)
        # configuration keyword arguments for the hierarchical sampling
        kwargs_likelihood = {'z_lens': self._z_lens, 'z_source': self._z_source, 'likelihood_type': 'TDKinGaussian',
                             'ddt_mean': self._ddt_mean, 'ddt_sigma': self._ddt_sigma,
                             'dd_mean': dd_mean,  'dd_sigma': dd_sigma,
                             'ani_param_array': ani_param_array, 'ani_scaling_array': ani_scaling_array}
        return kwargs_likelihood
