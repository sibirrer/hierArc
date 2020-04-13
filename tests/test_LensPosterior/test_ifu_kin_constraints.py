from hierarc.LensPosterior.ifu_kin_constraints import IFUKin


class TestIFUKinPosterior(object):

    def setup(self):
        pass

    def test_likelihoodconfiguration(self):
        anisotropy_model = 'OM'
        kwargs_aperture = {'aperture_type': 'shell', 'r_in': 0, 'r_out': 3 / 2., 'center_ra': 0.0, 'center_dec': 0}
        kwargs_seeing = {'psf_type': 'GAUSSIAN', 'fwhm': 1.4}

        # numerical settings (not needed if power-law profiles with Hernquist light distribution is computed)
        kwargs_numerics_galkin = {'interpol_grid_num': 1000,  # numerical interpolation, should converge -> infinity
                                  'log_integration': True,
                                  # log or linear interpolation of surface brightness and mass models
                                  'max_integrate': 100,
                                  'min_integrate': 0.001}  # lower/upper bound of numerical integrals

        # compute kinematics


        # compute likelihood
        ifu_kin = IFUKin(z_lens=0.2, z_source=1.5, theta_E=1, theta_E_error=0.01, gamma=2, gamma_error=0.02,
                         r_eff=1, r_eff_error=0.05, sigma_v=[250],
                 sigma_v_error_independent=[10], sigma_v_error_covariant=0, kwargs_aperture=kwargs_aperture,
                         kwargs_seeing=kwargs_seeing, kwargs_numerics_galkin=kwargs_numerics_galkin,
                         anisotropy_model=anisotropy_model,
                 kwargs_lens_light=None, lens_light_model_list=['HERNQUIST'], MGE_light=False, kwargs_mge_light=None,
                 hernquist_approx=True, sampling_number=1000, num_psf_sampling=100, num_kin_sampling=1000,
                 multi_observations=False)
        kwargs_posterior = ifu_kin.hierarchy_configuration(num_sample_model=5)

        # test likelihood
        assert kwargs_posterior['j_model'] > 0
