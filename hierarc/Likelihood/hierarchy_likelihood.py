from hierarc.Likelihood.transformed_cosmography import TransformedCosmography
from hierarc.Likelihood.lens_likelihood import LensLikelihoodBase
import numpy as np


class LensLikelihood(TransformedCosmography, LensLikelihoodBase):
    """
    master class containing the likelihood definitions of different analysis
    """
    def __init__(self, z_lens, z_source, name='name', likelihood_type='TDKin', ani_param_array=None,
                 num_distribution_draws=20, **kwargs_likelihood):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param name: string (optional) to name the specific lens
        :param likelihood_type: string to specify the likelihood type
        :param ani_param_array: array of anisotropy parameter values for which the kinematics are predicted
        :param ani_scaling_array: velocity dispersion sigma**2 scaling of anisotropy parameter relative to default prediction
        :param num_distribution_draws: int, number of distribution draws from the likelihood that are being averaged over
        :param kwargs_likelihood: keyword arguments specifying the likelihood function,
        see individual classes for their use
        """
        TransformedCosmography.__init__(self, z_lens=z_lens, z_source=z_source)
        LensLikelihoodBase.__init__(self, z_lens=z_lens, z_source=z_source, likelihood_type=likelihood_type, name=name,
                                    ani_param_array=ani_param_array, **kwargs_likelihood)
        self._num_distribution_draws = num_distribution_draws
        if ani_param_array is not None:
            self._ani_param_min = np.min(ani_param_array)
            self._ani_param_max = np.max(ani_param_array)

    def lens_log_likelihood(self, cosmo, **kwargs):
        """

        :param cosmo: astropy.cosmology instance
        :param kwargs: keyword arguments containing the hyper parameters
        :return: log likelihood of the data given the model
        """

        # here we compute the unperturbed angular diameter distances of the lens system given the cosmology
        # Note: Distances are in physical units of Mpc. Make sure the posteriors to evaluate this likelihood is in the
        # same units
        dd = cosmo.angular_diameter_distance(z=self._z_lens).value
        ds = cosmo.angular_diameter_distance(z=self._z_source).value
        dds = cosmo.angular_diameter_distance_z1z2(z1=self._z_lens, z2=self._z_source).value
        ddt = (1. + self._z_lens) * dd * ds / dds

        # here we effectively change the posteriors of the lens, but rather than changing the instance of the KDE we
        # displace the predicted angular diameter distances in the opposite direction
        return self.hyper_param_likelihood(ddt, dd, **kwargs)

    def hyper_param_likelihood(self, ddt, dd, **kwargs):
        """

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :param kwargs: keyword arguments containing the hyper parameters
        :return: log likelihood given the single lens analysis for the given hyper parameter
        """
        gamma_ppn = kwargs.get('gamma_ppn', 1)  # post-Newtonian parameter (optional)
        lambda_mst = kwargs.get('lambda_mst', 1)  # overall mass-sheet transform parameter mean
        lambda_mst_sigma = kwargs.get('lambda_mst_sigma', 0)  # scatter in MST
        kappa_ext = kwargs.get('kappa_ext', 0)  # overall external convergence mean
        kappa_ext_sigma = kwargs.get('kappa_ext_sigma', 0)
        aniso_param = kwargs.get('aniso_param', None)  # stellar anisotropy parameter mean
        aniso_param_sigma =kwargs.get('aniso_param_sigma', 0)

        if kappa_ext_sigma == 0 and lambda_mst_sigma == 0 and aniso_param_sigma == 0:  # sharp distributions
            ddt_, dd_ = self._displace_prediction(ddt, dd, gamma_ppn=gamma_ppn, lambda_mst=lambda_mst,
                                                  kappa_ext=kappa_ext)
            lnlog = self._lens_type.log_likelihood(ddt_, dd_, a_ani=aniso_param)
            return lnlog
        else:
            likelihood = 0
            for i in range(self._num_distribution_draws):
                lambda_mst_draw = np.random.normal(lambda_mst, lambda_mst_sigma)
                kappa_ext_draw = np.random.normal(kappa_ext, kappa_ext_sigma)
                if aniso_param is not None:
                    aniso_param_draw = self.draw_anisotropy(aniso_param, aniso_param_sigma)
                else:
                    aniso_param_draw = aniso_param
                ddt_, dd_ = self._displace_prediction(ddt, dd, gamma_ppn=gamma_ppn,
                                                      lambda_mst=lambda_mst_draw,
                                                      kappa_ext=kappa_ext_draw)
                logl = self._lens_type.log_likelihood(ddt_, dd_, a_ani=aniso_param_draw)
                exp_logl = np.exp(logl)
                if np.isfinite(exp_logl):
                    likelihood += exp_logl
            return np.log(likelihood)

    def draw_anisotropy(self, aniso_param, aniso_param_sigma):
        """
        draw Gaussian distribution and re-sample if outside bounds

        :param aniso_param: mean of the distribution
        :param aniso_param_sigma: std of the distribution
        :return: random draw from the distribution
        """
        if aniso_param < self._ani_param_min or aniso_param > self._ani_param_max:
            raise ValueError('anisotropy parameter is out of bounds of the interpolated range!')
        draw = np.random.normal(aniso_param, aniso_param_sigma)
        if draw < self._ani_param_min or draw > self._ani_param_max:
            draw = self.draw_anisotropy(aniso_param, aniso_param_sigma)
        return draw
