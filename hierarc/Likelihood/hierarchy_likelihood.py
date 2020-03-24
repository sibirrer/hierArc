from hierarc.Likelihood.transformed_cosmography import TransformedCosmography
from hierarc.Likelihood.lens_likelihood import LensLikelihoodBase
import numpy as np


class LensLikelihood(TransformedCosmography, LensLikelihoodBase):
    """
    master class containing the likelihood definitions of different analysis
    """
    def __init__(self, z_lens, z_source, name='name', likelihood_type='TDKin', anisotropy_model='NONE', ani_param_array=None,
                 num_distribution_draws=50, kappa_ext_bias=False, **kwargs_likelihood):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param name: string (optional) to name the specific lens
        :param likelihood_type: string to specify the likelihood type
        :param ani_param_array: array of anisotropy parameter values for which the kinematics are predicted
        :param ani_scaling_array: velocity dispersion sigma**2 scaling of anisotropy parameter relative to default prediction
        :param num_distribution_draws: int, number of distribution draws from the likelihood that are being averaged over
        :param kappa_ext_bias: bool, if True incorporates the global external selection function into the likelihood.
        If False, the likelihood needs to incorporate the individual selection function with sufficient accuracy.
        :param kwargs_likelihood: keyword arguments specifying the likelihood function,
        see individual classes for their use
        """
        TransformedCosmography.__init__(self, z_lens=z_lens, z_source=z_source)
        LensLikelihoodBase.__init__(self, z_lens=z_lens, z_source=z_source, likelihood_type=likelihood_type, name=name,
                                    anisotropy_model=anisotropy_model, ani_param_array=ani_param_array, **kwargs_likelihood)
        self._num_distribution_draws = num_distribution_draws
        self._kappa_ext_bias = kappa_ext_bias
        if ani_param_array is not None:
            if isinstance(ani_param_array, list):
                self._dim_scaling = len(ani_param_array)
            else:
                self._dim_scaling = 1
            if self._dim_scaling == 1:
                self._ani_param_min = np.min(ani_param_array)
                self._ani_param_max = np.max(ani_param_array)
            elif self._dim_scaling == 2:
                self._ani_param_min = [min(ani_param_array[0]), min(ani_param_array[1])]
                self._ani_param_max = [max(ani_param_array[0]), max(ani_param_array[1])]
            else:
                raise ValueError('anisotropy scaling with dimension %s not supported.' % self._dim_scaling)

    def lens_log_likelihood(self, cosmo, kwargs_lens=None, kwargs_kin=None):
        """

        :param cosmo: astropy.cosmology instance
        :param kwargs_lens: keywords of the hyper parameters of the lens model
        :param kwargs_kin: keyword arguments of the kinematic model hyper parameters
        :return: log likelihood of the data given the model
        """

        # here we compute the unperturbed angular diameter distances of the lens system given the cosmology
        # Note: Distances are in physical units of Mpc. Make sure the posteriors to evaluate this likelihood is in the
        # same units
        ddt, dd = self.angular_diameter_distances(cosmo)
        # here we effectively change the posteriors of the lens, but rather than changing the instance of the KDE we
        # displace the predicted angular diameter distances in the opposite direction
        return self.hyper_param_likelihood(ddt, dd, kwargs_lens=kwargs_lens, kwargs_kin=kwargs_kin)

    def hyper_param_likelihood(self, ddt, dd, kwargs_lens, kwargs_kin):
        """

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :param kwargs_lens: keywords of the hyper parameters of the lens model
        :param kwargs_kin: keyword arguments of the kinematic model hyper parameters
        :return: log likelihood given the single lens analysis for the given hyper parameter
        """
        #a_ani = kwargs_kin.get('a_ani', None)  # stellar anisotropy parameter mean

        if self.check_dist(kwargs_lens, kwargs_kin):  # sharp distributions
            lambda_mst, kappa_ext, gamma_ppn = self.draw_lens(**kwargs_lens)
            ddt_, dd_ = self.displace_prediction(ddt, dd, gamma_ppn=gamma_ppn, lambda_mst=lambda_mst,
                                                 kappa_ext=kappa_ext)
            aniso_param_array = self.draw_anisotropy(**kwargs_kin)
            lnlog = self._lens_type.log_likelihood(ddt_, dd_, aniso_param_array=aniso_param_array)
            return lnlog
        else:
            likelihood = 0
            for i in range(self._num_distribution_draws):
                lambda_mst_draw, kappa_ext_draw, gamma_ppn = self.draw_lens(**kwargs_lens)
                aniso_param_draw = self.draw_anisotropy(**kwargs_kin)
                ddt_, dd_ = self.displace_prediction(ddt, dd, gamma_ppn=gamma_ppn,
                                                     lambda_mst=lambda_mst_draw,
                                                     kappa_ext=kappa_ext_draw)
                logl = self._lens_type.log_likelihood(ddt_, dd_, aniso_param_array=aniso_param_draw)
                exp_logl = np.exp(logl)
                if np.isfinite(exp_logl):
                    likelihood += exp_logl
            return np.log(likelihood)

    def angular_diameter_distances(self, cosmo):
        """

        :param cosmo: astropy.comsmology instance (or equivalent with interpolation
        :return: ddt, dd in units Mpc
        """
        dd = cosmo.angular_diameter_distance(z=self._z_lens).value
        ds = cosmo.angular_diameter_distance(z=self._z_source).value
        dds = cosmo.angular_diameter_distance_z1z2(z1=self._z_lens, z2=self._z_source).value
        ddt = (1. + self._z_lens) * dd * ds / dds
        return ddt, dd

    @staticmethod
    def check_dist(kwargs_lens, kwargs_kin):
        """
        checks if the provided keyword arguments describe a distribution function of hyper parameters or are single
        values

        :param kwargs_lens: lens model hyper parameter keywords
        :param kwargs_kin: kinematic model hyper parameter keywords
        :return: bool, True if delta function, else False
        """
        lambda_mst_sigma = kwargs_lens.get('lambda_mst_sigma', 0)  # scatter in MST
        kappa_ext_sigma = kwargs_lens.get('kappa_ext_sigma', 0)
        a_ani_sigma = kwargs_kin.get('a_ani_sigma', 0)
        beta_inf_sigma = kwargs_kin.get('beta_inf_sigma', 0)
        if a_ani_sigma == 0 and lambda_mst_sigma == 0 and kappa_ext_sigma == 0 and beta_inf_sigma == 0:
            return True
        return False

    def draw_lens(self, lambda_mst=1, lambda_mst_sigma=0, kappa_ext=0, kappa_ext_sigma=0, gamma_ppn=1):
        """

        :param lambda_mst: MST transform
        :param lambda_mst_sigma: spread in the distribution
        :param kappa_ext: external convergence mean in distribution
        :param kappa_ext_sigma: spread in the distribution
        :param gamma_ppn: Post-Newtonian parameter
        :return: draw from the distributions
        """
        lambda_mst_draw = np.random.normal(lambda_mst, lambda_mst_sigma)
        if self._kappa_ext_bias is True:
            kappa_ext_draw = np.random.normal(kappa_ext, kappa_ext_sigma)
        else:
            kappa_ext_draw = 0
        return lambda_mst_draw, kappa_ext_draw, gamma_ppn

    def draw_anisotropy(self, a_ani=None, a_ani_sigma=0, beta_inf=None, beta_inf_sigma=0):
        """
        draw Gaussian distribution and re-sample if outside bounds

        :param a_ani: mean of the distribution
        :param a_ani_sigma: std of the distribution
        :return: random draw from the distribution
        """
        if self._anisotropy_model in ['OM']:
            if a_ani < self._ani_param_min or a_ani > self._ani_param_max:
                raise ValueError('anisotropy parameter is out of bounds of the interpolated range!')
            a_ani_draw = np.random.normal(a_ani, a_ani_sigma)
            if a_ani_draw < self._ani_param_min or a_ani_draw > self._ani_param_max:
                return self.draw_anisotropy(a_ani, a_ani_sigma)
            return np.array([a_ani_draw])
        elif self._anisotropy_model in ['GOM']:
            if a_ani < self._ani_param_min[0] or a_ani > self._ani_param_max[0] or beta_inf < self._ani_param_min[1] or beta_inf > self._ani_param_max[1]:
                raise ValueError('anisotropy parameter is out of bounds of the interpolated range!')
            a_ani_draw = np.random.normal(a_ani, a_ani_sigma)
            beta_inf_draw = np.random.normal(beta_inf, beta_inf_sigma)
            if a_ani_draw < self._ani_param_min[0] or a_ani_draw > self._ani_param_max[0] or beta_inf_draw < self._ani_param_min[1] or beta_inf_draw > self._ani_param_max[1]:
                return self.draw_anisotropy(a_ani, a_ani_sigma, beta_inf, beta_inf_sigma)
            return np.array([a_ani_draw, beta_inf_draw])
        return None

