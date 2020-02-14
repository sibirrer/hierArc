from hierarc.Likelihood.transformed_cosmography import TransformedCosmography
from hierarc.Likelihood.lens_likelihood import LensLikelihoodBase


class LensLikelihood(TransformedCosmography, LensLikelihoodBase):
    """
    master class containing the likelihood definitions of different analysis
    """
    def __init__(self, z_lens, z_source, likelihood_type='TDKin', ani_param_array=None, ani_scaling_array=None,
                 name='name', **kwargs_likelihood):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param name: string (optional) to name the specific lens
        :param likelihood_type: string to specify the likelihood type
        :param ani_param_array: array of anisotropy parameter values for which the kinematics are predicted
        :param ani_scaling_array: velocity dispersion sigma**2 scaling of anisotropy parameter relative to default prediction
        :param kwargs_likelihood: keyword arguments specifying the likelihood function,
        see individual classes for their use
        """
        self._name = name
        self._z_lens = z_lens
        self._z_source = z_source
        TransformedCosmography.__init__(self, z_lens=z_lens, z_source=z_source, ani_param_array=ani_param_array,
                                             ani_scaling_array=ani_scaling_array)
        LensLikelihoodBase.__init__(self, z_lens=z_lens, z_source=z_source, likelihood_type=likelihood_type, name=name,
                                    **kwargs_likelihood)

    def lens_log_likelihood(self, cosmo, gamma_ppn=1, lambda_mst=1, kappa_ext=0, aniso_param=None):
        """

        :param cosmo: astropy.cosmology instance
        :param lambda_mst: overall global mass-sheet transform applied on the sample,
        lambda_mst=1 corresponds to the input model
        :param gamma_ppn: post-newtonian gravity parameter (=1 is GR)
        :param kappa_ext: external convergence to be added on top of the D_dt posterior
        :param aniso_param: global stellar anisotropy parameter
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
        ddt_, dd_ = self._displace_prediction(ddt, dd, gamma_ppn=gamma_ppn, lambda_mst=lambda_mst, kappa_ext=kappa_ext,
                                              aniso_param=aniso_param)
        return self._lens_type.log_likelihood(ddt_, dd_)
