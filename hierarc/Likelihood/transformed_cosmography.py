__author__ = "sibirrer"
import numpy as np


class TransformedCosmography(object):
    """Class to manage hierarchical hyper-parameter that impact the cosmographic
    posterior interpretation of individual lenses."""

    def __init__(self, gamma_pl_pivot=None, gamma_pl_ddt_slope=None):
        """

        :param gamma_pl_pivot: pivote power-law slope relative to which the Ddt scaling operates
        :param gamma_pl_ddt_slope: slope of Ddt / gamma_pl such that Ddt(gamma_pl) = Ddt_0 + slope * gamma_pl
        """
        if gamma_pl_ddt_slope is None or gamma_pl_ddt_slope is None:
            self._gamma_pl_scaling = False
        else:
            self._gamma_pl_scaling = True
        self._gamma_pl_pivot = gamma_pl_pivot
        self._gamma_pl_ddt_slope = gamma_pl_ddt_slope

    def displace_prediction(
        self, ddt, dd, gamma_ppn=1, lambda_mst=1, kappa_ext=0, mag_source=0, gamma_pl=None,
    ):
        """Here we effectively change the posteriors of the lens, but rather than
        changing the instance of the KDE we displace the predicted angular diameter
        distances in the opposite direction The displacements form different effects are
        multiplicative and thus invariant under the order those displacements are
        applied.

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :param lambda_mst: overall global mass-sheet transform applied on the sample,
            lambda_mst=1 corresponds to the input model
        :param gamma_ppn: post-newtonian gravity parameter (=1 is GR)
        :param kappa_ext: external convergence to be added on top of the D_dt posterior
        :param mag_source: source magnitude (attention, log scale, thus transform needs
            to be changed!)
        :param gamma_pl: power-law density slope
        :returns: ddt, dd, mag_source
        """
        ddt_, dd_ = self._displace_ppn(ddt, dd, gamma_ppn=gamma_ppn)
        # TODO scale source with ds, make sure definition is either linear or magnitudes (log) consistently
        ddt_, dd_, mag_source_ = self._displace_lambda_mst(
            ddt_, dd_, lambda_mst=lambda_mst, kappa_ext=kappa_ext, mag_source=mag_source
        )
        ddt_ = self._displace_gamma_pl(ddt_, gamma_pl)
        return ddt_, dd_, mag_source_

    @staticmethod
    def _displace_ppn(ddt, dd, gamma_ppn=1):
        """Post-Newtonian parameter sampling. The deflection terms remain the same as
        those are measured by lensing. The dynamical term changes and affects the
        kinematic prediction and thus dd.

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :param gamma_ppn: post-newtonian gravity parameter (=1 is GR)
        :return: ddt_, dd_
        """
        dd_ = dd * (1 + gamma_ppn) / 2.0
        return ddt, dd_

    @staticmethod
    def _displace_kappa_ext(ddt, dd, kappa_ext=0):
        """Assumes an additional mass-sheet of kappa_ext is present at the lens LOS
        (effectively mimicing an overall selection bias in the lenses that is not
        visible in the individual LOS analyses of the lenses. This is speculative and
        should only be considered if there are specific reasons why the current LOS
        analysis is insufficient.

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :param kappa_ext: external convergence to be added on top of the D_dt posterior
        :return: ddt_, dd_
        """
        ddt_ = ddt * (1.0 - kappa_ext)
        return ddt_, dd

    @staticmethod
    def _displace_lambda_mst(ddt, dd, lambda_mst=1, kappa_ext=0, mag_source=0):
        """Approximate internal mass-sheet transform on top of the assumed profiles
        inferred in the analysis of the individual lenses. The effect is to first order
        the same as for a pure mass sheet as a kappa_ext term. However the change here
        affects the 3-dimensional mass profile and thus the kinematics predictions is
        affected.

        We showed that for a set of profiles, the kinematics of a 3-d approximate mass sheet can still be very well
        approximated as d sigma_v_lambda**2 /d lambda = lambda * sigma_v0

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :param lambda_mst: overall global mass-sheet transform applied on the sample,
         lambda_mst = 1 corresponds to the input model, 0.9 corresponds to a positive mass sheet of 0.1
        :param kappa_ext: external convergence to be added on top of the D_dt posterior kappa_ext = 1 - lambda_mst
        :param mag_source: source magnitude (-2.5 log10 base)
        :return: ddt_, dd_, mag_source
        """
        lambda_tot = lambda_mst * (1 - kappa_ext)  # combine internal and external MST
        lambda_tot = np.maximum(
            lambda_tot, 0.0001
        )  # lambda can not get negative and zero is leading to infinite magnitudes
        ddt_ = (
            ddt * lambda_tot
        )  # the actual posteriors needed to be corrected by Ddt_true = Ddt_mst / (1-kappa_ext)
        # this line can be changed in case the physical 3-d approximation of the chosen profile does scale differently with the kinematics
        sigma_v2_scaling = lambda_mst
        dd_ = (
            dd * sigma_v2_scaling / lambda_mst
        )  # the kinematics constrain Dd/Dds and thus the constraints on Dd is not affected by lambda

        # amp_source_ = amp_source / lambda_tot ** 2  # inverse MST transform of magnification
        mag_source_ = mag_source + 5 * np.log10(
            lambda_tot
        )  # inverse MST transform of magnification in magnitude
        return ddt_, dd_, mag_source_

    def _displace_gamma_pl(self, ddt, gamma_pl):
        """
        shifts Ddt prediction as a function of the power-law density slope.
        This shift is the inverse such that the Ddt predictions in the likelihood are unchanged.

        :param ddt: model-predicted time-delay distance
        :param gamma_pl: model-predicted power-law density slope
        :return: shifted ddt
        """
        if self._gamma_pl_scaling:
            # minus sign is as we move the model prediction and not the posteriors
            ddt -= (gamma_pl - self._gamma_pl_pivot) * self._gamma_pl_ddt_slope
        return ddt
