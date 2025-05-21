import numpy as np
from astropy.constants import c
import astropy.units as u

_twopi = 2 * np.pi


class CustomBAOLikelihood(object):
    """Class method for an arbitrary BAO measurements. Distances measurements (scaled by rs) and the
     covariance matrix must be provided in the constructor. The likelihood is
     assumed to be Gaussian."""

    def __init__(self, z, d, distance_type, cov):
        """

        :param z: array of redshifts of the BAO measurements
        :param d: array of BAO measurements, scaled by rs
        :param distance_type: string, either 'DV_over_rs' or 'DM_over_rs' or 'DH_over_rs'

        """
        self.z = z
        self.d = d
        self.distance_type = distance_type
        self.cov = cov
        self._inv_cov = np.linalg.inv(cov)
        self.num_d = len(d)
        assert len(z) == len(d), "z and d must have the same length"

    def log_likelihood_bao(self, cosmo, rd):
        """
        :param cosmo: instance of a class to compute angular diameter distances on arrays
        :param rd: comoving sound horizon at the drag epoch
        :return: log likelihood of the data given the specified cosmology
        """
        distance_theory = np.zeros(self.num_d)

        for i in range(self.num_d):
            if self.distance_type[i] == "DV_over_rs":
                distance_theory[i] = self._compute_DV(cosmo, self.z[i])
            elif self.distance_type[i] == "DM_over_rs":
                distance_theory[i] = self._compute_DM(cosmo, self.z[i])
            elif self.distance_type[i] == "DH_over_rs":
                distance_theory[i] = self._compute_DH(cosmo, self.z[i])
            else:
                raise ValueError("Unsupported distance type: {}".format(self.distance_type))
            #scale by the comoving sound horizon
            distance_theory[i] /= rd

        # Compute the log likelihood
        diff = self.d - distance_theory
        inv_cov = self._inv_cov
        logL = -0.5 * np.dot(diff, np.dot(inv_cov, diff))
        return logL

    def _compute_DV(self, cosmo, z):
        """Compute the DV distance at redshift z. (see Section III.A of https://arxiv.org/pdf/2503.14738)"""

        DV = (z * self._compute_DM(cosmo, z)**2 * self._compute_DH(cosmo, z))**(1./3.)
        return DV

    def _compute_DM(self, cosmo, z):
        """Compute the DM distance (transverse comoving distance) at redshift z."""
        return cosmo.comoving_transverse_distance(z).value

    def _compute_DH(self, cosmo, z):
        """Compute the DH (Hubble distance) distance at redshift z."""
        Hz = cosmo.H(z)  # in km/s/Mpc
        D_H = (c / Hz).to(u.Mpc)
        return D_H.value


