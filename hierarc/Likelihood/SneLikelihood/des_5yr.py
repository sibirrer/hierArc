import os
import pandas as pd
import numpy as np
import hierarc

_PATH_2_DATA = os.path.join(os.path.dirname(hierarc.__file__), "Data", "SNe")


class DES5YRData(object):
    """This class is a lightweight version of the DES-SN Year 5 analysis presented in DES collab et al. (2024):
     https://ui.adsabs.harvard.edu/abs/2024ApJ...973L..14D/abstract.

    The data covariances that are stored in hierArc are originally from `DES 5YR`_.

    If you make use of these products, please cite `DES collaboration et al. (2024)`_

    .. _DES 5YR Data products: https://github.com/des-science/DES-SN5YR/tree/main/4_DISTANCES_COVMAT
    .. _DES 5YR likelihood: https://github.com/des-science/DES-SN5YR/tree/main/5_COSMOLOGY

    The Dark Energy Survey: Cosmology Results With ~1500 New High-redshift Type Ia Supernovae Using The Full 5-year Dataset. The Astrophysical Journal Letters, Volume 973, Issue 1, id.L14, 20 pp. DES Collaboration (2024).
    The Dark Energy Survey Supernova Program: Cosmological Analysis and Systematic Uncertainties. The Astrophysical Journal, Volume 975, Issue 1, id.86, 31 pp. Vincenzi et al (2024).
    Light curve and ancillary data release for the full Dark Energy Survey Supernova Program. The Astrophysical Journal, Volume 975, Issue 1, id.5, 12 pp Sánchez et al. (2024)
    """

    def __init__(self):
        self._data_file = os.path.join(_PATH_2_DATA, "DES-SN5YR", "DES-SN5YR_HD.csv")
        self._cov_file = os.path.join(_PATH_2_DATA, "DES-SN5YR", "STAT+SYS.txt")

        print("Loading DES Y5 SN data from {}".format(self._data_file))
        data = pd.read_csv(self._data_file, comment="#")
        self.origlen = len(data)
        # The only columns that we actually need here are the redshift,
        # distance modulus and distance modulus error

        self.ww = data["zHD"] > 0.00
        # use the vpec corrected redshift for zCMB
        self.zCMB = data["zHD"][self.ww]
        self.zHEL = data["zHEL"][self.ww]
        # distance modulus and relative stat uncertainties
        self.mu_obs = data["MU"][self.ww]
        self.mu_obs_err = data["MUERR_FINAL"][self.ww]

        # Return this to the parent class, which will use it
        # when working out the likelihood
        print(
            f"Found {len(self.zCMB)} DES SN 5 supernovae (or bins if you used the binned data file)"
        )

        self.cov_mag_b = self.build_covariance()

    def build_covariance(self):
        """Run once at the start to build the covariance matrix for the data."""
        filename = self._cov_file
        print("Loading DESY5 SN covariance from {}".format(filename))
        # The file format for the covariance has the first line as an integer
        # indicating the number of covariance elements, and the the subsequent
        # lines being the elements.
        # This data file is just the systematic component of the covariance -
        # we also need to add in the statistical error on the magnitudes
        # that we loaded earlier
        f = open(filename)
        line = f.readline()
        n = int(line)
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                C[i, j] = float(f.readline())

        # Now add in the statistical error to the diagonal
        for i in range(n):
            C[i, i] += self.mu_obs_err[i] ** 2
        f.close()

        # Return the covariance; the parent class knows to invert this
        # later to get the precision matrix that we need for the likelihood.

        C = C[self.ww][:, self.ww]

        return C
