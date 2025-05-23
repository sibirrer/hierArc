import os
import pandas as pd
import numpy as np
import hierarc

_PATH_2_DATA = os.path.join(os.path.dirname(hierarc.__file__), "Data", "BAO")


class DESIDR2Data(object):
    """
    This class collect the data from teh DESI DR2 analysis presented in DESI collaboration et al. 2025 (https://arxiv.org/abs/2503.14738)

    The data covariances that are stored in hierArc are originally from `DESI DR2`_.
    https://github.com/CobayaSampler/bao_data/tree/master/desi_bao_dr2

    If you make use of these products, please cite `DESI collaboration et al. 2025`_
    """

    def __init__(self):
        self._data_file = os.path.join(
            _PATH_2_DATA, "desi_bao_dr2", "desi_gaussian_bao_ALL_GCcomb_mean.txt"
        )
        self._cov_file = os.path.join(
            _PATH_2_DATA, "desi_bao_dr2", "desi_gaussian_bao_ALL_GCcomb_cov.txt"
        )

        data = pd.read_csv(self._data_file, sep=r"\s+")

        self.origlen = len(data)
        print(f"Importing {self.origlen} distances from DESI DR2 data.")

        self.z = data.iloc[:, 0].to_numpy()
        self.d = data.iloc[:, 1].to_numpy()
        self.distance_type = data.iloc[:, 2].to_numpy()

        self.cov = np.loadtxt(self._cov_file)
