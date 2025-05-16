import numpy as np
import numpy.testing as npt

from hierarc.Likelihood.SneLikelihood.des_5yr import DES5YRData


class TestDES5YRData(object):
    def setup_method(self):
        pass

    def test_import(self):
        data = DES5YRData()
        mag_mean = data.mu_obs
        cov_mag = data.cov_mag_b
        zhel = data.zHEL
        zcmb = data.zCMB
        npt.assert_almost_equal(zcmb[0], 0.24605)
        assert len(mag_mean) == len(zhel)
        assert len(zhel) == len(zcmb)
        nx, ny = np.shape(cov_mag)
        assert nx == ny
        assert nx == len(mag_mean)
