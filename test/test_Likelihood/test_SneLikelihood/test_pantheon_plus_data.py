import numpy as np
import numpy.testing as npt

from hierarc.Likelihood.SneLikelihood.sne_pantheon_plus import PantheonPlusData


class TestPantheonPlusData(object):

    def setup(self):
        pass

    def test_import(self):
        data = PantheonPlusData()
        mag_mean = data.m_obs
        cov_mag = data.cov_mag_b
        zhel = data.zHEL
        zcmb = data.zCMB
        print(zcmb, 'zcmb')
        npt.assert_almost_equal(zcmb[0], 0.01016)
        assert len(mag_mean) == len(zhel)
        assert len(zhel) == len(zcmb)
        nx, ny = np.shape(cov_mag)
        assert nx == ny
        assert nx == len(mag_mean)
