import numpy as np
import numpy.testing as npt

from hierarc.Likelihood.BAOLikelihood.desi_dr2 import DESIDR2Data

class TestDESIDR2Data(object):
    def setup_method(self):
        pass

    def test_import(self):
        data = DESIDR2Data()
        z = data.z
        d = data.d
        distance_type = data.distance_type
        cov = data.cov

        npt.assert_almost_equal(z[0], 0.29500)
        assert len(z) == len(d)
        assert len(z) == len(distance_type)
        nx, ny = np.shape(cov)
        assert nx == ny
        assert nx == len(d)
