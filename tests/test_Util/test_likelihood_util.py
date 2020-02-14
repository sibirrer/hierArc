import numpy as np
import numpy.testing as npt
import pytest


from hierarc.Util import likelihood_util


class TestLikelihoodUtil(object):

    def setup(self):
        pass

    def test_get_truncated_normal(self):
        np.random.seed(seed=42)
        draw = likelihood_util.get_truncated_normal(mean=0, sd=1, low=0, upp=10, size=1)
        npt.assert_almost_equal(draw, 0.48812700907868467, decimal=3)

        draw = likelihood_util.get_truncated_normal(mean=0, sd=1, low=0, upp=10, size=10)
        assert len(draw) == 10


if __name__ == '__main__':
    pytest.main()
