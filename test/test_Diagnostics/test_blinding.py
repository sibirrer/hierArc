import numpy as np
import numpy.testing as npt
from hierarc.Diagnostics.blinding import blind_posterior


def test_blind_posterior():
    param_names = ['h0', 'lambda_mst', 'test']
    posterior = np.random.random((100, 3))
    posterior_blinded = blind_posterior(posterior=posterior, param_names=param_names)
    assert len(posterior[:, 0]) == 100
    npt.assert_almost_equal(np.median(posterior_blinded[:, 0]), 70, decimal=8)
    npt.assert_almost_equal(np.median(posterior_blinded[:, 1]), 1, decimal=8)

