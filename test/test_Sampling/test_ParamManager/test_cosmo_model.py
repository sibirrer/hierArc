from hierarc.Sampling.ParamManager.cosmo_model import wPhiCDM
import pytest
from scipy.special import exp1
from astropy.cosmology import FlatLambdaCDM
import numpy as np


class TestwPhiCDM(object):
    def setup_class(self):
        self.cosmo = wPhiCDM(w0=-1, alpha=1.45, H0=70, Om0=0.3)

    def test_w(self):
        z = np.array([0, 1, 2])
        expected_w = -1 + (1 + self.cosmo.w0) * np.exp(-self.cosmo.alpha * z)
        np.testing.assert_array_almost_equal(self.cosmo.w(z), expected_w)

    def test_de_density_scale(self):
        z = np.array([0, 1, 2])
        zp1 = z + 1.0
        expected_de_density_scale = np.exp(
            3
            * (1 + self.cosmo.w0)
            * np.exp(self.cosmo.alpha)
            * (exp1(self.cosmo.alpha) - exp1(self.cosmo.alpha * zp1))
        )
        np.testing.assert_array_almost_equal(
            self.cosmo.de_density_scale(z), expected_de_density_scale
        )

    def test_against_lcdm(self):
        cosmo = wPhiCDM(w0=-1, alpha=1.45, H0=70, Om0=0.3)
        lcdm = FlatLambdaCDM(H0=70, Om0=0.3)
        z = np.linspace(0, 2, 100)
        np.testing.assert_allclose(cosmo.de_density_scale(z), lcdm.de_density_scale(z))
        np.testing.assert_allclose(cosmo.w(z), lcdm.w(z))


if __name__ == "__main__":
    pytest.main()
