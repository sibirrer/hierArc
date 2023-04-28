import pytest
from hierarc.Likelihood.transformed_cosmography import TransformedCosmography
from lenstronomy.Util.data_util import magnitude2cps
import numpy.testing as npt
from hierarc.LensPosterior import power_law_marginalization


class TestTransformedCosmography(object):

    def setup(self):
        z_lens = 0.5
        z_source = 1.5
        self.transform = TransformedCosmography(z_lens=z_lens, z_source=z_source)

    def test_displace_prediction(self):

        ddt, dd = 1, 1
        mag_source = 16
        magnitude_zero_point = 20
        amp_source = magnitude2cps(magnitude=mag_source, magnitude_zero_point=magnitude_zero_point)
        # case where nothing happens
        ddt_, dd_, mag_source_ = self.transform.displace_prediction(ddt, dd, gamma_ppn=1, lambda_mst=1, kappa_ext=0,
                                                                    mag_source=mag_source)
        assert ddt == ddt_
        assert dd == dd_
        assert mag_source_ == mag_source

        # case where kappa_ext is displaced
        kappa_ext = 0.1
        ddt_, dd_, mag_source_ = self.transform.displace_prediction(ddt, dd, gamma_ppn=1, lambda_mst=1,
                                                                    kappa_ext=kappa_ext, mag_source=mag_source)
        assert ddt_ == ddt * (1 - kappa_ext)
        assert dd_ == dd
        amp_source_ = magnitude2cps(magnitude=mag_source_, magnitude_zero_point=magnitude_zero_point)
        npt.assert_almost_equal(amp_source_, amp_source / (1 - kappa_ext)**2, decimal=7)

        # case where lambda_mst is displaced
        lambda_mst = 0.9
        ddt_, dd_, mag_source_ = self.transform.displace_prediction(ddt, dd, gamma_ppn=1, lambda_mst=lambda_mst,
                                                                    kappa_ext=0, mag_source=mag_source)
        assert ddt_ == ddt * lambda_mst
        assert dd == dd_
        amp_source_ = magnitude2cps(magnitude=mag_source_, magnitude_zero_point=magnitude_zero_point)
        npt.assert_almost_equal(amp_source_, amp_source / lambda_mst ** 2, decimal=7)
        # case for gamma_ppn
        gamma_ppn = 1.1
        ddt_, dd_, mag_source_ = self.transform.displace_prediction(ddt, dd, gamma_ppn=gamma_ppn, lambda_mst=1,
                                                                    kappa_ext=0, mag_source=mag_source)
        assert ddt_ == ddt
        assert dd_ == dd * (1 + gamma_ppn) / 2.
        assert mag_source_ == mag_source

    def test_displace_kappa_ext(self):
        ddt, dd = 1, 1
        kappa_ext = 0.1
        ddt_, dd_ = self.transform._displace_kappa_ext(ddt, dd, kappa_ext=kappa_ext)
        assert ddt_ == ddt * (1 - kappa_ext)
        assert dd == dd_

    def test_displace_gamma(self):
        z_lens = 0.5
        z_source = 1.5
        gamma_pl_baseline = 2
        ddt_dgamma_cov = power_law_marginalization.d_ddt_d_gamma(gamma_pl_baseline)
        print(ddt_dgamma_cov, 'test ddt_d_gamma_cov')
        transform = TransformedCosmography(z_lens=z_lens, z_source=z_source, power_law_scaling=True,
                                           gamma_pl_baseline=gamma_pl_baseline, ddt_d_gamma_cov=None)
        transform_cov = TransformedCosmography(z_lens=z_lens, z_source=z_source, power_law_scaling=True,
                                               gamma_pl_baseline=gamma_pl_baseline, ddt_d_gamma_cov=ddt_dgamma_cov)
        ddt = 1
        delta_gamma = 0.1
        ddt_ = transform._displace_gamma_pl(ddt=ddt, gamma_pl=gamma_pl_baseline + delta_gamma)
        ddt_cov = transform_cov._displace_gamma_pl(ddt=ddt, gamma_pl=gamma_pl_baseline + delta_gamma)
        npt.assert_almost_equal(ddt_, ddt_cov, decimal=1)


if __name__ == '__main__':
    pytest.main()
