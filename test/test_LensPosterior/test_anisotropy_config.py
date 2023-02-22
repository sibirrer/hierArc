from hierarc.LensPosterior.anisotropy_config import AnisotropyConfig
import unittest
import pytest


class TestAnisotropyConfig(object):

    def setup(self):
        self.r_eff = 2
        self.config_om = AnisotropyConfig(anisotropy_model='OM', r_eff=self.r_eff)
        self.config_gom = AnisotropyConfig(anisotropy_model='GOM', r_eff=self.r_eff)
        self.config_const = AnisotropyConfig(anisotropy_model='const', r_eff=self.r_eff)

    def test_kwargs_anisotropy_base(self):
        kwargs = self.config_om.kwargs_anisotropy_base
        assert kwargs['r_ani'] == self.r_eff

        kwargs = self.config_gom.kwargs_anisotropy_base
        assert kwargs['r_ani'] == self.r_eff
        assert kwargs['beta_inf'] == 1

        kwargs = self.config_const.kwargs_anisotropy_base
        assert kwargs['beta'] == 0.1

    def test_ani_param_array(self):
        ani_param_array = self.config_om.ani_param_array
        assert len(ani_param_array) == 6

        ani_param_array = self.config_gom.ani_param_array
        assert len(ani_param_array[0]) == 6
        assert len(ani_param_array[1]) == 4

        ani_param_array = self.config_const.ani_param_array
        assert len(ani_param_array) == 7

    def test_anisotropy_kwargs(self):
        a_ani = 2
        beta_inf = 0.5
        kwargs = self.config_om.anisotropy_kwargs(a_ani)
        assert kwargs['r_ani'] == a_ani * self.r_eff

        kwargs = self.config_gom.anisotropy_kwargs(a_ani, beta_inf)
        assert kwargs['r_ani'] == a_ani * self.r_eff
        assert kwargs['beta_inf'] == beta_inf

        kwargs = self.config_const.anisotropy_kwargs(a_ani)
        assert kwargs['beta'] == a_ani


class TestRaise(unittest.TestCase):

    def test_raise(self):

        with self.assertRaises(ValueError):
            AnisotropyConfig(anisotropy_model='BAD', r_eff=1)

        with self.assertRaises(ValueError):
            conf = AnisotropyConfig(anisotropy_model='OM', r_eff=1)
            conf._anisotropy_model = 'BAD'
            kwargs = conf.kwargs_anisotropy_base

        with self.assertRaises(ValueError):
            conf = AnisotropyConfig(anisotropy_model='OM', r_eff=1)
            conf._anisotropy_model = 'BAD'
            kwargs = conf.anisotropy_kwargs(a_ani=1, beta_inf=1)


if __name__ == '__main__':
    pytest.main()
