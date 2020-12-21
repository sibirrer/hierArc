from hierarc.Likelihood.SneLikelihood.sne_pantheon import SneBaseLikelihood
import os
import unittest
import pytest


class TestSnePantheon(object):

    def setup(self):
        pass

    def test_import_pantheon(self):
        base = SneBaseLikelihood(sample_name='Pantheon')
        assert os.path.exists(base._data_path)


class TestRaise(unittest.TestCase):

    def test_raise(self):

        with self.assertRaises(ValueError):
            base = SneBaseLikelihood(sample_name='BAD')


if __name__ == '__main__':
    pytest.main()
