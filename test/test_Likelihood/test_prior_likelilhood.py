from hierarc.Likelihood.prior_likelihood import PriorLikelihood
import numpy.testing as npt


class TestPriorLikelihood(object):

    def test_likelihood(self):
        prior_list = [["a", 0, 1], ["b", 1, 0.1]]
        prior_likelihood = PriorLikelihood(prior_list=prior_list)

        kwargs = {"a": 0}
        ln_l = prior_likelihood.log_likelihood(kwargs)
        npt.assert_almost_equal(ln_l, 0, decimal=5)

        kwargs = {"a": 1}
        ln_l = prior_likelihood.log_likelihood(kwargs)
        npt.assert_almost_equal(ln_l, -1 / 2, decimal=5)

        kwargs = {"a": 1, "b": 1}
        ln_l = prior_likelihood.log_likelihood(kwargs)
        npt.assert_almost_equal(ln_l, -1 / 2, decimal=5)

        prior_likelihood = PriorLikelihood(prior_list=None)
        ln_l = prior_likelihood.log_likelihood(kwargs)
        npt.assert_almost_equal(ln_l, 0, decimal=5)
