import math
import numpy as np
from scipy.stats import truncnorm


def get_truncated_normal(mean=0, sd=1, low=0, upp=10, size=1):
    """

    :param mean: mean of normal distribution
    :param sd: standard deviation
    :param low: lower bound
    :param upp: upper bound
    :return: float, draw of distribution
    """
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs(size)


def sklogn_likelihood(x, mu, lam, sigma, explim=100):
    """

    :param x: value to evaluate log(p(x))
    :param mu:
    :param lam:
    :param sigma:
    :param explim: exponent limit for evaluating the likelihood
    :return: log likelihood
    """
    if (x <= lam) or ((-mu + math.log(x - lam)) ** 2 / (2. * sigma ** 2) > explim):
        return -np.inf
    else:
        llh = sklogn(x, mu, lam, sigma)

        if np.isnan(llh):
            return -np.inf
        else:
            return np.log(llh)


def sklogn(x, mu, lam, sigma):
    """
    skewed log normal distribution

    :param x: value to evaluate p(x)
    :param mu:
    :param lam:
    :param sigma:
    :return:
    """
    return math.exp(-((-mu + math.log(x - lam)) ** 2 / (2. * sigma ** 2))) / (math.sqrt(2 * math.pi) * (x - lam) * sigma)
