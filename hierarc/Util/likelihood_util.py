import math
import numpy as np
from scipy.stats import truncnorm


def log_likelihood_cov(data, model, cov_error):
    """
    log likelihood of the data given a model

    :param data: data vector
    :param model: model vector
    :param cov_error: covariance matrix
    :return: log likelihood
    """
    delta = data - model
    return -delta.dot(cov_error.dot(delta)) / 2.


def cov_error_create(error_independent, error_covariance):
    """
    generates an error covariance matrix from a set of independent uncertainties combined with a fully covariant term

    :param error_independent: array of Gaussian 1-sigma uncertainties
    :param error_covariance: float, shared covariant error among all data points. So if all data points are off by
    1-sigma, then the log likelihood is 1-sigma
    :return: error covariance matrix
    """
    error_covariance_array = np.ones_like(error_independent) * error_covariance
    error = np.outer(error_covariance_array, error_covariance_array) + np.diag(error_independent**2)
    return np.linalg.inv(error)


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
