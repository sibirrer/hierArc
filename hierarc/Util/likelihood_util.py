import numpy as np
from scipy.stats import truncnorm


def log_likelihood_cov(data, model, cov_error):
    """
    log likelihood of the data given a model

    :param data: data vector
    :param model: model vector
    :param cov_error: inverse covariance matrix
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
