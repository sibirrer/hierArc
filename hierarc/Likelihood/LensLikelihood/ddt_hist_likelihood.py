import numpy as np
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity


class DdtHistLikelihood(object):
    """
    Evaluates the likelihood of a time-delay distance ddt (in Mpc) against the model predictions, using a
         loglikelihood sampled from a Kernel Density Estimator. the KDE is constructed using a binned version of the
         full samples. Greatly improves speed at the cost of a (tiny) loss in precision
        __warning:: you should adjust bandwidth and nbins_hist to the spacing and size of your samples chain!

    original source: https://github.com/shsuyu/H0LiCOW-public/blob/master/H0_inference_code/lensutils.py
    credits to Martin Millon, Aymeric Galan
    """
    def __init__(self, z_lens, z_source, ddt_samples, kde_kernel=None, ddt_weights=None, bandwidth=20, nbins_hist=200):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param ddt_samples: numpy array of Ddt values
        :param ddt_weights: optional weights for the samples in Ddt
        :param kde_kernel: string of KDE kernel type
        :param bandwidth: bandwith of kernel
        :param nbins_hist: number of bins in the histogram
        """

        hist = np.histogram(ddt_samples, bins=nbins_hist, weights=ddt_weights)
        vals = hist[0]
        bins = [(h + hist[1][i + 1]) / 2.0 for i, h in enumerate(hist[1][:-1])]

        # ignore potential zero weights, sklearn does not like them
        kde_bins = [b for v, b in zip(vals, bins) if v > 0]
        kde_weights = [v for v in vals if v > 0]
        print(np.shape(kde_weights), np.shape(kde_bins))
        self._kde = gaussian_kde(dataset=kde_bins, weights=kde_weights[:])

    def log_likelihood(self, ddt, dd=None, aniso_scaling=None):
        """
        Note: kinematics + imaging data can constrain Ds/Dds. The input of Ddt, Dd is transformed here to match Ds/Dds

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :return: log likelihood given the single lens analysis
        """
        return np.log(self._kde(ddt))


class DdtHistKDELikelihood(object):
    """
    Evaluates the likelihood of a time-delay distance ddt (in Mpc) against the model predictions, using a
         loglikelihood sampled from a Kernel Density Estimator. the KDE is constructed using a binned version of the
         full samples. Greatly improves speed at the cost of a (tiny) loss in precision
        __warning:: you should adjust bandwidth and nbins_hist to the spacing and size of your samples chain!

    original source: https://github.com/shsuyu/H0LiCOW-public/blob/master/H0_inference_code/lensutils.py
    credits to Martin Millon, Aymeric Galan
    """
    def __init__(self, z_lens, z_source, ddt_samples, kde_kernel=None, ddt_weights=None, bandwidth=20, nbins_hist=200):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param ddt_samples: numpy array of Ddt values
        :param ddt_weights: optional weights for the samples in Ddt
        :param kde_kernel: string of KDE kernel type
        :param bandwidth: bandwith of kernel
        :param nbins_hist: number of bins in the histogram
        """

        hist = np.histogram(ddt_samples, bins=nbins_hist, weights=ddt_weights)
        vals = hist[0]
        bins = [(h + hist[1][i + 1]) / 2.0 for i, h in enumerate(hist[1][:-1])]

        # ignore potential zero weights, sklearn does not like them
        kde_bins = [(b,) for v, b in zip(vals, bins) if v > 0]
        kde_weights = [v for v in vals if v > 0]

        kde = KernelDensity(kernel=kde_kernel, bandwidth=bandwidth).fit(kde_bins, sample_weight=kde_weights)
        self._score = kde.score

    def log_likelihood(self, ddt, dd=None, aniso_scaling=None):
        """
        Note: kinematics + imaging data can constrain Ds/Dds. The input of Ddt, Dd is transformed here to match Ds/Dds

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :return: log likelihood given the single lens analysis
        """
        return self._score(np.array(ddt).reshape(1, -1))
