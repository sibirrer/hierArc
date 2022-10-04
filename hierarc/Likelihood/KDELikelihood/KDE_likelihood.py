__author__ = 'martin-millon'

import pandas as pd
from sklearn.neighbors import KernelDensity

LIKELIHOOD_TYPES = ["kde_hist_nd", "kde_full"]


class KDELikelihood(object):
    """
    KDE likelihood class. Provide Chain object that will be used as your likelihood.
    """

    def __init__(self, chain, likelihood_type="kde_hist_nd",
                 weight_type='default',
                 kde_kernel=None,
                 bandwidth=0.01, nbins_hist=30):
        self.chain = chain
        self.loglikelihood_type = likelihood_type
        self.weight_type = weight_type
        self.kde_kernel = kde_kernel
        self.bandwidth = bandwidth
        self.nbins_hist = nbins_hist
        self.init_loglikelihood()

    def init_loglikelihood(self):
        if self.loglikelihood_type == "kde_full":
            self.kde = self.init_kernel_full(kde_kernel=self.kde_kernel, bandwidth=self.bandwidth)
            self.loglikelihood = self.kdelikelihood()
        elif self.loglikelihood_type == "kde_hist_nd":
            self.kde = self.init_kernel_kdelikelihood_hist_nd(kde_kernel=self.kde_kernel, bandwidth=self.bandwidth,
                                                              nbins_hist=self.nbins_hist)
            self.loglikelihood = self.kdelikelihood()
        else:
            raise ValueError(
                'likelihood_type %s not supported! Supported are %s.' % (likelihood_type, LIKELIHOOD_TYPES))

    def kdelikelihood(self):
        """
        Evaluates the likelihood. Return a function.

        __ warning:: you should adjust bandwidth to the spacing of your samples chain!
        """
        return self.kde.score

    def kdelikelihood_samples(self, samples):
        """
        Evaluates the likelihood on an array. Return an array

        __ warning:: you should adjust bandwidth to the spacing of your samples chain!
        """
        return self.kde.score_samples(samples)

    def init_kernel_full(self, kde_kernel, bandwidth):
        data = pd.DataFrame.from_dict(self.params)
        kde = KernelDensity(kernel=kde_kernel, bandwidth=bandwidth).fit(data.values,
                                                                        sample_weight=self.weights[self.weight_type])
        return kde

    def init_kernel_kdelikelihood_hist_nd(self, kde_kernel, bandwidth, nbins_hist):
        """
        Evaluates the likelihood from a Kernel Density Estimator. The KDE is constructed using a binned version of the full samples. Greatly improves speed at the cost of a (tiny) loss in precision

        __warning:: you should adjust bandwidth and nbins_hist to the spacing and size of your samples chain!

        __note:: nbins_hist refer to the number of bins per dimension. Hence, the final number of bins will be nbins_hist**n

        """
        samples = np.asarray([self.chain.params[keys] for keys in self.chain.params.keys()])
        hist, edges = np.histogramdd(samples.T, bins=nbins_hist)
        edges = np.asarray(edges)

        dic = {}
        ndim = len(self.chain.params.keys())
        keys = list(self.chain.params.keys())
        for i, key in enumerate(keys):
            dic[key] = [(p + edges[i, j + 1]) / 2.0 for j, p in enumerate(edges[i, :-1])]

        # for some reasons that are not obvious, ravel do not reverse meshgrid, indexing argument is really needed here
        mesh_tuple = np.meshgrid(*[dic[key] for key in keys], indexing='ij')
        meshdic = []
        for i, key in enumerate(keys):
            meshdic.append(mesh_tuple[i].flatten())
        meshdic = np.asarray(meshdic)
        kde_bins = pd.DataFrame(meshdic.T, columns=keys)
        kde_weights = np.ravel(hist)

        # remove the zero weights values
        kde_bins = kde_bins[kde_weights > 0]
        kde_weights = kde_weights[kde_weights > 0]

        # fit the KDE
        kde = KernelDensity(kernel=kde_kernel, bandwidth=bandwidth).fit(kde_bins.values, sample_weight=kde_weights)
        return kde
