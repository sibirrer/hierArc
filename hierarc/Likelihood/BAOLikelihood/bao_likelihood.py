import numpy as np
from hierarc.Likelihood.BAOLikelihood.bao_likelihood_custom import CustomBAOLikelihood


class BAOLikelihood(object):
    """BAO likelihood.This class supports custom likelihoods as well as likelihoods from
    the DESI BAO files."""

    def __init__(self, sample_name="DESI_DR2", **kwargs_bao_likelihood):
        """

        :param sample_name: string, either 'CUSTOM' or a specific name supported by SneLikelihoodFromFile() class
        :param kwargs_sne_likelihood: keyword arguments to initiate likelihood class
        """
        if sample_name == "CUSTOM":
            self._likelihood = CustomBAOLikelihood(**kwargs_bao_likelihood)
        elif sample_name == "DESI_DR2":
            from hierarc.Likelihood.BAOLikelihood.desi_dr2 import DESIDR2Data

            data = DESIDR2Data()

            self._likelihood = CustomBAOLikelihood(
                z=data.z,
                d=data.d,
                distance_type=data.distance_type,
                cov=data.cov,
            )

        else:
            raise ValueError("Unsupported sample name: {}".format(sample_name))

    def log_likelihood(self, cosmo, rd=None):
        """

        :param cosmo: instance of a class to compute angular diameter distances on arrays

        :return: log likelihood of the data given the specified cosmology
        """
        # TODO compute here the default case if rd is not sampled.
        if rd is None:
            raise NotImplementedError(
                "Computation of rd is not implemented yet. Please provide rd in the kwargs_cosmo and turn rd_sampling=True in the kwargs_model."
            )

        return self._likelihood.log_likelihood_bao(cosmo, rd)
