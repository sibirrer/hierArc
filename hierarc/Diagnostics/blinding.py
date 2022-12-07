import numpy as np
import copy


def blind_posterior(posterior, param_names):
    """
    blinds H0 and lambda_int to default values

    :param posterior: posterior samples of hierArc
    :type posterior: flattened posterior with num_param x num_samples array
    :param param_names: names of parameters being sampled in hierArc conventions
    :type param_names: list of strings
    :return: posterior_blind
    """
    posterior_blind = copy.deepcopy(posterior)
    for i, param_name in enumerate(param_names):
        if param_name == 'lambda_mst':
            # shift all lambda_int posteriors to a median = 1
            posterior_blind[:, i] *= 1 / np.median(posterior_blind[:, i])
        if param_name == 'h0':
            # shift all H0 posteriors to a mean = 70
            posterior_blind[:, i] *= 70 / np.median(posterior_blind[:, i])
    return posterior_blind
