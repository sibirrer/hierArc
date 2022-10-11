__author__ = 'martin-millon'

import glob
import os

import numpy as np


class Chain(object):
    """
    Chain class to have a convenient way to manipulate posteriors distributions of some experiments.
    """

    def __init__(self, kw, probe, params, default_weights, cosmology, loglsamples=None, rescale=True):
        """

        :param kw: (str). Planck base cosmology keyword. For example, "base" or "base_omegak". See https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/Cosmological_Parameters.
        :param probe: (str). Planck probe combination. For example, "plikHM_TTTEEE_lowl_lowE" for default Planck results
        :param params: (dictionnary). Dictionnary containing the samples.
        :param default_weights: (numpy array). Default weights associated to the samples.
        :param cosmology: (str). Astropy cosmology
        :param loglsamples: (numpy array). Corresponding Loglikelihood of the samples (optionnal).
        :param rescale: (bool). Rescale the chains between 0 and 1 for all parameters. This is absolutely necessary if you want to evaluate a KDE on these chains.
        """
        self.kw = kw
        self.probe = probe
        self.params = params
        self.weights = {"default": default_weights}
        self.cosmology = cosmology
        self.loglsamples = loglsamples
        self.rescale = rescale
        if self.rescale:
            self.rescale_dic = {'rescaled': False}
            self.rescale_to_unity()

    def __str__(self):
        """
        Print the identifier of the Chain.
        :return:
        """
        return "%s_%s" % (self.kw, self.probe)

    def list_params(self):
        """
        List the cosmo parameters that are not empty
        :return: List of parameter name
        """
        return [p for p in self.params.keys() if len(self.params[p]) > 0]

    def list_weights(self):
        """
        List the existing weights
        :return: Array of weights
        """
        return [w for w in self.weights.keys() if len(self.weights[w]) > 0]

    def fill_default(self, param, default_val, nsamples=None, verbose=False):
        """
        Fill an empty default param with a default value

        :param param: (string) Name of the parameter to fill with default value
        :param default_val: (float). Default value.
        :param nsamples: (int). Number of samples in the Chain. If None, it will take the same number of samples as for the other parameters
        :param verbose: (bool).
        """
        assert (len(self.params[param]) == 0)
        if nsamples is None:
            lp = self.list_params()
            assert (len(lp) > 0)
            nsamples = len(self.params[lp[0]])

        self.params[param] = np.ones(nsamples) * default_val
        if verbose:
            print("filled %s with value %f" % (param, default_val))

    def fill_default_array(self, param, default_array, verbose=False):
        """
        Fill an empty default param with a default array

        :param param: (str). Name of the parameter
        :param default_array: (numpy array). Must have the same dimension as the samples.
        :param verbose: (bool).
        """

        lp = self.list_params()
        assert (len(lp) > 0)
        nsamples = len(self.params[lp[0]])
        assert (len(default_array) == nsamples)

        if param not in self.params.keys():
            if verbose:
                print('Creating a new parameter.')

        self.params[param] = default_array
        if verbose:
            print("filled %s" % (param))

    def create_param(self, param_key):
        """
        Add a new parameter to the Chain

        :param param_key: (str). Parameter name.
        """
        self.params[param_key] = []

    def rescale_to_unity(self, verbose=False):
        """
        Rescale all parameter chains between 0 and 1.

        :param verbose: (bool).
        """
        if self.rescale_dic['rescaled']:
            raise RuntimeError('Your data are already rescaled!')
        else:
            for p in self.params.keys():
                max, min = np.max(self.params[p]), np.min(self.params[p])
                self.params[p] = ((self.params[p] - min) / (max - min))
                self.rescale_dic[p] = [max, min]

                if verbose:
                    print("Rescaled parameter %s between 0 and 1" % p)
            self.rescale_dic['rescaled'] = True

    def rescale_from_unity(self, verbose=False):
        """
        Rescale all parameter chains to their original values.

        :param verbose: (bool).
        """
        if not self.rescale_dic['rescaled']:
            raise RuntimeError('Your data are not rescaled, call rescale_to_unity() first')
        else:
            for p in self.params.keys():
                max, min = self.rescale_dic[p]
                self.params[p] = ((max - min)) * self.params[p] + min
                if verbose:
                    print("Rescaled parameter %s to original scale." % p)

            self.rescale_dic['rescaled'] = False


def import_Planck_chain(datapath, kw, probe, params, cosmology, rescale=True):
    """
    Special function to parse Planck files. Return a Chain object.

    :param datapath: (str). Path to the Planck chain
    :param kw: (str). Planck base cosmology keyword. For example, "base" or "base_omegak". See https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/Cosmological_Parameters.
    :param probe: (str). Planck probe combination. For example, "plikHM_TTTEEE_lowl_lowE" for default Planck results
    :param params: (list). List of cosmological parameters. ["h0", "om"] for FLCDM.
    :param cosmology: (str). Astropy cosmology
    :param rescale: (bool). Rescale the chains between 0 and 1 for all parameters. This is absolutely necessary if you want to evaluate a KDE on these chains.
    :return: Chain object.
    """
    # the planck chains are usually split in four files
    chainspaths = glob.glob("%s/%s_%s_?.txt" % (os.path.join(datapath, kw, probe), kw, probe))
    allchainspath = "%s/%s_%s-all.txt" % (os.path.join(datapath, kw, probe), kw, probe)

    # read paramfiles and collect indexes of the interesting cosmological parameters
    params_all = open("%s/%s_%s.paramnames" % (os.path.join(datapath, kw, probe), kw, probe)).readlines()

    params_values, params_index = {}, {}
    for p in params:
        params_values[p] = []
        params_index[p] = None

    # Not all the chains have all the parameters we are interested in. We want to standardise them for our use
    # Here, we browse the chains params, looking for the parameters of interest.
    # We corresponding index is +2, since the first two parameters of each sample (weights ) are not in params
    for ind, line in enumerate(params_all):
        if 'omegal*\t\\Omega_\\Lambda\n' in line:
            params_index["ol"] = ind + 2

        elif 'ns\tn_s\n' in line:
            params_index["ns"] = ind + 2

        elif 'H0*\tH_0\n' in line:
            params_index["h0"] = ind + 2

        elif 'omegam*\t\\Omega_m\n' in line:
            params_index["om"] = ind + 2

        elif 'mnu\t\\Sigma m_\\nu\n' in line:  # pragma: no cover
            params_index["mnu"] = ind + 2

        elif 'nnu\tN_{eff}\n' in line:  # pragma: no cover
            params_index["nnu"] = ind + 2

        elif 'omegak\t\\Omega_K\n' in line:  # pragma: no cover
            params_index["ok"] = ind + 2

        elif 'w\tw\n' in line:  # pragma: no cover
            params_index["w"] = ind + 2
            params_index["w0"] = ind + 2

        elif 'wa\tw_a\n' in line:  # pragma: no cover
            params_index["wa"] = ind + 2

        elif 'meffsterile\tm_{\\nu,{\\rm{sterile}}}^{\\rm{eff}}\n' in line:  # pragma: no cover
            params_index["meffsterile"] = ind + 2

    default_weights = []
    logl_samples = []
    for chainspath in chainspaths:
        with open(chainspath, "r") as chain:
            samples = [line.split() for line in chain.readlines()]
            for p in params:
                if params_index[p] is not None:
                    params_values[p] += [s[params_index[p]] for s in samples]
            default_weights += [s[0] for s in samples]  # weights are always the first element of a chain
            logl_samples += [s[1] for s in samples]  # loglikelihood is the second element

    # fallback to numpy arrays with floats...should be default but is not...?
    for p in params:
        params_values[p] = np.array([float(v) for v in params_values[p]])

    default_weights = np.array([float(v) for v in default_weights])
    logl_samples = np.array([float(v) for v in logl_samples])

    return Chain(kw=kw, probe=probe, params=params_values, default_weights=default_weights, cosmology=cosmology,
                 loglsamples=logl_samples, rescale=rescale)


def rescale_vector_from_unity(vector, rescale_dic, keys):
    """
    Restore the original scaling of the samples, given the value in `rescale_dic`.

    :param vector: (numpy array). Vector to be rescaled.
    :param rescale_dic: (dictionnary). Contains the min and max value for each parameters.
    :param keys: (list). Contain the name of the parameter to be rescaled. all keys must be in the rescale_dic.
    :return:
    """
    for i, key in enumerate(keys):
        max, min = rescale_dic[key]
        vector[:, i] = ((max - min)) * vector[:, i] + min
    return vector


def rescale_vector_to_unity(vector, rescale_dic, keys):
    """
    Rescale a vector accroding to the min adn max value provided in rescale_dic

    :param vector: (numpy array). Vector to be rescaled.
    :param rescale_dic: (dictionnary). Contains the min and max value for each parameters.
    :param keys: Contain the name of the parameter to be rescaled. all keys must be in the rescale_dic.
    :return:
    """
    for i, key in enumerate(keys):
        max, min = rescale_dic[key]
        vector[:, i] = ((vector[:, i] - min) / (max - min))
    return vector
