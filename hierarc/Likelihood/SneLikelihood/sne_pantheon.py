"""
This is a lightweight version of the COSMOMC/Cobaya sampler: https://github.com/CobayaSampler/cobaya/blob/71b87842d12c6a04eec182c39b6bef1cd9a987af/cobaya/likelihoods/_base_classes/_sn_prototype.py#L287
It uses the binned Pantheon data: https://github.com/dscolnic/Pantheon/blob/master/Binned_data/lcparam_DS17f.txt
And computes the cosmographic likelihood.
The main difference is that this class is compatible with the hierArc cosmology module for evaluating likelihoods.

.. |br| raw:: html
   <br />
.. note::
   - If you use ``sn.pantheon``, please cite:|br|
     Scolnic, D. M. et al,
     `The Complete Light-curve Sample of Spectroscopically
     Confirmed Type Ia Supernovae from Pan-STARRS1 and
     Cosmological Constraints from The Combined Pantheon Sample`
     `(arXiv:1710.00845) <https://arxiv.org/abs/1710.00845>`_


:Synopsis: Supernovae likelihood, from CosmoMC's JLA module, for Pantheon and JLA samples.
:Author: Alex Conley, Marc Betoule, Antony Lewis (see source for more specific authorship)

"""
import os
import hierarc


_SAMPLE_NAME_SUPPORTED = ['Pantheon']


class SneBaseLikelihood(object):
    """
    Base likelihood class for evaluating Sne likelihoods
    """
    def __init__(self, sample_name='Pantheon'):
        if sample_name not in _SAMPLE_NAME_SUPPORTED:
            raise ValueError('Sample name %s not supported. Please chose the Sne sample name among %s.'
                             % (sample_name, _SAMPLE_NAME_SUPPORTED))
        if sample_name == 'Pantheon':
            self._data_path = os.path.join(os.path.dirname(hierarc.__file__), 'Data', 'SNe', 'pantheon_binned_lcparam_DS17f.txt')
