"""
This is a lightweight version of the COSMOMC/Cobaya sampler: https://github.com/CobayaSampler/cobaya/blob/71b87842d12c6a04eec182c39b6bef1cd9a987af/cobaya/likelihoods/_base_classes/_sn_prototype.py#L287
It uses the binned Pantheon data: https://github.com/dscolnic/Pantheon/blob/master/Binned_data/lcparam_DS17f.txt
And computes the cosmographic likelihood.
The main difference is that this class is compatible with the hierArc cosmology module for evaluating likelihoods.
This likelihood does NOT include systematics!

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
__author__ = 'sibirrer'

# Global
import numpy as np
import os

# Local
import hierarc

_twopi = 2 * np.pi
_SAMPLE_NAME_SUPPORTED = ['Pantheon_binned', 'Pantheon']
_PATH_2_DATA = os.path.join(os.path.dirname(hierarc.__file__), 'Data', 'SNe')


class SneLikelihood(object):
    """
    Base likelihood class for evaluating Sne likelihoods
    """
    def __init__(self, sample_name='Pantheon_binned', pec_z=0.001):
        """

        :param sample_name: string, name of data sample
        :param pec_z: float, peculiar velocity in units of redshift (ignored when binned data products are used)
        """
        if sample_name not in _SAMPLE_NAME_SUPPORTED:
            raise ValueError('Sample name %s not supported. Please chose the Sne sample name among %s.'
                             % (sample_name, _SAMPLE_NAME_SUPPORTED))
        if sample_name == 'Pantheon_binned':
            self._data_file = os.path.join(_PATH_2_DATA, 'pantheon_binned_lcparam_DS17f.txt')
            self._cov_file = None
            pec_z = 0

        if sample_name == 'Pantheon':
            self._data_file = os.path.join(_PATH_2_DATA, 'pantheon_lcparam_full_long_zhel.txt')
            self._cov_file = os.path.join(_PATH_2_DATA, 'pantheon_sys_full_long.txt')
        self._pec_z = pec_z
        cols = None

        self.names = []
        ix = 0
        with open(self._data_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if '#' in line:
                    cols = line[1:].split()
                    for rename, new in zip(
                            ['mb', 'color', 'x1', '3rdvar', 'd3rdvar',
                             'cov_m_s', 'cov_m_c', 'cov_s_c'],
                            ['mag', 'colour', 'stretch', 'third_var',
                             'dthird_var', 'cov_mag_stretch',
                             'cov_mag_colour', 'cov_stretch_colour']):
                        if rename in cols:
                            cols[cols.index(rename)] = new

                    zeros = np.zeros(len(lines) - 1)
                    self.set = zeros.copy()
                    for col in cols:
                        setattr(self, col, zeros.copy())
                elif line.strip():
                    if cols is None: raise ImportError('Data file must have comment header')
                    vals = line.split()
                    for i, (col, val) in enumerate(zip(cols, vals)):
                        if col == 'name':
                            self.names.append(val)
                        else:
                            getattr(self, col)[ix] = np.float64(val)
                    ix += 1
        # Check whether required instances are read in
        assert hasattr(self, 'dz')
        # TODO: make read-in such that the arguments required are explicitly matched ('zcmb', 'zhel', 'dz', 'mag', 'dmb')
        # spectroscopic redshift error. ATTENTION! This value =0 for binned data. In this code the value is not used.
        # Cobaya also does not use it!
        self.z_var = self.dz ** 2
        self.mag_var = self.dmb ** 2

        self.nsn = ix
        self._cov = self._read_covmat(self._cov_file)

        # jla_prep
        zfacsq = 25.0 / np.log(10.0) ** 2
        self.pre_vars = self.mag_var + zfacsq * self._pec_z ** 2 * (
                (1.0 + self.zcmb) / (self.zcmb * (1 + 0.5 * self.zcmb))) ** 2

        self._inv_cov = self._inverse_covariance_matrix()

    def _read_covmat(self, filename):
        """
        reads in covariance matrix file and returns it as a numpy matrix

        :param filename: string, absolute path of covariance matrix file
        :return: nxn covariance matrix
        """
        if filename is None:
            return np.zeros((self.nsn, self.nsn))
        cov = np.loadtxt(filename)
        if np.isscalar(cov[0]) and cov[0] ** 2 + 1 == len(cov):
            cov = cov[1:]
        return cov.reshape((self.nsn, self.nsn))

    def _inverse_covariance_matrix(self):
        """
        inverse error covariance matrix. Combines redshift uncertainties (to first order) and magnitude uncertainties

        :return: inverse covariance matrix (2d numpy array)
        """
        # here is the option for adding an additional covariance matrix term of the calibration and/or systematic
        # errors in the evolution of the Sne population
        invcovmat = self._cov
        invcovmat_diag = invcovmat.diagonal()  # if invcovmat is a matrix, then this is invcovmat.diagonal()

        delta = self.pre_vars
        np.fill_diagonal(invcovmat, invcovmat_diag + delta)
        invcov = np.linalg.inv(invcovmat)
        return invcov

    def logp_lum_dist(self, lum_dists, estimated_scriptm=None):
        """

        :param lum_dists: numpy array of luminosity distances to the measured supernovae bins
         (units do not matter since normalization is subtracted off for the likelihood)
        :param estimated_scriptm: mean magnitude at lum_dist=0 (optional)
        :return: log likelihood of the data given the luminosity distances
        """

        invvars = 1.0 / self.pre_vars
        wtval = np.sum(invvars)
        # uncertainty weighted estimated normalization of magnitude (maximum likelihood value)

        if estimated_scriptm is None:
            estimated_scriptm = np.sum((self.mag - lum_dists) * invvars) / wtval
        diffmag = self.mag - lum_dists - estimated_scriptm
        invcovmat = self._inv_cov

        invvars = invcovmat.dot(diffmag)
        amarg_A = invvars.dot(diffmag)

        amarg_B = np.sum(invvars)
        amarg_E = np.sum(invcovmat)
        chi2 = amarg_A + np.log(amarg_E / _twopi)  # - amarg_B ** 2 / amarg_E

        return - chi2 / 2

    def log_likelihood(self, cosmo, apparent_m_z=None, z_anchor=0.1):
        """

        :param cosmo: instance of a class to compute angular diameter distances on arrays
        :param apparent_m_z: mean apparent magnitude of SN Ia at z=z_anchor (optional)
        :param z_anchor: redshift where definition of apparent_m_z is set (only applicable when apparent_m_z != None)
        :return: log likelihood of the data given the specified cosmology
        """
        angular_diameter_distances = cosmo.angular_diameter_distance(self.zcmb).value
        lum_dists = (5 * np.log10((1 + self.zhel) * (1 + self.zcmb) * angular_diameter_distances))

        ang_dist_anchor = cosmo.angular_diameter_distance(z_anchor).value
        lum_dist_anchor = (5 * np.log10((1 + z_anchor) * (1 + z_anchor) * ang_dist_anchor))

        return self.logp_lum_dist(lum_dists - lum_dist_anchor, apparent_m_z)
