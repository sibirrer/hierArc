import numpy as np
from astropy.cosmology import FLRW, FlatFLRWMixin
from scipy.special import exp1
from astropy.cosmology.core import dataclass_decorator
from astropy.cosmology.parameter import Parameter
from astropy.cosmology._utils import aszarr


@dataclass_decorator
class wPhiCDM(FlatFLRWMixin, FLRW):
    """
    This class implements the wphiCDM cosmology from Shajib & Frieman (2025), https://arxiv.org/abs/2502.06929.
    """

    w0: Parameter = Parameter(
        default=-1.0, doc="Dark energy equation of state at z=0.", fvalidate="float"
    )
    alpha: Parameter = Parameter(
        default=1.45,
        doc="Negative derivative of dark energy equation of state w.r.t. a.",
        fvalidate="float",
    )

    def __post_init__(self):
        super().__post_init__()

    def w(self, z):
        r"""Returns dark energy equation of state at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'] or array-like
            Input redshift.

            .. versionchanged:: 7.0
            Passing z as a keyword argument is deprecated.

        Returns
        -------
        w : ndarray or float
            The dark energy equation of state
            Returns `float` if the input is scalar.

        Notes
        -----
        The dark energy equation of state is defined as
        :math:`w(z) = P(z)/\rho(z)`, where :math:`P(z)` is the pressure at
        redshift z and :math:`\rho(z)` is the density at redshift z, both in
        units where c=1. Here this is
        :math:`w(z) =  -1 + (1 + w_0) \exp(-\alpha  z)`.
        """
        z = aszarr(z)
        return -1 + (1 + self.w0) * np.exp(
            -self.alpha * z
        )  # self.w0 + self.wa * z / (z + 1.0)

    def de_density_scale(self, z):
        r"""Evaluates the redshift dependence of the dark energy density.

        Parameters
        ----------
        z : Quantity-like ['redshift'] or array-like, positional-only
            Input redshift.

            .. versionchanged:: 7.0
                Passing z as a keyword argument is deprecated.

        Returns
        -------
        I : ndarray or float
            The scaling of the energy density of dark energy with redshift.
            Returns `float` if the input is scalar.

        Notes
        -----
        The scaling factor, I, is defined by :math:`\rho(z) = \rho_0 I`,
        and in this case is given by

        .. math::
            I = \exp\left(-3 (1 + w_0) \exp(\alpha) \left[E_1(\alpha) - E_1(\alpha (1 + z))\right]\right)
        """
        z = aszarr(z)
        zp1 = z + 1.0
        return np.exp(
            3
            * (1 + self.w0)
            * np.exp(self.alpha)
            * (exp1(self.alpha) - exp1(self.alpha * zp1))
        )
