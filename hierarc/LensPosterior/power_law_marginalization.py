

def theory_fermat_pot_scaling(gamma, gamma_base):
    """

    :param gamma: power-law slope
    :param gamma_base: baseline power-law slope
    :return: scaling factor of Fermat potential relative to prediction with baseline power-law slope
    """
    delta_ddt = 1. / fermat_pl_slope_scaling(gamma) - 1. / fermat_pl_slope_scaling(gamma_base)
    return 1 / (1. + delta_ddt)


def theory_ddt_gamma_scaling(gamma, gamma_base):
    """

    :param gamma: power-law slope
    :param gamma_base: baseline power-law slope
    :return: scaling factor of Time-Delay distance relative to prediction with baseline power-law slope
    """
    return 1 + 1. / fermat_pl_slope_scaling(gamma) - 1. / fermat_pl_slope_scaling(gamma_base)


def fermat_pl_slope_scaling(gamma):
    """
    scaling of the fermat potential with power law slope gamma

    :param gamma: power-law slope of single power-law lens profile
    :return: proportionality factor of Fermat potential with the expected convergence at the Einstein radius
    """
    kappa_e = (3 - gamma) / 2.  # see e.g., Kochanek 2021
    h0_scaling = 1 - kappa_e  # see e.g., Kochanek 2021
    fermat_scaling = h0_scaling  # the Fermat potential scales proportional to the Hubble constant
    return fermat_scaling
