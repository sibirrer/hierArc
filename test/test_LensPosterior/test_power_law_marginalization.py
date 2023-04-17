from hierarc.LensPosterior import power_law_marginalization
import copy
import numpy as np
import numpy.testing as npt


def test_power_law_marginalization():
    from lenstronomy.LensModel.lens_model import LensModel

    lens_model = LensModel(lens_model_list=['EPL', 'SHEAR'])
    gamma_pl_base = 2.

    # define parameter values of lens models #
    kwargs_epl = {'theta_E': 1.1, 'e1': 0.1, 'e2': 0.1, 'gamma': gamma_pl_base, 'center_x': 0.1, 'center_y': 0}
    kwargs_shear = {'gamma1': -0.01, 'gamma2': .03}
    kwargs_lens = [kwargs_epl, kwargs_shear]

    beta_ra, beta_dec = 0.00, -0.01

    # import the lens equation solver class #
    from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
    from lenstronomy.LensModel.Solver.solver import Solver

    # specifiy the lens model class to deal with #
    solver = LensEquationSolver(lens_model)

    # solve for image positions provided a lens model and the source position #
    theta_ra, theta_dec = solver.image_position_from_source(beta_ra, beta_dec, kwargs_lens)

    # calculate Fermat potential differences
    fermat_pot = lens_model.fermat_potential(theta_ra, theta_dec, kwargs_lens)
    fermat_pot_diff_base = fermat_pot[1:] - fermat_pot[0]

    solver = Solver(solver_type='PROFILE_SHEAR', num_images=len(theta_ra), lensModel=lens_model)

    def fermat_pot_gamma_difference(gamma):
        """
        """

        # change power-law slope

        kwargs_lens_g = copy.deepcopy(kwargs_lens)
        kwargs_lens_g[0]['gamma'] = gamma

        # apply 4-point solver

        kwargs_lens_g, _ = solver.constraint_lensmodel(theta_ra, theta_dec, kwargs_lens_g)

        # recalculate Fermat potential differences
        fermat_pot = lens_model.fermat_potential(theta_ra, theta_dec, kwargs_lens_g)
        fermat_pot_diff_g = fermat_pot[1:] - fermat_pot[0]

        return fermat_pot_diff_g

    gamma_list = np.linspace(start=1.8, stop=2.2, num=11)

    x = gamma_list
    # x = 1 / np.sqrt((gamma_list - 1))

    fermat_pot_ratios = [[] for i in range(len(fermat_pot_diff_base))]
    ddt_ratios = [[] for i in range(len(fermat_pot_diff_base))]

    for gamma in gamma_list:
        fermat_pot_diff_g = fermat_pot_gamma_difference(gamma)
        for i, fermat_pot_diff in enumerate(fermat_pot_diff_g):
            fermat_pot_ratios[i].append(fermat_pot_diff/fermat_pot_diff_base[i])
            ddt_ratios[i].append(fermat_pot_diff_base[i]/fermat_pot_diff)

    fermat_ratio_theory = power_law_marginalization.theory_fermat_pot_scaling(gamma_list, gamma_pl_base)
    ddt_ratio_theory = power_law_marginalization.theory_ddt_gamma_scaling(gamma_list, gamma_pl_base)

    """
    import matplotlib.pyplot as plt
    plt.plot(x, fermat_pot_ratios[0], label='AB')
    plt.plot(x, fermat_pot_ratios[1], label='AC')
    plt.plot(x, fermat_pot_ratios[2], label='AD')
    plt.plot(x, fermat_ratio_theory, label='theory')
    plt.xlabel(r'$\gamma_{\rm pl}$')
    plt.ylabel(r'$\Delta \phi(\gamma) / \Delta \phi(\gamma=2)$')
    plt.legend()
    plt.show()

    plt.plot(x, fermat_pot_ratios[0] / fermat_ratio_theory, label='AB')
    plt.plot(x, fermat_pot_ratios[1] / fermat_ratio_theory, label='AC')
    plt.plot(x, fermat_pot_ratios[2] / fermat_ratio_theory, label='AD')
    plt.legend()
    plt.show()
    """

    npt.assert_almost_equal(fermat_pot_ratios[0] / fermat_ratio_theory, 1, decimal=1)
    npt.assert_almost_equal(ddt_ratios[0] / ddt_ratio_theory, 1, decimal=1)
