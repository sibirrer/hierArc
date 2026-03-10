import numpy as np


class JAMAnisotropy:
    _supported_types = (
        "const",
        "radial",
        "isotropic",
        "OM",
        "GOM",
        "Colin",
        "logistic",
    )

    """
    Manager for stellar velocity anisotropy parameterizations used by Jampy.

    This class defines different radial anisotropy profiles beta(r) that can be passed
    to the Jampy code. Depending on the selected anisotropy type, the model may yield either:

    - Logistic function given by beta_0, beta_inf, r_ani, alpha parameters
    - Single value for constant anisotropy models

    The velocity anisotropy is defined as:

        beta(r) = 1 − sigma_t^2 / sigma_r^2

    where sigma_t and sigma_r are the tangential and radial velocity dispersions.

    Supported anisotropy models
    ----------------------------
    const
        Constant beta (one free parameter: beta)

    radial
        Purely radial orbits (beta = 1)

    isotropic
        Isotropic orbits (beta = 0)

    OM
        Osipkov–Merritt model:
            beta(r) = r^2 / (r^2 + r_ani^2)

    GOM
        Generalized Osipkov–Merritt:
            beta(r) = beta_inf * r^2 / (r^2 + r_ani^2)

    Colin
        Colin et al. (2000):
            beta(r) = 0.5 * r / (r + r_ani)

    logistic
        Fully flexible logistic model:
            beta(r) transitions from beta_0 to beta_inf with scale r_ani and slope alpha

    Parameters
    ----------
    anisotropy_type : str
        Anisotropy model type. Must be one of:

        ("const", "radial", "isotropic", "OM", "GOM", "Colin", "logistic")
    """

    def __init__(self, anisotropy_type):
        """

        :param anisotropy_type: string, anisotropy model type
        """
        self._type = anisotropy_type
        self.num_params = None
        self.param_names = None
        self.use_logistic = False
        self._logistic_kwargs = None
        self._constant_beta = None

        if self._type not in self._supported_types:
            raise ValueError(
                f"anisotropy type {self._type} not supported!"
                f"\nchoose from {self._supported_types}"
            )

        if self._type == "const":
            self.num_params = 1
            self.param_names = ["beta"]
            self.use_logistic = False
        elif self._type == "radial":
            self.num_params = 0
            self.param_names = []
            self.use_logistic = False
            self._constant_beta = 1.0
        elif self._type == "isotropic":
            self.num_params = 0
            self.param_names = []
            self.use_logistic = False
            self._constant_beta = 0.0
        elif self._type == "OM":
            # Osipkov&Merrit: beta(r) = r**2 / (r**2 + r_ani**2)
            self.num_params = 1
            self.param_names = ["r_ani"]
            self.use_logistic = True
            self._logistic_kwargs = {"beta_0": 0.0, "beta_inf": 1.0, "alpha": 2.0}
        elif self._type == "GOM":
            # generalized Osipkov&Merrit: beta(r) = beta_inf * r**2 / (r**2 + r_ani**2)
            self.num_params = 2
            self.param_names = ["r_ani", "beta_inf"]
            self.use_logistic = True
            self._logistic_kwargs = {"beta_0": 0.0, "alpha": 2.0}
        elif self._type == "Colin":
            # Colin et al. 2000: beta(r) = 0.5 * r / (r + r_ani)
            self.num_params = 1
            self.param_names = ["r_ani"]
            self.use_logistic = True
            self._logistic_kwargs = {"beta_0": 0.0, "beta_inf": 0.5, "alpha": 1.0}
        elif self._type == "logistic":
            # logistic function anisotropy
            self.num_params = 4
            self.param_names = ["beta_0", "beta_inf", "r_ani", "alpha"]
            self.use_logistic = True
            self._logistic_kwargs = {}

    def beta_params(self, kwargs_anisotropy):
        """
        :param kwargs_anisotropy : dict
            Dictionary containing anisotropy parameters. Only parameters relevant
            for the selected anisotropy model are used.

        :return: float or list
            Anisotropy parameters converted to the Jampy beta(r) format
            - For constant/radial/isotropic models: a scalar beta value.
            - For logistic-type models: list of parameters in JAM format:
              [r_ani, beta_0, beta_inf, alpha].
        """
        anisotropy_params = {
            k: v for k, v in kwargs_anisotropy.items() if k in self.param_names
        }
        if self.use_logistic:
            return self.logistic_function_params(
                **anisotropy_params, **self._logistic_kwargs
            )
        else:
            if self._type == "const":
                constant_beta = anisotropy_params["beta"]
            else:
                constant_beta = self._constant_beta
            return np.minimum(constant_beta, 0.999)  # Jampy requires beta < 1

    @staticmethod
    def logistic_function_params(beta_0, beta_inf, r_ani, alpha):
        """Return the logistic function parameters for Jampy beta(r) format.

        :param beta_0: float, central anisotropy value (beta at r=0)
        :param beta_inf: float, asymptotic anisotropy value (beta at r
        :param r_ani: float, anisotropy radius (scale of transition)
        :param alpha: float, slope of the transition

        :return: list
        """
        return [r_ani, beta_0, beta_inf, alpha]
