class JAMAnisotropy:
    _supported_types = ("const", "radial", "isotropic", "OM", "GOM", "Colin", "logistic")

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
            raise ValueError("anisotropy type %s not supported!" % self._type)

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
            # Osipkov&Merrit
            self.num_params = 1
            self.param_names = ["r_ani"]
            self.use_logistic = True
            self._logistic_kwargs = {"beta_0": 0.0, "beta_inf": 1.0, "alpha": 2.0}
        elif self._type == "GOM":
            # generalized Osipkov&Merrit: beta(r) = beta_inf * r**2 / (r**2 + r_ani**2)
            self.num_params = 2
            self.param_names = ["r_ani", "beta_inf"]
            self.use_logistic = True
            self._logistic_kwargs = {"beta_0": 0.0, "beta_inf": 0.5, "alpha": 2.0}
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

    def beta_params(self, kwargs_anisotropy, n_gauss=None):
        """
        anisotropy parameters converted to the JAM beta(r) format
        it can be a logistic function or one beta per MGE component
        """
        anisotropy_params = {k: v for k, v in kwargs_anisotropy.items() if k in self.param_names}
        if self.use_logistic:
            return self.logistic_function_params(**anisotropy_params, **self._logistic_kwargs)
        else:
            if self._type == "constant":
                constant_beta = anisotropy_params["beta"]
            else:
                constant_beta = self._constant_beta
            return [constant_beta] * n_gauss

    @staticmethod
    def logistic_function_params(beta_0, beta_inf, r_ani, alpha):
        """
        return the logistic function parameters for JAM beta(r) format
        """
        return [beta_0, beta_inf, r_ani, alpha]
