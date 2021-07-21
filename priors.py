from collections import namedtuple


InverseGammaPriors = namedtuple("InverseGammaPriors", ["a", "b"])


class Priors:

    def __init__(self, alpha, sigma_sq_mu_prior, sigma_sq_n_prior):
        """
        A priors class for the gaussian mixture model.

        Arguments:
            alpha (list): param for Dirichlet prior over pi
            sigma_sq_mu_prior (list): params for InverseGamma prior over sigma_sq_mu
            sigma_sq_n_prior (list): params for InverseGamma prior over sigma_sq_n
        """

        self.alpha = alpha

        self.sigma_sq_mu_prior = InverseGammaPriors(a=sigma_sq_mu_prior[0], b=sigma_sq_mu_prior[1])

        self.sigma_sq_n_prior = InverseGammaPriors(a=sigma_sq_n_prior[0], b=sigma_sq_n_prior[1])