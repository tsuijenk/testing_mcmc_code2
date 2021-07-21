import numpy as np
from scipy.special import loggamma
from scipy.stats import invgamma


class GaussianDistribution:
    """
    A class for a Gaussian distribution.

    Attributes:
        mu: mean of gaussian distribution
        sigma_sq: variance of gaussian distribution
    """

    def __init__(self, mu, sigma_sq):
        self.mu = mu

        self.sigma_sq = sigma_sq

    def log_p(self, x):
        term0 = -0.5 * np.log(2 * np.pi)

        term1 = -0.5 * np.log(self.sigma_sq)

        term2 = -0.5 * (x - self.mu) ** 2 / self.sigma_sq

        return term0 + term1 + term2

    def sample(self):
        return np.random.normal(loc=self.mu, scale=np.sqrt(self.sigma_sq))


class DirichletDistribution:
    """
    A class for a Dirichlet distribution.

    Attributes:
        alpha: vector of concentration params for Dirichlet distribution
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def log_p(self, x):
        term0 = -np.sum(loggamma(self.alpha))

        term1 = loggamma(np.sum(self.alpha))

        term2 = np.sum((self.alpha - 1) * np.log(x))

        return term0 + term1 + term2

    def sample(self):
        return np.random.dirichlet(self.alpha)


class MultinomialDistribution:
    """
    A class for a Multinomial distribution.

    Attributes:
        p: simplex

    Notes:
        number of trials is set to 1 hence don't include it as an attribute.
    """

    def __init__(self, p):
        self.p = p

    @classmethod
    def from_log_odds(cls, log_odds):
        """
        Args:
            log_odds: vector of log odds (don't need the denominator so values turn into vector of un-normalized log probabilities)

        Returns:
            instance of a Multinomial distribution initialized using log odds
        """

        p = calc_probs(log_odds)

        return cls(p)

    @classmethod
    def from_probabilities(cls, p):
        return cls(p)

    def log_p(self, z):
        return np.log(self.p[z])

    def sample(self, size=1):
        # return np.where(np.random.multinomial(n=1, pvals=self.p))[0][0]
        return np.where(np.random.multinomial(n=1, pvals=self.p, size=size))[1]


class InverseGammaDistribution:
    """
    A class for a Inverse Gamma distribution.

    Attributes:
        alpha: shape of inverse gamma distribution
        beta: scale of inverse gamma distribution
    """

    def __init__(self, alpha, beta):
        self.a = alpha

        self.b = beta

    def log_p(self, x):
        t0 = self.a * np.log(self.b)

        t1 = -loggamma(self.a)

        t2 = -(self.a + 1) * np.log(x)

        t3 = -(self.b / x)

        return t0 + t1 + t2 + t3

    def sample(self, size=1):
        return invgamma.rvs(self.a, scale=self.b, size=size)


def calc_probs(log_p):
    """
    Compute probabilities given log of un-normalized probabilities.

    Arguments:
        log_p (np.array): an array of un-normalized log probs

    Returns:
        p (np.array): an array of probs
    """

    N = log_p.shape[0]

    log_Z_per_N = np.zeros(shape=(N, 1))

    for i in range(N):

        log_Z_per_N[i] = log_norm(log_p[i])

    log_p_new = log_p - log_Z_per_N

    p = np.exp(log_p_new)

    # log_Z = log_norm(log_p)

    # p = np.exp(log_p - log_Z)

    return p


def log_norm(log_x):
    """
    Calculates the log normalization constant, log_Z, given un-normalized log probabilities, log_x,
    using the log-sum-trick.

    Arguments:
        log_x (np.array): an array of un-normalized log probabilities

    Returns:
        log_Z (float): returns log normalization constant of probabilities
    """
    c = np.max(log_x)

    if np.isinf(c):
        return c

    sum_exp = 0

    for x in log_x:
        sum_exp += np.exp(x - c)

    log_sum_exp = np.log(sum_exp)

    log_Z = log_sum_exp + c

    return log_Z
