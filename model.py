import numpy as np

from distributions import *


class Model:
    """
    A class for a gaussian mixture model.

    Attributes:
        priors: an instance of the Priors class
        K: number clusters
    """

    def __init__(self, priors, K):

        self.alpha = priors.alpha

        # self.sigma_sq_mu_prior = priors.sigma_sq_mu_prior
        #
        # self.sigma_sq_n_prior = priors.sigma_sq_n_prior

        self.sigma_sq_mu_prior = InverseGammaDistribution(priors.sigma_sq_mu_prior.a,
                                                          priors.sigma_sq_mu_prior.b)

        self.sigma_sq_n_prior = InverseGammaDistribution(priors.sigma_sq_n_prior.a,
                                                         priors.sigma_sq_n_prior.b)

        self.K = K

    def gibbs_step(self, state, X):

        state.pi = self.cond_pi(state).sample()

        state.pi = self.cond_z(state, X).sample()

        state.mu = self.cond_mu(state, X).sample()

        state.sigma_sq_mu = self.cond_sigma_sq_mu(state).sample()

        state.sigma_sq_n = self.cond_sigma_sq_n(state, X).sample()

    def cond_pi(self, state):

        counts = np.bincount(state.z)

        counts.resize(self.K)

        return DirichletDistribution(self.alpha + counts)

    # def cond_z(self, state, X):
    #
    #     prior = np.log(state.pi)
    #
    #     evidence = GaussianDistribution(state.mu, state.sigma_sq_n).log_p(X).sum()
    #
    #     return MultinomialDistribution.from_log_odds(prior + evidence)

    def cond_z(self, state, X):
        nax = np.newaxis
        prior = np.log(state.pi)
        evidence = GaussianDistribution(state.mu[nax, :, :], state.sigma_sq_n).log_p(X[:, nax, :]).sum(2)
        return MultinomialDistribution.from_log_odds(prior[nax, :] + evidence)

    def cond_mu(self, state, X):

        ndata, ndim = X.shape

        h = np.zeros((self.K, ndim))

        lam = np.zeros((self.K, ndim))

        for k in range(self.K):

            idxs = np.where(state.z == k)[0]

            if idxs.size > 0:

                h[k, :] = X[idxs, :].sum(0) / state.sigma_sq_n

                lam[k, :] = idxs.size / state.sigma_sq_n + 1. / state.sigma_sq_mu

            else:

                h[k, :] = 0.

                lam[k, :] = 1. / state.sigma_sq_mu

        return GaussianDistribution(h / lam, 1. / lam)

    def cond_sigma_sq_mu(self, state):

        ndim = state.mu.shape[1]

        a = self.sigma_sq_mu_prior.a + 0.5 * self.K * ndim

        b = self.sigma_sq_mu_prior.b + 0.5 * np.sum(state.mu ** 2)

        return InverseGammaDistribution(a, b)

    def cond_sigma_sq_n(self, state, X):

        ndata, ndim = X.shape

        a = self.sigma_sq_n_prior.a + 0.5 * ndata * ndim

        b = self.sigma_sq_n_prior.b + 0.5 * np.sum((X - state.mu[state.z, :]) ** 2)

        return InverseGammaDistribution(a, b)

    def joint_log_p(self, state, X):

        term0 = self.sigma_sq_mu_prior.log_p(state.sigma_sq_mu)

        term1 = self.sigma_sq_n_prior.log_p(state.sigma_sq_n)

        term2 = DirichletDistribution(self.alpha * np.ones(self.K)).log_p(state.pi)

        term3 = MultinomialDistribution.from_probabilities(state.pi).log_p(state.z).sum()

        term4 = GaussianDistribution(0., state.sigma_sq_mu).log_p(state.mu).sum()

        term5 = GaussianDistribution(state.mu[state.z, :], state.sigma_sq_n).log_p(X).sum()

        return term0 + term1 + term2 + term3 + term4 + term5

    def plot(self):

        raise NotImplementedError


class State:
    """
    A class for the current state of the model

    Attributes:
        pi: mixture probabilities
        sigma_sq_mu: between-cluster variance
        sigma_sq_n: within-cluster variance
        z: cluster assignments
        mu: cluster means
    """

    def __init__(self, z, mu, sigma_sq_mu, sigma_sq_n, pi):
        self.z = z

        self.mu = mu

        self.sigma_sq_mu = sigma_sq_mu

        self.sigma_sq_n = sigma_sq_n

        self.pi = pi
