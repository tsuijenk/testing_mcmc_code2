import matplotlib.pyplot as plt
from distributions import *


def forward_generate(N, M, priors, plot=False):
	"""
	Args:
		priors: an instance of a priors class
		N (int): number of data points
		M (int): dimension of data points

	Returns:
		data (np.array): forward generated data
	"""

	# initialize prior distributions

	K = len(priors.alpha)

	pi_dist = DirichletDistribution(priors.alpha)

	sigma_sq_mu_dist = InverseGammaDistribution(alpha=priors.sigma_sq_mu_prior.a, beta=priors.sigma_sq_mu_prior.b)

	sigma_sq_n_dist = InverseGammaDistribution(alpha=priors.sigma_sq_n_prior.a, beta=priors.sigma_sq_n_prior.b)

	# sample from priors

	sigma_sq_mu = sigma_sq_mu_dist.sample()

	sigma_sq_n = sigma_sq_n_dist.sample()

	mu_k = np.zeros(shape=(K, M))

	for k in range(K):

		mu_k[k] = GaussianDistribution(mu_k[k], sigma_sq_mu).sample()

	pi = pi_dist.sample()

	z = np.zeros(shape=(N,), dtype=np.int64)

	X = np.zeros(shape=(N, M))

	for i in range(N):

		z[i] = MultinomialDistribution(pi).sample()

		X[i] = GaussianDistribution(mu_k[z[i]], sigma_sq_n).sample()

	if plot:

		fig, ax = plt.subplots()

		for c in np.unique(z):

			idx = np.where(z == c)[0][0]

			ax.scatter(x=X[idx], y=X[idx])

		plt.show()

	return X
