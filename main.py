from priors import *
from data import *
from model import *

# forward generate data
K = 4

priors = Priors(alpha=[1] * K,
                sigma_sq_mu_prior=[1, 1],
                sigma_sq_n_prior=[1, 1])

X = forward_generate(N=50,
                     M=2,
                     priors=priors,
                     plot=True)

# perform inference

model = Model(priors=priors, K=K)

state = State(z=np.ones(shape=(X.shape[0],), dtype=int),
              mu=np.zeros(shape=(K,)),
              sigma_sq_mu=np.ones(shape=(K,)),
              sigma_sq_n=1,
              pi=np.ones(shape=(K,)) / K)

model.gibbs_step(state=state, X=X)

# post-processing

model.plot()
