from model import *
from priors import *
from data import *
from distributions import *
import numpy as np
import copy

K = 4

D = 2

N = 50


def setup():
    priors = Priors(alpha=[1] * K,
                    sigma_sq_mu_prior=[1, 1],
                    sigma_sq_n_prior=[1, 1])

    X = forward_generate(N=N,
                         M=D,
                         priors=priors,
                         plot=False)

    model = Model(priors=priors, K=K)

    state = State(z=np.zeros(shape=(X.shape[0],), dtype=int),
                  mu=np.zeros(shape=(K, X.shape[1])),
                  sigma_sq_mu=np.array([1]),
                  sigma_sq_n=np.array([1]),
                  pi=np.ones(shape=(K,)) / K)

    return model, X, state


def test_cond_mu():
    # model = random_model()
    # TODO: implement random_model fxn

    test_model, X, state = setup()
    new_state = copy.deepcopy(state)
    new_state.mu = np.random.normal(size=(K, D))
    cond = test_model.cond_mu(state, X)

    val0 = test_model.joint_log_p(new_state, X)

    val1 = test_model.joint_log_p(state, X)

    val2 = cond.log_p(new_state.mu).sum()

    val3 = cond.log_p(state.mu).sum()

    assert np.allclose(cond.log_p(new_state.mu).sum() - cond.log_p(state.mu).sum(),
                       test_model.joint_log_p(new_state, X) - test_model.joint_log_p(state, X))


def test_cond_sigma_sq_n():
    # model = random_model()
    # TODO: implement random_model fxn

    test_model, X, state = setup()
    new_state = copy.deepcopy(state)
    new_state.sigma_sq_n = InverseGammaDistribution(1, 1).sample(size=1)
    cond = test_model.cond_sigma_sq_n(state=state, X=X)

    assert np.allclose(cond.log_p(new_state.sigma_sq_n) - cond.log_p(state.sigma_sq_n),
                       test_model.joint_log_p(new_state, X) - test_model.joint_log_p(state, X))


def test_cond_sigma_sq_mu():
    # model = random_model()
    # TODO: implement random_model fxn

    test_model, X, state = setup()
    new_state = copy.deepcopy(state)
    new_state.sigma_sq_mu = InverseGammaDistribution(1, 1).sample(size=1)
    cond = test_model.cond_sigma_sq_mu(state=state)

    assert np.allclose(cond.log_p(new_state.sigma_sq_mu) - cond.log_p(state.sigma_sq_mu),
                       test_model.joint_log_p(new_state, X) - test_model.joint_log_p(state, X))


def test_cond_pi():
    # model = random_model()
    # TODO: implement random_model fxn

    test_model, X, state = setup()
    new_state = copy.deepcopy(state)
    new_state.pi = np.random.dirichlet([1] * K)
    cond = test_model.cond_pi(state)

    assert np.allclose(cond.log_p(new_state.pi) - cond.log_p(state.pi),
                       test_model.joint_log_p(new_state, X) - test_model.joint_log_p(state, X))


def test_cond_z():
    # model = random_model()
    # TODO: implement random_model fxn

    test_model, X, state = setup()
    new_state = copy.deepcopy(state)
    random_pvals = np.random.dirichlet([1] * K)
    # new_state.z = np.random.multinomial(N, random_pvals, size=(N,))
    new_state.z = MultinomialDistribution.from_probabilities(random_pvals).sample(N)
    cond = test_model.cond_z(state, X)

    assert np.allclose(cond.log_p(new_state.z).sum() - cond.log_p(state.z).sum(),
                       test_model.joint_log_p(new_state, X) - test_model.joint_log_p(state, X))

# test_cond_mu()

# test_cond_sigma_sq_n()

test_cond_z()
