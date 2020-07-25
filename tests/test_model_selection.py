import numpy as np
from numpy.testing import assert_raises

from gemmr.model_selection import *


def test_max_min_detector():

    np.random.seed(0)
    n = 10000
    signal = np.cos(np.arange(n)).reshape(-1, 1)
    X = np.c_[signal, signal, np.random.normal(size=(n, 2))]
    Y = np.c_[np.random.normal(size=(n, 1)), signal, signal, np.random.normal(size=(n, 2))]

    assert_raises(ValueError, max_min_detector, X, Y, 4)
    pXs, pYs, d, best_s = max_min_detector(X, Y, 3)
    assert pXs == [2]
    assert pYs == [2]
    assert d == 1


def test_n_components_to_explain_variance():
    n_ftrs = 10
    evals = np.arange(1, n_ftrs+1)[::-1]
    cumsum = np.cumsum(evals)
    eval_sum = np.sum(evals)
    covariance_matrix = np.diag(evals)
    assert_raises(ValueError, n_components_to_explain_variance, covariance_matrix, -.1)
    assert_raises(ValueError, n_components_to_explain_variance, covariance_matrix, 1.1)
    n_comps = np.array([n_components_to_explain_variance(covariance_matrix, cumsum[i]/eval_sum) for i in range(n_ftrs)])
    n_comps_true = np.arange(1, n_ftrs+1)

    cov_nonsymmetric = np.arange(4).reshape(2, 2)
    assert_raises(ValueError, n_components_to_explain_variance, cov_nonsymmetric)

    assert np.all(n_comps == n_comps_true)