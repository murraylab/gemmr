import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold

from numpy.testing import assert_allclose, assert_raises
from xarray.testing import assert_allclose as assert_xr_allclose

from unittest.mock import patch, create_autospec

from scipy.stats import pearsonr

from gemmr.generative_model import generate_data
from gemmr.util import _calc_true_loadings
from gemmr.sample_analysis.addon import *
# don't confuse pytest:
del test_scores, test_scores_true_spearman, test_scores_true_pearson, test_loadings, test_loadings_true_pearson
from gemmr.sample_analysis.addon import \
    test_scores as calc_test_scores, \
    test_scores_true_spearman as calc_test_scores_true_spearman, \
    test_loadings_true_pearson as calc_test_loadings_true_pearson


def test_remove_weights_loadings():
    ds = xr.Dataset(dict(
        x_weights=xr.DataArray(np.arange(6).reshape(2, 3), dims=('a', 'b')),
        y_weights=xr.DataArray(np.arange(6).reshape(2, 3), dims=('a', 'b')),
        x_loadings=xr.DataArray(np.arange(6).reshape(2, 3), dims=('a', 'b')),
        y_loadings=xr.DataArray(np.arange(6).reshape(2, 3), dims=('a', 'b')),
    ))
    remove_weights_loadings(None, None, None, None, None, None, None, ds)
    assert 'x_weights' not in ds
    assert 'y_weights' not in ds
    assert 'x_loadings' not in ds
    assert 'y_loadings' not in ds


def test_remove_cv_weights_loadings():
    ds = xr.Dataset(dict(
        x_weights_cv=xr.DataArray(np.arange(6).reshape(2, 3), dims=('a', 'b')),
        y_weights_cv=xr.DataArray(np.arange(6).reshape(2, 3), dims=('a', 'b')),
        x_loadings_cv=xr.DataArray(np.arange(6).reshape(2, 3), dims=('a', 'b')),
        y_loadings_cv=xr.DataArray(np.arange(6).reshape(2, 3), dims=('a', 'b')),
    ))
    remove_cv_weights_loadings(None, None, None, None, None, None, None, ds)
    assert 'x_weights_cv' not in ds
    assert 'y_weights_cv' not in ds
    assert 'x_loadings_cv' not in ds
    assert 'x_loadings_cv' not in ds


def test_weights_true_cossim():
    x_weights_true = np.array([1, 0]).reshape(-1, 1)
    x_weights = np.array([.1, np.sqrt(1-.1**2)]).reshape(-1, 1)
    results = xr.Dataset(dict(
        x_weights=xr.DataArray(x_weights, dims=('x_feature', 'mode')),
        y_weights=xr.DataArray(x_weights, dims=('y_feature', 'mode')),
    ))
    weights_true_cossim(None, None, None, None, None, x_weights_true, x_weights_true, results)
    assert np.allclose(results.x_weights_true_cossim, .1)
    assert np.allclose(results.y_weights_true_cossim, .1)


def test_test_scores():
    class MockEstr():
        def transform(self, X, Y):
            return X, Y
    estr = MockEstr()
    Xtest = np.array([
        [0, 1, 2]
    ]).T
    Ytest = Xtest

    results = xr.Dataset()

    assert_raises(KeyError, calc_test_scores, estr, None, None, None, None, None, None, results, Xtest=Xtest)
    assert_raises(KeyError, calc_test_scores, estr, None, None, None, None, None, None, results, Ytest=Ytest)

    calc_test_scores(estr, None, None, None, None, None, None, results, Xtest=Xtest, Ytest=Ytest)
    assert results.x_test_scores.dims == ('test_sample', 'mode')
    assert results.y_test_scores.dims == ('test_sample', 'mode')
    assert_allclose(results.x_test_scores.values, Xtest)
    assert_allclose(results.y_test_scores.values, Xtest)

    results = xr.Dataset()
    calc_test_scores(estr, None, None, None, None, None, None, results, Xtest=Xtest[:0], Ytest=Ytest[:0])
    assert 'x_test_scores' not in results
    assert 'y_test_scores' not in results


def test_test_scores_true_spearman():
    class MockEstr():
        def transform(self, X, Y):
            return X, Y
    estr = MockEstr()
    Xtest = np.array([
        [0, 1, 2]
    ]).T
    Ytest = Xtest

    test_statistics = mk_test_statistics_scores(Xtest, Xtest, np.eye(1), np.eye(1))
    results = xr.Dataset()
    assert_raises(KeyError, calc_test_scores_true_spearman, estr, None, None, None, None, None, None, results, Xtest=Xtest, Ytest=Xtest)
    assert_raises(KeyError, calc_test_scores_true_spearman, estr, None, None, None, None, None, None, results, test_statistics=test_statistics)

    # required to run before scores_true_spearman
    calc_test_scores(estr, None, None, None, None, None, None, results, Xtest=Xtest, Ytest=Ytest)
    calc_test_scores_true_spearman(estr, None, None, None, None, None, None, results, Xtest=Xtest, Ytest=Xtest, test_statistics=test_statistics)
    assert results.x_test_scores_true_spearman == 1.
    assert results.y_test_scores_true_spearman == 1.


def test_test_loadings_true_pearson():

    px, py = 5, 6
    pmin = min(px, py)
    Sigmaxy = np.zeros((px, py))
    Sigmaxy.flat[::py+1] = 1./np.arange(1, px+1)
    Sigma = np.vstack([
        np.hstack([np.eye(px), Sigmaxy]),
        np.hstack([Sigmaxy.T, np.eye(py)])
    ])
    Sigma[0, px] = Sigma[px, 0] = 1
    Sigma[1, px+1] = Sigma[px+1, 1] = .1
    U_latent = np.eye(px)[:, :pmin]
    V_latent = np.eye(py)[:, :pmin]
    Xtest, Ytest = generate_data(Sigma, px, n=100000, random_state=0)

    true_loadings = _calc_true_loadings(Sigma, px, U_latent, V_latent)

    x_test_scores = np.dot(Xtest, U_latent)
    y_test_scores = np.dot(Ytest, V_latent)

    results = xr.Dataset()

    assert_raises(KeyError, calc_test_loadings_true_pearson, None, None, None, None, None, None, None, results,
                  true_loadings=true_loadings, Xtest=Xtest, Ytest=Xtest)

    results['x_test_scores'] = xr.DataArray(
        x_test_scores, dims=('test_subject', 'mode'),
    )
    results['y_test_scores'] = xr.DataArray(
        y_test_scores, dims=('test_subject', 'mode'),
    )

    assert_raises(KeyError, calc_test_loadings_true_pearson, None, None, None, None, None, None, None, results,
                  true_loadings=true_loadings)
    assert_raises(KeyError, calc_test_loadings_true_pearson, None, None, None, None, None, None, None, results,
                  Xtest=Xtest, Ytest=Ytest)

    assert_raises(ValueError, calc_test_loadings_true_pearson, None, Xtest, None, None, None, None, None, results,
                  true_loadings=true_loadings, Xtest=Xtest, Ytest=Xtest)

    calc_test_loadings_true_pearson(None, Xtest, None, None, None, None, None, results,
                               true_loadings=true_loadings, Xtest=Xtest, Ytest=Ytest)

    assert_allclose(results.x_test_loadings_true_pearson.values, 1, rtol=1e-2)
    assert_allclose(results.y_test_loadings_true_pearson.values, 1, rtol=1e-2)
    assert_allclose(results.x_test_crossloadings_true_pearson.values, 1, rtol=1e-2)
    assert_allclose(results.y_test_crossloadings_true_pearson.values, 1, rtol=1e-2)


def test_remove_test_scores():
    ds = xr.Dataset(dict(
        x_test_scores=xr.DataArray(np.arange(6).reshape(2, 3), dims=('a', 'b')),
        y_test_scores=xr.DataArray(np.arange(6).reshape(2, 3), dims=('a', 'b')),
    ))
    remove_test_scores(None, None, None, None, None, None, None, ds)
    assert 'x_test_scores' not in ds
    assert 'y_test_scores' not in ds


def test_assoc_test():
    class MockEstr():
        def transform(self, X, Y):
            return X, Y
    estr = MockEstr()
    np.random.seed(0)
    n = 16
    Xtest = np.random.normal(size=(n, 1))
    Ytest = np.random.normal(size=(n, 1))
    results = xr.Dataset()
    assoc_test(estr, None, None, None, None, None, None, results,
               Xtest=Xtest, Ytest=Ytest)
    assert np.isclose(
        float(results['between_assoc_test']),
        pearsonr(Xtest[:, 0], Ytest[:, 0])[0]
    )


class MockPCA():
    def fit(self, X):
        print('mock called')
        self.components_ = np.eye(X.shape[1])
        return self


@patch('gemmr.sample_analysis.addon.PCA', return_value=MockPCA())
def test_weights_pc_cossim(MockPCA):

    np.random.seed(0)
    X = np.random.normal(size=(10000, 2))
    Y = X
    weights = xr.DataArray([
        [1, 0],
        [1./np.sqrt(2), 1./np.sqrt(2)],
        [3/5, 4/5],
    ], dims=('mode', 'x_feature')).T
    results = xr.Dataset(dict(
        x_weights=weights,
        y_weights=weights.rename(x_feature='y_feature')
    ))

    weights_pc_cossim(None, X, Y, None, None, None, None, results)

    target_da = xr.DataArray([
        [1, 1./np.sqrt(2), 3./5],
        [0, 1./np.sqrt(2), 4./5]
    ], dims=('x_pc', 'mode'), coords=dict(x_pc=np.arange(2)))

    assert_xr_allclose(results.x_weights_pc_cossim, target_da)
    assert_xr_allclose(results.y_weights_pc_cossim, target_da.rename(x_pc='y_pc'))


def test_sparseCCA_penalties():
    class MockEstr:
        pass
    estr = MockEstr()
    estr.penaltyx_ = .1
    estr.penaltyy_ = .2
    results = xr.Dataset()
    sparseCCA_penalties(estr, None, None, None, None, None, None, results)
    assert results['x_penalty'] == .1
    assert results['y_penalty'] == .2


def test_mk_scorers_for_cv():
    class MockEstr:
        def transform(self, x, y):
            return x, y
    estr = MockEstr()
    scorers = mk_scorers_for_cv(n_between_modes=1)
    x = np.arange(10)
    y = np.arange(10)**2
    assert scorers['cov_m0'](estr, x.reshape(-1, 1), y.reshape(-1, 1)) == np.cov(x, y)[0, 1]
    assert scorers['corr_m0'](estr, x.reshape(-1, 1), y.reshape(-1, 1)) == pearsonr(x, y)[0]


def test_cv():
    class MockEstr(BaseEstimator):
        def __init__(self, n_components=1):
            self.n_components = n_components
        def fit(self, X, Y):
            self.x_rotations_ = np.arange(2).reshape(-1, self.n_components)
            self.y_rotations_ = self.x_rotations_
            return self
        def transform(self, X, Y):
            return X, Y
    estr = MockEstr()
    X = np.array([
        [-1, 0, 1, 2],
        [-1, 0, 1, 2]
    ]).T
    Y = X
    results = xr.Dataset()
    cvs = [
        ('kfold2', KFold(2))
    ]
    scorers = mk_scorers_for_cv(n_between_modes=1)

    assert_raises(KeyError, cv, estr, X, Y, None, None, None, None, results, cvs=cvs)
    assert_raises(KeyError, cv, estr, X, Y, None, None, None, None, results, scorers=scorers)
    cv(estr, X, Y, None, None, None, None, results, cvs=cvs, scorers=scorers)
    assert 'between_covs_cv' in results
    assert results['between_corrs_cv'].sel(cv='kfold2', mode=0) == 1

    assert_allclose(results.x_weights_cv.sel(cv='kfold2', mode=0).values, [np.arange(2)]*2)
    assert_allclose(results.y_weights_cv.sel(cv='kfold2', mode=0).values, [np.arange(2)]*2)

    assert results.x_weights_cv.dims == ('cv', 'fold', 'x_feature', 'mode')
    assert results.y_weights_cv.dims == ('cv', 'fold', 'y_feature', 'mode')


def test_mk_test_statistics_scores():
    Xtest = np.array([
        [-1, 0, 1],
        [-1, 0, 1]
    ]).T
    U_latent = np.eye(2)
    test_stats = mk_test_statistics_scores(Xtest, Xtest, U_latent, U_latent)

    assert_allclose(test_stats['x_test_scores_true'].values, Xtest[:, 0])
    assert_allclose(test_stats['y_test_scores_true'].values, Xtest[:, 0])
    assert test_stats['x_test_scores_true'].dims[0] == 'test_sample'
    assert test_stats['y_test_scores_true'].dims[0] == 'test_sample'

    # assert_allclose(test_stats['x_test_loadings_true'].values, 1)
    # assert_allclose(test_stats['y_test_loadings_true'].values, 1)
    # assert test_stats['x_test_loadings_true'].dims[0] == 'x_orig_feature'
    # assert test_stats['y_test_loadings_true'].dims[0] == 'y_orig_feature'
    #
    # assert_allclose(test_stats['x_test_crossloadings_true'].values, 1)
    # assert_allclose(test_stats['y_test_crossloadings_true'].values, 1)
    # assert test_stats['x_test_crossloadings_true'].dims[0] == 'x_orig_feature'
    # assert test_stats['y_test_crossloadings_true'].dims[0] == 'y_orig_feature'
