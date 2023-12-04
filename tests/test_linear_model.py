import warnings

import numpy as np
import xarray as xr
from numpy.testing import assert_raises, assert_allclose

from sklearn.linear_model import LinearRegression

from gemmr.data import load_outcomes
from gemmr.sample_size.linear_model import *
from gemmr.sample_size.linear_model import do_fit_lm, _check_pxy, \
    _check_axy, get_lm_coefs


def test__do_fit_lm():
    tmp = xr.DataArray(np.ones((2,3,4)), dims=('r', 'px', 'Sigma_id',), coords={'r': [.1, .2]})
    n_reqs = xr.DataArray(np.arange(2*3*4).reshape((2,3,4))+1, dims=('r', 'px', 'Sigma_id',), coords={'r': [.1, .2]})
    ds = xr.Dataset(dict(
        py=tmp,
        ax=-tmp,
        ay=-tmp,
        latent_expl_var_ratios_x=tmp,#.expand_dims(mode=[0]),
        latent_expl_var_ratios_y=tmp,#.expand_dims(mode=[0]),
        between_corrs_true=tmp.r
    ))
    lm, X, y, coeff_names = do_fit_lm(
        ds, n_reqs, include_pc_var_decay_constants=True, include_latent_explained_vars=False, verbose=True
    )
    assert isinstance(lm, LinearRegression)
    assert coeff_names[0] == 'const'
    assert len(coeff_names) == 4

    lm, X, y, coeff_names = do_fit_lm(
        ds, n_reqs, include_pc_var_decay_constants=False, include_latent_explained_vars=False
    )
    assert len(coeff_names) == 3


def test_fit_linear_model():

    def _check_lm(lm, n_coefs):
        assert lm.intercept_ > 0
        assert len(lm.coef_) == n_coefs

    try:
        load_outcomes('cca', fetch=False)
    except:
        warnings.warn("Couldn't load outcome data for CCA")
    else:
        _check_lm(fit_linear_model('combined', 'cca'), 2)
        _check_lm(fit_linear_model('power', 'cca'), 2)
        _check_lm(fit_linear_model('weight', 'cca'), 2)
        _check_lm(fit_linear_model('combined', 'pls'), 3)
        assert_raises(ValueError, fit_linear_model, 'NOTAMETRIC', 'cca')


def test_save_linear_models():
    pass  # Nothing to test (?)


def test__save_linear_model():
    pass  # Nothing to test (?)


def test__check_pxy():
    _check_pxy(2, 2)
    assert_raises(ValueError, _check_pxy, 1, 2)
    assert_raises(ValueError, _check_pxy, 2, 1)
    assert_raises(ValueError, _check_pxy, 2, 3.2)
    assert_raises(ValueError, _check_pxy, 4.5, 3)
    assert_raises(ValueError, _check_pxy, '2', 3)
    assert_raises(ValueError, _check_pxy, 3, None)

    px, py = _check_pxy(np.arange(3).reshape(1, -1), np.arange(4).reshape(1, -1))
    assert (px == 3) and (py == 4)


def test__check_axy():
    ax, ay = _check_axy(None, None, -.5, -.2)
    assert_raises(ValueError, _check_axy, None, None, -.5, 2)
    assert_raises(ValueError, _check_axy, None, None, 2, -.2)
    assert_raises(ValueError, _check_axy, None, None, '2', 3)
    assert_raises(ValueError, _check_axy, None, None, 3, None)

    assert_raises(ValueError, _check_axy, 2, 3, None, None)

    np.random.seed(0)
    n = 128
    X = np.random.normal(size=(n, 10))
    Y = np.random.normal(size=(n, 10))
    assert_raises(ValueError, _check_axy, X, Y, -.5, -.2)
    ax, ay = _check_axy(X, Y, None, None)
    assert isinstance(ax, float)
    assert isinstance(ay, float)


def test__get_lm_coefs():

    def _check(res, n_coefs):
        intercept, coefs = res
        assert intercept > 0
        assert len(coefs) == n_coefs

    _check(get_lm_coefs('cca', 'combined', target_error=.1, target_power=0.9, data_home=None), 3)
    _check(get_lm_coefs('pls', 'combined', target_error=.1, target_power=0.9, data_home=None), 3)

    try:
        load_outcomes('cca', fetch=False)
    except:
        warnings.warn("Couldn't load outcome data for CCA")
    else:
        _check(get_lm_coefs('cca', 'combined', target_error=.1, target_power=0.8, data_home=None), 2)
        _check(get_lm_coefs('cca', 'weight', target_error=.1, target_power=0.8, data_home=None), 2)

    try:
        load_outcomes('pls', fetch=False)
    except:
        warnings.warn("Couldn't load outcome data for PLS")
    else:
        _check(get_lm_coefs('pls', 'combined', target_error=.1, target_power=0.8, data_home=None), 3)
        _check(get_lm_coefs('pls', 'weight', target_error=.1, target_power=0.8, data_home=None), 3)


def test_cca_sample_size():
    # assert_raises(NotImplementedError, cca_sample_size, 2, 3, target_power=.5)
    # assert_raises(NotImplementedError, cca_sample_size, 2, 3, target_error=.5)

    assert_raises(NotImplementedError, cca_sample_size, 2, 3, -1, -1.2, algorithm='generative_model')
    assert_raises(ValueError, cca_sample_size, 2, 3, 0, -3, algorithm='invalid_algorithm')

    rs = (1./2, 1./4, 3./4)
    res = cca_sample_size(2, 3, -.1, -10.2, rs=rs)
    assert_allclose(np.asarray(sorted(list(res.keys()))), np.asarray(sorted(rs)))
    assert np.all(np.array(list(res.values())) > 0)

    np.random.seed(0)
    X = np.random.normal(size=(5, 2))
    Y = np.random.normal(size=(5, 3))
    res2 = cca_sample_size(X, Y, rs=rs)
    for r in rs:
        assert r in res2


def test_pls_sample_size():
    # assert_raises(NotImplementedError, pls_sample_size, 2, 3, -.5, -1.2, target_power=.5)
    # assert_raises(NotImplementedError, pls_sample_size, 2, 3, -.5, -1.2, target_error=.5)

    assert_raises(NotImplementedError, pls_sample_size, 2, 3, -.5, -1.2, algorithm='generative_model')
    assert_raises(ValueError, pls_sample_size, 2, 3, -.5, -1.2, algorithm='invalid_algorithm')

    rs = (1. / 2, 1. / 4, 3. / 4)
    res = pls_sample_size(2, 3, -.5, -1.2, rs=rs)
    assert_allclose(np.asarray(sorted(list(res.keys()))), np.asarray(sorted(rs)))
    assert np.all(np.array(list(res.values())) > 0)

    np.random.seed(0)
    X = np.random.normal(size=(5, 2))
    Y = np.random.normal(size=(5, 3))
    res2 = pls_sample_size(X, Y, rs=rs)
    for r in rs:
        assert r in res2


def test_cca_req_corr():

    assert_raises(NotImplementedError, cca_req_corr, 2, 3, -1, -1, 10, algorithm='generative_model')
    assert_raises(ValueError, cca_req_corr, 2, 3, -1, -1, 10, algorithm='invalid_algorithm')

    r_req = cca_req_corr(2, 3, -.8, -1.1, 0)
    assert 0 <= r_req <= 1

    r_req = cca_req_corr(2, 3, -.2, -10, 1e6)
    assert 0 <= r_req <= 1


def test_pls_req_corr():

    assert_raises(NotImplementedError, pls_req_corr, 2, 3, -.5, -1.2, 10, algorithm='generative_model')
    assert_raises(ValueError, pls_req_corr, 2, 3, -.5, -1.2, 10, algorithm='invalid_algorithm')

    r_req = pls_req_corr(2, 3, -.5, -1.2, 0)
    assert 0 <= r_req <= 1

    r_req = pls_req_corr(2, 3, -.5, -1.2, 1e6)
    assert 0 <= r_req <= 1
