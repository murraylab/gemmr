import numpy as np
import xarray as xr

from scipy.spatial.distance import cdist

from numpy.testing import assert_raises, assert_array_almost_equal, \
    assert_equal
from xarray.testing import assert_allclose as assert_xr_allclose

import gemmr.generative_model
from gemmr.data import generate_example_dataset
from gemmr.estimators import SVDCCA, SVDPLS
from gemmr.util import _check_float, check_positive_definite, align_weights, \
    nPerFtr2n, rank_based_inverse_normal_trafo, pc_spectrum_decay_constant, \
    _calc_true_loadings

from testtools import assert_array_almost_equal_up_to_sign

def test_check_float():
    _check_float(1, 'jk', .5, 1.5, 'inclusive')  # this should work
    _check_float(-1, 'jk', -1.5, .5, 'inclusive')  # this should work
    assert_raises(ValueError, _check_float, 0, 'alekj', 1, 2, 'inclusive')
    assert_raises(NotImplementedError, _check_float, -2, 'wer', 0, 1, 'OTHER_BOUNDARIES')


def test_check_positive_defintie():
    noise_factor = 1e-6  # default value in _check_positive_definite
    noise_level = 1e-10
    m = np.zeros((3, 3))
    assert_raises(ValueError, check_positive_definite, m, noise_level)
    m[0, 0] = -noise_level
    assert_raises(ValueError, check_positive_definite, m, noise_level)
    m = np.diag([1. * noise_factor * noise_level * 3])
    check_positive_definite(m, noise_level, noise_factor)  # this should work and not through an exception


def test_align_weights():
    n_ftrs = 3

    vtrue = np.arange(n_ftrs, dtype=float)
    vtrue /= np.linalg.norm(vtrue)
    v = np.arange(n_ftrs * 2, dtype=float).reshape(-1, n_ftrs)
    assert_raises(ValueError, align_weights, v, vtrue)  # test if vectors in v are unit vectors

    vtrue = np.arange(n_ftrs, dtype=float)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    assert_raises(ValueError, align_weights, v, vtrue)  # test if vtrue is unit vectors

    vtrue = vtrue / np.linalg.norm(vtrue)
    vtrue[-1] *= -1

    assert_raises(ValueError, align_weights, v, vtrue[:, np.newaxis])  # test if vtrue has a single dimension
    assert_raises(ValueError, align_weights, v[:, :2], vtrue)  # test if last dim of v has same length as vtrue

    aligned = align_weights(v, vtrue)
    aligned_one = align_weights(v[0], vtrue)
    assert_array_almost_equal(aligned[[0]], aligned_one)

    aligned_nd = align_weights(v[np.newaxis, np.newaxis, ...], vtrue)
    assert_array_almost_equal(aligned, aligned_nd)

    aligned, signs = align_weights(v, vtrue, return_sign=True)
    assert np.allclose(signs, -1)  # as vtrue[-1] is negative (see above) all signs should be negative


def test_nPerFtr2n():

    da = xr.DataArray(
        1.5 * np.ones((2, 3, 4)),
        dims=('px', 'r', 'Sigma_id'),
        coords=dict(px=[1, 2])
    )
    py = xr.DataArray(
        2 * np.ones((2, 3, 4)),
        dims=('px', 'r', 'Sigma_id'),
        coords=dict(px=[1, 2])
    ).astype(int)

    assert_raises(NotImplementedError, nPerFtr2n, da.sel(px=1), py)

    n_req = nPerFtr2n(da, py)
    assert n_req.name == 'n_required'
    assert (n_req.dims == np.array(['ptot', 'r', 'Sigma_id'])).all()

    assert np.allclose(n_req.sel(ptot=3).values, 3 * 1.5)
    assert np.allclose(n_req.sel(ptot=4).values, 4 * 1.5)

    ###

    da['py'] = py
    n_req_2 = nPerFtr2n(da)  # n_req_2 should be identical to n_req except that it has an additional attribute py
    del n_req_2['py']
    assert_xr_allclose(n_req, n_req_2)


def test_rank_based_inverse_normal_trafo():
    np.random.seed(0)

    x = np.random.normal(size=10000000)
    out = rank_based_inverse_normal_trafo(x)

    xq = np.quantile(x, [.025, .25, .5, .75, .975])
    outq = np.quantile(out, [.025, .25, .5, .75, .975])

    assert_array_almost_equal(xq, outq, decimal=3)

    # test NaNs in input data

    x = np.arange(15.).reshape(-1, 3)
    x[1, 1] = np.nan
    x[3, 2] = np.nan
    x[4, 2] = np.nan
    out = rank_based_inverse_normal_trafo(x)
    print(out)
    assert np.isnan(out[1, 1]) & np.isnan(out[3,2]) & np.isnan(out[4, 2])
    print(np.mean(np.ma.masked_invalid(out), axis=0))
    assert_array_almost_equal(
        np.mean(np.ma.masked_invalid(out), axis=0),
        0
    )

    x = np.random.normal(size=(10, 3))
    x[1, 1] = np.nan
    x[3, 2] = np.nan
    x[8, 2] = np.nan
    out = rank_based_inverse_normal_trafo(x)
    assert np.isnan(out[1, 1]) & np.isnan(out[3,2]) & np.isnan(out[8, 2])
    assert_array_almost_equal(
       np.mean(np.ma.masked_invalid(out), axis=0),
       0
    )

    for col in range(3):
        out_mask = np.isfinite(out[:, col])

        out_col = rank_based_inverse_normal_trafo(x[:, col])
        out_col_mask = np.isfinite(out_col)

        assert_equal(out_mask, out_col_mask)
        assert_array_almost_equal(
            out[:,col][out_mask],
            out_col[out_col_mask]
        )

        x_col = x[:, col]
        out2 = rank_based_inverse_normal_trafo(x_col[np.isfinite(x_col)])
        assert_array_almost_equal(
            out2,
            out[:, col][out_mask]
        )


def test_pc_spectrum_decay_constant():

    def _check(ax_hat, ax):
        print(ax_hat, ax)
        assert np.isclose(ax_hat, ax, rtol=1e-2, atol=1e-2)

    ax, ay = -.5, -1.5
    X, Y = generate_example_dataset('cca', px=5, py=10, ax=ax, ay=ay, n=100000)
    _check(pc_spectrum_decay_constant(X, expl_var_ratios=(.99,))[0], ax)
    _check(pc_spectrum_decay_constant(Y, expl_var_ratios=(.99,))[0], ay)

    ax, ay = -1.2, -0.3
    X, Y = generate_example_dataset('pls', px=13, py=3, ax=ax, ay=ay, n=100000)
    _check(pc_spectrum_decay_constant(X, expl_var_ratios=(.99,))[0], ax)
    _check(pc_spectrum_decay_constant(Y, expl_var_ratios=(.99,))[0], ay)


def test__calc_true_loadings():
    px, py = 3, 5
    for model, estr in [
        ('cca', SVDCCA()),
        ('pls', SVDPLS())
    ]:
        gm = gemmr.generative_model.GEMMR(model, wx=px, wy=py)

        X, Y = gm.generate_data(1000000, random_state=0)
        estr.fit(X, Y)
        lXX = 1 - cdist(X.T, estr.x_scores_.T, metric='correlation')
        lXY = 1 - cdist(X.T, estr.y_scores_.T, metric='correlation')
        lYX = 1 - cdist(Y.T, estr.x_scores_.T, metric='correlation')
        lYY = 1 - cdist(Y.T, estr.y_scores_.T, metric='correlation')

        true_loadings = _calc_true_loadings(gm.Sigma_, px,
                                            gm.x_weights_, gm.y_weights_)

        decimal = 2
        assert_array_almost_equal_up_to_sign(lXX, true_loadings['x_loadings_true'], decimal=decimal)
        assert_array_almost_equal_up_to_sign(lXY, true_loadings['x_crossloadings_true'], decimal=decimal)
        assert_array_almost_equal_up_to_sign(lYX, true_loadings['y_crossloadings_true'], decimal=decimal)
        assert_array_almost_equal_up_to_sign(lYY, true_loadings['y_loadings_true'], decimal=decimal)