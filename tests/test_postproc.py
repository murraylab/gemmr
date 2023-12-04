import numpy as np
import xarray as xr

from numpy.testing import assert_allclose
from xarray.testing import assert_allclose as assert_xr_allclose

from gemmr.sample_analysis.postproc import *


def test_power():
    ds = xr.Dataset(dict(
        between_assocs=xr.DataArray(np.linspace(0, 1, 10).reshape(1, -1), dims=('dummy', 'rep')),
        between_assocs_perm=xr.DataArray(.05 * np.ones((1, 10, 100)), dims=('dummy', 'rep', 'perm')),
    ))

    alpha = 0.05
    power(ds, alpha=alpha)

    p_values = np.array([1] + [.01]*9)
    true_power = xr.DataArray(
        np.array([(p_values < alpha).mean()]),
        dims=('dummy',)
    )

    assert_xr_allclose(ds.power, true_power)


def test_remove_between_assocs_perm():
    ds = xr.Dataset(dict(
        between_assocs_perm=xr.DataArray(np.arange(6).reshape(2,3), dims=('a', 'b'))
    ))
    remove_between_assocs_perm(ds)
    assert 'between_assocs_perm' not in ds


def test_weights_pairwise_cossim_stats():
    np.random.seed(0)
    weights = np.linalg.qr(np.random.normal(size=(3, 3)))[0]
    ds = xr.Dataset(dict(
        x_weights=xr.DataArray(weights, dims=('rep', 'x_feature')),
        y_weights=xr.DataArray(weights, dims=('rep', 'y_feature')),
    ))
    weights_pairwise_cossim_stats(ds, qs=(.025, .5, .975))
    stats_true = xr.DataArray(np.zeros(4), dims=('stat',), coords=dict(stat=['q2.5%', 'q50.0%', 'q97.5%', 'mean']))
    assert_xr_allclose(ds.x_weights_pairwise_cossim_stats, stats_true)
    assert_xr_allclose(ds.y_weights_pairwise_cossim_stats, stats_true)


def test_scores_pairwise_spearmansim_stats():
    np.random.seed(0)
    scores = np.array([[.1, .2], [1, 2]])
    ds = xr.Dataset(dict(
        x_test_scores=xr.DataArray(scores, dims=('rep', 'test_sample')),
        y_test_scores=xr.DataArray(scores, dims=('rep', 'test_sample')),
    ))
    scores_pairwise_spearmansim_stats(ds, qs=(.025, .5, .975))
    stats_true = xr.DataArray(np.ones(4), dims=('stat',), coords=dict(stat=['q2.5%', 'q50.0%', 'q97.5%', 'mean']))
    assert_xr_allclose(ds.x_scores_pairwise_spearmansim_stats, stats_true)
    assert_xr_allclose(ds.y_scores_pairwise_spearmansim_stats, stats_true)


def test_remove_weights_loadings():
    ds = xr.Dataset(dict(
        x_weights=xr.DataArray(np.arange(6).reshape(2, 3), dims=('a', 'b')),
        y_weights=xr.DataArray(np.arange(6).reshape(2, 3), dims=('a', 'b')),
        x_loadings=xr.DataArray(np.arange(6).reshape(2, 3), dims=('a', 'b')),
        y_loadings=xr.DataArray(np.arange(6).reshape(2, 3), dims=('a', 'b')),
    ))
    remove_weights_loadings(ds)
    assert 'x_weights' not in ds
    assert 'y_weights' not in ds
    assert 'x_loadings' not in ds
    assert 'y_loadings' not in ds


def test_remove_test_scores():
    ds = xr.Dataset(dict(
        x_test_scores=xr.DataArray(np.arange(6).reshape(2, 3), dims=('a', 'b')),
        y_test_scores=xr.DataArray(np.arange(6).reshape(2, 3), dims=('a', 'b')),
    ))
    remove_test_scores(ds)
    assert 'x_test_scores' not in ds
    assert 'y_test_scores' not in ds


def test_bs_quantiles():
    ds = xr.Dataset(dict(
        some=xr.DataArray(np.arange(3).reshape(1, 3), dims=('a', 'b')),
        some_bs=xr.DataArray(np.arange(100).reshape(1, -1), dims=('a', 'bs')),
    ))
    qs = (.025, .5, .975)
    bs_quantiles(ds, qs=qs)
    assert 'some' in ds
    assert 'some_stats' not in ds
    assert 'some_bs' in ds
    assert 'bs' in ds.dims  # dimension name of 'some_bs'
    assert 'some_bs_stats' in ds
    assert ds['some_bs_stats'].dims == ('a', 'stat')
    assert len(ds['stat']) == len(qs) + 1
    
    
def test_remove_bs_datavars():
    ds = xr.Dataset(dict(
        some=xr.DataArray(np.arange(6).reshape(2, 3), dims=('a', 'b')),
        other=xr.DataArray(np.arange(6).reshape(2, 3), dims=('a', 'b')),
        some_bs=xr.DataArray(np.arange(6).reshape(2, 3), dims=('a', 'bs')),
        other_bs=xr.DataArray(np.arange(6).reshape(2, 3), dims=('a', 'bs')),
    ))
    remove_bs_datavars(ds)
    assert 'some' in ds
    assert 'other' in ds
    assert 'some_bs' not in ds
    assert 'other_bs' not in ds
    assert 'bs' not in ds