import numpy as np
import xarray as xr

from numpy.testing import assert_allclose, assert_raises

from gemmr.estimators import SVDPLS
from gemmr.data import generate_example_dataset
from gemmr.sample_analysis.macros import *


class MockEstr():
    def fit(self, X, Y):
        self.assocs_ = np.arange(2)
        return self


def test_calc_p_value():
    estr = MockEstr()
    X = np.arange(6).reshape(3, 2).astype(float)
    Y = X
    n_permutations=3
    p_value = calc_p_value(estr, X, Y, permutations=n_permutations)
    assert np.isclose(p_value, 1.)


def test_analyze_subsampled_and_resampled():
    estr = SVDPLS()
    X, Y = generate_example_dataset('pls', n=24)
    assert_raises(ValueError, analyze_subsampled_and_resampled, estr, X, Y, n_test_subsample=4)
    res = analyze_subsampled_and_resampled(estr, X, Y, permutations=3, n_perm_subsample=3, n_rep_subsample=3)
    assert isinstance(res['full_sample'], xr.Dataset)
    assert isinstance(res['subsampled'], xr.Dataset)
    assert 'p_value' in res['full_sample']
    assert 'p_value' not in res['subsampled']


def test_pairwise_weight_cosine_similarity():
    weights = xr.DataArray(np.arange(3), dims=('x_feature',)).expand_dims(n=[3], other=[14, 15])

    print(weights)

    ds = xr.Dataset(dict(
        x_weights=weights,
        y_weights=weights.rename(x_feature='y_feature')
    ))
    xy_weight_sims_mean, xy_weight_sims_q = pairwise_weight_cosine_similarity(ds)
    assert_allclose(xy_weight_sims_mean.values, 1)
    assert_allclose(xy_weight_sims_q.values, 1)
