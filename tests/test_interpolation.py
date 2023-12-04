import numpy as np
import xarray as xr

from numpy.testing import assert_raises, assert_allclose
from xarray.testing import assert_allclose as assert_xr_allclose

from gemmr.sample_size.interpolation import *
from gemmr.sample_size.interpolation import proc_roots, filter_interpol_roots, _calc_n_required
import gemmr.sample_size.interpolation


from unittest.mock import patch, create_autospec


def test_proc_roots():

    class MockInterpolator():
        def __init__(self, roots, fun):
            self._roots = roots
            self.fun = fun
        def __call__(self, x):
            return self.fun(x)
        def roots(self):
            return self._roots

    xs = np.arange(3)
    fun = lambda x: x
    ys = fun(xs)
    interpolator = MockInterpolator(np.arange(3), fun)
    admissible_region = -1

    interpolator._roots = None
    assert_raises(TypeError, proc_roots, xs, ys, interpolator, admissible_region)

    interpolator._roots = 1
    assert_raises(TypeError, proc_roots, xs, ys, interpolator, admissible_region)

    interpolator._roots = [xs[0]]
    assert np.isnan(
        proc_roots(xs, ys, interpolator, admissible_region)
    )

    interpolator._roots = [xs[0], xs[1]]
    assert np.isnan(
        proc_roots(xs, ys, interpolator, admissible_region)
    )

    interpolator._roots = [xs[1]]
    interpolator.fun = lambda x: x-1
    assert np.isnan(
        proc_roots(xs, ys-1, interpolator, admissible_region)
    )

    interpolator._roots = [xs[1]]
    interpolator.fun = lambda x: -x + 1
    assert proc_roots(xs, interpolator.fun(xs), interpolator, admissible_region) == xs[1]

    interpolator._roots = [xs[1], xs[2]]
    assert proc_roots(xs, interpolator.fun(xs), interpolator, admissible_region) == xs[1]


def test_filter_interpol_roots():
    xs = np.arange(13)
    ys = np.array([.1, 0, -.1, -.2, -1, -10, -1, 0, 1, 0, 1, 0, 1, 10])
    class MockInterpolator:
        def __init__(self, ys):
            self.ys = ys
            self._roots = np.where(ys == 0)[0]
        def __call__(self, x):
            return (.5 * x) ** 3
        def roots(self):
            return self._roots
    interpolator = MockInterpolator(ys)
    filtered_roots = filter_interpol_roots(xs, ys, interpolator, verbose=True)
    assert filtered_roots == [1.]


def test_calc_max_n_required():
    assert_raises(ValueError, calc_max_n_required)
    a = xr.DataArray(np.arange(3), dims=('dummy',))
    b = xr.DataArray(np.arange(3)[::-1], dims=('dummy',))
    mx_n_req = calc_max_n_required(2*a, 2*b, a, b)
    assert_xr_allclose(mx_n_req, xr.DataArray([4, 2, 4], dims=('dummy',)))


def test__calc_n_required():
    delta_x = .5
    x = delta_x * np.arange(9.)
    y = np.array([-10, -1, -.9, -.8, -.7, -.1, 0, .1, 0])
    y_target_min = -.1
    y_target_max = .1

    assert_raises(ValueError, _calc_n_required, x, y, 1, -1)

    n_req = _calc_n_required(np.empty(5) * np.nan, np.empty(5) * np.nan, y_target_min, y_target_max, verbose=True)
    assert np.isnan(n_req)

    n_req = _calc_n_required(x[3:6], y[3:6], y_target_min, y_target_max, verbose=True)
    assert np.isnan(n_req)

    n_req = _calc_n_required(x, y+100, y_target_min, y_target_max, verbose=True)
    assert np.isnan(n_req)

    n_req = _calc_n_required(x, y-100, y_target_min, y_target_max, verbose=True)
    assert np.isnan(n_req)

    n_req = _calc_n_required(x, np.zeros_like(x), y_target_min, y_target_max, verbose=True)
    assert np.isclose(n_req, x[0])

    n_req = _calc_n_required(x, y, y_target_min, y_target_max, verbose=True)
    assert np.isclose(n_req, delta_x * 5)

    n_req = _calc_n_required(x, -np.where(y <= 0, y, 0), y_target_min, y_target_max, verbose=True)
    assert np.isclose(n_req, x[5])

    n_req = _calc_n_required(x, np.where(y <= 0, y, 0), y_target_min, y_target_max, verbose=True)
    assert np.isclose(n_req, x[5])

    x[1] = np.nan
    n_req = _calc_n_required(x, y, y_target_min, y_target_max, verbose=True)
    print(n_req)
    assert np.isclose(n_req, x[5])


mocked__calc_n_required = create_autospec(gemmr.sample_size.interpolation._calc_n_required, return_value=3)


@patch('gemmr.sample_size.interpolation._calc_n_required', side_effect=mocked__calc_n_required)
def test_calc_n_required(mock__calc_required):
        #print(mock__calc_required(None, None, None, None))
        #print(gemmr.sample_size.interpolation._calc_n_required(None, None, None, None))
        n_other = 3
        metric = xr.DataArray(np.zeros((2, n_other)), dims=('search_dim', 'other_dim'))
        n_req = calc_n_required(metric, np.nan, np.nan, 'search_dim')
        assert_allclose(n_req.values, np.exp(mocked__calc_n_required.return_value))
        assert mock__calc_required.call_count == n_other
        assert n_req.name == 'search_dim_required'

        assert_raises(KeyError, calc_n_required, metric, np.nan, np.nan, 'not_a_dim')


def test_calc_n_required_all_metrics():
    ds = xr.Dataset()
    tmp = xr.DataArray(.5 * np.ones((2, 3)), dims=('search_dim', 'other'))
    ds['power'] = tmp
    tmp = tmp.expand_dims(rep=np.arange(4))
    ds['between_assocs'] = tmp
    ds['between_assocs_true'] = tmp
    ds['x_weights_true_cossim'] = tmp
    ds['y_weights_true_cossim'] = tmp
    ds['x_test_scores_true_pearson'] = tmp
    ds['y_test_scores_true_pearson'] = tmp
    ds['x_test_loadings_true_pearson'] = tmp
    ds['y_test_loadings_true_pearson'] = tmp
    assert_raises(ValueError, calc_n_required_all_metrics, ds, search_dim='not_a_dim')
    n_reqs = calc_n_required_all_metrics(ds, search_dim='search_dim')
    assert sorted(list(n_reqs.keys())) == sorted(['power', 'betweenAssoc', 'weightError', 'loadingError', 'scoreError', 'combined'])
