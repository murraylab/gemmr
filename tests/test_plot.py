import numpy as np
import xarray as xr

from numpy.testing import assert_raises

from gemmr.plot import *
from gemmr.plot import hv

hv.extension('matplotlib')


def test_mean_metric_curve():
    rs = np.asarray((.1, .2))
    n_per_ftrs=np.asarray((1,2,3))
    n_req = xr.DataArray(np.arange(6).reshape(2, 3), dims=('r', 'n_per_ftr'), coords=dict(r=rs, n_per_ftr=n_per_ftrs))
    assert isinstance(mean_metric_curve(n_req, rs=rs), hv.Overlay)

    n_req = xr.DataArray(np.arange(24).reshape(2, 3, 4), dims=('r', 'n_per_ftr', 'other'), coords=dict(r=rs, n_per_ftr=n_per_ftrs))
    assert isinstance(mean_metric_curve(n_req, rs=rs), hv.Overlay)

    n_req = xr.DataArray(np.arange(24).reshape(2, 3, 4), dims=('r', 'n_per_ftr', 'other'), coords=dict(r=rs, n_per_ftr=n_per_ftrs))
    assert isinstance(mean_metric_curve(n_req, rs=rs, ylabel='bla'), hv.Overlay)

    assert isinstance(mean_metric_curve(n_req, rs=rs, n_per_ftr_typical=None), hv.Overlay)


def test_heatmap_n_req():
    n_req = xr.DataArray(np.arange(6).reshape(2, 3), dims=('ptot', 'r'))
    assert isinstance(heatmap_n_req(n_req), hv.QuadMesh)

    n_req = xr.DataArray(np.arange(24).reshape(2, 3, 4), dims=('ptot', 'r', 'other'))
    assert isinstance(heatmap_n_req(n_req), hv.QuadMesh)

    n_req = xr.DataArray(np.arange(6).reshape(2, 3), dims=('not_ptot', 'r'))
    assert_raises(ValueError, heatmap_n_req, n_req)

    n_req = xr.DataArray(np.arange(6).reshape(2, 3), dims=('ptot', 'not_r'))
    assert_raises(ValueError, heatmap_n_req, n_req)


def test_polar_hist():
    np.random.seed(0)
    assert isinstance(polar_hist(np.random.uniform(0, 3, size=32)), hv.Overlay)
    assert isinstance(polar_hist(np.random.uniform(0, 3, size=32), bins='semicircle'), hv.Overlay)
