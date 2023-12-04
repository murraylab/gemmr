import numpy as np
import xarray as xr


from gemmr.metrics import *


def _setup_ds(data_vars, value_range):
    ds = xr.Dataset()
    np.random.seed(0)
    for v in data_vars:
        ds[v] = xr.DataArray(
            np.random.uniform(*value_range, size=(3, 4)),
            dims=('a', 'b')
        )
    return ds


def _test_metric(metric, ds, lo=0, up=1):
    e = metric(ds)
    assert np.all(lo <= e.values) and np.all(e.values <= up)


# def test_mk_latentCovRelError():
#     ds = _setup_ds(['true_covs'], (0, 1))
#     ds['sample_svals'] = 1.1 * ds['true_covs']
#     _test_metric(mk_latentCovRelError, ds, 0, np.inf)


def test_mk_betweenAssocRelError():
    ds = _setup_ds(['between_assocs_true'], (0, 1))
    ds['between_assocs'] = 1.1 * ds['between_assocs_true']
    _test_metric(mk_betweenAssocRelError, ds, 0, np.inf)


def test_mk_betweenAssocRelError_cv():
    ds = _setup_ds(['between_assocs_true'], (0, 1))
    ds['between_assocs_cv'] = 1.1 * ds['between_assocs_true']
    _test_metric(lambda ds: mk_betweenAssocRelError_cv(ds, 'between_assocs_cv'), ds, 0, np.inf)


def test_mk_meanBetweenAssocRelError():
    ds = _setup_ds(['between_assocs_true'], (0, 1))
    ds['between_assocs'] = 1.03 * ds['between_assocs_true']
    ds['between_assocs_cv'] = 1.1 * ds['between_assocs_true']
    _test_metric(lambda ds: mk_meanBetweenAssocRelError(ds, 'between_assocs_cv'), ds, 0, np.inf)


def test_mk_weightError():
    ds = _setup_ds(['x_weights_true_cossim', 'y_weights_true_cossim'], (-1, 1))
    _test_metric(mk_weightError, ds, 0, 1)
    ds = ds.rename(**{
        'x_weights_true_cossim': 'x_weights_true_cossim_perm',
        'y_weights_true_cossim': 'y_weights_true_cossim_perm',
    })
    _test_metric(lambda ds: mk_weightError(ds, '_perm'), ds, 0, np.inf)


def test_mk_scoreError():
    ds = _setup_ds(['x_test_scores_true_pearson', 'y_test_scores_true_pearson'], (-1, 1))
    _test_metric(mk_scoreError, ds, 0, 1)


def test_mk_loadingError():
    ds = _setup_ds(['x_test_loadings_true_pearson', 'y_test_loadings_true_pearson'], (-1, 1))
    _test_metric(mk_loadingError, ds, 0, 1)


def test_mk_crossloadingError():
    ds = _setup_ds(['x_test_crossloadings_true_pearson', 'y_test_crossloadings_true_pearson'], (-1, 1))
    _test_metric(mk_crossloadingError, ds, 0, 1)
