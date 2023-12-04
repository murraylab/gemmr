import numpy as np
import xarray as xr

from unittest.mock import patch, create_autospec

from numpy.testing import assert_allclose, assert_array_almost_equal, assert_raises, assert_warns
from xarray.testing import assert_equal as assert_xr_equal, assert_allclose as assert_xr_allclose
from testtools import assert_array_almost_equal_up_to_sign

from scipy.spatial.distance import cdist
from gemmr.estimators import *

import gemmr.generative_model
import gemmr.sample_analysis.analyzers
from gemmr.sample_analysis.analyzers import _calc_loadings, _get_py, _select_n_per_ftrs, \
    _prep_progressbar, _check_model_and_estr, _check_powerlaw_decay


def test_analyze_dataset():

    def addon_test(estr, X, Y, Xorig, Yorig, x_align_ref, y_align_ref, results, **kwargs):
        results['addon_var'] = 3./8

    estr = SVDPLS()
    X = np.arange(10).reshape(5, 2)
    Y = X

    result = gemmr.sample_analysis.analyze_dataset(estr, X, Y, addons=[addon_test])

    assert np.isclose(result.between_corrs_sample, 1.)
    assert np.isclose(result.addon_var, 3./8)

    estr.fit(X, Y)
    assert np.isclose(estr.assocs_[0], result.between_assocs)

    assert np.isclose(np.cov(estr.x_scores_[:, 0], estr.y_scores_[:, 0])[0, 1], result.between_covs_sample)

    sgn = np.sign(result.x_weights.values[0, 0])
    x_tgt_weights = xr.DataArray([[sgn * 1./np.sqrt(2)]*2], dims=('mode', 'x_feature'), coords=dict(x_feature=np.arange(2)), name='x_weights').T
    x_tgt_loadings = xr.DataArray(sgn * np.ones((1, 2), dtype=float), dims=('mode', 'x_feature'), coords=dict(x_feature=np.arange(2)), name='x_loadings').T
    assert_xr_allclose(result.x_weights, x_tgt_weights)
    assert_xr_allclose(result.x_loadings, x_tgt_loadings)

    y_tgt_weights = x_tgt_weights.rename('y_weights').rename(x_feature='y_feature')
    y_tgt_loadings = x_tgt_loadings.rename('y_loadings').rename(x_feature='y_feature')

    assert_xr_allclose(result.y_weights, y_tgt_weights)
    assert_xr_allclose(result.y_loadings, y_tgt_loadings)

    ###

    # assert np.all(result.x_weights.values > 0)
    # assert np.all(result.y_weights.values > 0)
    # i.e. if we now use rerun analyze_dataset with ``?_align_ref`` the weights should be negative
    
    # rerun with x/y_align_ref = sgn*result.x/y_weights and expect weights to have opposite signs    
    result = gemmr.sample_analysis.analyze_dataset(
        estr, X, Y, 
        x_align_ref=sgn * result.x_weights.values, 
        y_align_ref=sgn * result.y_weights.values,
    )
    print(result.x_weights)
    assert_xr_allclose(result.x_weights, sgn * x_tgt_weights)
    assert_xr_allclose(result.x_loadings, sgn * x_tgt_loadings)
    assert_xr_allclose(result.y_weights, sgn * y_tgt_weights)
    assert_xr_allclose(result.y_loadings, sgn * y_tgt_loadings)
    ###

    class MockEstr():
        def fit(self, X, Y):
            raise ValueError()
    mock_estr = MockEstr()
    result = gemmr.sample_analysis.analyze_dataset(mock_estr, X, Y, addons=[addon_test])

    da_nan = xr.DataArray(np.nan * np.empty((2, 1)), dims=('x_feature', 'mode'), coords=dict(x_feature=np.arange(2)))
    target_result = xr.Dataset(dict(
        between_assocs=np.nan,
        between_covs_sample=np.nan,
        between_corrs_sample=np.nan,
        addon_var=3./8,
        x_weights=da_nan,
        y_weights=da_nan.rename(x_feature='y_feature'),
        x_loadings=da_nan.rename(x_feature='x_feature'),
        y_loadings=da_nan.rename(x_feature='y_feature')
    ))
    assert_xr_equal(result, target_result)


mocked_analyze_dataset = create_autospec(
    gemmr.sample_analysis.analyzers.analyze_dataset,
    return_value=xr.Dataset(dict(
        output=xr.DataArray(np.arange(2), dims=('output_dim',))
    ))
)


@patch('gemmr.sample_analysis.analyzers.analyze_dataset', side_effect=mocked_analyze_dataset)
def test_analyze_resampled(mock_analyze_dataset):

    estr = SVDPLS()
    X = np.arange(10).reshape(5, 2)
    Y = X

    result = gemmr.sample_analysis.analyze_resampled(estr, X, Y, perm=0)
    assert mock_analyze_dataset.call_count > 0

    assert_xr_equal(result, mocked_analyze_dataset.return_value)

    n_perm = 3
    result = gemmr.sample_analysis.analyze_resampled(estr, X, Y, perm=n_perm)
    target_result = mocked_analyze_dataset.return_value.copy()
    target_result['output_perm'] = target_result.output.expand_dims(perm=range(n_perm))
    del target_result.coords['perm']
    assert_xr_equal(result, target_result)

    n_bs = 5
    result = gemmr.sample_analysis.analyze_resampled(estr, X, Y, perm=0, n_bs=n_bs, x_align_ref=np.arange(2), y_align_ref=np.arange(2))
    target_result = mocked_analyze_dataset.return_value.copy()
    target_result['output_bs'] = target_result.output.expand_dims(bs=range(n_bs))
    del target_result.coords['bs']
    assert_xr_equal(result, target_result)

    result = gemmr.sample_analysis.analyze_resampled(estr, X, Y, perm=0, loo=True, x_align_ref=np.arange(2), y_align_ref=np.arange(2))
    target_result = mocked_analyze_dataset.return_value.copy()
    target_result['output_loo'] = target_result.output.expand_dims(loo=range(len(X)))
    del target_result.coords['loo']
    assert_xr_equal(result, target_result)


mocked_analyze_resampled = create_autospec(
    gemmr.sample_analysis.analyzers.analyze_resampled,
    return_value=xr.Dataset(dict(
        var1=xr.DataArray(np.arange(2), dims=('dummy',))
    ))
)


@patch('gemmr.sample_analysis.analyzers.analyze_resampled', side_effect=mocked_analyze_resampled)
def test_analyze_subsampled(mock_analyze_resampled):

    estr = SVDPLS()
    ns = np.array([2, 3])
    n_rep = 4

    X = np.arange(10).reshape(5, 2)
    Y = X

    assert_raises(ValueError, gemmr.sample_analysis.analyze_subsampled, estr, X, Y, ns=[], n_rep=2)
    assert_raises(ValueError, gemmr.sample_analysis.analyze_subsampled, estr, X, Y, ns=ns, n_rep=0)
    assert_raises(ValueError, gemmr.sample_analysis.analyze_subsampled, estr, X[:2], Y[:2], ns=ns, n_rep=2)

    result = gemmr.sample_analysis.analyze_subsampled(estr, X, Y, ns=ns, n_rep=n_rep)
    assert mock_analyze_resampled.call_count > 0

    print(result)
    tgt_var1 = xr.DataArray(np.arange(2), dims=('dummy',)).expand_dims(n=np.asarray(ns), rep=np.arange(n_rep))
    del tgt_var1.coords['rep']
    target_ds = xr.Dataset(dict(
        var1=tgt_var1
    ))
    assert_xr_equal(result, target_ds)


@patch('gemmr.sample_analysis.analyzers.analyze_resampled', side_effect=mocked_analyze_resampled)
def test_analyze_model_light(mock_analyze_resampled):

    estr = SVDPLS()
    Sigma = np.eye(4)
    px = 2
    ns = np.array([3,4])
    assert_raises(ValueError, gemmr.sample_analysis.analyze_model_light, estr, Sigma, px, [], n_rep=2)
    assert_raises(ValueError, gemmr.sample_analysis.analyze_model_light, estr, Sigma, px, ns, n_rep=0)

    n_rep=5
    result = gemmr.sample_analysis.analyze_model_light(estr, Sigma, px, ns, n_rep=5)
    assert mock_analyze_resampled.call_count > 0

    tgt_var1 = xr.DataArray(np.arange(2), dims=('dummy',)).expand_dims(n=np.asarray(ns), rep=np.arange(n_rep))
    del tgt_var1.coords['rep']
    target_ds = xr.Dataset(dict(
        var1=tgt_var1
    ))
    assert_xr_equal(result, target_ds)


def test__check_powerlaw_decay():
    class MockRNG:
        def uniform(self, a, b):
            return 3.1
    rng = MockRNG()
    assert_raises(ValueError, next, _check_powerlaw_decay(2, rng, powerlaw_decay=(1,)))
    assert_raises(ValueError, next, _check_powerlaw_decay(2, rng, powerlaw_decay=(1, 2, 3)))
    assert_raises(ValueError, next, _check_powerlaw_decay(2, rng, powerlaw_decay=('random_sum', 4, 3)))

    assert next(_check_powerlaw_decay(2, rng, powerlaw_decay=(3, 1))) == (3, 1)

    axys = [x for x in _check_powerlaw_decay(5, rng, ('random_sum', 3, 4,))]
    assert(len(axys) == 5)
    axy_sums = np.array([np.sum(x) for x in axys])
    assert np.all(axy_sums >= 3)
    assert np.all(axy_sums <= 4)


@patch('gemmr.sample_analysis.analyzers.analyze_resampled', side_effect=mocked_analyze_resampled)
def test_analyze_model_parameters(
        # mock_setup_model,
        mock_analyze_resampled
):

    def mk_test_stats(Xtest, Ytest, U_latent, V_latent):
        return dict(
            test_stat1=2.5,
        )

    def postproc_test(ds):
        ds['postproc'] = 3.1

    rs = np.asarray((1./2, 1./4))
    pxs = np.asarray((8, 9))
    n_per_ftrs = np.asarray((6, 7))
    n_Sigmas = 10
    n_rep = 2
    ax, ay = -.5, -1.5
    kwargs = dict(
        model='cca',
        estr=None,
        n_rep=n_rep,
        n_bs=None,  # tested in analyze_resampled
        n_perm=None,  # tested in analyze_resampled
        n_per_ftrs=n_per_ftrs,
        pxs=pxs,
        pys='px',
        rs=rs,
        n_between_modes=1,
        n_Sigmas=n_Sigmas,
        powerlaw_decay=(ax, ay),
        n_test=2*7*9+1,
        mk_test_statistics=mk_test_stats,
        addons=[],  # test in analyze_dataset
        postprocessors=[postproc_test],
        random_state=0,
        show_progress=False,
    )
    result = gemmr.sample_analysis.analyzers.analyze_model_parameters(**kwargs)

    assert mock_analyze_resampled.call_count > 0

    assert set(result.data_vars.keys()) == set(['var1', 'between_assocs_true', 'between_corrs_true', 'x_weights_true', 'y_weights_true', 'ax', 'ay', 'x_loadings_true', 'x_crossloadings_true', 'y_loadings_true', 'y_crossloadings_true', 'py', 'test_stat1', 'postproc'])
    assert set(result.dims) == set(['Sigma_id', 'dummy', 'mode', 'n_per_ftr', 'px', 'r', 'rep', 'x_feature', 'y_feature'])

    assert np.allclose(result.postproc, 3.1)
    assert_allclose(result.r.values, np.sort(rs))
    assert np.all(result.px == pxs)
    assert np.all(result.n_per_ftr == n_per_ftrs)
    assert np.all(result.Sigma_id == np.arange(n_Sigmas))
    assert np.all(result.x_feature.values == np.arange(np.max(pxs)))
    assert np.all(result.y_feature.values == np.arange(np.max(pxs)))

    assert np.all(result.py.values == pxs)

    target_var1 = xr.DataArray(np.arange(2), dims=('dummy',)).expand_dims(
        px=pxs, r=np.sort(rs), Sigma_id=np.arange(n_Sigmas), n_per_ftr=n_per_ftrs, rep=np.arange(n_rep)
    )
    del target_var1.coords['rep']
    assert_xr_equal(result.var1, target_var1)

    assert result.between_assocs_true.dims == ('px', 'r', 'Sigma_id', 'mode')

    assert result.x_weights_true.dims == ('px', 'r', 'Sigma_id', 'x_feature', 'mode')
    assert result.y_weights_true.dims == ('px', 'r', 'Sigma_id', 'y_feature', 'mode')

    assert result.ax.dims == ('px', 'r', 'Sigma_id',)
    assert result.ay.dims == ('px', 'r', 'Sigma_id',)
    assert_allclose(result.ax.values, ax)
    assert_allclose(result.ay.values, ay)

    assert result.x_loadings_true.dims == ('px', 'r', 'Sigma_id', 'x_feature', 'mode')
    assert result.x_crossloadings_true.dims == ('px', 'r', 'Sigma_id', 'x_feature', 'mode')
    assert result.y_loadings_true.dims == ('px', 'r', 'Sigma_id', 'y_feature', 'mode')
    assert result.y_crossloadings_true.dims == ('px', 'r', 'Sigma_id', 'y_feature', 'mode')

    assert result.test_stat1.dims == ('px', 'r', 'Sigma_id')
    assert np.allclose(result.test_stat1.values, 2.5)

    kwargs['n_test'] = 1
    assert_warns(UserWarning, gemmr.sample_analysis.analyzers.analyze_model_parameters, **kwargs)


def _overlap_sign(a, b):
    overlap = np.dot(a, b)
    if overlap > 0:
        return +1
    elif overlap < 0:
        return -1
    else:
        return 0


def _check_alignment(da, da_true, core_dim):
    signs = xr.apply_ufunc(
        _overlap_sign,
        da,
        da_true,
        input_core_dims=[[core_dim], [core_dim]],
        vectorize=True
    )
    assert_allclose(signs.values, 1)


def test_analyze_model_parameters_align():
    px = 64
    ds = gemmr.sample_analysis.analyzers.analyze_model_parameters(
        'cca',
        n_perm=0,
        n_rep=10,
        n_Sigmas=10,
        n_test=0,  # 1000,
        pxs=[64],
        py='px',
        n_per_ftrs=[2*px+1],
        rs=(.1,),
        powerlaw_decay=(0, 0),
        random_state=0,
        qx=.9,
        qy=.9,
        show_progress=False
    )
    _check_alignment(ds.x_weights, ds.x_weights_true, 'x_feature')
    _check_alignment(ds.y_weights, ds.y_weights_true, 'y_feature')

    px = 64
    ds = gemmr.sample_analysis.analyzers.analyze_model_parameters(
        'pls',
        n_perm=0,
        n_rep=10,
        n_Sigmas=10,
        n_test=0,  # 1000,
        pxs=[64],
        py='px',
        n_per_ftrs=[2*px+1],
        rs=(.1,),
        powerlaw_decay=(0, 0),
        random_state=0,
        qx=.9,
        qy=.9,
        show_progress=False
    )
    _check_alignment(ds.x_weights, ds.x_weights_true, 'x_feature')
    _check_alignment(ds.y_weights, ds.y_weights_true, 'y_feature')


def test__calc_loadings():
    X = np.arange(12.).reshape(4, 3)
    scores = np.arange(8).reshape(4, 2)
    loadings = _calc_loadings(X, scores)
    assert loadings.shape == (3, 2)
    assert_allclose(loadings, 1.)

    X[0, -1] = np.nan
    loadings = _calc_loadings(X, scores)
    assert loadings.shape == (3, 2)
    assert_allclose(loadings, 1.)


def test__prep_progressbar():
    tqdm = _prep_progressbar(show_progress=False)
    assert tqdm('BLA', 1, None, 'str') == 'BLA'

    tqdm = _prep_progressbar(show_progress=True)
    vals = [1, 'str']
    for xi, x in tqdm(enumerate(vals)):
        assert x == vals[xi]


def test__check_model_and_estr():
    class DummyEstr:
        def __init__(self, n_components=1):
            self.n_components = n_components
        def fit(self, X, y):
            return self
    dummy_estr = DummyEstr()

    class DummyCcAEstr:
        def __init__(self, n_components=1):
            self.n_components = n_components
        def fit(self, X, y):
            return self
    dummy_cca = DummyCcAEstr()

    class DummyplsEstr:
        def __init__(self, n_components=1):
            self.n_components = n_components
        def fit(self, X, y):
            return self
    dummy_pls = DummyplsEstr()

    print(dummy_estr.__class__.__name__, dummy_cca.__class__.__name__, dummy_pls.__class__.__name__,)

    assert_raises(ValueError, _check_model_and_estr, 'NOT_A_MODEL', dummy_cca)

    assert isinstance(_check_model_and_estr('cca', None), SVDCCA)
    assert isinstance(_check_model_and_estr('cca', 'auto'), SVDCCA)
    assert isinstance(_check_model_and_estr('pls', None), SVDPLS)
    assert isinstance(_check_model_and_estr('pls', 'auto'), SVDPLS)

    assert isinstance(_check_model_and_estr('pls', 'cCa'), SVDCCA)
    assert isinstance(_check_model_and_estr('cca', 'PlS'), SVDPLS)
    assert_raises(ValueError, _check_model_and_estr, None, 'NOTAESTR')
    if 'SparsePLS' in locals():
        assert isinstance(_check_model_and_estr('cca', 'SPARSEPLS'), SparsePLS)

    assert_raises(ValueError, _check_model_and_estr, 'cca', 'NOTAESTR')
    assert_raises(ValueError, _check_model_and_estr, 'cca', dict())

    assert _check_model_and_estr('cca', dummy_cca) is dummy_cca
    assert _check_model_and_estr('pls', dummy_pls) is dummy_pls

    assert_warns(UserWarning, _check_model_and_estr, 'cca', dummy_pls)
    assert_warns(UserWarning, _check_model_and_estr, 'pls', dummy_cca)
    assert_warns(UserWarning, _check_model_and_estr, 'cca', dummy_estr)
    assert_warns(UserWarning, _check_model_and_estr, 'pls', dummy_estr)


def test__get_py():
    py = _get_py('px', 'DUMMY')
    assert py == 'DUMMY'

    pxs = np.arange(2)
    py = _get_py(lambda x: 2*x, pxs)
    assert_allclose(py, 2*pxs)

    assert _get_py(2.8, pxs) == 2

    assert_raises(ValueError, _get_py, 'NOT_A_NUMBER', pxs)


def test__select_n_per_ftrs():

    n_per_ftrs = _select_n_per_ftrs('DUMMY', None, None, None, None)
    assert n_per_ftrs == 'DUMMY'

    n_per_ftrs = _select_n_per_ftrs(None, None, None, None, None)
    assert n_per_ftrs is None

    n_per_ftrs = _select_n_per_ftrs('auto', None, None, None, r_between=.5)
    n_per_ftrs = np.array(n_per_ftrs)
    assert len(n_per_ftrs) > 0
    assert np.all(n_per_ftrs > 0)

    n_per_ftrs = _select_n_per_ftrs('auto', 'pls', None, None, r_between=.5)
    n_per_ftrs = np.array(n_per_ftrs)
    assert len(n_per_ftrs) > 0
    assert np.all(n_per_ftrs > 0)
