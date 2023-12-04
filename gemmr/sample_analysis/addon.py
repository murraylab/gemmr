"""Add-ons for sample analyzers.
"""

import warnings

from functools import partial

import numpy as np
import xarray as xr
from scipy.spatial.distance import cosine as cos_dist, cdist
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate

from .analyzers import _calc_loadings
from ..estimators.helpers import cov_transform_scorer, pearson_transform_scorer


__all__ = [
    'remove_weights_loadings',
    'remove_cv_weights_loadings',
    'weights_true_cossim',
    'test_scores',
    'test_scores_true_spearman',
    'test_scores_true_pearson',
    'test_loadings',
    'test_loadings_true_pearson',
    'loadings_true_pearson',
    'remove_test_scores',
    'assoc_test',
    'weights_pc_cossim',
    'loadings_pc_cossim',
    'loadings_pc_pearsonsim',
    'orig_loadings_pc_cossim',
    'orig_loadings_pc_pearsonsim',
    'sparseCCA_penalties',
    'mk_scorers_for_cv',
    'cv',
    'mk_test_statistics_scores'
]


def _unitvec(x):
    return x / np.linalg.norm(x)


def remove_weights_loadings(estr, X, Y, Xorig, Yorig, x_align_ref, y_align_ref,
                            results, **kwargs):
    """Removes weights and loadings from ``results`` dataset to save storage
    space.

    Parameters
    ----------
    estr : **sklearn**-style estimator
        fitted estimator
    X : np.ndarray (n_samples, n_features)
        dataset `X`
    Y : np.ndarray (n_samples, n_features)
        dataset `Y`
    Xorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `Y`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``Y`` and ``Yorig`` correspond to the same samples
        (subjects).
    x_align_ref : (n_features,)
        the sign of `X` weights is chosen such that the cosine-distance between
        fitted `X` weights and ``x_align_ref`` is positive
    y_align_ref : (n_features,)
        the sign of `Y` weights is chosen such that the cosine-distance between
        fitted `Y` weights and ``y_align_ref`` is positive
    results : xr.Dataset
        containing outcome features computed so far, and is modified with
        outcomes of this function
    kwargs : dict
        keyword arguments
    """
    del results['x_weights']
    del results['y_weights']
    del results['x_loadings']
    del results['y_loadings']

    try:
        del results['x_orig_loadings']
    except KeyError:
        pass

    try:
        del results['y_orig_loadings']
    except KeyError:
        pass


def remove_cv_weights_loadings(estr, X, Y, Xorig, Yorig,
                               x_align_ref, y_align_ref, results, **kwargs):
    """Removes ``x_weights_cv``, ``y_weights_cv``, ``x_loadings_cv`` and
     ``y_loadings_cv`` from ``results`` dataset to save storage space.

    ``x_weights_cv``, ``y_weights_cv``, ``x_loadings_cv`` and
     ``y_loadings_cv`` are created by :func:`addana_cv`

    Parameters
    ----------
    estr : **sklearn**-style estimator
        fitted estimator
    X : np.ndarray (n_samples, n_features)
        dataset `X`
    Y : np.ndarray (n_samples, n_features)
        dataset `Y`
    Xorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `Y`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``Y`` and ``Yorig`` correspond to the same samples
        (subjects).
    x_align_ref : (n_features,)
        the sign of `X` weights is chosen such that the cosine-distance between
        fitted `X` weights and ``x_align_ref`` is positive
    y_align_ref : (n_features,)
        the sign of `Y` weights is chosen such that the cosine-distance between
        fitted `Y` weights and ``y_align_ref`` is positive
    results : xr.Dataset
        containing outcome features computed so far, and is modified with
        outcomes of this function
    kwargs : dict
        keyword arguments
    """

    del results['x_weights_cv']
    del results['y_weights_cv']
    del results['x_loadings_cv']
    del results['y_loadings_cv']


def weights_true_cossim(estr, X, Y, Xorig, Yorig, x_align_ref, y_align_ref,
                        results, **kwargs):
    """Calculates cosine-distance between estimated and true weights.

    Provides outcome metrics ``x_weights_true_cossim`` and
    ``y_weights_true_cossim``

    Parameters
    ----------
    estr : **sklearn**-style estimator
        fitted estimator
    X : np.ndarray (n_samples, n_features)
        dataset `X`
    Y : np.ndarray (n_samples, n_features)
        dataset `Y`
    Xorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `Y`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``Y`` and ``Yorig`` correspond to the same samples
        (subjects).
    x_align_ref : (n_features,)
        the sign of `X` weights is chosen such that the cosine-distance between
        fitted `X` weights and ``x_align_ref`` is positive
    y_align_ref : (n_features,)
        the sign of `Y` weights is chosen such that the cosine-distance between
        fitted `Y` weights and ``y_align_ref`` is positive
    results : xr.Dataset
        containing outcome features computed so far, and is modified with
        outcomes of this function
    kwargs : dict
        keyword arguments
    """
    results['x_weights_true_cossim'] = 1 - cos_dist(
        results['x_weights'].values[:, 0],
        x_align_ref[:, 0]
    )
    results['y_weights_true_cossim'] = 1 - cos_dist(
        results['y_weights'].values[:, 0],
        y_align_ref[:, 0]
    )


def test_scores(estr, X, Y, Xorig, Yorig, x_align_ref, y_align_ref,
                results, **kwargs):
    """Calculates test scores.

    Requires keyword arguments:

    * kwargs['Xtest']
    * kwargs['Ytest']

    Provides outcome metrics ``x_test_scores``, ``y_test_scores``.
    Outcome is calculated only if ``len(Xtest) > 0``.

    Parameters
    ----------
    estr : **sklearn**-style estimator
        fitted estimator
    X : np.ndarray (n_samples, n_features)
        dataset `X`
    Y : np.ndarray (n_samples, n_features)
        dataset `Y`
    Xorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `Y`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``Y`` and ``Yorig`` correspond to the same samples
        (subjects).
    x_align_ref : (n_features,)
        the sign of `X` weights is chosen such that the cosine-distance between
        fitted `X` weights and ``x_align_ref`` is positive
    y_align_ref : (n_features,)
        the sign of `Y` weights is chosen such that the cosine-distance between
        fitted `Y` weights and ``y_align_ref`` is positive
    results : xr.Dataset
        containing outcome features computed so far, and is modified with
        outcomes of this function
    kwargs : dict
        keyword arguments
    """
    Xtest = kwargs['Xtest']
    Ytest = kwargs['Ytest']

    if len(Xtest) > 0:

        try:
            Xtest_t, Ytest_t = estr.transform(Xtest, Ytest)

        except ValueError:
            Xtest_t = np.nan * np.empty((len(Xtest), estr.n_components))
            Ytest_t = np.nan * np.empty((len(Xtest), estr.n_components))

        results['x_test_scores'] = xr.DataArray(
            Xtest_t, dims=('test_sample', 'mode'),
            coords=dict(test_sample=np.arange(len(Xtest)))
        )
        results['y_test_scores'] = xr.DataArray(
            Ytest_t, dims=('test_sample', 'mode'),
            coords=dict(test_sample=np.arange(len(Xtest)))
        )


def test_scores_true_spearman(estr, X, Y, Xorig, Yorig, x_align_ref, y_align_ref,
                              results, **kwargs):
    """Calculates Spearman correlations between estimated and true test scores.

    Requires that addon :func:`test_scores` is run before.

    Requires keyword arguments:

    * kwargs['test_statistics']['x_test_scores_true']
    * kwargs['test_statistics']['x_test_scores_true']

    The latter two can be created by :func:`mk_test_statistics_scores`.

    Provides outcome metrics ``x_test_scores_true_spearman`` and
    ``y_test_scores_true_spearman``

    Parameters
    ----------
    estr : **sklearn**-style estimator
        fitted estimator
    X : np.ndarray (n_samples, n_features)
        dataset `X`
    Y : np.ndarray (n_samples, n_features)
        dataset `Y`
    Xorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `Y`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``Y`` and ``Yorig`` correspond to the same samples
        (subjects).
    x_align_ref : (n_features,)
        the sign of `X` weights is chosen such that the cosine-distance between
        fitted `X` weights and ``x_align_ref`` is positive
    y_align_ref : (n_features,)
        the sign of `Y` weights is chosen such that the cosine-distance between
        fitted `Y` weights and ``y_align_ref`` is positive
    results : xr.Dataset
        containing outcome features computed so far, and is modified with
        outcomes of this function
    kwargs : dict
        keyword arguments
    """
    x_test_scores_true = kwargs['test_statistics']['x_test_scores_true'].values
    y_test_scores_true = kwargs['test_statistics']['y_test_scores_true'].values

    results['x_test_scores_true_spearman'] = spearmanr(
        results['x_test_scores'][:, 0],
        x_test_scores_true
    )[0]
    results['y_test_scores_true_spearman'] = spearmanr(
        results['y_test_scores'][:, 0],
        y_test_scores_true
    )[0]


def test_scores_true_pearson(estr, X, Y, Xorig, Yorig, x_align_ref, y_align_ref,
                             results, **kwargs):
    """Calculates Pearson correlations between estimated and true test scores.

    Requires that addon :func:`test_scores` is run before.

    Requires keyword arguments:

    * kwargs['test_statistics']['x_test_scores_true']
    * kwargs['test_statistics']['x_test_scores_true']

    The latter two can be created by :func:`mk_test_statistics_scores`.

    Provides outcome metrics ``x_test_scores_true_spearman`` and
    ``y_test_scores_true_spearman``

    Parameters
    ----------
    estr : **sklearn**-style estimator
        fitted estimator
    X : np.ndarray (n_samples, n_features)
        dataset `X`
    Y : np.ndarray (n_samples, n_features)
        dataset `Y`
    Xorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `Y`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``Y`` and ``Yorig`` correspond to the same samples
        (subjects).
    x_align_ref : (n_features,)
        the sign of `X` weights is chosen such that the cosine-distance between
        fitted `X` weights and ``x_align_ref`` is positive
    y_align_ref : (n_features,)
        the sign of `Y` weights is chosen such that the cosine-distance between
        fitted `Y` weights and ``y_align_ref`` is positive
    results : xr.Dataset
        containing outcome features computed so far, and is modified with
        outcomes of this function
    kwargs : dict
        keyword arguments
    """
    x_test_scores_true = kwargs['test_statistics']['x_test_scores_true'].values
    y_test_scores_true = kwargs['test_statistics']['y_test_scores_true'].values

    results['x_test_scores_true_pearson'] = pearsonr(
        results['x_test_scores'][:, 0],
        x_test_scores_true
    )[0]
    results['y_test_scores_true_pearson'] = pearsonr(
        results['y_test_scores'][:, 0],
        y_test_scores_true
    )[0]


def test_loadings(estr, X, Y, Xorig, Yorig, x_align_ref, y_align_ref,
                          results, **kwargs):
    """Calculates test loadings with respect to (possibly transformed) data
    variables, i.e. the variables of X, Y (not Xorig, Yorig).

    Requires results:

    * ``x_test_scores``
    * ``x_test_scores``

    These are available if :func:`test_scores` was run before.

    Requires keyword arguments:

    * Xtest, Ytest

    Provides outcome metrics ``x_test_loadings``,
    ``y_test_loadings``, ``x_test_crossloadings``
    and ``y_test_crossloadings``.

    Parameters
    ----------
    estr : **sklearn**-style estimator
        fitted estimator
    X : np.ndarray (n_samples, n_features)
        dataset `X`
    Y : np.ndarray (n_samples, n_features)
        dataset `Y`
    Xorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `Y`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``Y`` and ``Yorig`` correspond to the same samples
        (subjects).
    x_align_ref : (n_features,)
        the sign of `X` weights is chosen such that the cosine-distance between
        fitted `X` weights and ``x_align_ref`` is positive
    y_align_ref : (n_features,)
        the sign of `Y` weights is chosen such that the cosine-distance between
        fitted `Y` weights and ``y_align_ref`` is positive
    results : xr.Dataset
        containing outcome features computed so far, and is modified with
        outcomes of this function
    kwargs : dict
        keyword arguments
    """
    x_test_scores = results['x_test_scores'].values
    y_test_scores = results['y_test_scores'].values

    #print(estr.x_rotations_[:, 0].shape, x_align_ref.shape)
    #sgn = np.sign(estr.x_rotations_[:, 0] @ x_align_ref)

    Xtest = kwargs['Xtest']
    Ytest = kwargs['Ytest']

    n = len(X)
    x_coords = dict(x_feature=np.arange(Xtest.shape[1]))
    y_coords = dict(y_feature=np.arange(Ytest.shape[1]))
    results['x_test_loadings'] = xr.DataArray(
        _calc_loadings(Xtest[:n], x_test_scores[:n]),
        dims=('x_feature', 'mode'), coords=x_coords
    )
    results['y_test_loadings'] = xr.DataArray(
        _calc_loadings(Ytest[:n], y_test_scores[:n]),
        dims=('y_feature', 'mode'), coords=y_coords
    )
    results['x_test_crossloadings'] = xr.DataArray(
        _calc_loadings(Xtest[:n], y_test_scores[:n]),
        dims=('x_feature', 'mode'), coords=x_coords
    )
    results['y_test_crossloadings'] = xr.DataArray(
        _calc_loadings(Ytest[:n], x_test_scores[:n]),
        dims=('y_feature', 'mode'), coords=y_coords
    )


def test_loadings_true_pearson(estr, X, Y, Xorig, Yorig, x_align_ref, y_align_ref,
                               results, **kwargs):
    """Calculates Pearson correlations between estimated and true test
    loadings (loadings with respect to possibly transformed variables, i.e.
    those in columns of X, Y (not Xorig, Yorig).

    Requires results:

    * ``x_test_scores``
    * ``x_test_scores``

    These are available if :func:`test_scores` was run before.

    Requires keyword arguments:

    * kwargs['true_loadings'] - which is a ``dict`` constructed with
      :func:`gemmr.util._calc_true_loadings`.
    * Xtest, Ytest

    Provides outcome metrics ``x_loadings_true_pearson``,
    ``y_loadings_true_pearson``, ``x_crossloadings_true_pearson``
    and ``y_crossloadings_true_pearson``

    Parameters
    ----------
    estr : **sklearn**-style estimator
        fitted estimator
    X : np.ndarray (n_samples, n_features)
        dataset `X`
    Y : np.ndarray (n_samples, n_features)
        dataset `Y`
    Xorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `Y`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``Y`` and ``Yorig`` correspond to the same samples
        (subjects).
    x_align_ref : (n_features,)
        the sign of `X` weights is chosen such that the cosine-distance between
        fitted `X` weights and ``x_align_ref`` is positive
    y_align_ref : (n_features,)
        the sign of `Y` weights is chosen such that the cosine-distance between
        fitted `Y` weights and ``y_align_ref`` is positive
    results : xr.Dataset
        containing outcome features computed so far, and is modified with
        outcomes of this function
    kwargs : dict
        keyword arguments
    """

    x_loadings_true = kwargs['true_loadings']['x_loadings_true']
    y_loadings_true = kwargs['true_loadings']['y_loadings_true']
    x_crossloadings_true = kwargs['true_loadings']['x_crossloadings_true']
    y_crossloadings_true = kwargs['true_loadings']['y_crossloadings_true']

    x_test_scores = results['x_test_scores'].values
    y_test_scores = results['y_test_scores'].values

    Xtest = kwargs['Xtest']
    Ytest = kwargs['Ytest']

    n = len(X)
    x_test_loadings = _calc_loadings(Xtest[:n], x_test_scores[:n])
    y_test_loadings = _calc_loadings(Ytest[:n], y_test_scores[:n])
    x_test_crossloadings = _calc_loadings(Xtest[:n], y_test_scores[:n])
    y_test_crossloadings = _calc_loadings(Ytest[:n], x_test_scores[:n])

    results['x_test_loadings_true_pearson'] = _mk_da_pearson_loadings(
        x_test_loadings, x_loadings_true)
    results['y_test_loadings_true_pearson'] = _mk_da_pearson_loadings(
        y_test_loadings, y_loadings_true)
    results['x_test_crossloadings_true_pearson'] = _mk_da_pearson_loadings(
        x_test_crossloadings, x_crossloadings_true)
    results['y_test_crossloadings_true_pearson'] = _mk_da_pearson_loadings(
        y_test_crossloadings, y_crossloadings_true)


def loadings_true_pearson(estr, X, Y, Xorig, Yorig, x_align_ref, y_align_ref,
                               results, **kwargs):
    """Calculates Pearson correlations between estimated and true
    loadings (loadings with respect to possibly transformed variables, i.e.
    those in columns of X, Y (not Xorig, Yorig).

    Requires results:

    * ``x_scores``
    * ``x_scores``

    These are available if :func:`test_scores` was run before.

    Requires keyword arguments:

    * kwargs['true_loadings'] - which is a ``dict`` constructed with
      :func:`gemmr.util._calc_true_loadings`.
    * Xtest, Ytest

    Provides outcome metrics ``x_loadings_true_pearson``,
    ``y_loadings_true_pearson``, ``x_crossloadings_true_pearson``
    and ``y_crossloadings_true_pearson``

    Parameters
    ----------
    estr : **sklearn**-style estimator
        fitted estimator
    X : np.ndarray (n_samples, n_features)
        dataset `X`
    Y : np.ndarray (n_samples, n_features)
        dataset `Y`
    Xorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `Y`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``Y`` and ``Yorig`` correspond to the same samples
        (subjects).
    x_align_ref : (n_features,)
        the sign of `X` weights is chosen such that the cosine-distance between
        fitted `X` weights and ``x_align_ref`` is positive
    y_align_ref : (n_features,)
        the sign of `Y` weights is chosen such that the cosine-distance between
        fitted `Y` weights and ``y_align_ref`` is positive
    results : xr.Dataset
        containing outcome features computed so far, and is modified with
        outcomes of this function
    kwargs : dict
        keyword arguments
    """

    x_loadings_true = kwargs['true_loadings']['x_loadings_true']
    y_loadings_true = kwargs['true_loadings']['y_loadings_true']
    x_crossloadings_true = kwargs['true_loadings']['x_crossloadings_true']
    y_crossloadings_true = kwargs['true_loadings']['y_crossloadings_true']

    x_scores = estr.x_scores_
    y_scores = estr.y_scores_

    n = len(X)
    x_loadings = _calc_loadings(X[:n], x_scores[:n])
    y_loadings = _calc_loadings(Y[:n], y_scores[:n])
    x_crossloadings = _calc_loadings(X[:n], y_scores[:n])
    y_crossloadings = _calc_loadings(Y[:n], x_scores[:n])

    results['x_loadings_true_pearson'] = _mk_da_pearson_loadings(
        x_loadings, x_loadings_true)
    results['y_loadings_true_pearson'] = _mk_da_pearson_loadings(
        y_loadings, y_loadings_true)
    results['x_crossloadings_true_pearson'] = _mk_da_pearson_loadings(
        x_crossloadings, x_crossloadings_true)
    results['y_crossloadings_true_pearson'] = _mk_da_pearson_loadings(
        y_crossloadings, y_crossloadings_true)


def _mk_da_pearson_loadings(l1, l2):
    return xr.DataArray(
        [
            pearsonr(
                l1[:, i],
                l2[:, i]
            )[0]
            for i in range(l2.shape[1])
        ],
        dims=('mode',)
    )


def remove_test_scores(estr, X, Y, Xorig, Yorig, x_align_ref, y_align_ref,
                       results, **kwargs):
    """Removes ``x_test_scores`` and ``y_test_scores`` from ``results``.

    ``x_test_scores`` and ``y_test_scores`` are created by
    :func:`scores_test`.

    Parameters
    ----------
    estr : **sklearn**-style estimator
        fitted estimator
    X : np.ndarray (n_samples, n_features)
        dataset `X`
    Y : np.ndarray (n_samples, n_features)
        dataset `Y`
    Xorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `Y`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``Y`` and ``Yorig`` correspond to the same samples
        (subjects).
    x_align_ref : (n_features,)
        the sign of `X` weights is chosen such that the cosine-distance between
        fitted `X` weights and ``x_align_ref`` is positive
    y_align_ref : (n_features,)
        the sign of `Y` weights is chosen such that the cosine-distance between
        fitted `Y` weights and ``y_align_ref`` is positive
    results : xr.Dataset
        containing outcome features computed so far, and is modified with
        outcomes of this function
    kwargs : dict
        keyword arguments
    """
    del results['x_test_scores']
    del results['y_test_scores']


def assoc_test(estr, X, Y, Xorig, Yorig, x_align_ref, y_align_ref, results,
               **kwargs):
    """Calculates Pearson correlations between test scores.

    Requires keyword arguments:

    * kwargs['Xtest']
    * kwargs['Ytest']

    Provides outcome metrics ``between_assoc_test``

    Parameters
    ----------
    estr : **sklearn**-style estimator
        fitted estimator
    X : np.ndarray (n_samples, n_features)
        dataset `X`
    Y : np.ndarray (n_samples, n_features)
        dataset `Y`
    Xorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `Y`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``Y`` and ``Yorig`` correspond to the same samples
        (subjects).
    x_align_ref : (n_features,)
        the sign of `X` weights is chosen such that the cosine-distance between
        fitted `X` weights and ``x_align_ref`` is positive
    y_align_ref : (n_features,)
        the sign of `Y` weights is chosen such that the cosine-distance
        between fitted `Y` weights and ``y_align_ref`` is positive
    results : xr.Dataset
        containing outcome features computed so far, and is modified with
        outcomes of this function
    kwargs : dict
        keyword arguments
    """

    Xtest = kwargs['Xtest']
    Ytest = kwargs['Ytest']

    Xtest_t, Ytest_t = estr.transform(Xtest, Ytest)

    results['between_assoc_test'] = pearsonr(Xtest_t[:, 0], Ytest_t[:, 0])[0]


def _maybe_nan_pca(X):
    if np.isfinite(X).all():
        return PCA().fit(X).components_
    else:
        X = X - np.nanmean(X, axis=0, keepdims=True)
        mask = np.isfinite(X)
        cov = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                _mask = mask[:, i] & mask[:, j]
                cov[i, j] = cov[j, i] = \
                    (X[_mask, i] @ X[_mask, j])/ (_mask.sum()+1)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        return evecs[:, order].T


def datavar_pc_similarity(results, X, Y, datavar, metric, metric_label=None):
    """General function to calculate similarities between a data-variable and
    principal component axes.

    Parameters
    ----------
    results : xr.Dataset
        containing outcome features computed so far, and is modified with
        outcomes of this function
    X : np.ndarray (n_samples, n_[orig]_features)
        dataset `X`
    Y : np.ndarray (n_samples, n_[orig]_features)
        dataset `Y`
    datavar : str
        E.g. "weights", "loadings", or "orig_loadings"
    metric : str
        E.g. "cosine" or "correlation", any metric that is compatible with
        ``scipy.spatial.distance.cdist``
    """
    # Xmask = np.isfinite(X).all(1)
    # Ymask = np.isfinite(Y).all(1)
    # good_subject_mask = Xmask & Ymask
    # #if good_subject_mask.sum() < len(X):
    # #    print(f'Using {good_subject_mask.sum()} / {len(X)} subjects for PCA')
    # pca_X = PCA().fit(X[good_subject_mask])
    # pca_Y = PCA().fit(Y[good_subject_mask])

    x_pca_components = _maybe_nan_pca(X)
    y_pca_components = _maybe_nan_pca(Y)

    dims = results[f'x_{datavar}'].dims
    assert len(dims) == 2
    if dims[0] != 'mode':
        assert dims[1] == 'mode'
        dims = (dims[1], dims[0])
    assert dims[1][:2] == 'x_'
    other_dim = dims[1][2:]
    x_datavar_pc_sim = 1 - cdist(
        x_pca_components,
        results[f'x_{datavar}'].transpose('mode', f'x_{other_dim}').values,
        metric=metric
    )
    y_datavar_pc_sim = 1 - cdist(
        y_pca_components,
        results[f'y_{datavar}'].transpose('mode', f'y_{other_dim}').values,
        metric=metric
    )
    if metric_label is None:
        metric_label = metric
    results[f'x_{datavar}_pc_{metric_label}sim'] = xr.DataArray(
        x_datavar_pc_sim, dims=('x_pc', 'mode'),
        coords=dict(x_pc=np.arange(len(x_datavar_pc_sim))))
    results[f'y_{datavar}_pc_{metric_label}sim'] = xr.DataArray(
        y_datavar_pc_sim, dims=('y_pc', 'mode'),
        coords=dict(y_pc=np.arange(len(y_datavar_pc_sim))))


def weights_pc_cossim(estr, X, Y, Xorig, Yorig, x_align_ref, y_align_ref,
                      results, **kwargs):
    """Calculates cosine-similarities of principal component axes of X and Y
    with corresponding weights.

    Requires keyword arguments: None

    Provides outcome metrics ``x_weights_pc_cossim``, ``y_weights_pc_cossim``.

    Parameters
    ----------
    estr : **sklearn**-style estimator
        fitted estimator
    X : np.ndarray (n_samples, n_features)
        dataset `X`
    Y : np.ndarray (n_samples, n_features)
        dataset `Y`
    Xorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `Y`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``Y`` and ``Yorig`` correspond to the same samples
        (subjects).
    x_align_ref : (n_features,)
        the sign of `X` weights is chosen such that the cosine-distance between
        fitted `X` weights and ``x_align_ref`` is positive
    y_align_ref : (n_features,)
        the sign of `Y` weights is chosen such that the cosine-distance between
        fitted `Y` weights and ``y_align_ref`` is positive
    results : xr.Dataset
        containing outcome features computed so far, and is modified with
        outcomes of this function
    kwargs : dict
        keyword arguments
    """
    datavar_pc_similarity(results, X, Y, datavar='weights', metric='cosine',
                          metric_label='cos')


def loadings_pc_cossim(estr, X, Y, Xorig, Yorig, x_align_ref, y_align_ref,
                       results, **kwargs):
    """Calculates cosine-similarities of principal component axes of X and Y
    with corresponding loadings.

    This function uses ``x/y_loadings`` (not ``x/y_orig_loadings``).

    Requires keyword arguments: None

    Provides outcome metrics ``x_loadings_pc_cossim``,
    ``y_loadings_pc_cossim``.

    Parameters
    ----------
    estr : **sklearn**-style estimator
        fitted estimator
    X : np.ndarray (n_samples, n_features)
        dataset `X`
    Y : np.ndarray (n_samples, n_features)
        dataset `Y`
    Xorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `Y`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``Y`` and ``Yorig`` correspond to the same samples
        (subjects).
    x_align_ref : (n_features,)
        the sign of `X` weights is chosen such that the cosine-distance between
        fitted `X` weights and ``x_align_ref`` is positive
    y_align_ref : (n_features,)
        the sign of `Y` weights is chosen such that the cosine-distance between
        fitted `Y` weights and ``y_align_ref`` is positive
    results : xr.Dataset
        containing outcome features computed so far, and is modified with
        outcomes of this function
    kwargs : dict
        keyword arguments
    """
    datavar_pc_similarity(results, X, Y, datavar='loadings', metric='cosine',
                          metric_label='cos')


def loadings_pc_pearsonsim(estr, X, Y, Xorig, Yorig,
                           x_align_ref, y_align_ref, results, **kwargs):
    """Calculates correlation-similarities of principal component axes of X
    and Y with corresponding loadings.

    This function uses ``x/y_loadings`` (not ``x/y_orig_loadings``).

    Requires keyword arguments: None

    Provides outcome metrics ``x_loadings_pc_pearsonsim``,
    ``y_loadings_pc_pearsonsim``.

    Parameters
    ----------
    estr : **sklearn**-style estimator
        fitted estimator
    X : np.ndarray (n_samples, n_features)
        dataset `X`
    Y : np.ndarray (n_samples, n_features)
        dataset `Y`
    Xorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `Y`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``Y`` and ``Yorig`` correspond to the same samples
        (subjects).
    x_align_ref : (n_features,)
        the sign of `X` weights is chosen such that the cosine-distance between
        fitted `X` weights and ``x_align_ref`` is positive
    y_align_ref : (n_features,)
        the sign of `Y` weights is chosen such that the cosine-distance between
        fitted `Y` weights and ``y_align_ref`` is positive
    results : xr.Dataset
        containing outcome features computed so far, and is modified with
        outcomes of this function
    kwargs : dict
        keyword arguments
    """
    datavar_pc_similarity(results, X, Y, datavar='loadings',
                          metric='correlation', metric_label='pearson')


def orig_loadings_pc_cossim(estr, X, Y, Xorig, Yorig, x_align_ref, y_align_ref,
                            results, **kwargs):
    """Calculates cosine-similarities of principal component axes of X and Y
    with corresponding original-variable loadings.

    This function uses ``x/y_orig_loadings`` (not ``x/y_loadings``).

    Requires keyword arguments: None

    Provides outcome metrics ``x_orig_loadings_pc_cossim``,
    ``y_orig_loadings_pc_cossim``.

    Parameters
    ----------
    estr : **sklearn**-style estimator
        fitted estimator
    X : np.ndarray (n_samples, n_features)
        dataset `X`
    Y : np.ndarray (n_samples, n_features)
        dataset `Y`
    Xorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `Y`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``Y`` and ``Yorig`` correspond to the same samples
        (subjects).
    x_align_ref : (n_features,)
        the sign of `X` weights is chosen such that the cosine-distance between
        fitted `X` weights and ``x_align_ref`` is positive
    y_align_ref : (n_features,)
        the sign of `Y` weights is chosen such that the cosine-distance between
        fitted `Y` weights and ``y_align_ref`` is positive
    results : xr.Dataset
        containing outcome features computed so far, and is modified with
        outcomes of this function
    kwargs : dict
        keyword arguments
    """
    datavar_pc_similarity(results, Xorig, Yorig, datavar='orig_loadings',
                          metric='cosine', metric_label='cos')


def orig_loadings_pc_pearsonsim(estr, X, Y, Xorig, Yorig,
                                x_align_ref, y_align_ref, results, **kwargs):
    """Calculates correlation-similarities of principal component axes of X
    and Y with corresponding original-variable loadings.

    This function uses ``x/y_orig_loadings`` (not ``x/y_loadings``).

    Requires keyword arguments: None

    Provides outcome metrics ``x_orig_loadings_pc_pearsonsim``,
    ``y_orig_loadings_pc_pearsonsim``.

    Parameters
    ----------
    estr : **sklearn**-style estimator
        fitted estimator
    X : np.ndarray (n_samples, n_features)
        dataset `X`
    Y : np.ndarray (n_samples, n_features)
        dataset `Y`
    Xorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `Y`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``Y`` and ``Yorig`` correspond to the same samples
        (subjects).
    x_align_ref : (n_features,)
        the sign of `X` weights is chosen such that the cosine-distance between
        fitted `X` weights and ``x_align_ref`` is positive
    y_align_ref : (n_features,)
        the sign of `Y` weights is chosen such that the cosine-distance between
        fitted `Y` weights and ``y_align_ref`` is positive
    results : xr.Dataset
        containing outcome features computed so far, and is modified with
        outcomes of this function
    kwargs : dict
        keyword arguments
    """
    datavar_pc_similarity(results, Xorig, Yorig, datavar='orig_loadings',
                          metric='correlation', metric_label='pearson')


def mk_scorers_for_cv(n_between_modes=1):
    """Create scorers to use with :func:`cv`.

    Parameters
    ----------
    n_between_modes : number of between-set association modes for which
        cross-vaidation statistics shall be calculated

    Returns
    -------
    dict with scorers
    """
    scorers = dict()
    for mi in range(n_between_modes):
        scorers['cov_m{}'.format(mi)] = partial(cov_transform_scorer, ftr=mi)
        scorers['corr_m{}'.format(mi)] = \
            partial(pearson_transform_scorer, ftr=mi)
    return scorers


def cv(estr, X, Y, Xorg, Yorig, x_align_ref, y_align_ref, results, **kwargs):
    """Calculates cross-validated outcome metrics.

    Required keyword-arguments:

    * kwargs['cvs']: list of (label, cross-validator)
    * kwargs['scorers']: can be created by :func:`mk_scorers_for_cv`

    Optional keyword-arguments:

    * kwargs['fit_params']: dict
        passed to cross_validate

    Provides outcome metrics ``between_covs_cv``, ``between_corrs_cv``,
    ``x_weights_cv``, ``y_weights_cv``, ``x_loadings_cv`` and
    ``y_loadingscv``.

    Parameters
    ----------
    estr : **sklearn**-style estimator
        fitted estimator
    X : np.ndarray (n_samples, n_features)
        dataset `X`
    Y : np.ndarray (n_samples, n_features)
        dataset `Y`
    Xorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `Y`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``Y`` and ``Yorig`` correspond to the same samples
        (subjects).
    x_align_ref : (n_features,)
        the sign of `X` weights is chosen such that the cosine-distance between
        fitted `X` weights and ``x_align_ref`` is positive
    y_align_ref : (n_features,)
        the sign of `Y` weights is chosen such that the cosine-distance between
        fitted `Y` weights and ``y_align_ref`` is positive
    results : xr.Dataset
        containing outcome features computed so far, and is modified with
        outcomes of this function
    kwargs : dict
        keyword arguments
    """

    cvs = kwargs['cvs']
    scorers = kwargs['scorers']
    # assume `scorers` contains cov_m? and corr_m? scorers
    n_score_modes = len(scorers) // 2

    try:
        fit_params = kwargs['fit_params']
    except KeyError:
        fit_params = dict()

    max_n_splits = max([cv.get_n_splits() for _, cv in cvs])

    with warnings.catch_warnings():
        warnings.simplefilter(category=RuntimeWarning, action='ignore')
        covs = np.nan * np.empty((len(cvs), n_score_modes))
        corrs = np.nan * np.empty((len(cvs), n_score_modes))
        x_weights = np.nan * np.empty((len(cvs), max_n_splits, X.shape[1], estr.n_components))
        y_weights = np.nan * np.empty((len(cvs), max_n_splits, Y.shape[1], estr.n_components))
        x_loadings = np.nan * np.empty((len(cvs), max_n_splits, X.shape[1], estr.n_components))
        y_loadings = np.nan * np.empty((len(cvs), max_n_splits, Y.shape[1], estr.n_components))

    for cvi, (cv_label, cv) in enumerate(cvs):
        scores = cross_validate(
            estr,
            X,
            Y,
            scoring=scorers,
            cv=cv,
            return_estimator=True,
            verbose=0,
            fit_params=fit_params,
            error_score=np.nan,  # 'raise'
        )

        covs[cvi] = [scores['test_cov_m{}'.format(mi)].mean()
                     for mi in range(n_score_modes)]
        corrs[cvi] = [scores['test_corr_m{}'.format(mi)].mean()
                      for mi in range(n_score_modes)]

        try:
            # x_weights[cvi] = _unitvec(np.asarray([
            #     _estr.x_rotations_ for _estr in scores['estimator']
            # ]).mean(0))
            # y_weights[cvi] = _unitvec(np.asarray([
            #     _estr.y_rotations_ for _estr in scores['estimator']
            # ]).mean(0))
            n_splits = cv.get_n_splits()
            x_weights[cvi, :n_splits] = np.asarray([
                _estr.x_rotations_ for _estr in scores['estimator']
            ])
            y_weights[cvi, :n_splits] = np.asarray([
                _estr.y_rotations_ for _estr in scores['estimator']
            ])
            x_loadings[cvi, :n_splits] = np.asarray([
                _estr.x_loadings_ for _estr in scores['estimator']
            ])
            y_loadings[cvi, :n_splits] = np.asarray([
                _estr.y_loadings_ for _estr in scores['estimator']
            ])
        except AttributeError:
            # can happen if a problem ocurred during fitting, so that
            # attribute doesn't exist
            pass

    coords = dict(cv=np.asarray(list(zip(*cvs))[0]))
    results['between_covs_cv'] = xr.DataArray(
        covs, dims=('cv', 'mode'), coords=coords
    )
    results['between_corrs_cv'] = xr.DataArray(
        corrs, dims=('cv', 'mode'), coords=coords
    )
    results['x_weights_cv'] = xr.DataArray(
        x_weights, dims=('cv', 'fold', 'x_feature', 'mode')
    )
    results['y_weights_cv'] = xr.DataArray(
        y_weights, dims=('cv', 'fold', 'y_feature', 'mode'),
    )
    results['x_loadings_cv'] = xr.DataArray(
        x_loadings, dims=('cv', 'fold', 'x_feature', 'mode')
    )
    results['y_loadings_cv'] = xr.DataArray(
        y_loadings, dims=('cv', 'fold', 'y_feature', 'mode'),
    )


def redundancies(estr, X, Y, Xorg, Yorig, x_align_ref, y_align_ref, results,
                 **kwargs):
    """Calculates redundancies (fraction of variance in Xorig or Yorig
    explained by scores of opposite set).

    if Xorig / Yorig is None, reduncancies will be ``np.nan``.

    Provides outcome metrics ``xy_redundancies``, ``yx_redundancies``.

    Parameters
    ----------
    estr : **sklearn**-style estimator
        fitted estimator
    X : np.ndarray (n_samples, n_features)
        dataset `X`
    Y : np.ndarray (n_samples, n_features)
        dataset `Y`
    Xorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        can be ``None``. Allows to provide an alternative set of `Y`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``Y`` and ``Yorig`` correspond to the same samples
        (subjects).
    x_align_ref : (n_features,)
        the sign of `X` weights is chosen such that the cosine-distance between
        fitted `X` weights and ``x_align_ref`` is positive
    y_align_ref : (n_features,)
        the sign of `Y` weights is chosen such that the cosine-distance between
        fitted `Y` weights and ``y_align_ref`` is positive
    results : xr.Dataset
        containing outcome features computed so far, and is modified with
        outcomes of this function
    kwargs : dict
        keyword arguments
    """
    n_modes = len(results['mode'])
    assert n_modes == 1  # currently analyze_dataset only stores 1 mode

    corrs = [results['between_corrs_sample'].values]  # convert to list so that
    # it can be sliced like lodaings

    if 'x_orig_loadings' in results:
        xl = results['x_orig_loadings'].values
        xy_redundancies = [(corrs[m]**2) * (xl[:, m]**2).mean()
                           for m in range(n_modes)]
    else:
        xy_redundancies = np.nan * np.empty(len(results['mode']))

    if 'y_orig_loadings' in results:
        yl = results['x_orig_loadings'].values
        yx_redundancies = [(corrs[m]**2) * (yl[:, m]**2).mean()
                           for m in range(n_modes)]
    else:
        yx_redundancies = np.nan * np.empty(len(results['mode']))

    results['xy_redundancies'] = xr.DataArray(xy_redundancies, dims=('mode',))
    results['yx_redundancies'] = xr.DataArray(yx_redundancies, dims=('mode',))


def sparseCCA_penalties(estr, X, Y, Xorg, Yorig, x_align_ref, y_align_ref,
                        results, **kwargs):
    """Store penalties of a fitted SparseCCA estimator.

    Provides outcome metrics ``x_penalty``, ``y_penalty``

    Parameters
    ----------
    estr : **sklearn**-style estimator
        fitted SparseCCA instance
    X : ignored
    Y : ignored
    Xorig : ignored
    Yorig : ignored
    x_align_ref : ignored
    y_align_ref : ignored
    results : ignored
    kwargs : ignored
    """
    try:
        results['x_penalty'] = estr.penaltyx_
        results['y_penalty'] = estr.penaltyy_
    except AttributeError:
        results['x_penalty'] = np.nan
        results['y_penalty'] = np.nan


def mk_test_statistics_scores(Xtest, Ytest, x_weights, y_weights):
    """Calculate scores for test subjects.

    Parameters
    ----------
    Xtest : np.ndarray (n_samples, n_x_features)
        test dataset `X`
    Ytest : np.ndarray (n_samples, n_y_features)
        test dataset `Y`
    x_weights : np.ndarray (n_x_features, n_components)
        true weight vectors for dataset `X`
    y_weights : np.ndarray (n_y_features, n_components)
        true weight vectors for dataset `Y`

    Returns
    -------
    dict with test-statistics
    """

    x_test_scores_true = np.dot(Xtest, x_weights)
    y_test_scores_true = np.dot(Ytest, y_weights)

    res = dict(
        x_test_scores_true=xr.DataArray(
            x_test_scores_true[:, 0],
            dims=('test_sample',),
            coords=dict(test_sample=np.arange(len(Xtest)))
        ),
        y_test_scores_true=xr.DataArray(
            y_test_scores_true[:, 0],
            dims=('test_sample',),
            coords=dict(test_sample=np.arange(len(Xtest)))
        ),
    )

    return res
