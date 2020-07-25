"""Add-ons for sample analyzers.
"""

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
    'remove_cv_weights',
    'weights_true_cossim',
    'test_scores',
    'scores_true_spearman',
    'loadings_true_pearson',
    'remove_test_scores',
    'assoc_test',
    'weights_pc_cossim',
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
        if ``None`` set to ``X``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        if ``None`` set to ``Y``. Allows to provide an alternative set of `Y`
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


def remove_cv_weights(estr, X, Y, Xorig, Yorig, x_align_ref, y_align_ref,
                      results, **kwargs):
    """Removes ``x_weights_cv`` and ``y_weights_cv`` from ``results`` dataset
    to save storage space.

    ``x_weights_cv`` and ``y_weights_cv`` are created by :func:`addana_cv`

    Parameters
    ----------
    estr : **sklearn**-style estimator
        fitted estimator
    X : np.ndarray (n_samples, n_features)
        dataset `X`
    Y : np.ndarray (n_samples, n_features)
        dataset `Y`
    Xorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        if ``None`` set to ``X``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        if ``None`` set to ``Y``. Allows to provide an alternative set of `Y`
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
        if ``None`` set to ``X``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        if ``None`` set to ``Y``. Allows to provide an alternative set of `Y`
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
        results['x_weights'].values,
        x_align_ref
    )
    results['y_weights_true_cossim'] = 1 - cos_dist(
        results['y_weights'].values,
        y_align_ref
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
        if ``None`` set to ``X``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        if ``None`` set to ``Y``. Allows to provide an alternative set of `Y`
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

        Xtest_t, Ytest_t = estr.transform(Xtest, Ytest)

        results['x_test_scores'] = xr.DataArray(
            Xtest_t, dims=('test_sample', 'mode'),
            coords=dict(test_sample=np.arange(len(Xtest)))
        )
        results['y_test_scores'] = xr.DataArray(
            Ytest_t, dims=('test_sample', 'mode'),
            coords=dict(test_sample=np.arange(len(Xtest)))
        )


def scores_true_spearman(estr, X, Y, Xorig, Yorig, x_align_ref, y_align_ref,
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
        if ``None`` set to ``X``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        if ``None`` set to ``Y``. Allows to provide an alternative set of `Y`
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


def loadings_true_pearson(estr, X, Y, Xorig, Yorig, x_align_ref, y_align_ref,
                          results, **kwargs):
    """Calculates Pearson correlations between estimated and true test
    loadings.

    Requires results:

    * ``x_test_scores``
    * ``x_test_scores``

    These are available if :func:`addana_scores_true_spearman` was run before.

    Requires keyword arguments:

    * kwargs['true_loadings'] - which is a ``dict`` constructed with
      :func:`ccapwr.sample_analysis.analyzers._calc_true_loadings`.
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
        if ``None`` set to ``X``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        if ``None`` set to ``Y``. Allows to provide an alternative set of `Y`
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
        if ``None`` set to ``X``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        if ``None`` set to ``Y``. Allows to provide an alternative set of `Y`
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
        if ``None`` set to ``X``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        if ``None`` set to ``Y``. Allows to provide an alternative set of `Y`
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
        if ``None`` set to ``X``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        if ``None`` set to ``Y``. Allows to provide an alternative set of `Y`
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

    pca_X = PCA().fit(X)
    pca_Y = PCA().fit(Y)

    x_weights_pc_cossim = 1 - cdist(
        pca_X.components_,
        results.x_weights.transpose('mode', 'x_feature').values,
        metric='cosine'
    )
    y_weights_pc_cossim = 1 - cdist(
        pca_Y.components_,
        results.y_weights.transpose('mode', 'y_feature').values,
        metric='cosine'
    )
    results['x_weights_pc_cossim'] = \
        xr.full_like(results.x_weights, x_weights_pc_cossim)
    results['y_weights_pc_cossim'] = \
        xr.full_like(results.y_weights, y_weights_pc_cossim)


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

    Provides outcome metrics ``between_covs_cv``, ``between_corrs_cv``,
    ``x_weights_cv`` and ``y_weights_cv``.

    Parameters
    ----------
    estr : **sklearn**-style estimator
        fitted estimator
    X : np.ndarray (n_samples, n_features)
        dataset `X`
    Y : np.ndarray (n_samples, n_features)
        dataset `Y`
    Xorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        if ``None`` set to ``X``. Allows to provide an alternative set of `X`
        features for calculating loadings. I.e. an implicit assumption is that
        the rows in ``X`` and ``Xorig`` correspond to the same samples
        (subjects).
    Yorig : ``None`` or np.ndarray (n_samples, n_orig_features)
        if ``None`` set to ``Y``. Allows to provide an alternative set of `Y`
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

    covs = np.nan * np.empty((len(cvs), n_score_modes))
    corrs = np.nan * np.empty((len(cvs), n_score_modes))
    x_weights = np.nan * np.empty((len(cvs), X.shape[1], estr.n_components))
    y_weights = np.nan * np.empty((len(cvs), Y.shape[1], estr.n_components))

    for cvi, (cv_label, cv) in enumerate(cvs):
        scores = cross_validate(
            estr,
            X,
            Y,
            scoring=scorers,
            cv=cv,
            return_estimator=True,
            verbose=0,
        )

        covs[cvi] = [scores['test_cov_m{}'.format(mi)].mean()
                     for mi in range(n_score_modes)]
        corrs[cvi] = [scores['test_corr_m{}'.format(mi)].mean()
                      for mi in range(n_score_modes)]

        try:
            x_weights[cvi] = _unitvec(np.asarray([
                _estr.x_rotations_ for _estr in scores['estimator']
            ]).mean(0))
            y_weights[cvi] = _unitvec(np.asarray([
                _estr.y_rotations_ for _estr in scores['estimator']
            ]).mean(0))
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
        x_weights, dims=('cv', 'x_feature', 'mode')
    )
    results['y_weights_cv'] = xr.DataArray(
        y_weights, dims=('cv', 'y_feature', 'mode')
    )


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
    results['x_penalty'] = estr.penaltyx_
    results['y_penalty'] = estr.penaltyy_


def mk_test_statistics_scores(Xtest, Ytest, U_latent, V_latent):
    """Calculate scores for test subjects.

    Parameters
    ----------
    Xtest : np.ndarray (n_samples, n_x_features)
        test dataset `X`
    Ytest : np.ndarray (n_samples, n_y_features)
        test dataset `Y`
    U_latent : np.ndarray (n_x_features, n_components)
        true weight vectors for dataset `X`
    V_latent : np.ndarray (n_y_features, n_components)
        true weight vectors for dataset `Y`

    Returns
    -------
    dict with test-statistics
    """

    x_test_scores_true = np.dot(Xtest, U_latent)
    y_test_scores_true = np.dot(Ytest, V_latent)

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
