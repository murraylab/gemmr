"""Metrics to evaluate analysis outcomes.
"""

import numpy as np

__all__ = [
    'mk_betweenAssocRelError', 'mk_betweenAssocRelError_cv',
    'mk_meanBetweenAssocRelError', 'mk_weightError', 'mk_scoreError',
    'mk_loadingError', 'mk_crossloadingError'
]


def mk_betweenAssocRelError(ds):
    """Evaluate the relative error of the between-set association strength.

    Parameters
    ----------
    ds : xr.Dataset
        outcome dataset

    Returns
    -------
    metric : xr.DataArray
        evaluated metric
    """
    return ds.between_assocs / ds.between_assocs_true - 1


def mk_betweenAssocRelError_cv(ds, cv_assoc):
    """Evaluate the relative error of the cross-validated between-set
    association strength.

    Parameters
    ----------
    ds : xr.Dataset
        outcome dataset
    cv_assoc : str
        identifier for the cross-validated association strength

    Returns
    -------
    metric : xr.DataArray
        evaluated metric
    """
    return ds[cv_assoc] / ds.between_assocs_true - 1


def mk_meanBetweenAssocRelError(ds, cv_assoc):
    """Evaluate the relative error of the between-set association strength as
    the average of the in-sample and cross-validated value.

    Parameters
    ----------
    ds : xr.Dataset
        outcome dataset
    cv_assoc : str
        identifier for the cross-validated association strength

    Returns
    -------
    metric : xr.DataArray
        evaluated metric
    """
    assoc = .5 * (ds[cv_assoc] + ds.between_assocs)
    return assoc / ds.between_assocs_true - 1


def mk_weightError(ds, suffix=''):
    """Evaluate the weight error.

    Weight error is the maximum cosine-distance between estimated and true
    weights across datasets :math:`X` and :math:`Y`.

    Parameters
    ----------
    ds : xr.Dataset
        outcome dataset
    suffix : str
        the calculation is based on the outcome metric
        ``'x/y_weights_true_cossim' + suffix``

    Returns
    -------
    metric : xr.DataArray
        evaluated metric
    """
    return np.maximum(
        1 - np.abs(ds['x_weights_true_cossim' + suffix]),
        1 - np.abs(ds['y_weights_true_cossim' + suffix])
    )


def mk_scoreError(ds):
    """Evaluate the score error.

    Score error is the maximum Spearman-distance between estimated and true
    test scores across datasets :math:`X` and :math:`Y`.

    Parameters
    ----------
    ds : xr.Dataset
        outcome dataset

    Returns
    -------
    metric : xr.DataArray
        evaluated metric
    """
    return np.maximum(
        1 - np.abs(ds.x_test_scores_true_spearman),
        1 - np.abs(ds.y_test_scores_true_spearman)
    )


def mk_loadingError(ds):
    """Evaluate the loading error.

    Loading error is the maximum correlation-distance between estimated test
    and true loadings across datasets :math:`X` and :math:`Y`.

    Parameters
    ----------
    ds : xr.Dataset
        outcome dataset

    Returns
    -------
    metric : xr.DataArray
        evaluated metric
    """
    return np.maximum(
        1 - np.abs(ds.x_test_loadings_true_pearson),
        1 - np.abs(ds.y_test_loadings_true_pearson)
    )


def mk_crossloadingError(ds):
    """Evaluate the cross-loading error.

    Cross-loading error is the maximum correlation-distance between estimated
    test and true cross-loadings across datasets :math:`X` and :math:`Y`.

    Parameters
    ----------
    ds : xr.Dataset
        outcome dataset

    Returns
    -------
    metric : xr.DataArray
        evaluated metric
    """
    return np.maximum(
        1 - np.abs(ds.x_test_crossloadings_true_pearson),
        1 - np.abs(ds.y_test_crossloadings_true_pearson)
    )


_metric_funs = dict(
    association_strength=mk_betweenAssocRelError,
    weight=mk_weightError,
    score=mk_scoreError,
    loading=mk_loadingError,
    crossloading=mk_crossloadingError,
)
