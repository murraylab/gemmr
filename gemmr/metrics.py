"""Metrics to evaluate analysis outcomes.
"""

import numpy as np


__all__ = [
    'mk_fnr', 'mk_absBetweenAssocRelError', 'mk_betweenAssocRelError',
    'mk_betweenAssocRelError_cv', 'mk_meanBetweenAssocRelError',
    'mk_betweenCorrRelError', 'mk_absBetweenCorrRelError',
    'mk_weightError', 'mk_scoreError',
    'mk_loadingError', 'mk_crossloadingError', 'mk_combinedError',
    'mk_combinedError_woPower'
]


def mk_fnr(ds, prefix=''):
    """Evaluate the false negative rate, i.e. ``1 - power``.

    Parameters
    ----------
    ds : xr.Dataset
        outcome dataset
    prefix : str
        will be prepended to the name of the property when looking it up in
        ``ds``

    Returns
    -------
    metric : xr.DataArray
        evaluated metric
    """
    return 1 - ds[f'{prefix}power']


def mk_betweenAssocRelError(ds, prefix='', datavar='between_assocs'):
    """Evaluate the relative error of the between-set association strength.

    Parameters
    ----------
    ds : xr.Dataset
        outcome dataset
    prefix : str
        will be prepended to the name of the property when looking it up in
        ``ds``

    Returns
    -------
    metric : xr.DataArray
        evaluated metric
    """
    return \
        ds[f'{prefix}{datavar}'] / ds[f'{prefix}between_assocs_true'] - 1


def mk_absBetweenAssocRelError(ds, prefix=''):
    """Evaluate the absolute value of the relative error of the between-set
    association strength.

    Parameters
    ----------
    ds : xr.Dataset
        outcome dataset
    prefix : str
        will be prepended to the name of the property when looking it up in
        ``ds``

    Returns
    -------
    metric : xr.DataArray
        evaluated metric
    """
    return np.abs(
        ds[f'{prefix}between_assocs'] / ds[f'{prefix}between_assocs_true'] - 1
    )


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


def mk_betweenCorrRelError(ds, prefix='', datavar='between_corrs_sample'):
    """Evaluate the relative error of the between-set correlations.

    Technical note: sample correlation strengths are taken from dataset
    attribute "between_corrs_sample" (meaning they are calculated as
    Pearson correlations between `X` and `Y` scores)

    Parameters
    ----------
    ds : xr.Dataset
        outcome dataset
    prefix : str
        will be prepended to the name of the property when looking it up in
        ``ds``

    Returns
    -------
    metric : xr.DataArray
        evaluated metric
    """
    return ds[f'{prefix}{datavar}'] / \
           ds[f'{prefix}between_corrs_true'] - 1


def mk_absBetweenCorrRelError(ds, prefix=''):
    """Evaluate the relative error of the between-set correlations.

    Technical note: sample correlation strengths are taken from dataset
    attribute "between_corrs_sample" (meaning they are calculated as
    Pearson correlations between `X` and `Y` scores)

    Parameters
    ----------
    ds : xr.Dataset
        outcome dataset
    prefix : str
        will be prepended to the name of the property when looking it up in
        ``ds``

    Returns
    -------
    metric : xr.DataArray
        evaluated metric
    """
    return np.abs(
        ds[f'{prefix}between_corrs_sample'] / ds[f'{prefix}between_corrs_true']
        - 1
    )


def mk_weightError(ds, suffix='', prefix=''):
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
    prefix : str
        will be prepended to the name of the property when looking it up in
        ``ds``

    Returns
    -------
    metric : xr.DataArray
        evaluated metric
    """
    return np.maximum(
        1 - np.abs(ds[f'{prefix}x_weights_true_cossim{suffix}']),
        1 - np.abs(ds[f'{prefix}y_weights_true_cossim{suffix}'])
    )


def mk_scoreError(ds, prefix=''):
    """Evaluate the score error.

    Score error is the maximum Spearman-distance between estimated and true
    test scores across datasets :math:`X` and :math:`Y`.

    Parameters
    ----------
    ds : xr.Dataset
        outcome dataset
    prefix : str
        will be prepended to the name of the property when looking it up in
        ``ds``

    Returns
    -------
    metric : xr.DataArray
        evaluated metric
    """
    return np.maximum(
        1 - np.abs(ds[f'{prefix}x_test_scores_true_pearson']),
        1 - np.abs(ds[f'{prefix}y_test_scores_true_pearson'])
    )


def mk_loadingError(ds, prefix=''):
    """Evaluate the loading error.

    Loading error is the maximum correlation-distance between estimated test
    and true loadings across datasets :math:`X` and :math:`Y`.

    Parameters
    ----------
    ds : xr.Dataset
        outcome dataset
    prefix : str
        will be prepended to the name of the property when looking it up in
        ``ds``

    Returns
    -------
    metric : xr.DataArray
        evaluated metric
    """
    return np.maximum(
        1 - np.abs(ds[f'{prefix}x_test_loadings_true_pearson']),
        1 - np.abs(ds[f'{prefix}y_test_loadings_true_pearson'])
    )


def mk_crossloadingError(ds, prefix=''):
    """Evaluate the cross-loading error.

    Cross-loading error is the maximum correlation-distance between estimated
    test and true cross-loadings across datasets :math:`X` and :math:`Y`.

    Parameters
    ----------
    ds : xr.Dataset
        outcome dataset
    prefix : str
        will be prepended to the name of the property when looking it up in
        ``ds``

    Returns
    -------
    metric : xr.DataArray
        evaluated metric
    """
    return np.maximum(
        1 - np.abs(ds[f'{prefix}x_test_crossloadings_true_pearson']),
        1 - np.abs(ds[f'{prefix}y_test_crossloadings_true_pearson'])
    )


def mk_combinedError(ds, prefix='', assoc_metric='assoc',
                     abs_assoc_error=False, ignore_power=False):
    """Evaluate the combined error.

    Parameters
    ----------
    ds : xr.Dataset
        outcome dataset
    prefix : str
        will be prepended to the name of the property when looking it up in
        ``ds``
    assoc_metric : str
        "assoc" refers to dataset attribute "between_assocs", "corrs" to
        dataset attribute "between_corrs_sample"
    abs_assoc_error : bool
        if ``True`` the absolute value of the relative association error will
        be considered
    ignore_power : bool
        if ``True`` power (i.e. 1 - FNR) will *not* be considered for the
        combined error

    Returns
    -------
    metric : xr.DataArray
        evaluated metric
    """
    metrics = [
        mk_weightError, mk_scoreError, mk_loadingError
    ]

    if assoc_metric == 'assoc':
        if abs_assoc_error:
            metrics.append(mk_absBetweenAssocRelError)
        else:
            metrics.append(mk_betweenAssocRelError)
    elif assoc_metric == 'corr':
        if abs_assoc_error:
            metrics.append(mk_absBetweenCorrRelError)
        else:
            metrics.append(mk_betweenCorrRelError)
    else:
        raise ValueError(f"Invalid 'assoc_metric': {assoc_metric}")

    if not ignore_power:
        metrics.append(mk_fnr)

    error = metrics[0](ds, prefix=prefix)
    for metric in metrics[1:]:
        error = np.fmax(  # ignore NaNs
            error,
            metric(ds, prefix=prefix)
        )
    return error


def mk_combinedError_woPower(ds, prefix=''):
    """Evaluate the combined error, ignoring metric `power`.

    Parameters
    ----------
    ds : xr.Dataset
        outcome dataset
    prefix : str
        will be prepended to the name of the property when looking it up in
        ``ds``

    Returns
    -------
    metric : xr.DataArray
        evaluated metric
    """
    return mk_combinedError(ds, prefix, ignore_power=True)


_metric_funs = dict(
    association_strength=mk_betweenAssocRelError,
    weight=mk_weightError,
    score=mk_scoreError,
    loading=mk_loadingError,
    crossloading=mk_crossloadingError,
)
