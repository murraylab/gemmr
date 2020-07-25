"""Postprocessors for sample analyzers."""

import numpy as np
import xarray as xr
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


__all__ = ['power', 'remove_between_assocs_perm',
           'weights_pairwise_cossim_stats',
           'scores_pairwise_spearmansim_stats',
           'remove_weights_loadings', 'remove_test_scores']


def power(res, alpha=0.05):
    """Calculate power

    Provides outcome metric ``power``.

    Parameters
    ----------
    res : xr.Dataset
        comprising all analysis outcomes
    alpha : float between 0 and 1
        significance level
    """
    p_values = \
        ((res.between_assocs_perm > res.between_assocs).sum('perm') + 1) / \
        (len(res.perm) + 1)
    res['power'] = (p_values < alpha).mean('rep')


def remove_between_assocs_perm(res):
    """Removes between_assocs_perm from results dataset

    Parameters
    ----------
    res : xr.Dataset
        comprising all analysis outcomes
    """
    del res['between_assocs_perm']


def _pairwise_similarities(w, metric='cosine'):
    """Calculate pairwise similarities.

    Only features that are present for all samples are considered.

    Parameters
    ----------
    w : np.ndarray (n_samples, n_features)
        data matrix

    Returns
    -------
    pairwise_similarities : np.ndarray (n_pairs,)
        array of similarities
    """
    mask = np.isfinite(w).all(0)
    sims = np.abs(1 - pdist(w[:, mask], metric=metric))
    return sims


def _calc_pairwise_similarity_stats(pairwise_similarities, qs=(.025, .5, .975)):
    """Calculate statistics of pairwise similarities.

    Calculated statistics are:
        - mean
        - quantiles indicated in parameter ``qs``

    Parameters
    ----------
    pairwise_similarities : xr.DataArray
        statistics are calculated across dimension "reprep"
    qs : tuple of floats between 0 and 1
        quantiles to calculate

    Returns
    -------
    statistics : xr.DataArray
        like ``pairwise_similarities``, but without dimension "reprep", instead
        with dimension "quantile"
    """

    mean = pairwise_similarities.mean('reprep').expand_dims(quantile=['mean'])

    quantiles = pairwise_similarities.quantile(qs, 'reprep')
    quantiles.coords['quantile'] = [
        'q{:.1f}%'.format(100*q)
        for q in quantiles['quantile'].values
    ]

    stats = xr.concat(
        [quantiles, mean], 'quantile'
    ).rename(
        quantile='stat'
    )
    stats = stats.transpose(*np.roll(stats.dims, -1))

    return stats


def pairwise_similarity_stats(res, outcome, x_feature_dim, y_feature_dim,
                              result_label, metric='cosine',
                              qs=(.025, .5, .975)):
    """Calculate pairwise similarities between outcomes for all pairs of
    repetitions.

    Provides outcome metrics ``x_[result_label`` and ``y_[result_label]``.

    Parameters
    ----------
    res : xr.Dataset
        comprising all analysis outcomes
    outcome : str
        similarities are calculated for outcomes ``x_[outcome]`` and
        ``y_[outcome]``
    x_feature_dim : str
        `X` similarities are calculated across dimension ``x_feature_dim``
    y_feature_dim : str
        `Y` similarities are calculates across dimension``y_feature_dim``
    result_label : str
        pairwise similarities are stored in ``res`` as ``x_[result_label`` and
        ``y_[result_label]``
    metric : str
        metric used to calculate similarities. Can be anything understood by
        :func:`scipy.spatial.distance.pdist`.
    qs : tuple of floats between 0 and 1
        quantiles to calculate
    """
    x_sims = xr.apply_ufunc(
        _pairwise_similarities,
        res['x_{}'.format(outcome)],
        kwargs=dict(metric=metric),
        input_core_dims=[['rep', x_feature_dim]],
        output_core_dims=[['reprep']],
        vectorize=True,
    )
    y_sims = xr.apply_ufunc(
        _pairwise_similarities,
        res['y_{}'.format(outcome)],
        kwargs=dict(metric=metric),
        input_core_dims=[['rep', y_feature_dim]],
        output_core_dims=[['reprep']],
        vectorize=True,
    )
    res['x_{}'.format(result_label)] = \
        _calc_pairwise_similarity_stats(x_sims, qs=qs)
    res['y_{}'.format(result_label)] = \
        _calc_pairwise_similarity_stats(y_sims, qs=qs)


def weights_pairwise_cossim_stats(res, qs=(.025, .5, .975)):
    """Calculate cosine similarity between weights for all pairs of
    repetitions.

    Provides outcome metrics ``x_weights_pairwise_cossim_stats`` and
    ``y_weights_pairwise_cossim_stats``.

    Parameters
    ----------
    res : xr.Dataset
        comprising all analysis outcomes
    qs : tuple of floats between 0 and 1
        quantiles to calculate
    """
    pairwise_similarity_stats(res, outcome='weights',
                              x_feature_dim='x_feature',
                              y_feature_dim='y_feature',
                              result_label='weights_pairwise_cossim_stats',
                              metric='cosine', qs=qs)


def scores_pairwise_spearmansim_stats(res, qs=(.025, .5, .975)):
    """Calculate cosine similarity between weights for all pairs of
    repetitions.

    Provides outcome metrics ``x_scores_pairwise_spearmansim_stats`` and
    ``y_scores_pairwise_spearmansim_stats``.

    Parameters
    ----------
    res : xr.Dataset
        comprising all analysis outcomes
    qs : tuple of floats between 0 and 1
        quantiles to calculate
    """
    pairwise_similarity_stats(res, outcome='test_scores',
                              x_feature_dim='test_sample',
                              y_feature_dim='test_sample',
                              result_label='scores_pairwise_spearmansim_stats',
                              metric=lambda x, y: 1 - spearmanr(x, y)[0],
                              qs=qs)


def remove_weights_loadings(res):
    """Removes weights and loadings from result dataset.

    Parameters
    ----------
    res : xr.Dataset
        comprising all analysis outcomes
    """
    del res['x_weights']
    del res['y_weights']
    del res['x_loadings']
    del res['y_loadings']


def remove_test_scores(res):
    """Removes test scores from result dataset.

    Parameters
    ----------
    res : xr.Dataset
        comprising all analysis outcomes
    """
    del res['x_test_scores']
    del res['y_test_scores']
