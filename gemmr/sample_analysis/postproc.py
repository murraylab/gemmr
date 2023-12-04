"""Postprocessors for sample analyzers."""

import numpy as np
import xarray as xr
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


__all__ = ['power', 'remove_between_assocs_perm',
           'weights_pairwise_cossim_stats',
           'weights_cv_pairwise_cossim_stats',
           'weights_pairwise_jaccard_stats',
           'scores_pairwise_spearmansim_stats',
           'loadings_pairwise_pearsonsim_stats',
           'loadings_pairwise_cossim_stats',
           'remove_weights_loadings', 'remove_cv_weights_loadings',
           'remove_test_scores', 'bs_quantiles', 'remove_bs_datavars']


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
        ((res.between_assocs_perm >= res.between_assocs).sum('perm') + 1) / \
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


def _pairwise_similarities(w, distance_metric='cosine'):
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
    sims = np.abs(1 - pdist(w[:, mask], metric=distance_metric))
    return sims


def _calc_pairwise_similarity_stats(pairwise_similarities, stat_dim,
                                    qs=(.025, .5, .975)):
    """Calculate statistics of pairwise similarities.

    Calculated statistics are:
        - mean
        - quantiles indicated in parameter ``qs``

    Parameters
    ----------
    pairwise_similarities : xr.DataArray
        statistics are calculated across dimension ``stat_dim``
    stat_dim : str
        dimension along which statistics are calculated. E.g. "reprep"
    qs : tuple of floats between 0 and 1
        quantiles to calculate

    Returns
    -------
    statistics : xr.DataArray
        like ``pairwise_similarities``, but without dimension ``stat_dim``,
        instead with dimension "quantile"
    """

    mean = pairwise_similarities.mean(stat_dim).expand_dims(quantile=['mean'])

    quantiles = pairwise_similarities.quantile(qs, stat_dim)
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


def pairwise_similarity_stats(res, outcome, cmp_dim,
                              x_feature_dim, y_feature_dim,
                              result_label, distance_metric='cosine',
                              single_stat=None, qs=(.025, .5, .975),
                              trafo=None):
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
    cmp_dim : str
        e.g. "rep". Similarities will be calculated between all pairs of
        observations of this dimension
    x_feature_dim : str
        `X` similarities are calculated across dimension ``x_feature_dim``
    y_feature_dim : str
        `Y` similarities are calculates across dimension``y_feature_dim``
    result_label : str
        pairwise similarities are stored in ``res`` as ``x_[result_label`` and
        ``y_[result_label]``
    distance_metric : str
        metric used to calculate similarities. Can be anything understood by
        :func:`scipy.spatial.distance.pdist`.
    single_stat : str of None
        if not None, only the statistic ``single_stat`` will be kept, and
        the dimension ``stat`` in the dataset will be dropped
    qs : tuple of floats between 0 and 1
        quantiles to calculate
    trafo: None or function
        applied to outcome variables before similarity-computation
    """

    if trafo is None:
        trafo = lambda x: x

    stat_dim = cmp_dim+cmp_dim
    x_sims = xr.apply_ufunc(
        _pairwise_similarities,
        trafo(res['x_{}'.format(outcome)]),
        kwargs=dict(distance_metric=distance_metric),
        input_core_dims=[[cmp_dim, x_feature_dim]],
        output_core_dims=[[stat_dim]],
        vectorize=True,
    )
    y_sims = xr.apply_ufunc(
        _pairwise_similarities,
        trafo(res['y_{}'.format(outcome)]),
        kwargs=dict(distance_metric=distance_metric),
        input_core_dims=[[cmp_dim, y_feature_dim]],
        output_core_dims=[[stat_dim]],
        vectorize=True,
    )
    x_stats = _calc_pairwise_similarity_stats(x_sims, stat_dim=stat_dim, qs=qs)
    y_stats = _calc_pairwise_similarity_stats(y_sims, stat_dim=stat_dim, qs=qs)

    if single_stat is not None:
        x_stats = x_stats.sel(stat=single_stat, drop=True)
        y_stats = y_stats.sel(stat=single_stat, drop=True)

    res['x_{}'.format(result_label)] = x_stats
    res['y_{}'.format(result_label)] = y_stats


def weights_pairwise_cossim_stats(res, qs=(.025, .5, .975), datavar_suffix=''):
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
    datavar_suffix : str
        if not ``''``, ``'_'+suffix`` is appended to outcome variable to
        identify outcome and result label
    """
    if datavar_suffix != '':
        datavar_suffix = '_' + datavar_suffix
    pairwise_similarity_stats(res, outcome='weights' + datavar_suffix,
                              cmp_dim='rep',
                              x_feature_dim='x_feature',
                              y_feature_dim='y_feature',
                              result_label=f'weights{datavar_suffix}_pairwise_cossim_stats',
                              distance_metric='cosine', qs=qs)


def weights_splithalf_cossim(res, datavar_suffix=''):
    """Calculate cosine similarity between weights for all split-half pairs.

    Provides outcome metrics ``x_weights_splithalf_cossim`` and
    ``y_weights_splithalf_cossim``.

    Parameters
    ----------
    res : xr.Dataset
        comprising all analysis outcomes
    datavar_suffix : str
        if not ``''``, ``'_'+suffix`` is appended to outcome variable to
        identify outcome and result label
    """
    if datavar_suffix != '':
        datavar_suffix = '_' + datavar_suffix
    # NOTE: analyze_splithalf uses analyze_subsampled internally
    # analyze_subsamped concatenates along dimension "rep", calls postprocs
    # and only then, within analyze_splithalf, is rep dimension renamed to
    # splithalf. Therefore here cmp_dim='rep'
    pairwise_similarity_stats(res, outcome='weights' + datavar_suffix,
                              cmp_dim='rep',
                              x_feature_dim='x_feature',
                              y_feature_dim='y_feature',
                              result_label=f'weights{datavar_suffix}_splithalf_cossim',
                              distance_metric='cosine', single_stat='mean',
                              qs=[])


def weights_cv_pairwise_cossim_stats(res, qs=(.025, .5, .975)):
    """Calculate cosine similarity between cross-validation-weights for all
    pairs of repetitions.

    Provides outcome metrics ``x_weights_cv_pairwise_cossim_stats`` and
    ``y_weights_cv_pairwise_cossim_stats``.

    Parameters
    ----------
    res : xr.Dataset
        comprising all analysis outcomes
    qs : tuple of floats between 0 and 1
        quantiles to calculate
    """
    pairwise_similarity_stats(res, outcome='weights_cv',
                              cmp_dim='rep',
                              x_feature_dim='x_feature',
                              y_feature_dim='y_feature',
                              result_label='weights_cv_pairwise_cossim_stats',
                              distance_metric='cosine', qs=qs)


def weights_pairwise_jaccard_stats(res, qs=(.025, .5, .975), datavar_suffix=''):
    """Calculate Jaccard similarity of non-zero weights weights for all pairs
    of repetitions.

    Provides outcome metrics ``x_weights_pairwise_jaccard_stats`` and
    ``y_weights_pairwise_jaccard_stats``.

    Parameters
    ----------
    res : xr.Dataset
        comprising all analysis outcomes
    qs : tuple of floats between 0 and 1
        quantiles to calculate
    datavar_suffix : str
        if not ``''``, ``'_'+suffix`` is appended to outcome variable to
        identify outcome and result label
    """
    if datavar_suffix != '':
        datavar_suffix = '_' + datavar_suffix
    pairwise_similarity_stats(res, outcome='weights' + datavar_suffix,
                              cmp_dim='rep',
                              x_feature_dim='x_feature',
                              y_feature_dim='y_feature',
                              result_label=f'weights{datavar_suffix}_pairwise_jaccard_stats',
                              distance_metric='jaccard', qs=qs,
                              trafo=lambda x: x != 0)


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
                              cmp_dim='rep',
                              x_feature_dim='test_sample',
                              y_feature_dim='test_sample',
                              result_label='scores_pairwise_spearmansim_stats',
                              distance_metric=lambda x, y: 1 - spearmanr(x, y)[0],
                              qs=qs)


def loadings_pairwise_cossim_stats(res, qs=(.025, .5, .975),
                                       datavar_suffix=''):
    """Calculate cosine-similarity between loadings for all pairs of
    repetitions.

    This function uses ``x/y_loadings`` (not ``x/y_orig_loadings``).

    Provides outcome metrics ``x_loadings_pairwise_cossim_stats`` and
    ``y_loadings_pairwise_cossim_stats``.

    Parameters
    ----------
    res : xr.Dataset
        comprising all analysis outcomes
    qs : tuple of floats between 0 and 1
        quantiles to calculate
    datavar_suffix : str
        if not ``''``, ``'_'+suffix`` is appended to outcome variable to
        identify outcome and result label
    """
    if datavar_suffix != '':
        datavar_suffix = '_' + datavar_suffix
    pairwise_similarity_stats(res, outcome='loadings' + datavar_suffix,
                              cmp_dim='rep',
                              x_feature_dim='x_feature',
                              y_feature_dim='y_feature',
                              result_label=f'loadings{datavar_suffix}_pairwise_cossim_stats',
                              distance_metric='cosine', qs=qs)


def loadings_pairwise_pearsonsim_stats(res, qs=(.025, .5, .975),
                                       datavar_suffix=''):
    """Calculate Pearson correlation between loadings for all pairs of
    repetitions.

    This function uses ``x/y_loadings`` (not ``x/y_orig_loadings``).

    Provides outcome metrics ``x_loadings_pairwise_pearsonsim_stats`` and
    ``y_loadings_pairwise_pearsonsim_stats``.

    Parameters
    ----------
    res : xr.Dataset
        comprising all analysis outcomes
    qs : tuple of floats between 0 and 1
        quantiles to calculate
    datavar_suffix : str
        if not ``''``, ``'_'+suffix`` is appended to outcome variable to
        identify outcome and result label
    """
    if datavar_suffix != '':
        datavar_suffix = '_' + datavar_suffix
    pairwise_similarity_stats(res, outcome='loadings' + datavar_suffix,
                              cmp_dim='rep',
                              x_feature_dim='x_feature',
                              y_feature_dim='y_feature',
                              result_label=f'loadings{datavar_suffix}_pairwise_pearsonsim_stats',
                              distance_metric='correlation', qs=qs)


def orig_loadings_pairwise_pearsonsim_stats(res, qs=(.025, .5, .975),
                                       datavar_suffix=''):
    """Calculate Pearson correlation between loadings for all pairs of
    repetitions.

    This function uses ``x/y_orig_loadings`` (not ``x/y_loadings``).

    Provides outcome metrics ``x_orig_loadings_pairwise_pearsonsim_stats`` and
    ``y_orig_loadings_pairwise_pearsonsim_stats``.

    Parameters
    ----------
    res : xr.Dataset
        comprising all analysis outcomes
    qs : tuple of floats between 0 and 1
        quantiles to calculate
    datavar_suffix : str
        if not ``''``, ``'_'+suffix`` is appended to outcome variable to
        identify outcome and result label
    """
    if datavar_suffix != '':
        datavar_suffix = '_' + datavar_suffix
    pairwise_similarity_stats(res, outcome='orig_loadings' + datavar_suffix,
                              cmp_dim='rep',
                              x_feature_dim='x_orig_feature',
                              y_feature_dim='y_orig_feature',
                              result_label=f'orig_loadings{datavar_suffix}_pairwise_pearsonsim_stats',
                              distance_metric='correlation', qs=qs)


def loadings_splithalf_pearsonsim(res, datavar_suffix=''):
    """Calculate Pearson correlation between loadings for all split-half pairs.

    This function uses ``x/y_loadings`` (not ``x/y_orig_loadings``).

    Provides outcome metrics ``x_loadings_splithalf_pearsonsim`` and
    ``y_loadings_splithalf_pearsonsim``.

    Parameters
    ----------
    res : xr.Dataset
        comprising all analysis outcomes
    datavar_suffix : str
        if not ``''``, ``'_'+suffix`` is appended to outcome variable to
        identify outcome and result label
    """
    if datavar_suffix != '':
        datavar_suffix = '_' + datavar_suffix
    # NOTE: analyze_splithalf uses analyze_subsampled internally
    # analyze_subsamped concatenates along dimension "rep", calls postprocs
    # and only then, within analyze_splithalf, is rep dimension renamed to
    # splithalf. Therefore here cmp_dim='rep'
    pairwise_similarity_stats(res, outcome='loadings' + datavar_suffix,
                              cmp_dim='rep',
                              x_feature_dim='x_feature',
                              y_feature_dim='y_feature',
                              result_label=f'loadings{datavar_suffix}_splithalf_pearsonsim',
                              distance_metric='correlation', single_stat='mean',
                              qs=[])


def orig_loadings_splithalf_pearsonsim(res, datavar_suffix=''):
    """Calculate Pearson correlation between loadings for all split-half pairs.

    This function uses ``x/y_orig_loadings`` (not ``x/y_loadings``).

    Provides outcome metrics ``x_orig_loadings_splithalf_pearsonsim`` and
    ``y_orig_loadings_splithalf_pearsonsim``.

    Parameters
    ----------
    res : xr.Dataset
        comprising all analysis outcomes
    datavar_suffix : str
        if not ``''``, ``'_'+suffix`` is appended to outcome variable to
        identify outcome and result label
    """
    if datavar_suffix != '':
        datavar_suffix = '_' + datavar_suffix
    # NOTE: analyze_splithalf uses analyze_subsampled internally
    # analyze_subsamped concatenates along dimension "rep", calls postprocs
    # and only then, within analyze_splithalf, is rep dimension renamed to
    # splithalf. Therefore here cmp_dim='rep'
    pairwise_similarity_stats(res, outcome='orig_loadings' + datavar_suffix,
                              cmp_dim='rep',
                              x_feature_dim='x_orig_feature',
                              y_feature_dim='y_orig_feature',
                              result_label=f'orig_loadings{datavar_suffix}_splithalf_pearsonsim',
                              distance_metric='correlation', single_stat='mean',
                              qs=[])


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


def remove_cv_weights_loadings(res):
    """Removes CV weights and loadings from result dataset.

    Parameters
    ----------
    res : xr.Dataset
        comprising all analysis outcomes
    """
    del res['x_weights_cv']
    del res['y_weights_cv']
    del res['x_loadings_cv']
    del res['y_loadings_cv']


def remove_test_scores(res):
    """Removes test scores from result dataset.

    Parameters
    ----------
    res : xr.Dataset
        comprising all analysis outcomes
    """
    del res['x_test_scores']
    del res['y_test_scores']


def bs_quantiles(res, qs=(.025, .5, .975)):
    """Calculate quantiles of bootstrap variables.

    Args:
        res (_type_): _description_
        qs (tuple, optional): _description_. Defaults to (.025, .5, .975).
    """
    for datavar in list(res.data_vars.keys()):
        if datavar.endswith('_bs'):
            
            quantiles = res[datavar].quantile(
                qs, 'bs'
            ).rename(
                quantile='stat'
            )
            print(quantiles)
            quantiles = quantiles.assign_coords(
                stat=[
                    'q{:.1f}%'.format(100*q)
                    for q in quantiles['stat'].values
                ]
            )
            
            mean = res[datavar].mean('bs').expand_dims(stat=['mean'])

            stats = xr.concat(
                [quantiles, mean], 'stat'
            )
            stats = stats.transpose(*np.roll(stats.dims, -1))
            res[f'{datavar}_stats'] = stats
    
def remove_bs_datavars(res):
    """Removes bootstrap datavars.

    Removes all datavars with names ending in "_bs".
    
    Parameters
    ----------
    res : xr.Dataset
        comprising all analysis outcomes
    """
    for datavar in list(res.data_vars.keys()):
        if datavar.endswith('_bs'):
            del res[datavar]
