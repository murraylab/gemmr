"""Analysis pipelines."""

import numbers

import numpy as np
from scipy.spatial.distance import pdist
import pandas as pd
import xarray as xr

from sklearn.utils import check_random_state, deprecated
from sklearn.model_selection import KFold

from .analyzers import analyze_resampled, analyze_splithalf
from ..sample_analysis import addon, postproc


def naive_permutations(Y, rng, n):
    for _ in range(n):
        yield rng.permutation(len(Y))


__all__ = ['calc_p_value', 'analyze_subsampled_and_resampled',
           'pairwise_weight_cosine_similarity']


def calc_p_value(estr, X, Y, permutations=999, fit_params=None,
                 random_state=0):
    """Calculate permutation-based p-value.

    The p-value is calculated based on the attribute "assocs_[0]" of the fitted
    estimator.

    Parameters
    ----------
    estr : sklearn-style estimator
        estimator used to analyze the data, needs to be compatible with
        analyzers in ccapwr.sample_analysis.analyzers
    X : np.ndarray (n_samples, n_X_features)
        dataset X
    Y : np.ndarray (n_samples, n_Y_features)
        dataset Y
    permutations : int > 1 or np.ndarray (n_permutations, n_samples)
        if ``int`` indicates the number of permutations to perform, and all
        possible permutations are allowed, if == 0 then np.nan is returned;
        if ``np.ndarray`` each row gives one set of permutation indices (i.e.
        the set of values in each row must be a permutation of [0, n_samples)
        and the order of columns are assumed to correspond to the order
        of rows of X and Y), if length of array is 0 then np.nan is returned
    fit_params : dict
        keyword-arguments for estr.fit
    random_state : None, int or rng-instance
        random seed

    Returns
    -------
    p_value : float between 0 and 1
        the permutation-based p-value
    """

    if fit_params is None:
        fit_params = dict()

    rng = check_random_state(random_state)

    estr.fit(X, Y, **fit_params)
    score = estr.assocs_[0]

    if isinstance(permutations, numbers.Integral):
        n_permutations = permutations
        if n_permutations == 0:
            return np.nan
        perm_iter = naive_permutations(Y, rng, n_permutations)
    else:
        n_permutations = len(permutations)
        if n_permutations == 0:
            return np.nan
        perm_iter = permutations

    perm_scores = np.nan * np.empty(n_permutations)
    for permi, permuted_indices in enumerate(perm_iter):
        Yperm = Y[permuted_indices]
        estr.fit(X, Yperm, **fit_params)
        perm_scores[permi] = estr.assocs_[0]

    return ((perm_scores >= score).sum() + 1) / (n_permutations + 1)


def analyze_subsampled_and_resampled(
        estr, X, Y, Xorig=None, Yorig=None, permutations=1000, ns=None,
        n_min_subsample=None, frac_max_subsample=0.5, n_subsample_ns=5,
        n_rep_subsample=100, n_perm_subsample=1000, n_test_subsample=0,
        cv=True, n_jobs=1, fit_params=None, random_state=0):
    """Analyzes the given data with the given estimator.

    Specifially:

    * calculates the permutation-based p-value
    * analyzes the full sample, and its permutations
    * analyzes non-overlapping subsamples of the data

    Parameters
    ----------
    estr : sklearn-style estimator
        estimator used to analyze the data, needs to be compatible with
        analyzers in ccapwr.sample_analysis.analyzers
    X : np.ndarray (n_samples, n_X_features)
        dataset X
    Y : np.ndarray (n_samples, n_Y_features)
        dataset Y
    Xorig : np.ndarray (n_samples, n_X_features)
        X dataset of original variables (for calculating loadings)
    Yorig : np.ndarray (n_samples, n_Y_features)
        Y dataset of original variables (for calculating loadings)
    permutations : int or iterable
        used for calculating p-value and the whole-sample analysis. If int,
        gives the number of permutations used, if iterable each element
        gives one set of permutation indices
    ns : list of int
        number of samples to which the data are subsampled. If ``None``
        calculated from ``n_min_subsample``, ``frac_max_subsample`` and
        ``n_subsample_ns``
    n_min_subsample : None or int
        minimum number of samples to which the data are subsampled. If None
        ``X.shape[1]+Y.shape[1]+2`` is used. Ignored if ``ns`` is not ``None``
    frac_max_subsample : float between 0 and 1
        the maximum number of samples to which the data are subsampled is
        ``frac_max_subsample * len(X)``. Ignored if ``ns`` is not ``None``
    n_subsample_ns : int
        the list of sample sizes to which the data are subsampled is a
        ``np.logspace`` with this many entries. Ignored if ``ns`` is not
        ``None``
    n_rep_subsample : int
        number of times a subsampled dataset of a given size is generated
    n_perm_subsample : int
        number of permutations for each subsampled datasets
    n_test_subsample : int or 'auto'
        number of subjects to use as test set in subsampled datasets. If
        ``n_test == 'auto'`` then ``n_test = n_samples - max(ns)`` will be
        used.
    cv : bool
        if ``True`` run cross-validations
    n_jobs : int or None
        number of parallel jobs (see :class:`joblib.Parallel`)
    fit_params : dict
        keyword-arguments for estr.fit
    random_state : None, int or rng-instance
        random seed

    Returns
    -------
    results : dict
        with items:

        * full_sample : xr.Dataset (output of analyze_resampled)
            This also contains ``p-value`` as a data-variable
        * subsampled : xr.Dataset (output of analyze_subsampled)
    """

    if n_test_subsample > 0:
        raise ValueError('n_test_subsample is deprecated')

    assert len(X) == len(Y)
    assert 0 <= frac_max_subsample <= 1
    if n_rep_subsample <= 1:
        raise ValueError(f"n_rep_subsample must be >= 2, "
                         f"got {n_rep_subsample}")
    if frac_max_subsample > .5:
        raise ValueError("To obtain only non-overlapping subsamples, "
                         "frac_max_subsample must be <= 0.5")

    rng = check_random_state(random_state)

    addons = [
        addon.weights_pc_cossim,
        addon.loadings_pc_cossim,
        addon.loadings_pc_pearsonsim,
        addon.redundancies,
    ]
    postprocs = [
        #postproc.weights_pairwise_cossim_stats,
        postproc.weights_splithalf_cossim,
        postproc.loadings_splithalf_pearsonsim,
        # postproc.scores_pairwise_spearmansim_stats,
        # postproc.remove_test_scores,
    ]
    if (Xorig is not None) and (Yorig is not None):
        addons += [
            addon.orig_loadings_pc_cossim,
            addon.orig_loadings_pc_pearsonsim,
        ]
        postprocs += [
            postproc.orig_loadings_splithalf_pearsonsim,
        ]
    if cv:
        addons += [
            addon.cv,
            addon.remove_cv_weights_loadings,
        ]

    # p-value
    p_value = calc_p_value(estr, X, Y, permutations=permutations,
                           fit_params=fit_params, random_state=rng)

    # whole sample analysis with permutations
    ds_tmp = analyze_resampled(estr, X, Y, random_state=0, perm=0,
                               fit_params=fit_params)
    ds_full_sample = analyze_resampled(
        estr,
        X, Y,
        Xorig=Xorig, Yorig=Yorig,
        x_align_ref=ds_tmp.x_weights.values,
        y_align_ref=ds_tmp.y_weights.values,
        perm=permutations,
        addons=addons,
        cvs=[
            ('kfold5', KFold(5)),
        ],
        scorers=addon.mk_scorers_for_cv(),
        n_jobs=n_jobs,
        random_state=rng,
        fit_params=fit_params,
    )
    ds_full_sample.attrs['n_samples'] = len(X)
    ds_full_sample['p_value'] = p_value

    # analysis of subsampled data
    if ns is None:
        if n_min_subsample is None:
            n_min_subsample = (X.shape[1] + Y.shape[1]) + 2
        ns = np.logspace(np.log10(n_min_subsample),
                         np.log10(frac_max_subsample*len(X)),
                         n_subsample_ns, dtype=int)
    # else: use ``ns`` as is

    ds_subsampled = analyze_splithalf(
        estr,
        X, Y,
        Xorig=Xorig, Yorig=Yorig,
        ns=ns,
        n_rep=n_rep_subsample,
        n_perm=n_perm_subsample,
        n_test=0,
        addons=addons,
        cvs=[
            ('kfold5', KFold(5)),
        ],
        scorers=addon.mk_scorers_for_cv(),
        postprocessors=postprocs,
        n_jobs=n_jobs,
        random_state=rng,
        fit_params=fit_params,
    )

    return dict(
        full_sample=ds_full_sample,
        subsampled=ds_subsampled,
    )


@deprecated()
def pairwise_weight_cosine_similarity(ds, qs=(.025, .975)):
    """Calculates statistics of the weight-similarities from pairs of
    synthetic datasets.

    All dimensions except ``n`` and ``x/y_features`` are stacked, and
    statistics are calculated across all these other dimensions.

    For each pair of dataests the minimum cosine similarity across X and Y is
    used.

    Parameters
    ----------
    ds : xr.Dataset
        containing variables ``x_weights`` and ``y_weights`` at the minimum,
        and must have dimension ``n``, ``x/y_feature``, as well as at least one
        other dimension
    qs : 2-tuple of floats between 0 and 1
        quantiles of statistic

    Returns
    -------
    xy_weight_cossim_mean : pd.Series (subsampled_n,)
        mean cosine similarity across pairs of datasets for each subsampled
        sample size
    xy_weight_cossim_q : xr.DataArray (quantile, subsampled_n)
        quantiles of cosine similarity across pairs of datasets for each
        subsampled sample size
    """

    x_weights = ds.x_weights
    y_weights = ds.y_weights

    x_other_dims = [d for d in x_weights.dims if d not in ['n', 'x_feature']]
    y_other_dims = [d for d in y_weights.dims if d not in ['n', 'y_feature']]
    assert x_other_dims == y_other_dims
    assert len(x_other_dims) > 0
    print('Stacking dims:', x_other_dims)
    x_weights = x_weights.stack(OTHERDIM=x_other_dims) \
        .transpose('n', 'OTHERDIM', 'x_feature')
    y_weights = y_weights.stack(OTHERDIM=y_other_dims) \
        .transpose('n', 'OTHERDIM', 'y_feature')

    xy_weight_sims = {}
    for n in ds.n.values:
        xy_weight_sims[n] = np.minimum(
            # this is cosine-similarity
            np.abs(1 - pdist(x_weights.sel(n=n).values, metric='cosine')),
            np.abs(1 - pdist(y_weights.sel(n=n).values, metric='cosine'))
        )
    xy_weight_sims = pd.DataFrame(xy_weight_sims)

    xy_weight_sims_mean = xy_weight_sims.mean(axis=0).rename('y')
    xy_weight_sims_q = xy_weight_sims.quantile(qs, axis=0)
    xy_weight_sims_q.index.names = ['quantile']
    xy_weight_sims_q.columns.names = ['n']

    xy_weight_sims_q = xr.DataArray(xy_weight_sims_q)

    return xy_weight_sims_mean, xy_weight_sims_q
