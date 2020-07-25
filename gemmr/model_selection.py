"""Methods to select among potential models.
"""

import numpy as np
import scipy.stats
from sklearn.decomposition import PCA


__all__ = ['max_min_detector', 'n_components_to_explain_variance']


def _calc_bartlett_lawley_statistic(n, between_set_pc_cov, pX, pY, s):
    """Calculates the Bartlett-Lawley statistic.

    Parameters
    ----------
    n : int
        number of samples
    between_set_pc_cov : (n_pca_components for X, n_pca_components for Y)
        dot-products between X and Y PCA scores
    pX : int
        number of X PCA components to use
    pY : int
        number of Y PCA components to use
    s : int
        number of between-set components

    Returns
    -------
    C : float
        Bartlett-Lawley statistic

    References
    ----------
    Song Y et al., Canonical correlation analysis of high-dimensional data with
    very small sample support, Signal Processing (2016)
    """
    canonical_corrs = np.linalg.svd(between_set_pc_cov[:pX, :pY],
                                    compute_uv=False)
    assert np.all(np.abs(canonical_corrs) < 1)
    p = min(pX, pY)
    # eq (11) in Song et al. (2016)
    C = -2 * (
            n - s - .5 * (pX + pY + 1) + np.sum(1. / canonical_corrs[:s] ** 2)
    ) * np.sum(np.log(1 - canonical_corrs[s:p] ** 2))
    return C


def max_min_detector(X, Y, p_max, alpha=0.01):
    """Hypothesis-test based method to jointly determine number of PCA and
    between-set components.

    Parameters
    ----------
    X : (n_samples, n_X_features)
        data matrix for X
    Y : (n_samples, n_Y_features)
        data matrix for Y
    p_max : int < min(n_X_features, n_Y_features)
        maximum number of components to try for both X and Y

    Returns
    -------
    pXs : list
        best number of PCA components for X, if there are multiple best options
        list contains all of them
    pYs : list
        best number of PCA components for Y, if there are multiple best options
        list contains all of them (i.e. i-th element of ``pXs`` and ``pYs``
        belong together)
    d : int
        best number of between-set components
    best_s : np.ndarray (p_max, p_max)
        inferred number of between-set modes for given number of within-set
        principal components (along axis of matrix)

    References
    ----------
    Song Y et al., Canonical correlation analysis of high-dimensional data with
    very small sample support, Signal Processing (2016)
    """

    if p_max >= min(X.shape[1], Y.shape[1]):
        raise ValueError('p_max must be < min(n_X_features, n_Y_features)')

    n = len(X)

    X_scores = PCA(whiten=True, random_state=0).fit_transform(X)
    Y_scores = PCA(whiten=True, random_state=0).fit_transform(Y)

    X_scores /= np.linalg.norm(X_scores, axis=0, keepdims=True)
    Y_scores /= np.linalg.norm(Y_scores, axis=0, keepdims=True)

    # eq. (4) in Song et al. (2016)
    between_set_pc_cov = np.dot(X_scores.T, Y_scores)

    # NOTE: I use "p" to denote number of features, not "r" as in
    # Song et al. (2016)

    best_s = np.nan * np.empty((p_max, p_max))
    for pX in range(1, p_max + 1):
        for pY in range(1, p_max + 1):
            p = min(pX, pY)
            for s in range(0, p):
                C = _calc_bartlett_lawley_statistic(
                    n, between_set_pc_cov, pX, pY, s)
                df = 2 * (pX - s) * (pY - s)
                T = scipy.stats.chi2(df=df).isf(alpha)
                if C < T:
                    best_s[pX - 1, pY - 1] = s
                    break

    # find best number of PCA components for X and Y, as well as best number of
    # between-set components
    d = int(np.nanmax(best_s))  # best numnber of between-set components
    pX, pY = np.where(best_s == d)
    pX += 1
    pY += 1

    pTot = pX + pY
    min_pTot = np.min(pTot)
    inds = np.where(pTot == min_pTot)[0]

    pXs, pYs = pX[inds], pY[inds]
    return pXs, pYs, d, best_s


def n_components_to_explain_variance(covariance_matrix, variance_threshold=.9):
    """Given a covariance matrix, find the number of components necessary to
    explain at least `variance_threshold` variance.

    Parameters
    ----------
    covariance_matrix : np.ndarray (n_features, n_features)
        a covariance matrix, needs to be symmetric and positive definite
    variance_threshold : float between 0 and 1
        amount of variance to be explained by the determined number of
        components

    Returns
    -------
    n_components : int
        number of components required to explain `variance_threshold` of
        variance
    """
    if not 0 <= variance_threshold <= 1:
        raise ValueError('variance_threshold must be between 0 and 1, '
                         'got {}'.format(variance_threshold))
    if not np.allclose(covariance_matrix, covariance_matrix.T):
        raise ValueError('covariance_matrix must be symmetric')
    evals = np.sort(np.linalg.eigvalsh(covariance_matrix))[::-1]
    assert (evals >= 0).all()
    evals_cum = np.cumsum(evals)
    expl_var = evals_cum / evals_cum[-1]
    n_components = next(i + 1
                        for i in range(len(expl_var))
                        if expl_var[i] >= variance_threshold)
    return n_components
