"""Helpers for estimators.
"""

import numpy as np
from scipy.stats import pearsonr


def _center_scale_Xs(*Xs, scale=True, ddof=1):
    """Center X, Y and scale if the scale parameter==True.

    This function is essentially a copy of
    :func:`sklearn.cross_decomposition.pls_._center_scale_xy`,
    generalized to more than 2 inputs and allowing to specify ddof (forwarded
    to std).

    CAREFUL: Default behavior in
    :func:`sklearn.cross_decomposition.pls_._center_scale_xy` is ddof=1,
    but sklearn.preprocessing.StandardScaler uses ddof=0.

    CAREFUL: scaling is done INPLACE

    Returns
    -------
        X, Y, x_mean, y_mean, x_std, y_std
    """

    Xs = list(Xs)

    means, stds = [], []
    for i in range(len(Xs)):
        mean = Xs[i].mean(axis=0)
        Xs[i] -= mean
        means.append(mean)

        if scale:
            std = Xs[i].std(axis=0, ddof=ddof)
            std[std == 0.0] = 1.0
            Xs[i] /= std
        else:
            std = np.ones(Xs[i].shape[1])
        stds.append(std)
    return Xs, means, stds


def _center_scale_xy(X, Y, scale=True, ddof=1):
    """Center X, Y and scale if the scale parameter==True.

    This function is essentially a copy of
    :func:`sklearn.cross_decomposition.pls_._center_scale_xy`,
    generalized to allow specifying ddof (forwarded to std).

    CAREFUL: Default behavior in
    :func:`sklearn.cross_decomposition.pls_._center_scale_xy` is ddof=1,
    but sklearn.preprocessing.StandardScaler uses ddof=0.

    CAREFUL: scaling is done INPLACE

    Returns
    -------
        X, Y, x_mean, y_mean, x_std, y_std
    """
    Xs, means, stds = _center_scale_Xs(X, Y, scale=scale, ddof=ddof)
    return Xs[0], Xs[1], means[0], means[1], stds[0], stds[1]


def _calc_cov(x, y):
    """Calculate covariance between x and y.

    Parameters
    ----------
    x : np.ndarray (n_samples,)
    y : np.ndarray (n_samples,)

    Returns
    -------
    cov : float
    """
    if (x.ndim != 1) or (y.ndim != 1):
        raise ValueError('x and y must have exactly 1 dimension')
    return np.dot(x - x.mean(), y - y.mean()) / (len(x) - 1)


def _calc_corr(x, y):
    """Calculate Pearson correlation between `x` and `y`

    Parameters
    ----------
    x : np.ndarray (n_samples,)
    y : np.ndarray (n_samples,)

    Returns
    -------
    Pearson correlation : float
    """
    return pearsonr(x, y)[0]


def cov_transform_scorer(estimator, X, Y, ftr=0):
    """Calculates the covariance between an estimator's scores.

    Parameters
    ----------
    estr : sklearn-style estimator
        needs to be fitted and provide method ``transform``
    X : np.ndarray (n_samples, n_X_features)
        data matrix X
    Y : np.ndarray (n_samples, n_Y_features)
        data matrix Y
    ftr : int >= 0
        covariance is calculated between the ``ftr``-th columns of the scores

    Returns
    -------
    cov : float
    """
    X_scores, Y_scores = estimator.transform(X, Y)
    if Y_scores.ndim == 1:
        Y_scores = Y_scores.reshape(-1, 1)
    return _calc_cov(X_scores[:, ftr], Y_scores[:, ftr])


def pearson_transform_scorer(estimator, X, Y, ftr=0):
    """Calculates the Pearson correlation between an estimator's scores.

    Parameters
    ----------
    estr : sklearn-style estimator
        needs to be fitted and provide method ``transform``
    X : np.ndarray (n_samples, n_X_features)
        data matrix X
    Y : np.ndarray (n_samples, n_Y_features)
        data matrix Y
    ftr : int >= 0
        covariance is calculated between the ``ftr``-th columns of the scores

    Returns
    -------
    corr : float
    """
    X_scores, Y_scores = estimator.transform(X, Y)
    if Y_scores.ndim == 1:
        Y_scores = Y_scores.reshape(-1, 1)
    return pearsonr(X_scores[:, ftr], Y_scores[:, ftr])[0]


class CanonicalCorrelationScorerMixin:
    """Mixin class for CCA-like estimators"""
    _estimator_type = "transformer"

    def score(self, X, Y, ftr=0):
        """Returns the pearson correlation of the `ftr`-th canonical variates
        (scores).

        Parameters
        ----------
        X, Y : array-like, shape = (n_samples, n_features)
            Test samples

        ftr : int
            The `ftr`-th canonical variates' correlation will be returned

        Returns
        -------
        score : float
            Pearson correlation of `ftr`-th canonical variates'.
        """

        if ftr > self.n_components:
            raise ValueError('`ftr` can be at most self.n_components_')

        return pearson_transform_scorer(self, X, Y, ftr=ftr)

    def fit_transform(self, X, Y, **fit_params):
        """Fit estimator and return predictions on training data.

        Parameters
        ----------
        X, Y : nd-arrays with dimensions ``(n_samples x n_features_[X or y])``
            input data

        Returns
        -------
        X_scores : np.ndarray (n_samples, n_components_)
            Transformed training data for X, i.e. the scores / canonical
            variates
        Y_scores :  np.ndarray (n_samples, n_components_)
            Transformed training data for Y, i.e. the scores / canonical
            variates
        """

        return self.fit(X, Y, **fit_params).transform(X, Y)


class SingularMatrixError(Exception):
    pass
