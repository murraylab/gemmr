"""CCA and PLS estimators.
"""

import warnings

import numpy as np
from scipy.linalg import svd
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist

from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
# from sklearn.utils.extmath import svd_flip
import sklearn.cross_decomposition

from .helpers import _center_scale_xy, _calc_cov, \
    CanonicalCorrelationScorerMixin, SingularMatrixError

__all__ = [
    'SVDPLS', 'SVDCCA', 'NIPALSPLS', 'NIPALSCCA', 'BiViewTransformer'
]


class BiViewTransformer:

    def _preprocess_data_for_fit(self, X, Y, copy):
        """Check if data conforms to expectations and possibly center-scale it.

        Parameters
        ----------
        X : np.ndarray (n_samples, n_X_features)
            data matrix X
        Y : np.ndarray (n_samples, n_Y_features)
            data matrix Y
        copy : bool
            whether a copy of the data is returned

        Returns
        -------
        prepared_X : np.ndarray (n_samples, n_X_features)
            data matrix X
        prepared_Y : np.ndarray (n_samples, n_Y_features)
            data matrix Y
        """

        check_consistent_length([X, Y])
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        X = check_array(X, dtype=np.float64, copy=copy, ensure_min_samples=2)
        Y = check_array(Y, dtype=np.float64, copy=copy, ensure_min_samples=2)

        # Scale (in place)
        X, Y, self.x_mean_, self.y_mean_, self.x_std_, self.y_std_ = \
            _center_scale_xy(X, Y, scale=self.scale, ddof=self.std_ddof)

        return X, Y

    def _preprocess_data_for_transform(self, X, Y, copy=True):
        """Apply same transformations to new data as used before fitting.

        Parameters
        ----------
        X : np.ndarray (n_samples, n_X_features) or None
            data matrix X
        Y : np.ndarray (n_samples, n_Y_features) or None
            data matrix Y
        copy : bool
            whether a copy of the data is returned

        Returns
        -------
        prepared_X : np.ndarray (n_samples, n_X_features)
            data matrix X
        prepared_Y : np.ndarray (n_samples, n_Y_features)
            data matrix Y
        """

        if X is not None:
            X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
            # Normalize
            X -= self.x_mean_
            X /= self.x_std_

        if Y is not None:
            Y = check_array(Y, ensure_2d=False, copy=copy, dtype=FLOAT_DTYPES)
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
            Y -= self.y_mean_
            Y /= self.y_std_

        return X, Y

    def transform(self, X, Y=None, copy=True):
        """Apply the previously fitted estimator to new data.

        Parameters
        ----------
        X : np.ndarray (n_samples, n_X_features)
            data matrix X
        Y : np.ndarray (n_samples, n_Y_features)
            data matrix Y
        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.

        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise. If X is
        None returns (None, y_scores).
        """

        check_is_fitted(self)
        X, Y = self._preprocess_data_for_transform(X, Y, copy)

        if X is not None:
            Xt = X @ self.x_rotations_
        else:
            Xt = None

        if Y is not None:
            Yt = Y @ self.y_rotations_

            return Xt, Yt

        return Xt


def _select_weight_signs(wX, wY, x_align_ref):
    if x_align_ref is None:
        return wX, wY  # do nothing
    else:
        # implicitly assume that wX and wY have same number of columns
        n_modes = min(wX.shape[1], x_align_ref.shape[1])
        for m in range(n_modes):
            sgn = np.sign(wX[:, m] @ x_align_ref[:, m])
            wX[:, m] *= sgn
            wY[:, m] *= sgn
        return wX, wY


class SVDPLS(BaseEstimator, BiViewTransformer):
    """Partial Least Squares estimators based on singular value decomposition.

    Parameters
    ----------
    n_components : int >= 1
        number of between-set components to estimate
    covariance : str
        must be 'empirical'
    scale : bool
        whether to divide each feature by its standard deviation before fitting
    std_ddof : int >= 0
        when calculating standard deviations and covariances, they are
        normalized by ``1 / (n - std_ddof)``


    Attributes
    ----------
    covs_: np.ndarray (n_components,)
        contains the covariances between scores. This is the quantity that is
        maximized by PLS
    assocs_: np.ndarray (n_components,)
        Identical to covs_. ``assocs_`` is the common identifier used in in
        ``SVDPLS``, ``SVDCCA``, ``NIPALSPLS`` and ``NIPALSCCA`` for the
        association strength that is optimized by each particular method
    corrs_ : np.ndarray (n_components_,)
        Pearson correlations between `X` and `Y` scores for each component
    """

    def __init__(self, n_components=1, covariance='empirical', scale=False,
                 std_ddof=0, calc_loadings=False):
        self.n_components = n_components
        self.covariance = covariance
        self.scale = scale
        self.std_ddof = std_ddof
        self.calc_loadings = calc_loadings

    def _postprocess(self, X, Y, U, V, s):
        self.covs_ = s[:self.n_components]

        x_scores = np.dot(X, U[:, :self.n_components])
        y_scores = np.dot(Y, V[:, :self.n_components])
        self.corrs_ = np.array([
            pearsonr(x_scores[:, c], y_scores[:, c])[0]
            for c in range(self.n_components)
        ])

        return U, V, s

    def _calc_K(self, between_cov, X, Y):
        return between_cov

    def fit(self, X, Y, copy=True, x_align_ref=None):
        """Fit the estimator.

        Parameters
        ----------
        X : np.ndarray (n_samples, n_X_features)
            data matrix X
        Y : np.ndarray (n_samples, n_Y_features)
            data matrix Y
        copy : bool
            Whether to copy X and Y, or perform in-place normalization.
        x_align_ref : np.ndarray (n_X_features, n_modes) or None
            if not None, sign ambiguity of weights is resolved by picking their
            sign such that they have positive overlap with columns in
            ``x_align_ref``. Ignored if None.

        Returns
        -------
        self : instance of this estimator
            fitted estimator
        """

        y_is_1d = (Y.ndim == 1)  # in this case Y-rotations, scores will also
        # be returned as 1d
        X, Y = self._preprocess_data_for_fit(X, Y, copy)

        if self.covariance == 'empirical':
            between_cov = np.dot(X.T, Y) / (len(X) - self.std_ddof)
        else:
            raise ValueError('Invalid covariance: {}'.format(self.covariance))

        K = self._calc_K(between_cov, X, Y)

        try:
            U, s, Vh = svd(K, full_matrices=False)
        except np.linalg.LinAlgError:
            print('SVD not converged:', X.shape, Y.shape)
            assert np.isfinite(K).all()
            raise SingularMatrixError('SVD not converged')

        V = Vh.T
        U, V = _select_weight_signs(U, V, x_align_ref)

        U, V, s = self._postprocess(X, Y, U, V, s)

        self.x_rotations_ = U[:, :self.n_components]
        self.y_rotations_ = V[:, :self.n_components]
        self.x_scores_ = np.dot(X, self.x_rotations_)
        self.y_scores_ = np.dot(Y, self.y_rotations_)
        if self.calc_loadings:
            self.x_loadings_ = self.get_x_loadings(X)
            self.y_loadings_ = self.get_y_loadings(Y)
            self.yx_redundancies_ = (self.corrs_**2) * \
                                    (self.y_loadings_**2).mean(0)
            self.xy_redundancies_ = (self.corrs_ ** 2) * \
                                    (self.x_loadings_ ** 2).mean(0)
        self.assocs_ = s

        if y_is_1d:  # for compatibility with sklearn.cross_decomposition
            self.y_rotations_ = self.y_rotations_[:, 0]
            self.y_scores_ = self.y_scores_[:, 0]

        return self

    def fit_transform(self, X, Y, **fit_params):
        """Fit the estimator and return the resulting scores

        Parameters
        ----------
        X : np.ndarray (n_samples, n_X_features)
            data matrix X
        Y : np.ndarray (n_samples, n_Y_features)
            data matrix Y
        fit_params : dict
            ignored

        Returns
        -------
        x_scores : np.ndarray (n_samples, n_modes)
            learned scores for X
        y_scores : np.ndarray (n_samples, n_modes)
            learned scores for Y
        """
        self.fit(X, Y, **fit_params)
        return self.x_scores_, self.y_scores_

    def get_x_loadings(self, X):
        return 1 - cdist(X.T, self.x_scores_.T, metric='correlation')

    def get_y_loadings(self, Y):
        return 1 - cdist(Y.T, self.y_scores_.T, metric='correlation')


class SVDCCA(SVDPLS, CanonicalCorrelationScorerMixin):
    """Canonical Correlation Analysis estimator based on singular value
    decomposition.

    Parameters
    ----------
    n_components : int >= 1
        number of between-set components to estimate
    covariance : str
        must be 'empirical'
    scale : bool
        whether to divide each feature by its standard deviation before fitting
    std_ddof : int >= 0
        when calculating standard deviations and covariances, they are
        normalized by ``1 / n-std_ddof``
    cov_out_of_bounds : str
        if fitting results in a canonical correlation > 1, which indicates some
        problem, potentially that too few samples were used raise an error if
        ``cov_out_of_bounds=='raise'``, set association strengths and weight
        vectors to ``np.nan`` if ``cov_out_of_bounds=='nan'``, or ignore the
        problem if ``cov_out_of_bounds == 'ignore'``
    normalize_weights : bool (default True)
        If ``normalize_weights == False`` weights are calculated as in Härdle
        and Simar (2015). In this case they are not normalized (i.e.
        || w ||_2 != 1). Set ``normalize_weights`` to True to get normalized
        weights.

    Attributes
    ----------
    corrs_ : np.ndarray (n_components,)
        contains the canonical correlations. This is the quantity that's
        maximized by CCA
    assocs_: np.ndarray (n_components,)
        Identical to corrs_. ``assocs_`` is the common identifier used in in
        ``SVDPLS``, ``SVDCCA``, ``NIPALSPLS`` and ``NIPALSCCA`` for the
        association strength that is optimized by each particular method


    References
    ----------
    Härdle and Simar, Applied Multivariate Statistical Analysis, Chapter 16,
    Springer (2015)
    """

    def __init__(self, n_components=1, covariance='empirical', scale=False,
                 std_ddof=1, cov_out_of_bounds='nan', normalize_weights=True,
                 calc_loadings=False):
        super().__init__(n_components=n_components, covariance=covariance,
                         scale=scale, std_ddof=std_ddof,
                         calc_loadings=calc_loadings)
        self.cov_out_of_bounds = cov_out_of_bounds
        self.normalize_weights = normalize_weights

    def _postprocess(self, X, Y, U, V, s):

        if np.any(np.abs(s) > 1):
            if self.cov_out_of_bounds == 'raise':
                # print(X.shape, Y.shape)
                raise ValueError("Canonical correlations > 1: something's "
                                 "rotten")
            elif self.cov_out_of_bounds == 'nan':
                s = np.nan * s
                U = np.nan * U[:, :self.n_components]
                V = np.nan * V[:, :self.n_components]
            elif self.cov_out_of_bounds == 'ignore':
                pass
            else:
                raise ValueError('Invalid cov_out_of_bounds: '
                                 '{}'.format(self.cov_out_of_bounds))

        U = np.dot(self.X_whitener_, U)
        V = np.dot(self.Y_whitener_, V)

        if self.normalize_weights:
            U /= np.linalg.norm(U, axis=0, keepdims=True)
            V /= np.linalg.norm(V, axis=0, keepdims=True)

        self.corrs_ = s[:self.n_components]

        return U, V, s

    def _calc_K(self, between_cov, X, Y):

        n_samples = len(X)
        if (n_samples < X.shape[1]) or (n_samples < Y.shape[1]):
            raise SingularMatrixError(
                'Not enough samples ({}) for given number of features (X: {}, '
                'Y: {})'.format(n_samples, X.shape[1], Y.shape[1]))

        self.X_whitener_ = _S_invsqrt(
            X, ddof=self.std_ddof, if_singular='raise')
        self.Y_whitener_ = _S_invsqrt(
            Y, ddof=self.std_ddof, if_singular='raise')
        return self.X_whitener_.dot(between_cov).dot(self.Y_whitener_)


def _S_invsqrt(X, ddof=0, min_sval=1e-10, if_singular='raise'):
    """Returns S_{XX} ^ {-1/2} where S_{XX} is covariance matrix of X.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_features)
        data matrix (NOT covariance matrix)
    ddof : int >= 0
        degrees of freedom for calculating covariance matrix
    min_sval : float
        ``X`` is considered singular if its minimum singular value is
        < ``min_sval``
    if_singular : str
        if 'raise' raise ``SingularMatrixError``, if 'warn' issue a warning

    Returns
    -------
    S_ : np.ndarray (n_features, n_features)
        square-root of inverse covariance matrix
    """
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    V = Vh.T

    sinv = np.where(s > min_sval, 1 / s, 0)

    msqrt = V.dot(np.diag(np.sqrt((len(X) - ddof)) * sinv)).dot(Vh)

    if s.min() < min_sval:
        msg = 'min sval = {} < {}'.format(s.min(), min_sval)
        if if_singular == 'raise':
            raise SingularMatrixError(msg)
        elif if_singular == 'warn':
            warnings.warn(msg, RuntimeWarning)
        else:
            raise ValueError('Invalid `if_singular`:', if_singular)

    assert np.allclose(msqrt.dot(1/(len(X)-ddof)*np.dot(X.T, X)).dot(msqrt),
                       np.eye(X.shape[1])
                       )

    return msqrt


class NIPALSPLS(sklearn.cross_decomposition.PLSCanonical):
    """Identical to `sklearn.cross_decomposition.PLSCanonical`, except that fit
    creates additional attributes for compatibility with `SVDPLS` and `SVDCCA`:

    Attributes
    ----------
    corrs_ : np.ndarray (n_components,)
        contains the canonical correlations
    covs_: np.ndarray (n_components,)
        contains the covariances between scores. This is the quantity that is
         maximized by PLS
    assocs_: np.ndarray (n_components,)
        Identical to corrs_. ``assocs_`` is the common identifier used in in
        ``SVDPLS``, ``SVDCCA``, ``NIPALSPLS`` and ``NIPALSCCA`` for the
        association strength that is optimized by each particular method

    """

    def fit(self, X, Y):

        # save, in order to return consistently shaped Y rotations, scores
        y_is_1d = (Y.ndim == 1)

        super().fit(X, Y)

        self.x_scores_, self.y_scores_ = super().transform(X, Y)
        if np.isfinite(self.x_scores_).all() and \
                np.isfinite(self.y_scores_).all():
            self.corrs_ = np.array(
                [pearsonr(self.x_scores_[:, i], self.y_scores_[:, i])[0]
                 for i in range(self.n_components)]
            )
            self.covs_ = np.array(
                [_calc_cov(self.x_scores_[:, i], self.y_scores_[:, i])
                 for i in range(self.n_components)]
            )
        else:
            self.corrs_ = np.nan * np.empty((self.n_components))
            self.covs_ = np.nan * np.empty((self.n_components))

        self.assocs_ = self.covs_

        # return consistently shaped Y rotations, scores
        if y_is_1d:
            # NOTE: sklearn has deprecated "x_scores_" and "y_scores_", remove eventually!
            self.y_scores_ = self.y_scores_[:, 0]
            self.y_rotations_ = self.y_rotations_[:, 0]

        return self


class NIPALSCCA(sklearn.cross_decomposition.CCA):
    """Identical to `sklearn.cross_decomposition.CCA`, except that fit creates
    additional attributes for compatibility with `SVDPLS` and `SVDCCA`:

    Attributes
    ----------
    corrs_ : np.ndarray (n_components,)
        contains the canonical correlations. This is the quantity that's
        maximized by CCA
    assocs_: np.ndarray (n_components,)
        Identical to corrs_. ``assocs_`` is the common identifier used in in
        ``SVDPLS``, ``SVDCCA``, ``NIPALSPLS`` and ``NIPALSCCA`` for the
        association strength that is optimized by each particular method
   """

    def fit(self, X, Y):

        # save, in order to return consistently shaped Y rotations, scores
        y_is_1d = (Y.ndim == 1)

        super().fit(X, Y)

        self.x_scores_, self.y_scores_ = super().transform(X, Y)
        if np.isfinite(self.x_scores_).all() and \
                np.isfinite(self.y_scores_).all():
            self.corrs_ = np.array(
                [pearsonr(self.x_scores_[:, i], self.y_scores_[:, i])[0]
                 for i in range(self.n_components)]
            )
        else:
            self.corrs_ = np.nan * np.empty((self.n_components))

        self.assocs_ = self.corrs_

        # return consistently shaped Y rotations, scores
        if y_is_1d:
            # NOTE: sklearn has deprecated "x_scores_" and "y_scores_", remove eventually!
            self.y_scores_ = self.y_scores_[:, 0]
            self.y_rotations_ = self.y_rotations_[:, 0]

        return self
