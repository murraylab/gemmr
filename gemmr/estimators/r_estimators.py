"""Estimators wrapped around R-functions.
"""
import warnings
import itertools
import numbers

import numpy as np
from scipy.stats import pearsonr

from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_consistent_length
from sklearn.model_selection import check_cv
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES

# from joblib import Parallel, delayed

from rpy2 import robjects
from rpy2.robjects.packages import importr

from rpy2.robjects import pandas2ri

from .helpers import _center_scale_xy, CanonicalCorrelationScorerMixin

__all__ = ['SparsePLS']

pandas2ri.activate()


class SparsePLS(BaseEstimator, CanonicalCorrelationScorerMixin):
    """SparsePLS (or sparse "CCA") algorithm from Witten et al. (2009).

    This is a sklearn-style wrapper for the function ``CCA`` from R-package
    ``PMA``.

    This estimator requires that ``rpy2`` is installed and that the package
    "PMA" is installed in R.

    Parameters
    ----------
    n_components : int >= 1
        number of between-set association modes to determine
    typex : str ('standard' or 'ordered')
        see PMA documentation
    typey : str ('standard' or 'ordered')
        see PMA documentation
    penaltyxs :
        penaltyx is a float between 0 and 1, larger values correspond to less
        peanlization
    penaltyys :
        penaltyy is a float between 0 and 1, larger values correspond to less
        penalization
    optimize_penalties : False or 'cv'
        if ``False``, penalties must be floats or None (in which case penalties
        are set to 1, i.e. no penalty). If ``'cv'`` penalties are determined
        through cross-validation to optimize
        the canonical correlation corr(Xu, Yv) (as suggested in Witten et al.
        (2009)
    cv : int
        if ``optimize_penalties == 'cv'``, number of folds for
        cross-validation, ignored otherwise
    verbose : bool or int
        verbosity level, if verbose > 0, the Python code will print some status
        messages, ``verbose > 1`` is passed as argument ``trace`` to
        ``PMA.CCA``

    References
    ----------
    * Witten et al, A penalized matrix decomposition, with applications to
      sparse principal components and canonical correlation analysis,
      Biostatistics (2009)
    * https://cran.r-project.org/web/packages/PMA/index.html
    """

    def __init__(self, n_components=1, typex='standard', typey='standard',
                 penaltyxs=None, penaltyys=None, penalty_pairing='product',
                 niter=15,
                 scale=False, std_ddof=1,
                 optimize_penalties='cv',
                 cv=5,
                 verbose=False, ):
        self.n_components = n_components
        self.typex = typex
        self.typey = typey
        self.penaltyxs = penaltyxs
        self.penaltyys = penaltyys
        self.penalty_pairing = penalty_pairing
        self.niter = niter
        self.scale = scale
        self.std_ddof = std_ddof
        self.optimize_penalties = optimize_penalties
        self.cv = cv
        self.verbose = verbose

    def fit(self, X, Y, copy=True, groups=None):
        """Fit estimator

        Parameters
        ----------
        X : np.ndarray (n_samples, n_X_features)
            data matrix X
        Y : np.ndarray (n_samples, n_Y_features)
            data matrix Y

        Returns
        -------
        self : instance of this estimator
            fitted estimator
        """

        # copy since this will contains the residuals (deflated) matrices
        check_consistent_length(X, Y)
        X = check_array(X, dtype=np.float64, copy=copy,
                        ensure_min_samples=2)
        Y = check_array(Y, dtype=np.float64, copy=copy, ensure_2d=False)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        if (X.shape[1] < 3):
            raise ValueError('SparsePLS requires at least 3 features in '
                             'dataset X, got {}'.format(X.shape[1]))
        if (Y.shape[1] < 2):
            raise ValueError('SparsePLS requires at least 2 features in '
                             'dataset Y, got {}'.format(Y.shape[1]))

        # Scale (in place)
        X, Y, self.x_mean_, self.y_mean_, self.x_std_, self.y_std_ = (
            _center_scale_xy(X, Y, self.scale, ddof=self.std_ddof))

        # set number of components to return
        _n_components = self.n_components if self.n_components is not None \
            else min(X.shape[1], Y.shape[1])
        if _n_components > min(Y.shape[1], X.shape[1]):
            raise ValueError("Invalid number of components n_components=%d"
                             " with X of shape %s and Y of shape %s."
                             % (_n_components, str(X.shape), str(Y.shape)))

        min_matrix_rank = min([np.linalg.matrix_rank(M) for M in [X, Y]])
        self.n_components_ = min(_n_components, min_matrix_rank)
        if self.n_components_ > 1:
            warnings.warn('More than 1 mode selected: NOT TESTED')

        robjects.r('rm(list = ls())')
        R_pma = importr('PMA')

        def fit_scca(X, Y, penaltyx, penaltyy, v_init=None):
            return R_pma.CCA(
                X, Y,
                typex=self.typex,
                typez=self.typey,
                penaltyx=penaltyx,
                penaltyz=penaltyy,
                K=self.n_components_,
                niter=self.niter,
                v=robjects.NULL if v_init is None else v_init,
                standardize=False,  # NOTE: explicitly done above if necessary
                trace=self.verbose > 1,
            )

        self.penaltyx_, self.penaltyy_ = \
            self._check_penalties(fit_scca, X, Y, groups)

        Rres = self.Rres_ = fit_scca(X, Y, self.penaltyx_, self.penaltyy_)

        # --- check for consistency ---
        assert Rres.rx2('penaltyx')[0] == self.penaltyx_
        assert Rres.rx2('penaltyz')[0] == self.penaltyy_

        assert Rres.rx2('typex')[0] == self.typex
        assert Rres.rx2('typez')[0] == self.typey

        assert Rres.rx2('K')[0] == self.n_components_
        assert Rres.rx2('niter')[0] == self.niter

        # --- expose interesting pieces of Rres ---
        self.corrs_ = np.asarray(Rres.rx2('cors'))
        self.d_ = np.asarray(Rres.rx2('d'))
        self.x_weights_ = np.asarray(Rres.rx2('u'))
        self.y_weights_ = np.asarray(Rres.rx2('v'))

        # ignored in Rres: call, upos, uneg, vpos, vneg, v.init

        # The returned correlations (stored in self.corrs_) are obtained by
        # matrix-multiplying original data matrices with weight matrices,
        # i.e. the weight matrices are the rotations
        self.x_rotations_ = self.x_weights_
        self.y_rotations_ = self.y_weights_

        self.x_scores_ = np.dot(X, self.x_rotations_)
        self.y_scores_ = np.dot(Y, self.y_rotations_)

        assert np.allclose(
            self.corrs_,
            [pearsonr(self.x_scores_[:, k], self.y_scores_[:, k])[0]
             for k in range(self.n_components_)]
        )

        self.covs_ = np.array(
            [np.cov(self.x_scores_[:, k], self.y_scores_[:, k])[0, 1]
             for k in range(self.n_components_)])

        self.assocs_ = self.covs_

        # ignore "v.init" in res

        return self

    def transform(self, X, Y, copy=True):
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
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
        """
        if self.n_components_ > 1:
            warnings.warn('More than 1 mode selected: NOT TESTED')

        check_is_fitted(self)
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        # Normalize
        X -= self.x_mean_
        X /= self.x_std_

        Y = check_array(Y, ensure_2d=False, copy=copy, dtype=FLOAT_DTYPES)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        Y -= self.y_mean_
        Y /= self.y_std_

        Xt = np.dot(X, self.x_rotations_)
        Yt = np.dot(Y, self.y_rotations_)

        return Xt, Yt

    def fit_transform(self, X, Y, copy=True, groups=None):
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
        self.fit(X, Y, copy=copy, groups=groups)
        return self.x_scores_, self.y_scores_

    def _check_penalties(self, fitfun, X, Y, groups):
        """

        Note: In case of ties, only one penalty combination will be returned:
            the one that ``np.argmax`` or ``np.argmin`` selects

        Parameters
        ----------
        fitfun : callable
            defined within :func:`SparsePLS.fit`
        X : np.ndarray (n_samples, n_X_features)
            :math:`X` data matrix
        Y : np.ndarray (n_samples, n_Y_features)
            :math:`Y` data matrix
        groups : groups
            group - parameter for cross-validator

        Returns
        -------
        best_penalty_pair : 2-tuple of floats between 0 and 1
            best penalties for X and Y
        """

        penaltyxs = self.penaltyxs
        penaltyys = self.penaltyys

        if self.optimize_penalties is False:

            penaltyxs = _check_penalty(penaltyxs)
            penaltyys = _check_penalty(penaltyys)
            return penaltyxs, penaltyys

        elif self.optimize_penalties == 'cv':  # do cross-validation

            penaltyxs = _check_penalty_cv(penaltyxs)
            penaltyys = _check_penalty_cv(penaltyys)
            if self.penalty_pairing == 'product':
                penalty_pairs = list(itertools.product(penaltyxs, penaltyys))
            elif self.penalty_pairing == 'zip':
                penalty_pairs = list(zip(penaltyxs, penaltyys))
            else:
                raise ValueError('Invalid penalty_pairing')

            if self.optimize_penalties == 'cv':
                cv = check_cv(self.cv)
                cv_scores = [
                    _fit_and_score_spls(fitfun, X, Y, train, test,
                                        penalty_pair)
                    for penalty_pair in penalty_pairs
                    for train, test in cv.split(X, Y, groups=groups)
                ]
                cv_scores = np.reshape(
                    cv_scores, (len(penalty_pairs), cv.get_n_splits())) \
                    .mean(1)  # average across CV-splits
                best_penalty_pair = np.asarray(penalty_pairs)[
                    np.where(cv_scores == cv_scores.max())[0]
                ]

                best_penalty_pair = _select_best_penalties(best_penalty_pair)

                if self.verbose:
                    print(
                        '[SparsePLS] penalties = {}'.format(best_penalty_pair))

                return best_penalty_pair

        else:
            raise ValueError('Invalid optimize_penalties')


def _check_penalty_cv(penalties):
    if penalties is None:
        penalties = np.linspace(0, 1, 6)
    elif isinstance(penalties, numbers.Number):
        penalties = [penalties]
    else:
        if not np.all([(0 <= p <= 1) for p in penalties]):
            raise ValueError('All penaltyxs must be between 0 and 1')
    return penalties


def _check_penalty(penalty):
    if penalty is None:
        penalty = 1
    elif not isinstance(penalty, numbers.Number):
        if len(penalty) == 1:
            penalty = penalty[0]
        else:
            raise ValueError('Invalid penaltyxs')
    if not (0 <= penalty <= 1):
        raise ValueError('penaltyx must be between 0 and 1')
    return penalty


def _fit_and_score_spls(fitfun, X, Y, train, test, penalties, component=0):
    """Fit SparsePLS and return the fit score

    Fitting is assumed to be done internally in ``fitfun`` (defined within
    :func:`SparsePLS.fit`). The score is the Pearson correlation between X and
    Y scores, i.e. :math:`corr(Xu, Yv)`, as suggested in Witten et al. (2009)

    Parameters
    ----------
    fitfun : callable
        fits the estimator and returns an object containing results. This
        function is defined within :func:`SparsePLS.fit`
    X : np.ndarray (n_samples, n_X_features)
        :math:`X` data matrix
    Y : np.ndarray (n_samples, n_Y_features)
        :math:`Y` data matrix
    train : np.ndarray (n_train_samples,)
        indices of training samples
    test : np.ndarray (n_test_samples,)
        indices of test samples
    penalties : 2-tuple of floats between 0 and 1
        penalties for X and Y
    component : int
        the between-set component that is used for scoring, should probably
        always be 0

    Returns
    -------
    corr : float
        Pearson correlation between X and Y scores
    """

    assert len(penalties) == 2

    Rres = fitfun(X[train], Y[train], *penalties)
    u = np.asarray(Rres.rx2('u'))[:, [component]]
    v = np.asarray(Rres.rx2('v'))[:, [component]]

    Xt = np.dot(X[test], u[:, 0])
    Yt = np.dot(Y[test], v[:, 0])

    return pearsonr(Xt, Yt)[0]


def _select_best_penalties(candidate_penalties):
    """Selects the best penalties

    The best pair of penalties is chosen such that:

    1. The sum of penalties is minimal, i.e. the most penalization in total
    2. If there are several such penalty pairs, the one in which the
       difference between X and Y penalties is minimal is chosen
    3. If there are several such penalty pairs, the first one in the list
       is chosen

    Parameters
    ----------
    candidate_penalties : np.ndarray (n_candidate_penalties, 2)
        The 2 columnns contain, respectively, candidate penalties for X and Y

    Returns
    -------
    best_penalties : 2-tuple of floats between 0 and 1
        best penalties for X and Y
    """

    regul_sums = candidate_penalties.sum(1)
    candidate_penalties = candidate_penalties[
        np.where(regul_sums == regul_sums.min())[0]
    ]

    regul_diffs = np.abs(
        candidate_penalties[:, 0] - candidate_penalties[:, 1])
    candidate_penalties = candidate_penalties[
        np.where(regul_diffs == regul_diffs.min())[0]
    ]

    # in case there's still more than one set of regularization parameters,
    # arbitrarily choose the first

    return candidate_penalties[0]
