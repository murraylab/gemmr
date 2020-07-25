"""Setup of model covariance matrices and generation of data.
"""

import numbers
import warnings

import numpy as np
import scipy.spatial
from sklearn.utils import check_random_state


from .util import check_positive_definite
from .model_selection import n_components_to_explain_variance

__all__ = ['setup_model', 'generate_data']


class SubspaceDim1To2Warning(Warning):
    pass


warnings.simplefilter('always', SubspaceDim1To2Warning)


def setup_model(model, random_state=42,
                px=5, py=5, qx=0.9, qy=0.9,
                m=1, c1x=1, c1y=1, ax=-1, ay=-1, r_between=0.3, a_between=-1,
                max_n_sigma_trials=10000, expl_var_ratio_thr=1./2,
                cx=None, cy=None, verbose=False, return_full=False
                ):
    r"""Generate a joint covariance matrix for `X` and `Y`.

    It is assumed that both datasets live in their respective principal
    component coordinate system, i.e. that the within-set covariance matrices
    :math:`\Sigma_{XX}` and :math:`\Sigma_{YY}` are diagonal. The entries of
    the diagonal are set to follow power laws with decay constants `ax` and
    `ay` for `X` and `Y`, respectively, and scaled by `cx` and `cy`.

    For generation of the between-set covariance matrix :math:`\Sigma_{XY}`
    :func:`_mk_Sigmaxy` is called, see there for details.

    Parameters
    ----------
    model: "pls" or "cca"
        whether to return a covariance matrix for CCA or PLS
    random_state : None or int or a random number generator instance
        For reproducibility, a random number generator is instantiated and all
        random numbers are drawn from that
    px : int
        number of features in `X`
    py : int
        number of features in `Y`
    qx : int or float between 0 and 1
        if float, gives the fraction of `px` to use
        (i.e. `q_x <- int(q_x * p_x)`). Specifies the number of dominant basis
        vectors from which to choose one component of the latent mode vectors
        for `X`. See :func:`_mk_Sigmaxy` for details
    qy : int or float between 0 and 1
        if float, gives the fraction of `py` to use
        (i.e. `q_y <- int(q_y * p_y)`). Specifies the number of dominant basis
        vectors from which to choose one component of the latent mode vectors
        for `Y`. See :func:`_mk_Sigmaxy` for details
    m : int
        number of latent cross-modality modes to encode
    c1x : float
        Should usually be 1. All `X` variances will be scaled by this number
    c1y : float
        Should usually be 1. All `Y` variances will be scaled by this number
    ax : float
        should usually be <= 0. Eigenvalues of within-modality covariance for
        `X` are assumed to follow a power-law with this exponent
    ay : float
        should usually be <= 0. Eigenvalues of within-modality covariance for
        `X` are assumed to follow a power-law with this exponent
    r_between : float between 0 and 1
        cross-modality correlation the latent mode vectors should have
    a_between : float
        should usually be <= 0. Higher-order cross-modality correlations are
        scaled by a power-law with this exponent
    max_n_sigma_trials : int >= 1
        number of times an attempt is made to find suitable latent mode
        vectors. See :func:`_mk_Sigmaxy` for details.
    expl_var_ratio_thr : float
        threshold for required within-modality variance along latent mode
        vectors
    cx : np.ndarray
        within-set variances for `X`
    cy : np.ndarray
        within-set variances for `Y`
    verbose : bool
        whether to print status messages
    return_full : bool
        if ``False`` returns only the joint covariance matrix, otherwise return
        more quantities of interest

    Returns
    -------
    Sigma : np.ndarray
        joint covariance matrix for `X` and `Y`
    Sigmaxy_svals : np.ndarray (m,)
        singular values of ``Sigmaxy``, these are the true canonical
        correlations or covariances (if ``model`` is 'cca' or 'pls',
        respectively)
    true_corrs : np.ndarray (m,)
        the encoded cross-modality correlations for each mode
    U : np.ndarray (n_X_features, n_X_features)
        basis vectors for `X`
    V : np.ndarray (n_Y_features, n_Y_features)
        basis vectors for `Y`
    latent_expl_var_ratios_x : np.ndarray (m,)
        explained variance ratio in `X` modality along the latent directions
    latent_expl_var_ratios_y : np.ndarray (m,)
        explained variance ratio in `Y` modality along the latent directions
    U_latent : np.ndarray (n_X_features, m)
        latent mode vectors for `X`
    V_latent : np.ndarray (n_Y_features, m)
        latent mode vectors for `Y`
    cosine_sim_pc1_latentMode_x : (m,)
        cosine similarities between latent mode vectors and PC1 for `X`
    cosine_sim_pc1_latentMode_y : (m,)
        cosine similarities between latent mode vectors and PC1 for `Y`
    latent_mode_vector_algo : str
        'qr__' or 'opti', algorithm with which the latent mode vectors were
        found

    Raises
    ------
    ValueError
        * if the number of requested between-set association modes `m` is
        greater than the minimum of the dimensions of the dominant subspaces
        (as encoded by `qx` and `qy`)
        * if the resulting joint covariance matrix is not positive definite
    NotImplementedError
        * if `model` == 'cca' and `m` > 1
    """
    rng = check_random_state(random_state)

    if max_n_sigma_trials < 1:
        raise ValueError('max_n_sigma_trials must be >= 1')

    if (c1x != 1) or (c1y != 1):
        raise ValueError('c1x != 1 or c1y != 1 is DISCOURAGED')

    if model == 'cca':
        if a_between == 0:
            warnings.warn('a_between == 0 with '
                          'normalization == "cca": ARE YOU SURE?',
                          category=UserWarning)

        assemble_Sigmaxy = _assemble_Sigmaxy_cca

    elif model == 'pls':
        assemble_Sigmaxy = _assemble_Sigmaxy_pls

    else:
        raise ValueError('Invalid model: {}'.format(model))

    if (ax > 0) or (ay > 0) or (a_between > 0):
        raise ValueError('ax, ay, a_latent must be <= 0, got '
                         '{}, {}, {}'.format(ax, ay, a_between))

    if not 0 <= r_between <= 1:
        raise ValueError('Invalid r_latent: {}, must be '
                         '>= 0 and <= 1'.format(r_between))

    # basis vectors for X and Y
    # w.l.o.g. can be assumed to be standard basis
    U = np.eye(px)
    V = np.eye(py)

    if cx is None:
        cx = c1x * np.arange(1, px + 1) ** float(ax)
    else:
        if len(cx) != px:
            raise ValueError('len(cx) != px')
        if verbose:
            print('using given cx, ignoring ax (except for estimating noise '
                  'for definiteness check)')

    if cy is None:
        cy = c1y * np.arange(1, py + 1) ** float(ay)
    else:
        if len(cy) != py:
            raise ValueError('len(cy) != py')
        if verbose:
            print('using given cy, ignoring ay (except for estimating noise '
                  'for definiteness check)')

    Sigmaxx = U.dot(np.diag(cx)).dot(U.T)
    Sigmayy = V.dot(np.diag(cy)).dot(V.T)

    qx = _check_subspace_dimension(Sigmaxx, qx)
    qy = _check_subspace_dimension(Sigmayy, qy)
    if verbose:
        print('qx={}, qy={}'.format(qx, qy))

    if m > min(qx, qy):
        raise ValueError(
            'Number of latent modes (m={}) must be <= min(qx, qy) = {}'.format(
                m, min(qx, qy)))

    true_corrs = r_between * np.arange(1, m + 1) ** float(a_between)
    Sigmaxy, Sigmaxy_svals, true_corrs, \
        latent_expl_var_ratios_x, latent_expl_var_ratios_y, \
        U_latent, V_latent, \
        cosine_sim_pc1_latentMode_x, cosine_sim_pc1_latentMode_y, \
        latent_mode_vector_algo = \
        _mk_Sigmaxy(assemble_Sigmaxy, Sigmaxx, Sigmayy, U, V, m,
                    max_n_sigma_trials, qx, qy, rng, true_corrs,
                    expl_var_ratio_thr=expl_var_ratio_thr, verbose=verbose)

    # assemble covariance matrix Sigma
    Sigma = np.vstack([
        np.hstack([Sigmaxx, Sigmaxy]),
        np.hstack([Sigmaxy.T, Sigmayy]),
    ])

    # check that covariance matrix Sigma is positive definite
    noisex = c1x * (px + 1) ** (ax)
    noisey = c1y * (py + 1) ** (ay)
    # raises ValueError if Sigma is not positive definite
    check_positive_definite(Sigma, min(noisex, noisey))

    if return_full:
        return Sigma, Sigmaxy_svals, true_corrs, U, V, \
               latent_expl_var_ratios_x, latent_expl_var_ratios_y, \
               U_latent, V_latent, \
               cosine_sim_pc1_latentMode_x, cosine_sim_pc1_latentMode_y, \
               latent_mode_vector_algo
    else:
        return Sigma


def _check_subspace_dimension(Sigmaxx, qx):
    """Interpret arguments `qx` and `qy` in :func:`mk_Sigma_model`.

    Parameters
    ----------
    Sigmaxx : np.ndarray (px, px)
        the covariance matrix of the dataset
    px : int
        number of variables in the dataset
    qx : None or int or float

        * if None returns `px`
        * if > `px` returns `px`
        * if 0 <= qx <= 1 returns the number of principle components necessary
            to explain a fraction`qx` of the variance in `Sigmaxx`
        * otherwise returns the same value `qx` that was given

    Returns
    -------
    Dimensionality of "dominant" subspace : int

    Raises
    ------
    ValueError
        - if `qx` < 0
        - if 1 <= px and qx is not an integer
    """

    px = Sigmaxx.shape[1]

    if (qx is None) or (qx == 'all'):
        qx = px
    elif qx == 'force_1':
        qx = 1
    elif isinstance(qx, numbers.Number):

        if qx > px:
            raise ValueError('q cannot be > p')
        elif qx < 0:
            raise ValueError('qx must be positive, got: {}'.format(qx))
        elif 0 <= qx < 1:  # don't allow 1 as this might be ambiguous
            qx = n_components_to_explain_variance(Sigmaxx, qx)

            if (qx == 1) and (px >= 2):
                qx = 2
                warnings.warn(
                    'Subspace dimension changed from 1 to 2',
                    category=SubspaceDim1To2Warning
                )

        elif qx == 1:
            raise ValueError("qx=1 is ambiguous, use either qx='all' or "
                             "qx=1/px")

        else:
            # make sure qx is an integer
            try:
                assert qx == int(qx)
            except (ValueError, AssertionError):
                raise ValueError('Invalid qx: {}'.format(qx))

            qx = int(qx)

    else:
        raise ValueError('Invalid qx: {}'.format(qx))

    return qx


def _mk_Sigmaxy(assemble_Sigmaxy, Sigmaxx, Sigmayy, U, V, m,
                max_n_sigma_trials, qx, qy, rng, true_corrs,
                expl_var_ratio_thr=1./2, verbose=True):
    r"""Generate the between-set covariance matrix :math:`\Sigma_{XY}` (i.e.
    the upper right block of the joint covariance matrix).

    Random directions are chosen for the `X` and `Y` latent mode vectors with
    the constraints that

    * the within-modality variance along these directions is at least
      `expl_var_ratio_thr` x the average variance along any dimension in
      this modality
    * the resulting joint cross-modality covariance matrix must be positive
      definite

    To increase chances of large within-modality variance for randomly chosen
    latent mode vectors they are calculated as a random linear combination of a
    random vector from the first `q_x` (for modality `X`, `q_y` for modality
    `Y`) modes and a random vector from the remaining modes.

    If this is not successful, i.e. if no between-set weight vectors could be
    found that explain enough variance and result in a positive definite
    :math:`\Sigma_{XY}`, an optimization procedure (using differential
    evolution algorithm) is used to maximize the minimum eigenvalue of
    :math:`\Sigma_{XY}`. If that doesn't succeed either, a ValueError is
    raised.

    Parameters
    ----------
    Sigmaxx : np.ndarray (n_X_features, n_X_features)
        covariance-matrix for modality `X`, i.e. the upper left block of the
        joint covariance matrix
    Sigmayy : np.ndarray (n_Y_features, n_Y_features)
        covariance-matrix for modality `Y`, i.e. lower right block of the joint
        covariance matrix
    U : np.ndarray (n_X_features, n_X_features)
        columns of `U` contain basis vectors for `X` data
    V : np.ndarray n_Y_features, n_Y_features)
        columns of `V` contain basis vectors for `Y` data
    m : int >= 1
        number of cross-modality modes to be encoded
    max_n_sigma_trials : int
        number of times an attempt is made to find latent mode vectors
        satisfying constraints
    qx : int
        latent mode vectors for modality `X` are calculated as a random linear
        combination of
        - a random linear combination of the first `q_x` columns of `U`
        - a random linear combination of the remaining columns of `U`
    qy : int
        latent mode vectors for modality `Y` are calculated as a random linear
        combination of
        - a random linear combination of the first `q_y` columns of `V`
        - a random linear combination of the remaining columns of `V`
    rng : random number generator instance
    true_corrs : np.ndarray (m,)
        cross-modality correlations that each latent mode should have
    expl_var_ratio_thr : float
        threshold for required within-modality variance along latent mode
        vectors
    verbose : bool
        whether to print status messages

    Returns
    -------
    Sigmaxy : np.ndarray (n_X_features, n_Y_features)
        cross-modality covariance matrix
    Sigmaxy_svals: np.ndarray (m,)
        singular values of ``Sigmaxy``, these are the true canonical
        correlations or covariances (for CCA or PLS, respectively)
    true_corrs : np.ndarray (m,)
        the cross-modality covariances are calculated as the true correlations
        (given by input argument `true_corrs` times the variances along these
        directions. Should the resulting cross-modality covariances not be in
        descending order, they will be reordered, as will input argument
        `true_corrs` to reflect the change in order
    latent_expl_var_ratios_x : np.ndarray (m,)
        explained variance ratio in `X` modality along the latent directions
    latent_expl_var_ratios_y : np.ndarray (m,)
        explained variance ratio in `Y` modality along the latent directions
    U_ : np.ndarray (n_X_features, m)
        latent mode vectors for `X`
    V_ : np.ndarray (n_Y_features, m)
        latent mode vectors for `Y`
    cosine_sim_pc1_latentMode_x : (m,)
        cosine similarities between latent mode vectors and PC1 for `X`
    cosine_sim_pc1_latentMode_y : (m,)
        cosine similarities between latent mode vectors and PC1 for `Y`
    latent_mode_vector_algo : str
        'qr__' or 'opti', algorithm with which the latent mode vectors were
        found

    Raises
    ------
    ValueError
        if no between-set weight vectors could be found that explain enough
        variance and result in a positive definite :math:`\Sigma_{XY}`
    """

    for _find_latent_mode_vectors_alg in [
        _find_latent_mode_vectors_qr,
        _find_latent_mode_vectors_opti,
        _find_latent_mode_vectors_pc1,
    ]:

        Sigmaxy, Sigmaxy_svals, U_, V_, \
            latent_expl_var_ratios_x, latent_expl_var_ratios_y, \
            min_eval, true_corrs, latent_mode_vector_algo = \
            _find_latent_mode_vectors_alg(
                Sigmaxx, Sigmayy, U, V, assemble_Sigmaxy, expl_var_ratio_thr,
                m, max_n_sigma_trials, qx, qy, rng, true_corrs, verbose)

        if min_eval > 0:
            break

    else:  # all algorithms failed
        raise ValueError("Couldn't find suitable weight vectors")

    if verbose:
        print('Found latent mode vectors with latent_mode_vector_algo',
              latent_mode_vector_algo)

    cosine_sim_pc1_latentMode_x = np.asarray(
        [1 - scipy.spatial.distance.cosine(U[:, 0], U_[:, modei])
         for modei in range(m)])
    cosine_sim_pc1_latentMode_y = np.asarray(
        [1 - scipy.spatial.distance.cosine(V[:, 0], V_[:, modei])
         for modei in range(m)])

    return Sigmaxy, Sigmaxy_svals, true_corrs, \
        latent_expl_var_ratios_x, latent_expl_var_ratios_y, \
        U_, V_, cosine_sim_pc1_latentMode_x, cosine_sim_pc1_latentMode_y, \
        latent_mode_vector_algo


def _find_latent_mode_vectors_pc1(Sigmaxx, Sigmayy, U, V, assemble_Sigmaxy,
                                  expl_var_ratio_thr, m, max_n_sigma_trials,
                                  qx, qy, rng, true_corrs, verbose):
    r"""Selects the first principal component axes as weight vectors.

    If only :math:`m=1` mode is sought, and if the variances in :math:`X` and
    :math:`Y` are standardized to be 1 along PC1 then, for PLS, the covariances
    along the weight vectors are identical to the correlation. Thus, for both
    PLS and CCA the between-set covariance matrix is given by :
    math:`\Sigma_{XY}=r_\mathrm{true}`

    .. math::
        \Sigma_{XY} = r_\mathrm{true} \vec{u}_1 \vec{v}_1^T

    where :math:`\vec{u}_1` and :math:`\vec{v}_1` are the first principal
    component axes for :math:`X` and :math:`Y`, respectively. If the overall
    coordinate system is the principal component coordinate system
    :math:`\Sigma_{XY}` is a :math:`p_x \times p_y` matrix with
    :math:`r_\mathrm{true}` in the top left corner and 0 everywhere else.

    The block matrix :math:`\Sigma` is positive definite if and only if its
    Schur complement
    :math:`\Sigma_{XX} - \Sigma_{XY} \Sigma_{YY} \Sigma_{XY}^T`
    is positive definite. :math:`\Sigma_{XY} \Sigma_{YY} \Sigma_{XY}^T`
    simplifies to :math:`r_\mathrm{true}^2 \vec{u}_1 \vec{u}_1^T`. As (in the
    principal component coordinate system) :math:`\Sigma_{XX}` is diagonal and,
    by assumption the top-left element is 1, and :math:`r_\mathrm{true}^2 < 1`,
    all entries on the diagonal of :math:`\Sigma_{XX}` are greater than 0.
    Thus, :math:`\Sigma` is positive definite when the weight vectors are
    chosen as the first principal component axes.

    Parameters
    ----------
    Sigmaxx : np.ndarray (px, px)
        within-set covariance matrix for `X`
    Sigmayy : np.ndarray (py, py)
        within-set covariance matrix for `Y`
    U : np.ndarray (px, m)
        weight vectors for `X`
    V : np.ndarray (py, m)
        weight vectors for `Y`
    assemble_Sigmaxy : function
        either `_assemble_Sigmaxy_pls` or `_assemble_Sigmaxy_cca`
    expl_var_ratio_thr : float
        the ratio of the amount of variance along the first mode vectors in
        `X` and `Y` to the mean variance along a mode in `X` and `Y` needs to
        surpass this number.
    m : int >= 1
        number of cross-modality modes to be encoded
    max_n_sigma_trials : int
        maximum number of attempts made to find a linear combination of
        dominant and low-variance subspace components for the weight vectors
        such that both enough variance is explained and the resulting joint
        covariance matrix :math:`\Sigma` is positive definite
    qx : int
        dimensionality of dominant subspace for `X`
    qy : int
        dimensionality of dominant subspace for `Y`
    rng : random number generator instance
        for reproducibility, all random numbers will be drawn from this
        generator
    true_corrs : np.ndarray (m,)
        true correlation of between-set association modes
    verbose : bool
        whether to print status messages

    Returns
    -------
    Sigmaxy : np.ndarray (px, py)
        between-set covariance matrix
    Sigmaxy_svals : np.ndarray (m,)
        singular values of ``Sigmaxy``, these are the true canonical
        correlations or covariances (for CCA or PLS, respectively)
    U_ : np.ndarray (px, m)
        between-set weight vectors
    V_ : np.ndarray (py, m)
        between-set weight vectors
    latent_expl_var_ratios_x : np.ndarray (m,)
        explained variance ratios for between-set weight vectors in set `X`
    latent_expl_var_ratios_y : np.ndarray (m,)
        explained variance ratios for between-set weight vectors in set `Y`
    min_eval : float
        smallest eigenvalue of Schur complement of joint covariance matrix
        :math:`\Sigma`. :math:`\Sigma` is positive definite if and only if
        `min_eval` > 0
    true_corrs : np.ndarray (m,)
        true correlations of between-set association modes
    latent_mode_vector_algo : str
        identifies the algorithm: is set to ``'pc1_'``
    """

    uvrots = None
    qx, qy = 1, 1
    if m > 1:
        raise ValueError('_find_latent_mode_vectors_pc1 requires m == 1')
    min_eval, Sigmaxy, Sigmaxy_svals, U_, V_, latent_expl_var_ratios_x, \
        latent_expl_var_ratios_y, true_corrs = _find_latent_mode_vectors(
            assemble_Sigmaxy, Sigmaxx, Sigmayy, U, V,
            _generate_random_dominant_subspace_rotations, expl_var_ratio_thr,
            m, max_n_sigma_trials, qx, qy, rng, true_corrs, uvrots,
            verbose=verbose)

    latent_mode_vector_algo = 'pc1_'
    return Sigmaxy, Sigmaxy_svals, U_, V_, \
        latent_expl_var_ratios_x, latent_expl_var_ratios_y, \
        min_eval, true_corrs, latent_mode_vector_algo


def _find_latent_mode_vectors_opti(Sigmaxx, Sigmayy, U, V, assemble_Sigmaxy,
                                   expl_var_ratio_thr, m, max_n_sigma_trials,
                                   qx, qy, rng, true_corrs, verbose):
    r"""Find latent mode vectors using an optimization algorithm that
    maximizes the minimum eigenvalue of the proposed joint covariance matrix.

    The minimum eigenvalue is monitored during optimization and as soon as it
    is positive, optimization is stopped.

    Parameters
    ----------
    Sigmaxx : np.ndarray (px, px)
        within-set covariance matrix for `X`
    Sigmayy : np.ndarray (py, py)
        within-set covariance matrix for `Y`
    U : np.ndarray (px, m)
        weight vectors for `X`
    V : np.ndarray (py, m)
        weight vectors for `Y`
    assemble_Sigmaxy : function
        either `_assemble_Sigmaxy_pls` or `_assemble_Sigmaxy_cca`
    expl_var_ratio_thr : float
        the ratio of the amount of variance along the first mode vectors in
        `X` and `Y` to the mean variance along a mode in `X` and `Y` needs to
        surpass this number.
    m : int >= 1
        number of cross-modality modes to be encoded
    max_n_sigma_trials : int
        maximum number of attempts made to find a linear combination of
        dominant and low-variance subspace components for the weight vectors
        such that both enough variance is explained and the resulting joint
        covariance matrix :math:`\Sigma` is positive definite
    qx : int
        dimensionality of dominant subspace for `X`
    qy : int
        dimensionality of dominant subspace for `Y`
    rng : random number generator instance
        for reproducibility, all random numbers will be drawn from this
        generator
    true_corrs : np.ndarray (m,)
        true correlation of between-set association modes
    verbose : bool
        whether to print status messages

    Returns
    -------
    Sigmaxy : np.ndarray (px, py)
        between-set covariance matrix
    Sigmaxy_svals : np.ndarray (m,)
        singular values of ``Sigmaxy``, these are the true canonical
        correlations or covariances (for CCA or PLS, respectively)
    U_ : np.ndarray (px, m)
        between-set weight vectors
    V_ : np.ndarray (py, m)
        between-set weight vectors
    latent_expl_var_ratios_x : np.ndarray (m,)
        explained variance ratios for between-set weight vectors in set `X`
    latent_expl_var_ratios_y : np.ndarray (m,)
        explained variance ratios for between-set weight vectors in set `Y`
    min_eval : float
        smallest eigenvalue of Schur complement of joint covariance matrix
        :math:`\Sigma`. :math:`\Sigma` is positive definite if and only if
        `min_eval` > 0
    true_corrs : np.ndarray (m,)
        true correlations of between-set association modes
    latent_mode_vector_algo : str
        identifies the algorithm: is set to ``'opti'``
    """
    # "opti" algorithm only implemented for m == 1 at the moment
    if m > 1:
        raise NotImplementedError("Couldn't find latent mode vectors, opti "
                                  "algorithm not implemented for m > 1")

    uvrots = []

    def opti_callback(xk, convergence):
        if len(uvrots) > 0:
            # signifying a solution has been found, cf. doc of
            # differential_evolution
            return True
        else:
            return False

    # interesting result of differential_evolution will be in uvrots
    _ = scipy.optimize.differential_evolution(
        _Sigmaxy_negative_min_eval,
        bounds=[(0, 1)] + [(-1, 1)] * (qx - 1) +
               [(0, 1)] + [(-1, 1)] * (qy - 1),
        args=(assemble_Sigmaxy, Sigmaxx, Sigmayy, U, V, m, qx, qy, rng,
              true_corrs, uvrots),
        # popsize=25,
        polish=False,
        callback=opti_callback,
        seed=rng,
        maxiter=max_n_sigma_trials,
    )
    if len(uvrots) > 0:
        min_eval, Sigmaxy, Sigmaxy_svals, U_, V_, \
            latent_expl_var_ratios_x, latent_expl_var_ratios_y, true_corrs = \
            _find_latent_mode_vectors(
                assemble_Sigmaxy, Sigmaxx, Sigmayy, U, V,
                _generate_dominant_subspace_rotations_from_opti,
                expl_var_ratio_thr, m, max_n_sigma_trials, qx, qy, rng,
                true_corrs, uvrots, verbose=verbose)
        latent_mode_vector_algo = 'opti'
    else:
        min_eval = -1  # value < 0 indicates that no solutino has been found
        Sigmaxy, Sigmaxy_svals, U_, V_, \
            latent_expl_var_ratios_x, latent_expl_var_ratios_y, true_corrs = \
            None, None, None, None, None, None, true_corrs
        latent_mode_vector_algo = 'opti_failed'
    return Sigmaxy, Sigmaxy_svals, U_, V_, \
        latent_expl_var_ratios_x, latent_expl_var_ratios_y, \
        min_eval, true_corrs, latent_mode_vector_algo


def _find_latent_mode_vectors_qr(Sigmaxx, Sigmayy, U, V, assemble_Sigmaxy,
                                 expl_var_ratio_thr, m, max_n_sigma_trials,
                                 qx, qy, rng, true_corrs, verbose):
    r"""Finds random latent mode vectors using the QR algorithm.

    Latent mode vectors are selected as the :math:`Q` factor in a
    QR-decomposition of a matrix with elements chosen i.i.d. from a standard
    normal distribution.

    Parameters
    ----------
    Sigmaxx : np.ndarray (px, px)
        within-set covariance matrix for `X`
    Sigmayy : np.ndarray (py, py)
        within-set covariance matrix for `Y`
    U : np.ndarray (px, m)
        weight vectors for `X`
    V : np.ndarray (py, m)
        weight vectors for `Y`
    assemble_Sigmaxy : function
        either `_assemble_Sigmaxy_pls` or `_assemble_Sigmaxy_cca`
    expl_var_ratio_thr : float
        the ratio of the amount of variance along the first mode vectors in
        `X` and `Y` to the mean variance along a mode in `X` and `Y` needs to
        surpass this number.
    m : int >= 1
        number of cross-modality modes to be encoded
    max_n_sigma_trials : int
        maximum number of attempts made to find a linear combination of
        dominant and low-variance subspace components for the weight vectors
        such that both enough variance is explained and the resulting joint
        covariance matrix :math:`\Sigma` is positive definite
    qx : int
        dimensionality of dominant subspace for `X`
    qy : int
        dimensionality of dominant subspace for `Y`
    rng : random number generator instance
        for reproducibility, all random numbers will be drawn from this
        generator
    true_corrs : np.ndarray (m,)
        true correlation of between-set association modes
    verbose : bool
        whether to print status messages

    Returns
    -------
    Sigmaxy : np.ndarray (px, py)
        between-set covariance matrix
    Sigmaxy_svals : np.ndarray (m,)
        singular values of ``Sigmaxy``, these are the true canonical
        correlations or covariances (for CCA or PLS, respectively)
    U_ : np.ndarray (px, m)
        between-set weight vectors
    V_ : np.ndarray (py, m)
        between-set weight vectors
    latent_expl_var_ratios_x : np.ndarray (m,)
        explained variance ratios for between-set weight vectors in set `X`
    latent_expl_var_ratios_y : np.ndarray (m,)
        explained variance ratios for between-set weight vectors in set `Y`
    min_eval : float
        smallest eigenvalue of Schur complement of joint covariance matrix
        :math:`\Sigma`. :math:`\Sigma` is positive definite if and only if
        `min_eval` > 0
    true_corrs : np.ndarray (m,)
        true correlations of between-set association modes
    latent_mode_vector_algo : str
        identifies the algorithm: is set to ``'qr__'``
    """
    uvrots = None
    min_eval, Sigmaxy, Sigmaxy_svals, U_, V_, \
        latent_expl_var_ratios_x, latent_expl_var_ratios_y, true_corrs = \
        _find_latent_mode_vectors(
            assemble_Sigmaxy, Sigmaxx, Sigmayy, U, V,
            _generate_random_dominant_subspace_rotations, expl_var_ratio_thr,
            m, max_n_sigma_trials, qx, qy, rng, true_corrs, uvrots,
            verbose=verbose)
    algo = 'qr__'
    return Sigmaxy, Sigmaxy_svals, U_, V_, \
        latent_expl_var_ratios_x, latent_expl_var_ratios_y, min_eval, \
        true_corrs, algo


def _find_latent_mode_vectors(assemble_Sigmaxy, Sigmaxx, Sigmayy, U, V,
                              _generate_random_subspace_rotations,
                              expl_var_ratio_thr, m, max_n_sigma_trials, qx,
                              qy, rng, true_corrs, uvrots, verbose=True):
    r"""Find between-set weight vectors.

    #. determine random between-set weight vectors from the dominant subspace,
        by calling `_generate_random_subspace_rotations`
    #. add a component from the complementary, low variance, :math:`px-qx`
        (for `X`) and :math:`py-qy` (for `Y`) dimensional subspace
    #. check that resulting latent mode vectors explain enough variance
    #. if so, assemble :math:`\Sigma_{XY}`, check if it is positive definite,
        and if it is return
    #. if either not enough variance is explained or :math:`\Sigma_{XY}` is not
        positive definite, make another attempt to find suitable weight vectors

    `max_n_sigma_trials` attempts are made to find suitable weight vectors in
    which the weight of the component from the dominant subspace is randomly
    chosen between 0 and 1. If this is not successful, another
    `max_n_sigma_trials` is made in which the the weight for the component from
    the dominant subspace is increased iteration by iteration from 0.5 to 1.
    This encourages to obtain a resulting weight vector that overlaps more and
    more with the dominant subspace so that more variance is explained.

    Parameters
    ----------
    assemble_Sigmaxy : function
        either `_assemble_Sigmaxy_pls` or `_assemble_Sigmaxy_cca`
    Sigmaxx : np.ndarray (px, px)
        within-set covariance matrix for `X`
    Sigmayy : np.ndarray (py, py)
        within-set covariance matrix for `Y`
    U : np.ndarray (px, m)
        weight vectors for `X`
    V : np.ndarray (py, m)
        weight vectors for `Y`
    _generate_random_subspace_rotations : function
        called to determine a random weight vector living in the dominant
        subspace
    expl_var_ratio_thr : float
        the ratio of the amount of variance along the first mode vectors in
        `X` and `Y` to the mean variance along a mode in `X` and `Y` needs to
        surpass this number.
    m : int >= 1
        number of cross-modality modes to be encoded
    max_n_sigma_trials : int
        maximum number of attempts made to find a linear combination of
        dominant and low-variance subspace components for the weight vectors
        such that both enough variance is explained and the resulting joint
        covariance matrix :math:`\Sigma` is positive definite
    qx : int
        dimensionality of dominant subspace for `X`
    qy : int
        dimensionality of dominant subspace for `Y`
    rng : random number generator instance
        for reproducibility, all random numbers will be drawn from this
        generator
    true_corrs : np.ndarray (m,)
        true correlation of between-set association modes
    uvrots : list of 3-tuples where the 2nd and 3rd entries are rotation
        matrices for `X` and `Y`, respectively, i.e. np.ndarrays with
        dimension (qx,) and (qy,)
    verbose : bool
        whether to print status messages

    Returns
    -------
    min_eval : float
        smallest eigenvalue of Schur complement of joint covariance matrix
        :math:`\Sigma`. :math:`\Sigma` is positive definite if and only if
        `min_eval` > 0
    Sigmaxy : np.ndarray (px, py)
        between-set covariance matrix
    Sigmaxy_svals : np.ndarray (m,)
        singular values of ``Sigmaxy``, these are the true canonical
        correlations or covariances (for CCA or PLS, respectively)
    U_ : np.ndarray (px, m)
        between-set weight vectors
    V_ : np.ndarray (py, m)
        between-set weight vectors
    latent_expl_var_ratios_x : np.ndarray (m,)
        explained variance ratios for between-set weight vectors in set `X`
    latent_expl_var_ratios_y : np.ndarray (m,)
        explained variance ratios for between-set weight vectors in set `Y`
    true_corrs : np.ndarray (m,)
        true correlations of between-set association modes
    """
    for _trial in range(2 * max_n_sigma_trials + 1):

        # choose the dominant parts of the latent mode vectors
        U_dominant, V_dominant = _generate_random_subspace_rotations(
            U, V, m, qx, qy, rng, uvrots)

        if _trial < max_n_sigma_trials:
            min_weight = 0
        else:
            min_weight = 0.5 + .5 * (_trial / max_n_sigma_trials - 1)
        U_, V_ = _add_lowvariance_subspace_components(
            U, U_dominant, V, V_dominant, m, qx, qy, rng,
            min_weight=min_weight)

        _enough_variance_explained, latent_expl_var_ratios_x, \
            latent_expl_var_ratios_y = _variance_explained_by_latent_modes(
                Sigmaxx, Sigmayy, U_, V_, expl_var_ratio_thr)

        if not _enough_variance_explained:
            continue

        Sigmaxy, Sigmaxy_svals, U_, V_, min_eval, true_corrs = \
            assemble_Sigmaxy(Sigmaxx, Sigmayy, U_, V_, m, true_corrs)

        if min_eval > 0:
            if verbose:
                print('success finding latent-mode vectors after '
                      '{} trials'.format(_trial))
            return min_eval, Sigmaxy, Sigmaxy_svals, U_, V_, \
                latent_expl_var_ratios_x, latent_expl_var_ratios_y, \
                true_corrs

    else:
        if 'min_eval' not in locals():
            min_eval = np.nan
        return min_eval, None, None, None, None, None, None, true_corrs


def _generate_random_dominant_subspace_rotations(U, V, m, qx, qy, rng, uvrots):
    r"""Generates random weight vectors in the dominant subspaces (of dimension
    `qx` and `qy`).

    Separately for `X` and `Y`, random rotation matrices are obtained via QR
    decomposition of a matrix with random entries drawn from a standard normal
    distribution and choosing the `Q` factor; the basis vectors `U` (for `X`)
    and `V` (for `Y` are then multiplied by these rotation matrices and the
    first `m` columns are selected and returned.

    Parameters
    ----------
    U : np.ndarray(px, px)
        basis vectors for `X`
    V : np.ndarray(py, py)
        basis vectors for `Y`
    m : int
        number of between-set association modes
    qx : int
        the dimensionality of the "dominant" subspace for `X`
    qy : int
        the dimensionality of the "dominant" subspace for `Y`
    rng : random number generator instance
    uvrots : IGNORED
        present for signature compatibility with
        :func:`_generate_dominant_subspace_rotations_from_opti`

    Returns
    -------
    U_dominant : np.ndarray (px, m)
        weight vectors for `X` living in "dominant" subspace
    V_dominant : np.ndarray (py, m)
        weight vectors for `Y` living in "dominant subspace
    """
    _Urot_dominant = np.linalg.qr(rng.normal(size=(qx, qx)))[0]
    _Vrot_dominant = np.linalg.qr(rng.normal(size=(qy, qy)))[0]

    # find latent mode vectors
    U_dominant = np.dot(U[:, :qx], _Urot_dominant)[:, :m]
    V_dominant = np.dot(V[:, :qy], _Vrot_dominant)[:, :m]

    return U_dominant, V_dominant


def _generate_dominant_subspace_rotations_from_opti(U, V, m, qx, qy, rng,
                                                    uvrots):
    r"""Generates weight vectors in the dominant subspaces (of dimension `qx`
    and `qy`) using a random entry in a list of predefined rotation matrices.

    Parameters
    ----------
    U : np.ndarray(px, px)
        basis vectors for `X`
    V : np.ndarray(py, py)
        basis vectors for `Y`
    m : int
        number of between-set association modes
    qx : int
        the dimensionality of the "dominant" subspace for `X`
    qy : int
        the dimensionality of the "dominant" subspace for `Y`
    rng : random number generator instance
    uvrots : list of 3-tuples
        where the 2nd and 3rd entries are rotation matrices for `X` and `Y`,
        respectively, i.e. np.ndarrays with dimension (qx,) and (qy,)

    Returns
    -------
    U_dominant : np.ndarray (px, m)
        weight vectors for `X` living in "dominant" subspace
    V_dominant : np.ndarray (py, m)
        weight vectors for `Y` living in "dominant subspace
    """
    # choose the dominant parts of the latent mode vectors from the
    # optimization results
    _Urot_dominant, _Vrot_dominant = uvrots[rng.randint(len(uvrots))][1:]
    _Urot_dominant = _Urot_dominant.reshape(-1, 1)
    _Vrot_dominant = _Vrot_dominant.reshape(-1, 1)

    # assert np.isclose(np.linalg.norm(_Urot_dominant), 1)
    # assert np.isclose(np.linalg.norm(_Vrot_dominant), 1)

    # find latent mode vectors
    U_dominant = np.dot(U[:, :qx], _Urot_dominant)[:, :m]
    V_dominant = np.dot(V[:, :qy], _Vrot_dominant)[:, :m]

    return U_dominant, V_dominant


def _add_lowvariance_subspace_components(U, U_dominant, V, V_dominant, m,
                                         qx, qy, rng, min_weight=0):
    """Add a component from the low-variance subspace to the weight vectors
    living in the dominant high-variance subspace.

    Separately for `X` and `Y`, a random rotation matrix is calculated for the
    low-variance subspace (of dimension `qx` and `qy`) via QR decomposition of
    a matrix with random entries drawn from a standard normal distribution. The
    `Q` factor is then post-multiplied with the last `qx` and `qy` columns of
    the basis vectors `U` and `V`. Then, a random weight is chosen (between
    `min_weight` and 1) and the weighted linear combination of the dominant
    subspace weights (`U_dominant` and `V_dominant`) with the low-variance
    subspace weights is computed and returned.

    Parameters
    ----------
    U : np.ndarray (px, px)
        basis vectors for `X`
    U_dominant : np.ndarray (px, m)
        weight vectors for `X` living in the dominant subspace
    V : np.ndarray (py, py)
        basis vectors for `Y`
    V_dominant : np.ndarray (py, m)
        weight vectors for `Y` living in the dominant subspace
    m : int
        number of between-set association modes
    qx : int
        dimensionality of dominant subspace for `X`
    qy : int
        dimensionality for dominant subspace for `Y`
    rng : random number generator instance
    min_weight : float, 0 <= min_weight <= 1
        minimum weight given to the weight vectors from the dominant subspace

    Returns
    -------
    U_ : np.ndarray (px, m)
        weight vectors for `X`
    V_ : np.ndarray (py, m)
        weight vectors for `Y`
    """

    assert 0 <= min_weight <= 1

    U_ = _add_lowvariance_subspace_component_1dim(
        U, U_dominant, m, min_weight, qx, rng)
    V_ = _add_lowvariance_subspace_component_1dim(
        V, V_dominant, m, min_weight, qy, rng)

    return U_, V_


def _add_lowvariance_subspace_component_1dim(U, U_dominant, m, min_weight, qx,
                                             rng):
    px = len(U)
    if qx < px:

        m_loVar = min(m, px - qx)  # by assumption, m <= qx

        _Urot_other = np.linalg.qr(rng.normal(size=(px - qx, px - qx)))[0]
        U_other = np.dot(U[:, qx:], _Urot_other)[:, :m_loVar]

        assert np.allclose(
            np.array([np.dot(U_dominant[:, mi], U_other[:, mi])
                      for mi in range(m_loVar)]),
            0)
        _U_weights = rng.uniform(min_weight, 1, size=(1, m_loVar))
        U_ = np.c_[
            _U_weights * U_dominant[:, :m_loVar] +
            np.sqrt(1 - _U_weights ** 2) * U_other,
            U_dominant[:, m_loVar:m]
            ]

        assert np.allclose(np.linalg.norm(U_, axis=0), 1)

    else:
        U_ = U_dominant
    return U_


def _variance_explained_by_latent_modes(Sigmaxx, Sigmayy, U_, V_,
                                        expl_var_ratio_thr):
    r"""Calculates the amount of explained variance by the between-set
    associations and checks if they surpass a threshold.

    The fraction of explained variance along a direction specified by a given
    weight vector is calculated as

    .. math::
        \frac{\vec{w}^T \Sigma \vec{w}}{\mathrm{tr}(\Sigma)}

    To decide whether the given weight vectors explain "enough" variance, the
    explained variance (separately for `X` and `Y`) of the first mode is
    compared to the mean variance of a mode and if the ratio of the 2 is above
    `expl_var_ratio_thr` we interpret this as "enough".

    Parameters
    ----------
    Sigmaxx : np.ndarray (px, px)
        within-set covariance for `X`
    Sigmayy : np.ndarray (py, py)
        within-set covariance for `Y`
    U_ : np.ndarray (px, m)
        weight vectors for `X`
    V_ : np.ndarray (py, m)
        weight vectors for `Y`
    expl_var_ratio_thr : float
        the ratio of the amount of variance along the first mode vectors in
        `X` and `Y` to the mean variance along a mode in `X` and `Y` needs to
        surpass this number.

    Returns
    -------
    enough_variance_explained : bool
        `True`, if "enough" variance is explained for both `X` and `Y`, `False`
        otherwise
    latent_expl_var_ratios_x : np.ndarray (m,)
        explained variance ratios for each mode in `X`
    latent_expl_var_ratios_y : np.ndarray (m,)
        explained variance ratios for each mode in `Y`
    """

    px, py = len(U_), len(V_)
    m = U_.shape[1]

    # check if resulting latent mode vectors U_, V_ explain enough variance
    expl_var_x = np.array(
        [np.dot(U_[:, modei], np.dot(Sigmaxx, U_[:, modei]))
         for modei in range(m)])
    tot_var_x = np.diag(Sigmaxx).sum()
    mean_tot_var_x = tot_var_x / px
    latent_expl_var_ratios_x = expl_var_x / tot_var_x  # will be returned

    expl_var_y = np.array(
        [np.dot(V_[:, modei], np.dot(Sigmayy, V_[:, modei]))
         for modei in range(m)])
    tot_var_y = np.diag(Sigmayy).sum()
    mean_tot_var_y = tot_var_y / py
    latent_expl_var_ratios_y = expl_var_y / tot_var_y  # will be returned

    if (expl_var_x[0] / mean_tot_var_x < expl_var_ratio_thr) or \
            (expl_var_y[0] / mean_tot_var_y < expl_var_ratio_thr):
        # not enough variance explained:
        enough_variance_explained = False
    else:
        enough_variance_explained = True

    return enough_variance_explained, \
        latent_expl_var_ratios_x, latent_expl_var_ratios_y


def _Sigmaxy_negative_min_eval(uvrot, assemble_Sigmaxy, Sigmaxx, Sigmayy,
                               U, V, m, qx, qy,
                               rng, true_corrs, uvrots, min_eval_thr=1e-5):
    r"""Find the negative of the minimum eigenvalue of (the Schur complement
    of) the joint covariance matrix :math:`\Sigma`.

    Random directions within the dominant subsapce are chosen for the `X` and
    `Y` between-set weight vectors. The weight vectors are calculated as linear
    combinations of the first `qx` (for `X`) and `qy` (for `Y`) basis vectors
    (given by `U` and `V`), and the corresponding expansion coefficients are
    given by `uvrots`.

    Parameters
    ----------
    uvrot : np.ndarray (qx+qy,)
        the first `qx` (`qy`) elements are interpreted as rotations for the
        `qx` (`qy`) dimensional dominant subspace for `X` and `Y`
    assemble_Sigmaxy : function
        either `_assemble_Sigmaxy_pls` or `_assemble_Sigmaxy_cca`
    Sigmaxx : np.ndarray (n_X_features, n_X_features)
        covariance-matrix for modality `X`, i.e. the upper left block of the
        joint covariance matrix
    Sigmayy : np.ndarray (n_Y_features, n_Y_features)
        covariance-matrix for modality `Y`, i.e. lower right block of the joint
        covariance matrix
    U : np.ndarray (n_X_features, n_X_features)
        columns of `U` contain basis vectors for `X` data
    V : np.ndarray n_Y_features, n_Y_features)
        columns of `V` contain basis vectors for `Y` data
    m : int >= 1
        number of cross-modality modes to be encoded
    qx : int
        latent mode vectors for modality `X` are calculated as a random linear
        combination of
        - a random linear combination of the first `q_x` columns of `U`
        - a random linear combination of the remaining columns of `U`
    qy : int
        latent mode vectors for modality `Y` are calculated as a random linear
        combination of
        - a random linear combination of the first `q_y` columns of `V`
        - a random linear combination of the remaining columns of `V`
    rng : random number generator instance
    true_corrs : np.ndarray (m,)
        cross-modality correlations that each latent mode should have
    expl_var_ratio_thr : float
        threshold for required within-modality variance along latent mode
        vectors
    uvrots : list
        In case the minimum eigenvalue is positive, (min_eval, urot, vrot) is
        appended to the list `uvrots`
    min_eval_thr : float
        minimum acceptable eigenvalue, i.e. differential evolution algorithm
        stopps, once an eigenvalue is found that is greater than
        ``min_eval_thr``. Should be >= 0.

    Returns
    -------
    negative_min_eval : float
        minus the minimum eigenvalue of the Schur-complement of the
        joint-covariance matrix. if negative_min_eval is <= 0, i.e. if
        min_eval is >= 0, Sigma will be positive definite.

    Side effect
    -----------
    In case the minimum eigenvalue is positive, (min_eval, urot, vrot) is
    appended to the list `uvrots`
    """

    assert m == 1
    assert min_eval_thr >= 0

    assert len(uvrot) == qx + qy
    urot = uvrot[:qx]
    urot /= np.linalg.norm(urot)
    vrot = uvrot[qx:]
    vrot /= np.linalg.norm(vrot)

    # find latent mode vectors
    U_dominant = np.dot(U[:, :qx], urot).reshape(-1, 1)
    V_dominant = np.dot(V[:, :qy], vrot).reshape(-1, 1)

    U_ = U_dominant
    V_ = V_dominant

    Sigmaxy, Sigmaxy_svals, U_, V_, min_eval, true_corrs = assemble_Sigmaxy(
        Sigmaxx, Sigmayy, U_, V_, m, true_corrs
    )

    if min_eval > min_eval_thr:
        uvrots.append((min_eval, urot, vrot))

    return -min_eval


def _assemble_Sigmaxy_pls(Sigmaxx, Sigmayy, U_, V_, m, true_corrs):
    r"""Generates the between-set covariance matrix.

    The between-set covariance matrix is given by

    .. math::
        \Sigma_{XY} = W_X \mathrm{diag}( \vec{\sigma}_{XY}) W_Y^T

    where :math:`W_X` and :math:`W_Y` are matrices whose columns are the weight
    vectors for the between-set associations (given by arguments `U_` and `V_`,
    respectively) and the entries of :math:`\vec{\sigma}_{XY}` give the
    associated covariance calculated as

    .. math::
        \sigma_{XY,i} = \rho_{XY,i} \sqrt{\mathrm{var}(X \vec{w}_{X,i})
            \mathrm{var}(Y \vec{w}_{Y,i})}

    The true correlations :math:`\rho_{XY,i}` are contained in the argument
    `true_corrs`.

    If the calculated vector of covariances :math:`\vec{\sigma}_{XY}` is not
    monotonously decreasing, modes (i. e. `true_corrs`, `U_` and `V_`) will be
    reordered to make it monotonously decreasing. Otherwise, the returned
    values of `true_corrs`, `U_` and `V_` are identical to the input values.

    With the generated :math:`\Sigma_{XY}` the joint covariance matrix
    :math:`\Sigma` can be assembled. To test its positive-definiteness, this
    function also returns the minimal eigenvalue of a Schur complement of
    :math:`\Sigma` (which is > 0 if and only if :math:`\Sigma` is positive
    definite).

    Parameters
    ----------
    Sigmaxx : np.ndarray (px, px)
        within-set covariance matrix for `X`
    Sigmayy : np.ndarray (py, py)
        within-set covariance matrix for `Y`
    U_ : np.ndarray (px, m)
        weight vectors for between-set association modes in `X`
    V_ : np.ndarray (py, m)
        weight vectors for between-set association modes in `Y`
    m : int
        number of between-set association modes
    true_corrs : np.ndarray (m,)
        assumed correlations for the between-set association modes

    Returns
    -------
    Sigmaxy : np.ndarray (px, py)
        between-set covariance matrix
    Sigmaxy_svals : np.ndarray (m,)
        singular values of ``Sigmaxy``, these are the true canonical
        covariances
    U_ : np.ndarray (px, m)
        weight vectors for between-set association modes in `X`
    V_ : np.ndarray (py, m)
        weight vectors for between-set association modes in `Y`
    min_eval : float
        smallest eigenvalue of the Schur complement of either `Sigmaxx` or
        `Sigmayy` (whichever has a larger dimension) of the joint covariance
        matrix :math:`\Sigma`. :math:`\Sigma` is positive definite if and only
        if `min_eval` is > 0.
    true_corrs : np.ndarray (m,)
        assumed correlations for the between-set association modes
    """

    # determine within-set variances along latent mode vectors
    latent_vars_x = np.asarray([(U_[:, mi].T).dot(Sigmaxx).dot(U_[:, mi])
                                for mi in range(m)])
    latent_vars_y = np.asarray([(V_[:, mi].T).dot(Sigmayy).dot(V_[:, mi])
                                for mi in range(m)])

    # determine between-set covariances of latent modes
    cxy_ = true_corrs * np.sqrt(latent_vars_x * latent_vars_y)

    # in case cxy_ is not sorted descendingly, reorder it and also U_, V_
    if (m > 1) and (not np.all(np.diff(cxy_) < 0)):

        sval_order = np.argsort(cxy_)[::-1]
        true_corrs = true_corrs[sval_order]
        U_, V_ = U_[:, sval_order], V_[:, sval_order]

        # recalculate to make sure I'm not missing anything
        # this should result in cxy_reordered being equivalent to
        # cxy_[sval_order]
        latent_vars_x = np.asarray(
            [(U_[:, mi].T).dot(Sigmaxx).dot(U_[:, mi]) for mi in range(m)])
        latent_vars_y = np.asarray(
            [(V_[:, mi].T).dot(Sigmayy).dot(V_[:, mi]) for mi in range(m)])
        cxy_reordered = true_corrs * np.sqrt(latent_vars_x * latent_vars_y)
        assert np.allclose(cxy_reordered, cxy_[sval_order])
        cxy_ = cxy_reordered

        if not np.all(np.diff(cxy_) <= 0):
            raise ValueError("cxy_ not in descending order. This should not "
                             "happen")

    # assemble cross-modality covariance matrix
    Sigmaxy = (U_).dot(np.diag(cxy_)).dot(V_.T)

    # check if Sigmaxy is positive definite
    schur_complement = calc_schur_complement(
        Sigmaxx, Sigmaxy, Sigmaxy.T, Sigmayy)
    min_eval = np.linalg.eigvalsh(schur_complement).min()
    return Sigmaxy, cxy_, U_, V_, min_eval, true_corrs


def _assemble_Sigmaxy_cca(Sigmaxx, Sigmayy, U_, V_, m, true_corrs):
    r"""Generates the between-set covariance matrix for CCA.

    To solve CCA consider the matrix

    .. math::
        K = \Sigma_{XX}^{-1/2} \Sigma_{XY} \Sigma_{YY}^{-1/2}

    and perform a singular value composition

    .. math::
        K = U \mathrm{diag}( \vec{\sigma}) V^T

    where :math:`U` and :math:`V` are unitary. The vector
    :math:`\vec{\sigma}` contains the canonical correlations
    and the weights are given by

    .. math::
        W_X = \Sigma_{XX}^{-1/2} U

        W_Y = \Sigma_{YY}^{-1/2} V

    Conversely, the covariance matrix associated with given canonical
    correlations :math:`\vec{\sigma}` and weights :math:`W_X` and
    :math:`W_Y` is given by

    .. math::
        \Sigma_{XY} = \Sigma_{XX}^{1/2} K \Sigma_{YY}^{1/2}

            = \Sigma_{XX}^{1/2} U \mathrm{diag}( \vec{\sigma}) V^T
                \Sigma_{YY}^{1/2}

            = \Sigma_{XX} W_X \mathrm{diag}( \vec{\sigma}) W_Y^T \Sigma_{YY}

    with the constraints that :math:`U=\Sigma_{XX}^{-1/2} W_X` and
    :math:`V=\Sigma_{YY}^{-1/2} W_Y` are unitary. If the data are white, i.e.
    if :math:`\Sigma_{XX}=\Sigma_{YY}=\mathrm{diag}(1, ..., 1)` the constraints
    are satisfied trivially. Also, if only one between-set mode is sought, i.e.
    if :math:`\vec{\sigma}` contains a single element and the weights
    :math:`W_X` and :math:`W_Y` have a single column each the constrained is
    satisfied. The general case is not implemented at the moment.

    With the generated :math:`\Sigma_{XY}` the joint covariance matrix
    :math:`\Sigma` can be assembled. To test its positive-definiteness, this
    function also returns the minimal eigenvalue of a Schur complement of
    :math:`\Sigma` (which is > 0 if and only if :math:`\Sigma` is positive
    definite).

    Parameters
    ----------
    Sigmaxx : np.ndarray (px, px)
        within-set covariance matrix for `X`
    Sigmayy : np.ndarray (py, py)
        within-set covariance matrix for `Y`
    U_ : np.ndarray (px, m)
        weight vectors for between-set association modes in `X`
    V_ : np.ndarray (py, m)
        weight vectors for between-set association modes in `Y`
    m : int
        number of between-set association modes
    true_corrs : np.ndarray (m,)
        assumed correlations for the between-set association modes

    Returns
    -------
    Sigmaxy : np.ndarray (px, py)
        between-set covariance matrix
    Sigmaxy_svals : np.ndarray (m,)
        singular values of ``Sigmaxy``, these are the true canonical
        correlations and identical to ``true_corrs``
    U_ : np.ndarray (px, m)
        weight vectors for between-set association modes in `X`
    V_ : np.ndarray (py, m)
        weight vectors for between-set association modes in `Y`
    min_eval : float
        smallest eigenvalue of the Schur complement of either `Sigmaxx` or
        `Sigmayy` (whichever has a larger dimension) of the joint covariance
        matrix :math:`\Sigma`. :math:`\Sigma` is positive definite if and only
        if `min_eval` is > 0.
    true_corrs : np.ndarray (m,)
        assumed correlations for the between-set association modes
    """

    if np.all(Sigmaxx == np.eye(len(Sigmaxx))) and \
            np.all(Sigmayy == np.eye(len(Sigmayy))):  # i.e. if data are white
        # assemble cross-modality covariance matrix
        Sigmaxy = (U_).dot(np.diag(true_corrs)).dot(V_.T)

    else:  # data are not white

        if m == 1:  # white data, 1 between-set mode

            sqrt_Sigmaxx = np.diag(np.sqrt(np.diag(Sigmaxx)))
            sqrt_Sigmayy = np.diag(np.sqrt(np.diag(Sigmayy)))

            U__ = np.dot(sqrt_Sigmaxx, U_)
            V__ = np.dot(sqrt_Sigmayy, V_)

            U__ /= np.linalg.norm(U__, axis=0, keepdims=True)
            V__ /= np.linalg.norm(V__, axis=0, keepdims=True)

            assert np.allclose(np.dot(U__.T, U__), np.eye(U__.shape[1]))
            assert np.allclose(np.dot(V__.T, V__), np.eye(V__.shape[1]))

            U__ = np.dot(sqrt_Sigmaxx, U__)
            V__ = np.dot(sqrt_Sigmayy, V__)

            Sigmaxy = U__.dot(np.diag(true_corrs)).dot(V__.T)

        else:  # non-white data, more than one between-set modes
            raise NotImplementedError(
                'Generation of between-set covariance matrix for CCA is not '
                'implemented for the case that more than 1 mode is sought and '
                'data are not white.')

    # check if Sigmaxy is positive definite
    schur_complement = calc_schur_complement(
        Sigmaxx, Sigmaxy, Sigmaxy.T, Sigmayy)
    min_eval = np.linalg.eigvalsh(schur_complement).min()
    return Sigmaxy, true_corrs, U_, V_, min_eval, true_corrs


def calc_schur_complement(A, B_or_px, C=None, D=None, kind='auto'):
    r"""Calculate Schur complement of a matrix.

    Parameters
    ----------
    A : np.ndarray
        Can either specify the complete 2x2 block matrix, in which case
        argument `B_or_px` needs to be an integer giving the dimension of the
        upper left block (i.e. A[:B_or_px, :B_or_px] is the upper left block)
        and `C` and `D` are ignored, or `A` is the upper left block, in which
        case `B_or_px` is assumed to be a matrix
    B_or_px : int or np.ndarray
        If int, gives the size of the upper left block, if array, specifies the
        upper right block.
    C : ignored or np.ndarray
        lower left block in case B_or_px is an array, ignored otherwise
    D : ignored or np.ndarray
        lower right block in case B_or_px is an array, ignored otherwise
    kind : "A" or "D"
        whether to calculate the Schur complement of block "A" or "D"

    Returns
    -------
    Schur complement : np.ndarray
        same shape as block `A` or `D`, depending on value of `kind`
    """

    if (C is None) and (D is None):
        M = A  # A is the complete 2x2 block matrix
        try:
            px = int(B_or_px)
        except ValueError:
            raise ValueError('If C and D are not given argument B_or_px needs '
                             'to be an integer')

        A = M[:px, :px]
        B = M[:px, px:]
        C = M[px:, :px]
        D = M[px:, px:]
    else:
        B = B_or_px

    if kind == 'auto':
        if len(A) <= len(D):
            kind = 'A'
        else:
            kind = 'D'
    if kind == 'A':
        schur_complement = A - B.dot(np.linalg.inv(D)).dot(C)
    elif kind == 'D':
        schur_complement = D - C.dot(np.linalg.inv(A)).dot(B)
    else:
        raise ValueError('Invalid kind: {}'.format(kind))

    return schur_complement


def generate_data(Sigma, px, n, random_state=42):
    r"""Generate synthetic data for a given model.

    The assumed model is a multivariate normal distribution with covariance
    matrix ``Sigma``. ``n`` samples are drawn and from this distribution and
    returned.

    Parameters
    ----------
    Sigma : np.ndarray (total number of features in `X` and `Y`, total number
        of features in `X` and `Y`) joint covariance matrix of model
    px : int
        number of features in dataset `X` (number of features in `Y` is
        inferred from size of ``Sigma``
    n : int
        number of samples to return
    random_state : ``None``, int or random number generator instance
        used to generate random numbers

    Returns
    -------
    X : np.ndarray (n_samples, n_features)
        dataset `X`
    Y : np.ndarray (n_samples, n_features)
        dataset `Y`
    """
    rng = check_random_state(random_state)
    XY = rng.multivariate_normal(
        np.zeros(len(Sigma)), Sigma, size=(n,), check_valid='raise')
    X, Y = XY[:, :px], XY[:, px:]
    return X, Y
