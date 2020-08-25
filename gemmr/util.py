"""Utility functions used throughout the rest of the package."""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
try:
    import holoviews as hv
except ModuleNotFoundError:
    pass


__all__ = ['_check_float', 'check_positive_definite',
           'align_weights', 'nPerFtr2n',
           'rank_based_inverse_normal_trafo', 'pc_spectrum_decay_constant']


def _check_float(v, label, lo, up, boundaries):
    if boundaries == 'inclusive':
        if not lo <= v <= up:
            raise ValueError(
                "{:.2f} <= {} <= {:.2f} not satisfied".format(lo, label, up)
            )
    else:
        raise NotImplementedError(
            'boundaries = {} not yet implemented'.format(boundaries)
        )


def check_positive_definite(Sigma, noise_level, noise_factor=1e-6):
    """Check if a matrix is positive definite.

    Parameters
    ----------
    Sigma : ndarray
        matrix or matrices for which to check positive definiteness. In case
        Sigma has more than 2 dimensions, the first dimensions are iterated
        over, and the last 2 dimensions identify the matrix for which to check
        positive definiteness.
    noise_level : float

    Returns
    -------
    Nothing

    Raises
    ------
    ValueError
        if Sigma is not positive definite
    """
    # eigvalsh iterates over first dimensions of input array, calculating
    # eigenvalues for each matrix specified by the last 2 dimensions of input
    # array

    # print(np.linalg.svd(Sigma[:5, 5:], full_matrices=False)[1])

    evals_Sigma = np.linalg.eigvalsh(Sigma)
    min_eval = evals_Sigma.min()
    if min_eval < noise_factor * noise_level:
        raise ValueError('Sigma is not positive definite: '
                         'min eval = {:.3f}'.format(min_eval))


def align_weights(v, vtrue, copy=True, return_sign=False):
    """Align vectors in rows of `v` such that they have a positive dot-product
    with `vtrue`.

    Parameters
    ----------
    v : np.ndarray (..., n_features)
        each vector of length n_feature will be compared to vtrue and if the
        dot-product is negative will be multiplied by -1
    vtrue : np.ndarray (n_features,)
        the reference vector
    copy : bool
        whether a copy of v is made before signs are changed
    return_sign : bool
        if ``True`` return signs of vector in addition to aligned vectors

    Returns
    -------
    aligned : np.ndarray (n_vectors, n_features)
        aligned vectors
    signed : np.ndarray (n_vectors, n_features)
        (only if ``return_sign == True``). Signs (-1, 0 or 1) of original
        vectors
    """
    if vtrue.ndim != 1:
        raise ValueError('vtrue must have a single dimension')
    if not v.shape[-1] == len(vtrue):
        raise ValueError('The last dimension of v must have same length as '
                         'vtrue')
    if not np.allclose(np.linalg.norm(vtrue), 1):
        raise ValueError('vtrue must be a unit vector')
    if not np.allclose(np.linalg.norm(v, axis=-1), 1):
        raise ValueError('all vectors in v must be unit vectors')

    if copy:
        v = v.copy()

    # vshape = v.shape
    v = v.reshape(-1, len(vtrue))
    sims = 1 - cdist(v, vtrue.reshape(1, -1), metric='cosine')
    signs = np.sign(sims)
    v *= signs
    assert np.all(1 - cdist(v, vtrue.reshape(1, -1), metric='cosine') > 0)

    if not return_sign:
        return v
    else:
        return v, signs


def nPerFtr2n(nPerFtr, py=None):
    """Convert a DataArray with elements representing "n_per_ftr" into "n"

    Parameters
    ----------
    nPerFtr : xr.DataArray
        the elements represent samples per feature. ``nPerFtr`` must have
        dimension ``'px'``, ``'r'`` and ``'Sigma_id'``
    py : None or xr.DataArray
        if None it is assumed that ``nPerFtr`` has an attribute ``py``. ``py``
        and  nPerFtr must have same dimensions.

    Returns
    -------
    n : xr.DataArray
        the elements represent samples
    """
    if (nPerFtr.ndim != 3) or \
            (np.sort(nPerFtr.dims) != np.array(['Sigma_id', 'px', 'r'])).any():
        raise NotImplementedError(
            'nPerFtr must have dimensions Sigma_id, px, r')

    if py is None:
        py = nPerFtr.py
    else:
        assert (py.px == nPerFtr.px).all()

    px = nPerFtr.px

    ptot = px + py
    assert nPerFtr.dims == ptot.dims

    n = nPerFtr * ptot

    ptot = ptot.stack(all_dims=ptot.dims)
    n = n.stack(all_dims=n.dims).dropna('all_dims')

    ptot = ptot.sel(all_dims=n.all_dims)
    assert (n.all_dims == ptot.all_dims).all()

    new_index = pd.MultiIndex.from_arrays(
        [ptot.values, ptot.r.values, ptot.Sigma_id.values],
        names=['ptot', 'r', 'Sigma_id']
    )

    n.coords['all_dims'] = new_index
    n = n.unstack('all_dims')

    return n.rename('n_required')


def rank_based_inverse_normal_trafo(x, c='blom'):
    """Rank-based inverse normal transformation.

    References
    ----------
    Beasley TM, Erickson S, Allison DB. Rank-Based Inverse Normal
        Transformations are Increasingly Used, But are They Merited?
        Behav Genet. 2009 Jun 14;39(5):580.


    Parameters
    ----------
    x : np.ndarray-like (n_samples,)
        data to be transformed
    c : float or str
        constant appearing in transformation, named values (cf. Beasley et al.)
        are
        - 'blom': c = 3/8
        - 'tukey': c = 1/3
        - 'rankit' c = 1/2
        - 'waerden' c = 0

    Returns
    -------
    x_INT : (n_samples,)
        transformed data
    """

    c = dict(
        blom=3./8,
        tukey=1./3,
        rankit=1./2,
        waerden=0
    ).get(c, c)

    ranks = np.argsort(np.argsort(np.asarray(x))) + 1

    return norm.ppf((ranks - c) / (len(ranks) - 2*c + 1))


def pc_spectrum_decay_constant(X=None, pc_spectrum=None,
                               expl_var_ratios=(.3, .9), plot=False):
    """Estimate powerlaw decay constant.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_features)
        the principal component spectrum will be estimated for data matrix
        ``X``. Either ``X`` or ``pc_spectrum`` must be given
    pc_spectrum : np.ndarray (n_pcs,)
        directly the principal component spectrum.
        Either ``X`` or ``pc_spectrum`` must be given
    expl_var_ratios : tuple of floats
        decay constants will be estimated by fitting a linear regression
        to the principal component spectrum using the number of components that
        explain this amount of variance. If multiple values for
        ``expl_var_ratios`` are given, multiple decay constants will be
        returned
    plot : bool
        if ``True``, a plot of the spectrum and illustrating the decay constant
        will be returned in addition to the decay constant

    Returns
    -------
    decay_constant : np.ndarray (len(expl_var_ratios),)
        estimated powerlaw decay constant
    panel : hv.Overlay
        if argument ``plot == True``, a panel showing the principal component
        spectrum and illustrating the decay constant
    """

    if not ((X is not None) ^ (pc_spectrum is not None)):
        raise ValueError('Exactly one of X and pc_spectrum must be given')

    if X is not None:

        # PCA works with demeaned data
        X = X - X.mean(0, keepdims=True)

        pc_spectrum = np.linalg.svd(
            X, full_matrices=False, compute_uv=False)**2 / (len(X) - 1)

    # otherwise use input variable pc_spectrum

    order = np.argsort(pc_spectrum)[::-1]
    pc_spectrum = pc_spectrum[order]
    assert np.all(np.diff(pc_spectrum) <= 0)

    pc_spectrum_nrm = pc_spectrum / pc_spectrum[0]
    assert pc_spectrum_nrm[0] == 1.
    expl_var_ratio = pc_spectrum / pc_spectrum.sum()
    expl_var_ratio_cum = np.cumsum(expl_var_ratio)

    n_components = [
        next(i+1
             for i in range(len(pc_spectrum))
             if expl_var_ratio_cum[i] > frac_expl_var)
        for frac_expl_var in expl_var_ratios
    ]
    for i in range(len(n_components)):
        if n_components[i] == 1:
            n_components[i] += 1

    powerlaw_decay_constants = []
    if plot:
        mx_nc = np.max(n_components)
        panel = hv.Curve(
            (np.arange(1, len(pc_spectrum_nrm)+1)[:10*mx_nc],
             pc_spectrum_nrm[:10*mx_nc]),
            label='data'
        )
    for nc in np.unique(n_components):

        X = np.arange(1, nc+1)
        y = pc_spectrum_nrm[:nc]

        lm = LinearRegression(
            fit_intercept=True, normalize=False
        ).fit(
            np.log(X).reshape(-1, 1),
            np.log(y)
        )

        powerlaw_decay_constants.append(
            lm.coef_[0]
        )

        if plot:
            xs = np.arange(1, nc+1)
            yhat = np.exp(lm.predict(np.log(xs).reshape(-1, 1)))
            panel *= hv.Curve(
                (xs, yhat),
                label='slope: %.1f' % lm.coef_[0],
            )

    powerlaw_decay_constants = np.array(powerlaw_decay_constants)
    # assert np.all(powerlaw_decay_constants < 0)

    if not plot:
        return powerlaw_decay_constants
    else:
        panel = panel.redim(
            x='Principal component id',
            y='Variance'
        ).opts(
            logx=True, logy=True
        )
        return powerlaw_decay_constants, panel
