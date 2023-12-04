"""Sample size predictions based on a linear model."""

import numbers
from pkg_resources import resource_filename

import numpy as np
import xarray as xr

from sklearn.linear_model import LinearRegression

# from ..sample_analysis import analyze_model_parameters
from .interpolation import calc_n_required_all_metrics, \
    _calc_n_req_for_power, _calc_n_req_for_metric
from ..data.loaders import load_outcomes
from ..metrics import _metric_funs
from ..util import pc_spectrum_decay_constant


__all__ = ['fit_linear_model',
           'cca_sample_size', 'pls_sample_size',
           'cca_req_corr', 'pls_req_corr']


def do_fit_lm(
        ds,
        n_reqs,
        include_pc_var_decay_constants=True,
        include_latent_explained_vars=True,
        include_pdiff=False,
        verbose=False,
        prefix=''
):
    """Fits a linear model to outcome data.

    First, prepares outcome data, i.e. selects predictor variables and stacks
    it along dimensions px, r, Sigma_id. Second, fits linear model.

    Parameters
    ----------
    ds : xr.Dataset
        outcome dataset
    n_reqs : xr.DataArray
        required sample sizes
    include_pc_var_decay_constants : bool
        whether to include a predictor for the principal component spectrum
        decay constant in the linear model
    include_latent_explained_vars : bool
        whether to include a predictor for the latent explained variance in
        the linear model
    include_pdiff : bool
        whether to include predictor for :math:`|p_X - p_Y|` in the linear
        model
    verbose : bool
        if ``True`` prints deltaAIC
    prefix : str
        prefix for outcome variables in ``ds``

    Returns
    -------
    lm : sklearn.LinearRegression instance
        fitted model
    X : (n_synth_datasets, n_predictors)
        predictor data matrix
    y : (n_synth_datasets,)
        dependent variable
    coef_names : list
        labels for included linear model coefficients (first one is "const")
    """
    X, y, coeff_names = prep_data_for_lm(
        ds, n_reqs, include_latent_explained_vars,
        include_pc_var_decay_constants, include_pdiff, prefix=prefix)

    lm = LinearRegression().fit(X, y)

    if verbose:
        y_hat = lm.predict(X)
        sse = np.sum((y_hat - y) ** 2)
        n_vars = X.shape[1] + 1  # "+1" for constant
        n_samples = len(X)
        print('deltaAIC=', 2 * n_vars + n_samples * np.log(sse))

    return lm, X, y, coeff_names


def prep_data_for_lm(ds, n_reqs, include_latent_explained_vars,
                     include_pc_var_decay_constants, include_pdiff,
                     prefix=''):
    """Prepare outcome data for use with linear model.

    Constructs a predictor data matrix with columns representing linear
    model predictors, and rows representing stacked synthetic datasets
    (stacked dimensions are 'px', 'r', 'Sigma_id').

    Parameters
    ----------
    ds : xr.Dataset
        outcome dataset
    n_reqs : xr.DataArray
        required sample sizes
    include_pc_var_decay_constants : bool
        whether to include a predictor for the principal component spectrum
        decay constant in the linear model
    include_latent_explained_vars : bool
        whether to include a predictor for the latent explained variance in
        the linear model
    include_pdiff : bool
        whether to include predictor for :math:`|p_X - p_Y|` in the linear
        model
    prefix : str
        prefix for outcome variables in ``ds``

    Returns
    -------
    X : (n_synth_datasets, n_predictors)
        predictor data matrix
    y : (n_synth_datasets,)
        dependent variable
    coef_names : list
        labels for included linear model coefficients (first one is "const")
    """

    if include_latent_explained_vars:
        raise ValueError("include_latent_explained_vars is deprecated")

    lm_vars = xr.Dataset(dict(
        log_n_reqs=np.log(n_reqs),
        ptot=ds.px + ds.py,
        pdiff=np.abs(ds.py - ds.px),
        r_true=ds[f'{prefix}between_corrs_true'],
        axPlusy=ds[f'{prefix}ax'] + ds[f'{prefix}ay'],
        #latent_vars_xTimesy=ds[f'{prefix}latent_expl_var_ratios_x']
        #                    * ds[f'{prefix}latent_expl_var_ratios_y'],
    )).stack(it=('px', 'r', 'Sigma_id'))
    X = [
        -np.log(lm_vars.r_true.values),
        np.log(lm_vars.ptot.values),  # np.log(log_n_crits_.px.values),
    ]
    coef_names = ['const', r'$-\log(r_\mathrm{true})$', r'$\log(p_X+p_Y)$', ]
    if include_pdiff:
        X.append(
            lm_vars.pdiff.values
        )
        coef_names.append(
            r'$|p_Y-p_X|$'
        )
    if include_pc_var_decay_constants:
        X.append(
            np.abs(lm_vars.axPlusy.values)
        )
        coef_names.append(
            r'$\|a_X+a_Y\|$'
        )
    if include_latent_explained_vars:
        X.append(
            -np.log(lm_vars.latent_vars_xTimesy.values)
        )
        coef_names.append(
            r'$-\log(v_X \cdot v_Y)$',
        )
    X = np.stack(X).T
    y = lm_vars.log_n_reqs.values
    mask = np.isfinite(X).all(1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    return X, y, coef_names


def fit_linear_model(criterion, model, estr=None, tag=None, target_power=0.9,
                     target_error=0.1, data_home=None,
                     include_pc_var_decay_constants=None,
                     include_latent_explained_vars=None):
    """Fit a linear model to outcome data.

    Parameters
    ----------
    criterion : str
        Can be:

        - ``'combined'``
        - ``'power'``
        - ``'association_strength'``
        - ``'weight'``
        - ``'score'``
        - ``'loading'``
        - ``'crossloading'``
    model : str
        'cca' or 'pls'
    estr : None or *sklearn*-style estimator instance
        if not ``None`` must be compatible with ``model``
    tag : str or None
        further specifies the outcome data file, cf.
        :func:`.data.load_outcomes`
    target_error : float between 0 and 1
        target error level
    target_power : float between 0 and 1
        target power level
    data_home : None or str
        path where outcome data are stored, ``None`` indicates default path
    include_pc_var_decay_constants : bool
        whether to include a predictor for the principal component spectrum
        decay constant in the linear model
    include_latent_explained_vars : bool
        whether to include a predictor for the latent explained variance in
        the linear model

    Returns
    -------
    lm : sklearn.LinearRegression instance
        fitted model
    """

    if model == 'cca':
        ds = load_outcomes('sweep_cca_cca_random_sum+-3+0_wOtherModel', model='cca', data_home=data_home).sel(mode=0)

    elif model == 'pls':
        ds = load_outcomes('sweep_pls_pls_random_sum+-3+0_wOtherModel', model='pls', data_home=data_home).sel(mode=0)

    else:
        raise ValueError(f'Invalid argument model: {model}')

    ds = ds.sel(px=(ds.px > 4) & (ds.px < 128))

    if criterion == 'combined':
        n_req_per_ftr = calc_n_required_all_metrics(
            ds, target_power=target_power, target_error=target_error,
            search_dim='n_per_ftr')['combined']
    elif criterion == 'power':
        n_req_per_ftr = _calc_n_req_for_power(ds, target_power,
                                              search_dim='n_per_ftr')
    else:
        try:
            metric_fun = _metric_funs[criterion]
        except KeyError:
            raise ValueError('Invalid metric: {}'.format(criterion))
        n_req_per_ftr = _calc_n_req_for_metric(
            metric_fun, ds, target_error, search_dim='n_per_ftr')

    n_req = n_req_per_ftr * (ds.px + ds.py)

    if model == 'cca':
        if include_pc_var_decay_constants is None:
            include_pc_var_decay_constants = True
        if include_latent_explained_vars is None:
            include_latent_explained_vars = False
    elif model == 'pls':
        if include_pc_var_decay_constants is None:
            include_pc_var_decay_constants = True
        if include_latent_explained_vars is None:
            include_latent_explained_vars = False
    else:
        raise ValueError('This should not happen')

    lm, X, y, coeff_names = do_fit_lm(
        ds, n_req,
        include_pc_var_decay_constants=include_pc_var_decay_constants,
        include_latent_explained_vars=include_latent_explained_vars,
        include_pdiff=False,
        verbose=False,
    )
    return lm


def _check_pxy(X, Y):

    def _check(X):
        if isinstance(X, np.ndarray):
            px = X.shape[1]
        else:  # assume X is already "number of features"
            px = X
        if not isinstance(px, numbers.Integral) or (px < 2):
            raise ValueError('Invalid input: need np.ndarray with at least 2 '
                             'dimensions or integer >= 2')
        return px

    px = _check(X)
    py = _check(Y)
    return px, py


def _check_axy(X, Y, ax, ay, expl_var_ratio=0.3):
    """Check validity of or determine ax and ay.

    Either X must be a data matrix or ax must be given. Analogous for Y.
    A data matrix here means a two-dimensional np.ndarray.
    If the data matrix is given the corresponding decay constant is estimated.
    If the decay constant is given it will be taken directly and checked that
    it is <= 0.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_X_features)
        data matrix X
    Y : np.ndarray (n_samples, n_Y_features)
        data matrix Y
    ax : float < 0
        principal component spectrum decay constant for X
    ay : float < 0
        prinicpal component spectrum decay constant for Y
    expl_var_ratio : float between 0 and 1
        if decay constant is estimated from a data matrix, the number of
        principal components employed is determined to explain this amount of
        variance

    Returns
    -------
    ax : float < 0
        decay constant for X
    ay : float < 0
        decay constant for Y
    """

    def _have_data_matrix(X):
        if (X is not None) and isinstance(X, np.ndarray) and (X.ndim == 2):
            return True

    def _check(X, ax, dataset_label):

        if _have_data_matrix(X):
            if ax is not None:
                raise ValueError(
                    'Either {} must be a data matrix or {} must be '
                    'given'.format(dataset_label, dataset_label.lower()))
            else:
                ax = pc_spectrum_decay_constant(
                    X=X, expl_var_ratios=(expl_var_ratio,), plot=False)[0]
                return ax
        else:
            if ax is None:
                raise ValueError(
                    'Either {} must be a data matrix or {} must be '
                    'given'.format(dataset_label, dataset_label.lower()))

            if isinstance(ax, numbers.Number):
                if ax > 0:
                    raise ValueError(
                        'a{} must be <= 0'.format(dataset_label.lower()))
            else:
                raise ValueError('a{} must be a number'.format(
                    dataset_label.lower()))

            return ax

    ax = _check(X, ax, 'X')
    ay = _check(Y, ay, 'Y')
    return ax, ay


def _save_linear_model(model, data_home=None, verbose=False):
    """Save linear model.

    Creates csv files with parameters for

    * criterion == 'combined'
    * target_power == 0.9
    * target_error == 0.1

    When these parameters are given in :func:`cca_sample_size` or
    :func:`pls_sample_size`, then linear model coefficients are read from file
    "gemmr.datasets.sample_size_lm_[model].csv". This allows to quickly
    calculate sample sizes without having the complete outcome data available.

    For internal use.

    Parameters
    ----------
    model : str
        'cca' or 'pls'
    data_home : None or str
        if ``str`` indicates path where outcome data are stored,
        ``None`` is interpreted as default path
    verbose : bool
        if ``True`` intercept and coefficients of fitted linear model are
        printed to stdout

    """

    lm = fit_linear_model(criterion='combined', model=model, target_power=0.9,
                          target_error=0.1, data_home=data_home)
    intercept, coefs = lm.intercept_, lm.coef_

    if verbose:
        print(f'intercept = {lm.intercept_}, coef = {lm.coef_}')

    fname = resource_filename(
        'gemmr', 'datasets/sample_size_lm_{}.csv'.format(model))

    with open(fname, 'wt') as f:
        f.write(
            ','.join(np.r_[[intercept], coefs].astype(str).tolist())
        )


def _read_sample_size_lm_csv(fname):
    with open(fname, 'rt') as f:
        all_coefs = f.read().split(',')
        intercept = float(all_coefs[0])
        coefs = np.asarray(all_coefs[1:]).astype(float)
    return coefs, intercept


def get_lm_coefs(model, criterion, target_error, target_power, data_home):
    """Get linear model coefficients.

    For default parameters, i.e. if ``criterion='combined'``,
    ``target_power=0.9`` and ``target_error=0.1``, coefficients are read from
    file ``'../datasets/sample_size_lm_[model].csv'``. Otherwise, they are
    calculated from outcome datasets.

    Parameters
    ----------
    model : str
        'cca' or 'pls'
    criterion : str
        Can be:

        - ``'combined'``
        - ``'power'``
        - ``'association_strength'``
        - ``'weight'``
        - ``'score'``
        - ``'loading'``
        - ``'crossloading'``
    target_error : float between 0 and 1
        target error level
    target_power : float between 0 and 1
        target power level
    data_home : None or str
        path where outcome data are stored, ``None`` indicates default path

    Returns
    -------
    intercept : float
        linear model intercept
    coefs : np.ndarray
        linear model coefficients
    """
    if (criterion == 'combined') and \
            (target_power == 0.9) and (target_error == 0.1):

        fname = resource_filename(
            'gemmr', 'datasets/sample_size_lm_{}.csv'.format(model))
        coefs, intercept = _read_sample_size_lm_csv(fname)

    else:
        lm = fit_linear_model(criterion, model, target_power=target_power,
                              target_error=target_error, data_home=data_home)
        intercept, coefs = lm.intercept_, lm.coef_
    return intercept, coefs


def cca_sample_size(
        X, Y, ax=None, ay=None,
        rs=(.1, .3, .5),
        criterion='combined',
        algorithm='linear_model',
        target_power=0.9,
        target_error=0.1,
        expl_var_ratio=0.5,
        data_home=None,
):
    """Suggest sample size for CCA.

    Suggested sample sizes are estimated using a linear model to to inter-
    and extrapolate parameters for which the generative model was used
    beforehand to calculate sample sizes.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_X_features) or int >= 2
        either a data matrix or directly the number of features for data matrix
        :math:`X`
    Y : np.ndarray (n_samples, n_Y_features) or int >= 2
        either a data matrix or directly the number of features for data matrix
        :math:`Y`
    ax : float < 0 or None
        principal component spectrum decay constant, if ``X`` is not a data
        matrix, ``None`` otherwise
    ay : float < 0 or None
        principal component spectrum decay constant, if ``Y`` is not a data
        matrix, ``None`` otherwise
    rs : list-like
        true correlations for which sample sizes are estimated
    criterion : str
        criterion according to which sample sizes are estimated.
        Can be:

        - ``'combined'``
        - ``'power'``
        - ``'association_strength'``
        - ``'weight'``
        - ``'score'``
        - ``'loading'``
        - ``'crossloading'``
    algorithm : str
        algorithm used to calculate sample sizes.
        Can be:

        - ``'linear_model'``
    target_power : float between o and 1
        if ``criterion`` is ``'combined'`` or ``'power'`` sample size is
        chosen to obtain at least ``target_power`` power
    target_error : float between 0 and 1
        if criterion is not ``'power'`` sample size is chosen to obtain at
        most ``target_error`` error in error metric(s)
    expl_var_ratio : float
        if ``X`` or ``Y`` is a data matrix, ``ax`` or ``ay``, respectively,
        will be estimated directly from the data using the number of principal
        components that explain this amount of variance
    data_home : None or str
        path where outcome data are stored, ``None`` indicates default path

    Returns
    -------
    suggested_sample_sizes : dict
        suggested sample sizes for correlations ``rs``
    """

    px, py = _check_pxy(X, Y)
    ax, ay = _check_axy(X, Y, ax, ay, expl_var_ratio=expl_var_ratio)

    if algorithm == 'linear_model':

        intercept, coefs = get_lm_coefs('cca', criterion, target_error,
                                        target_power, data_home)

        suggested_sample_sizes = {
            r: int(np.exp(
                intercept - np.log(r) * coefs[0] + np.log(px + py) * coefs[1]
                + np.abs(ax + ay) * coefs[2]
            ))
            for r in rs
        }

    elif algorithm == 'generative_model':
        raise NotImplementedError()
        # ds = analyze_model_parameters('cca', pxs=[px], pys=py, rs=rs)

    else:
        raise ValueError('Invalid algorithm: {}'.format(algorithm))

    return suggested_sample_sizes


def pls_sample_size(
        X, Y, ax=None, ay=None,
        rs=(.1, .3, .5),
        criterion='combined',
        algorithm='linear_model',
        target_power=0.9,
        target_error=0.1,
        expl_var_ratio=0.5,
        data_home=None,
):
    """Suggest sample size for PLS.

    Suggested sample sizes are estimated using a linear model to to inter-
    and extrapolate parameters for which the generative model was used
    beforehand to calculate sample sizes.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_X_features) or int >= 2
        either a data matrix or directly the number of features for data matrix
        :math:`X`
    Y : np.ndarray (n_samples, n_Y_features) or int >= 2
        either a data matrix or directly the number of features for data matrix
        :math:`Y`
    ax : float < 0 or None
        principal component spectrum decay constant, if ``X`` is not a data
        matrix, ``None`` otherwise
    ay : float < 0 or None
        principal component spectrum decay constant, if ``Y`` is not a data
        matrix, ``None`` otherwise
    rs : list-like
        true correlations for which sample sizes are estimated.
    criterion : str
        criterion according to which sample sizes are estimated.
        Can be:

        - ``'combined'``
        - ``'power'``
        - ``'association_strength'``
        - ``'weight'``
        - ``'score'``
        - ``'loading'``
        - ``'crossloading'``
    algorithm : str
        algorithm used to calculate sample sizes.
        Can be:

        - ``'linear_model'``
    target_power : float between o and 1
        if ``criterion`` is ``'combined'`` or ``'power'`` sample size is chosen
        to obtain at least ``target_power`` power
    target_error : float between 0 and 1
        if criterion is not ``'power'`` sample size is chosen to obtain at most
        ``target_error`` error in error metric(s)
    expl_var_ratio : float
        if ``X`` or ``Y`` is a data matrix, ``ax`` or ``ay``, respectively,
        will be estimated directly from the data using the number of principal
        components that explain this amount of variance
    data_home : None or str
        path where outcome data are stored, ``None`` indicates default path

    Returns
    -------
    suggested_sample_sizes : dict
        suggested sample sizes for correlations ``rs``
    """

    px, py = _check_pxy(X, Y)
    ax, ay = _check_axy(X, Y, ax, ay, expl_var_ratio=expl_var_ratio)

    if algorithm == 'linear_model':

        intercept, coefs = get_lm_coefs('pls', criterion, target_error,
                                        target_power, data_home)

        suggested_sample_sizes = {
            r: int(np.exp(
                intercept - np.log(r) * coefs[0] + np.log(px + py) * coefs[1]
                + np.abs(ax + ay) * coefs[2]
            ))
            for r in rs
        }

    elif algorithm == 'generative_model':
        raise NotImplementedError('TODO')

    else:
        raise ValueError('Invalid algorithm: {}'.format(algorithm))

    return suggested_sample_sizes


def cca_req_corr(
        X, Y, ax, ay,
        n_req,
        criterion='combined',
        algorithm='linear_model',
        target_power=0.9,
        target_error=0.1,
        expl_var_ratio=0.3,
        data_home=None,
):
    """Determines the minimum required true correlation to achieve power and
    error levels.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_X_features) or int >= 2
        either a data matrix or directly the number of features for data matrix
        :math:`X`
    Y : np.ndarray (n_samples, n_Y_features) or int >= 2
        either a data matrix or directly the number of features for data matrix
        :math:`Y`
    ax : float < 0 or None
        principal component spectrum decay constant, if ``X`` is not a data
        matrix, ``None`` otherwise
    ay : float < 0 or None
        principal component spectrum decay constant, if ``Y`` is not a data
        matrix, ``None`` otherwise
    n_req : sample_size
        available sample size
    criterion : str
        criterion according to which sample sizes are estimated.
        Can be:

        - ``'combined'``
        - ``'power'``
        - ``'association_strength'``
        - ``'weight'``
        - ``'score'``
        - ``'loading'``
        - ``'crossloading'``
    algorithm : str
        algorithm used to calculate sample sizes.
        Can be:

        - ``'linear_model'``
    target_power : float between o and 1
        if ``criterion`` is ``'combined'`` or ``'power'`` sample size is
        chosen to obtain at least ``target_power`` power
    target_error : float between 0 and 1
        if criterion is not ``'power'`` sample size is chosen to obtain at
        most ``target_error`` error in error metric(s)
    expl_var_ratio : float
        if ``X`` or ``Y`` is a data matrix, ``ax`` or ``ay``, respectively,
        will be estimated directly from the data using the number of principal
        components that explain this amount of variance
    data_home : None or str
        path where outcome data are stored, ``None`` indicates default path

    Returns
    -------
    req_corr : float
        minimum required true correlation
    """

    px, py = _check_pxy(X, Y)
    ax, ay = _check_axy(X, Y, ax, ay, expl_var_ratio=expl_var_ratio)

    if algorithm == 'linear_model':

        intercept, coefs = get_lm_coefs('cca', criterion, target_error,
                                        target_power, data_home)

        req_corr = np.exp((- np.log(n_req) + intercept
                           + np.log(px + py) * coefs[1]
                           + np.abs(ax + ay) * coefs[2]) / coefs[0])
        if req_corr >= 1:
            req_corr = 1

    elif algorithm == 'generative_model':
        raise NotImplementedError('TODO')

    else:
        raise ValueError('Invalid algorithm: {}'.format(algorithm))

    return req_corr


def pls_req_corr(
        X, Y, ax, ay,
        n_req,
        criterion='combined',
        algorithm='linear_model',
        target_power=0.9,
        target_error=0.1,
        expl_var_ratio=0.3,
        data_home=None,
):
    """Determines the minimum required true correlation to achieve power and
    error levels.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_X_features) or int >= 2
        either a data matrix or directly the number of features for data matrix
        :math:`X`
    Y : np.ndarray (n_samples, n_Y_features) or int >= 2
        either a data matrix or directly the number of features for data matrix
        :math:`Y`
    ax : float < 0 or None
        principal component spectrum decay constant, if ``X`` is not a data
        matrix, ``None`` otherwise
    ay : float < 0 or None
        principal component spectrum decay constant, if ``Y`` is not a data
        matrix, ``None`` otherwise
    n_req : sample_size
        available sample size
    criterion : str
        criterion according to which sample sizes are estimated.
        Can be:

        - ``'combined'``
        - ``'power'``
        - ``'association_strength'``
        - ``'weight'``
        - ``'score'``
        - ``'loading'``
        - ``'crossloading'``
    algorithm : str
        algorithm used to calculate sample sizes.
        Can be:

        - ``'linear_model'``
    target_power : float between o and 1
        if ``criterion`` is ``'combined'`` or ``'power'`` sample size is
        chosen to obtain at least ``target_power`` power
    target_error : float between 0 and 1
        if criterion is not ``'power'`` sample size is chosen to obtain at
        most ``target_error`` error in error metric(s)
    expl_var_ratio : float
        if ``X`` or ``Y`` is a data matrix, ``ax`` or ``ay``, respectively,
        will be estimated directly from the data using the number of principal
        components that explain this amount of variance
    data_home : None or str
        path where outcome data are stored, ``None`` indicates default path

    Returns
    -------
    req_corr : float
        minimum required true correlation
    """

    px, py = _check_pxy(X, Y)
    ax, ay = _check_axy(X, Y, ax, ay, expl_var_ratio=expl_var_ratio)

    if algorithm == 'linear_model':

        intercept, coefs = get_lm_coefs('pls', criterion, target_error,
                                        target_power, data_home)

        req_corr = np.exp((- np.log(n_req) + intercept
                           + np.log(px + py) * coefs[1]
                           + np.abs(ax + ay) * coefs[2]) / coefs[0])
        if req_corr >= 1:
            req_corr = 1

    elif algorithm == 'generative_model':
        raise NotImplementedError('TODO')

    else:
        raise ValueError('Invalid algorithm: {}'.format(algorithm))

    return req_corr
