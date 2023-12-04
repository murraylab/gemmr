"""Determine sample sizes by interpolating run simulation outcomes."""

import numpy as np
import xarray as xr
from scipy.stats import linregress
from scipy.interpolate import InterpolatedUnivariateSpline

from ..metrics import mk_betweenAssocRelError, mk_weightError, \
    mk_scoreError, mk_loadingError, mk_betweenCorrRelError


__all__ = ['calc_n_required', 'calc_max_n_required',
           'calc_n_required_all_metrics']


def proc_roots(x, y, interpolator, admissible_region, max_yrange_ratio=1./3,
               verbose=False):
    """Select one of the roots from interpolation

    Parameters
    ----------
    x : np.ndarray (n_xs,)
        x-values of interpolated curve
    y : np.ndarray (n_xs,)
        y-values of interpolated curve
    interpolator : interpolator-instance
        fitted interpolator
    admissible_region : +1 or -1
        if -1 interpolated values below 0 are interpreted as admissible (i.e.
        below the error margin); if +1 interpolated values above 0 are
        interpreted as admissible (i.e. below the error margin
    max_yrange_ratio : float
        see :func:`filter_interpol_roots`
    verbose : bool
        see :func:`filter_interpol_roots`

    Returns
    -------
    root : float
        best root, ``np.nan`` if no root could be determined.
        If ``roots`` contains multiple items they are filtered by
        :func:`filter_interpol_roots`. The smallest remaining root for which
        interpolated values remain below the error margins is then selected
    """
    roots = interpolator.roots()
    if len(roots) > 1:
        roots = np.sort(
            filter_interpol_roots(x, y, interpolator,
                                  max_yrange_ratio=max_yrange_ratio,
                                  verbose=verbose)
        )

    for root in roots:
        if (admissible_region * interpolator(x[x > root]) >= 0).all():
            return root

    else:  # no admissible root
        return np.nan


def filter_interpol_roots(xs, ys, interpolator, max_yrange_ratio=1./3,
                          x_range_ratio=0.1, verbose=False):
    """Drop roots for which interpolated curve is "noisy".

    If the data to be interpolated is not monotonic (e.g. due to sampling
    error) the cubic spline will follow the non-monotonicity and return more
    than 1 root. Assuming the non-monotonicity is caused by "spikes" in the
    data, the cubic spline will be "unstable" i.e. the interpolation will vary
    strongly close to the found root. This function will thus determine the
    y-range around the roots and if in relation to the total y-range it is
    larger than `max_yrange_ratio` will filter the root

    """
    x_range = xs.ptp()
    y_range = ys.ptp()
    good_roots = []
    for r in interpolator.roots():
        root_neighboring_range = interpolator(np.arange(
            r - x_range_ratio*x_range,
            r + x_range_ratio*x_range,
            x_range_ratio*x_range/20
        )).ptp()
        root_y_range_ratio = root_neighboring_range / y_range
        if root_y_range_ratio < max_yrange_ratio:
            good_roots.append(r)
        else:
            if verbose:
                print('Skipping root {:.3f} as interpolated neighboring '
                      'y-range is large ({:.1f}% of total '
                      'y-range)'.format(r, 100*root_y_range_ratio))
    return good_roots


def calc_max_n_required(*n_requireds):
    """Finds the maximum sample size across datasets.

    Parameters
    ----------
    n_requireds : list-like of ``xr.DataArray`` s with same dimensions

    Returns
    -------
    max_n_required : ``xr.DataArray``
        maximum across ``n_requireds``

    """
    if len(n_requireds) == 0:
        raise ValueError('Need at least 1 argument')
    #for xi, x in enumerate(n_requireds):
    #    if np.isnan(x.sel(r=0.9, px=64)).any():
    #        print(xi, x.sel(r=0.9, px=64))
    max_n_required = n_requireds[0]
    for other_n_required in n_requireds[1:]:
        max_n_required = np.maximum(
            max_n_required,
            other_n_required
        )
    return max_n_required


def _calc_n_required(x, y, y_target_min, y_target_max, verbose=False):
    """Helper function to find required sample size by interpolation.

    ``scipy.interpolate.InterpolatedUnivariateSpline`` performs interpolation.
    If :math:`y`-values remain outside the target region, or the interpolation
    is very noisy (see :func:`proc_roots`) ``np.nan`` is returned.

    Parameters
    ----------
    x : np.ndarray (n_samples,)
        :math:`x`-values
    y : np.ndarray (n_samples,)
        :math:`y`-values
    y_target_min : float
        minimum acceptable value for ``y``
    y_target_max : float
        maximum acceptable value for ``y``, must be >= ``y_target_min``
    verbose : float
        whether some status messages are printed

    Returns
    -------
    n_required : float
        smallest value along dimension :math:`x` for which dimension
        :math:`y` fall between and stays in range ``y_target_min`` and
        ``y_target_max``. If interpolation fails ``np.nan`` is returned.
    """

    if y_target_max < y_target_min:
        raise ValueError('y_target_max must be >= y_target_min')

    # 1) filter nans
    is_finite = np.isfinite(x) & np.isfinite(y)
    if is_finite.sum() == 0:
        return np.nan
    x = x[is_finite]
    y = y[is_finite]

    # 2) if all y-values are within error-bounds return smallest x-value
    if (y >= y_target_min).all() & (y <= y_target_max).all():
        # print('yea')
        return x[0]

    # 3) if y values are either all smaller than lower error bound or all
    # greater than upper error bound return nan
    elif (y <= y_target_min).all() | (y >= y_target_max).all():
        # print('y out of target range')
        return np.nan

    # 4) else (i.e. if there are y-values outside and inside error bounds) find
    # smallet x for which interpolated y-values don't leave error-bounds any
    # more
    else:

        # InterpolatedUnivariateSpline raises exception otherwise, not sure why
        if len(y) < 4:
            # use linear regression
            linreg = linregress(x, y)
            root1 = (y_target_max - linreg.intercept) / linreg.slope
            root2 = (y_target_min - linreg.intercept) / linreg.slope
            root = min([root1, root2])
            if (not np.isfinite(root)) or \
                    (not (np.min(x) <= root <= np.max(x))):
                print('[_calc_n_required] invalid root')
                return np.nan
            return root

        if (y > y_target_max).any() & (y < y_target_max).any():

            interpolator = InterpolatedUnivariateSpline(
                x,
                y - y_target_max,
                k=3
            )

            roots_max = proc_roots(x, y, interpolator, -1, verbose=verbose)
            if not np.isfinite(roots_max):
                roots_max = None

        else:
            roots_max = None

        if (y > y_target_min).any() & (y < y_target_min).any():

            interpolator = InterpolatedUnivariateSpline(
                x,
                y - y_target_min,
                k=3
            )

            roots_min = proc_roots(x, y, interpolator, +1, verbose=verbose)
            if not np.isfinite(roots_min):
                roots_min = None

        else:
            roots_min = None

        all_roots = [roots_min, roots_max]
        finite_roots = [r for r in all_roots if r is not None]
        if len(finite_roots) > 0:
            best_root = max(finite_roots)
            return best_root
        else:
            return np.nan


def calc_n_required(metric, y_target_min, y_target_max,
                    search_dim='n_per_ftr'):
    """Calculate required sample sizes for a given metric.

    Search is performed along dimension ``search_dim``.
    :func:`_calc_n_required` performs a 1-dimensional interpolation using the
    logarithms of dimensions ``search_dim`` as :math:`x`-values and a slice of
    ``metric`` along dimension ``search_dim`` as :math:`y`-values to determine
    an estimation for the smallest value of dimension ``search_dim`` for which
    each slice of `metric` falls into and remains the range between
    ``y_target_min`` and ``y_target_max``.

    Parameters
    ----------
    metric : xr.DataArray
        metric for which required sample size is computed. Must have dimension
        indicated in argument ``search_dim``, and can have arbitrary other
        dimensions
    y_target_min : float
        minimum acceptable metric value
    y_target_max : float
        maximum acceptable metric value
    search_dim : str
        dimension along which interpolation is performed to find required
        sample size

    Returns
    -------
    n_required : xr.DataArray
        required sample sizes. Apart from dimension ``search_dim``,
        ``DataArray`` has same dimensions as ``metric``

    """
    log_n_required = xr.apply_ufunc(
        _calc_n_required,
        np.log(metric[search_dim]), metric,
        input_core_dims=[[search_dim], [search_dim]],
        vectorize=True,
        kwargs=dict(y_target_min=y_target_min, y_target_max=y_target_max)
    ).rename(search_dim + '_required')
    return np.exp(log_n_required)


def calc_n_required_all_metrics(ds, target_power=0.9, target_error=0.1,
                                search_dim='n', prefix='', average_rep=True):
    """Calculate n_required for 5 commonly used metrics, as well as maximum
    across metrics.

    Required sample sizes are calculated with :func:`calc_n_required`.

    Used metrics are power, :func:`.metrics.mk_betweenAssocRelError`,
    :func:`.metrics.mk_weightError`, :func:`.metrics.mk_scoreError`, and
    :func:`.metrics.mk_loadingError`

    NOTE: removed ".sel(mode=0)" and ".dropna('iter', 'all')"

    Parameters
    ----------
    ds : ``xr.Dataset``
        dataset on which metrics will be computed.
    target_power : float between 0 and 1
        minimum acceptable power
    target_error : float between 0 and 1
        maximum accepted error
    average_rep : bool
        if ``True`` calculated values of metrics are averaged across dimension
        ``'rep'`` before calculating required sample size (doesn't apply to
        metric `power`)

    Returns
    -------
    all_n_requireds : dict
        entries are ``xr.DataArray`` s giving required sample sizes
    """

    if search_dim not in ds.dims:
        raise ValueError("search_dim ('{}') is not a dimension "
                         "of ds".format(search_dim))

    power_n_required = _calc_n_req_for_power(ds, target_power, search_dim,
                                             prefix)
    betweenAssocRelError_n_required = _calc_n_req_for_metric(
        mk_betweenAssocRelError, ds, target_error, search_dim, prefix,
        average_rep)
    weightError_n_required = _calc_n_req_for_metric(
        mk_weightError, ds, target_error, search_dim, prefix, average_rep)
    scoreError_n_required = _calc_n_req_for_metric(
        mk_scoreError, ds, target_error, search_dim, prefix, average_rep)
    loadingError_n_required = _calc_n_req_for_metric(
        mk_loadingError, ds, target_error, search_dim, prefix, average_rep)
    max_log_n_required = calc_max_n_required(
        power_n_required,
        betweenAssocRelError_n_required,
        weightError_n_required,
        loadingError_n_required,
        scoreError_n_required,
    )
    return dict(
        power=power_n_required,
        betweenAssoc=betweenAssocRelError_n_required,
        weightError=weightError_n_required,
        loadingError=loadingError_n_required,
        scoreError=scoreError_n_required,
        combined=max_log_n_required,
    )


def _calc_n_req_for_metric(metric, ds, target_error, search_dim, prefix='',
                           average_rep=True):
    _metric = metric(ds, prefix=prefix)
    if average_rep:
        _metric = _metric.mean('rep')
    n_required = calc_n_required(
        _metric, -target_error, target_error, search_dim=search_dim)
    return n_required


def _calc_n_req_for_power(ds, target_power, search_dim, prefix=''):
    power_n_required = calc_n_required(
        ds[f'{prefix}power'], target_power, 1, search_dim=search_dim)
    return power_n_required
