"""Plot functions."""

import numpy as np
import holoviews as hv


__all__ = ['mean_metric_curve', 'heatmap_n_req', 'polar_hist']


def mean_metric_curve(metric, rs=(.1, .3, .5), n_per_ftr_typical=5,
                      ylabel=None):
    """Plots mean curves for given ``rs`` as a function of ``n_per_ftr``.

    Parameters
    ----------
    metric : xr.DataArray
        must have dimensions ``r`` and ``n_per_ftr``, all other dimensions
        are averaged over
    rs : tuple-like
        separate curves are plotted for each entry of ``rs``
    n_per_ftr_typical : int or None
        if not ``None``, a vertical dashed line is plotted at this value
    ylabel : str or None
        y-label

    Returns
    -------
    panel : hv.Overlay
    """

    other_dims = [d for d in metric.dims if d not in ['r', 'n_per_ftr']]
    if len(other_dims) > 0:
        metric = metric.stack(DUMMYDIM=other_dims).mean('DUMMYDIM')

    if ylabel is not None:
        metric = metric.rename(ylabel)

    panel = hv.Overlay()
    for r in rs:
        panel *= hv.Curve(metric.sel(r=r))

    if n_per_ftr_typical is not None:
        panel *= hv.VLine(n_per_ftr_typical)

    return panel


def heatmap_n_req(n_req, clabel='Required sample size'):
    """Plots a heatmap of required number of samples as a function of number of
    features and true correlation.

    Parameters
    ----------
    n_req : xr.DataArray
        elements represent number of samples. Must have dimensions ``px`` and
        ``r``. All other dimensions will be averaged over.
    clabel : str
        label for colorbar

    Returns
    -------
    fig : hv.QuadMesh
    """

    if not (('ptot' in n_req.dims) and ('r' in n_req.dims)):
        raise ValueError('DataArray `n` must have dimensions `ptot` and `r`.')

    n_req = n_req.rename('y')  # so that "redim" below works

    other_dims = [d for d in n_req.dims if not d in ['ptot', 'r']]
    if len(other_dims) > 0:
        n_req = n_req.stack(DUMMYDIMENSION=other_dims).mean('DUMMYDIMENSION')

    return (
        hv.QuadMesh(
            n_req,
            kdims=['ptot', 'r'],
        )
    ).redim(
        r=r'$r_\mathrm{true}$',
        ptot='Number of features',
        y=clabel
    )


def polar_hist(angles, bins=None, mark_mean=False):
    """Plot a polar histogram.

    Parameters
    ----------
    angles : array-like
        angles for which to generate the histogram
    bins : None or "semicircle"
        if None the circular histogram will be generated between the minimum
        and maximum element of ``angles``, if "semicircle" the upper half
        of the circle will be plotted
    mark_mean : bool
        whether to mark the circular mean with a "o"

    Returns
    -------
    panel : hv.Overlay
    """
    if bins is None:
        mn, mx = angles.min(), angles.max()
        # mx *= 1 + (mx-mn) * 1e-6
        bins = np.linspace(mn, mx, 18)
    elif bins == 'semicircle':
        bins = np.linspace(0, np.pi, 18)
    else:
        raise ValueError('Invalid bins: {}'.format(bins))

    hist, _ = np.histogram(angles, bins=bins)

    thetas_ = np.r_[
        bins[0], np.stack([bins[1:-1], bins[1:-1]]).T.ravel(), bins[-1]
    ]
    rs_ = np.stack([hist, hist]).T.ravel()
    panel = hv.Overlay()
    panel *= hv.Curve(
        (thetas_, rs_), 'theta', 'r'
    ).opts(projection='polar', ylim=(0, 1.1 * hist.max()))

    if mark_mean:
        circ_mean = np.exp(1j * angles).mean()
        mean_angle = np.arctan(circ_mean.imag / circ_mean.real)
        if mean_angle < 0:
            mean_angle = np.pi + mean_angle
        panel *= hv.Scatter(([mean_angle], [hist.max()])).opts(marker='o')

    return panel
