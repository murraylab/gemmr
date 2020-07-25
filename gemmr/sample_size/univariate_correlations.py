"""Sample sizes for stable univariate correlatinons."""

import numpy as np
import scipy.stats
import scipy.optimize


__all__ = ['pearson_sample_size']


def pearson_sample_size(
        rs=(.1, .3, .5),
        criterion='power',
        # target_error=0.1,
        target_power=0.9,
        alpha=.05,
):
    """Calculate required sample sizes for accurate estimation of Pearson
    correlation.

    Parameters
    ----------
    rs : tuple-like of floats between 0 and 1
        power is calculated for these assumed true correlations
    criterion : str
        must be 'power'
    target_power : float between 0 and 1
        target power
    alpha : float between 0 and 1
        type 1 error rate

    Returns
    -------
    sample_sizes : dict {r: sample_size}
        gives the required sample size for each assumed true correlation

    References
    ----------
    Cohen (1988), Statistical power analysis for the behavioral sciences
    """
    if criterion == 'power':
        return {r: _cohen_pearson_sample_size(r, alpha, 1 - target_power)
                for r in rs}
    else:
        raise ValueError('Invalid criterion: {}'.format(criterion))


def _cohen_pearson_sample_size(r, alpha, beta):
    """Sample size for testing whether a Pearson correlation coefficient is 0

    Sample size is calculated for a one-sided test using equation (12.3.5) in
    Cohen (1988)

    Parameters
    ----------
    r : float between 0 and 1
        assumed true correlation
    alpha : float between 0 and 1
        false-positive rate
    beta : float between 0 and 1
        false-negative rate

    Returns
    -------
    n : int
        sample size

    References
    ----------
    Cohen (1988), Statistical power analysis for the behavioral sciences
    """
    z_power = scipy.stats.norm(0, 1).ppf(1 - beta)
    z_sig = scipy.stats.norm(0, 1).ppf(1 - alpha)

    def zp_(n):
        return np.arctanh(r) + r/(2*(n - 1))

    def fun(n):
        return ((z_power + z_sig) / zp_(n)) ** 2 + 3 - n

    res = scipy.optimize.fsolve(fun, 100)[0]
    return int(np.ceil(res))
