"""Functionality to load datasets."""

import warnings

import numpy as np
import xarray as xr
import pandas as pd
import pkg_resources
import atexit

from ..generative_model import setup_model, generate_data

import os
_default_data_home = os.path.expanduser('~/gemmr_data')

_remote_urls = {
    'synthanad_cca_cca.nc': 'https://osf.io/5cvxt/download',
}

__all__ = [
    'set_data_home',
    'load_outcomes',
    'load_metaanalysis_outcomes',
    'generate_example_dataset',
    'load_metaanalysis',
    'print_ds_stats',
]


atexit.register(pkg_resources.cleanup_resources)


def set_data_home(data_home):
    """Set directory in which outcome data is stored

    Parameters
    ----------
    data_home : str
        folder containaing outcome data
    """
    global _default_data_home
    _default_data_home = data_home


def _fetch_synthanad(fname, path):
    """Retrieve file from remote repository.

    Used remote URL is ``_remote_urls[fname]``.

    Parameters
    ----------
    fname : str
        name of file to retrieve
    path : str
        path where to store file

    Raises
    ------
    ValueError
        if some error ocurred during download
    """
    from urllib.request import urlretrieve
    urls = _remote_urls[fname]
    if isinstance(urls, list):  # download and concatenate a multi-part file

        part_fnames = []
        for i, _url in enumerate(urls):
            part_fname = path + ".{}".format(i)
            urlretrieve(_url, part_fname)

        with open(path, 'wb') as combined_file:
            for part_fname in part_fnames:
                chunk = open(part_fname, 'rb').read()
                combined_file.write(chunk)
                os.remove(part_fname)

    else:  # download a single-part file
        urlretrieve(_remote_urls[fname], path)

    if not os.path.exists(path):
        raise ValueError("Problem occurred during download of "
                         "{}".format(_remote_urls[fname]))


def _check_data_home(data_home, subfolder=None):
    """Check data_home directory

    Parameters
    ----------
    data_home : None or str
        folder in which outcome data resides, if ``None`` defaults to
        ``~/gemmr_data``
    subfolder : str or None
        if not None, append ``subfolder`` to ``data_home``
    """
    if data_home is None:
        data_home = _default_data_home

    if subfolder is not None:
        data_home = os.path.join(data_home, subfolder)

    return data_home


def load_outcomes(model, estr=None, tag=None, data_home=None, fetch=True):
    """Load previously generated outcome data.

    Loads the file ``{data_home}/synthanad_{model}_{estr}[_{tag}].nc``

    Parameters
    ----------
    model : str
        ''`cca`'' or ''`pls`''
    estr : None or str
        name of estimator used to analyzed the data, defaults to ``model``
    tag : None or str
        possibly an additional tag to identify the outcome file, defaults to
        an empty string
    data_home : None or str
        folder in which outcome data resides, if ``None`` defaults to
        ``~/gemmr_data``
    fetch : bool
        if True and data is not found in ``data_home`` attempts to retrieve it
        from repository

    Returns
    -------
    data : xr.Dataset
        outcome dataset
    """

    if estr is None:
        estr = model

    if tag is None:
        tag = ''
    else:
        tag = '_' + tag

    synthanad_home = _check_data_home(data_home)
    fname = 'synthanad_{}_{}{}.nc'.format(model, estr, tag)
    path = os.path.join(synthanad_home, fname)
    _check_outcome_data_exists(fname, path, fetch=fetch)
    return xr.open_dataset(path)


def load_metaanalysis_outcomes(px, py, n, data_home=None, fetch=True):
    """Load previously generated outcomes for a specific parameter set.

    Parameters
    ----------
    px : int
        number of features in dataset X
    py : int
        number of features in dataset Y
    n : int
        number of samples in datasets
    data_home : None or str
        folder in which outcome data resides, if ``None`` defaults to
        ``~/gemmr_data``
    fetch : bool
        if True and data is not found in ``data_home`` attempts to retrieve it
        from repository

    Returns
    -------
    data : xr.Dataset
        metaanalysis dataset
    """
    synthanad_home = _check_data_home(data_home)
    fname = os.path.join('metaanalysis',
                         'metaanalysis_paramset_'
                         'px{}_py{}_n{}.nc'.format(px, py, n))
    path = os.path.join(synthanad_home, fname)
    if not os.path.exists(path):
        raise FileNotFoundError("File ({}) not found. Please download it "
                                "from https://osf.io/8expj/".format(fname))
    return xr.open_dataset(path)


def _check_outcome_data_exists(fname, path, fetch=True):
    """Check if a given outcome data file exists and possibly retrieve it if
    not.

    Parameters
    ----------
    fname : str
        name of file which is checked for existence
    path : str
        local path where it is searched
    fetch : bool
        if True and data is not found in ``data_home`` attempts to retrieve it
        from repository

    Raises
    -------
    FileNotFoundError
        if file doesn't exist
    """
    if not os.path.exists(path) and fetch:

        warnings.warn("Couldn't find data file: {}. "
                      "Downloading it...".format(path))
        _fetch_synthanad(fname, path)

    if not os.path.exists(path):
        raise FileNotFoundError("Couldn't find data files")


def load_metaanalysis(data_home=None, fetch=True):
    """Load table of metaanalysis features.

    Parameters
    ----------
    data_home : None or str
        folder in which outcome data resides, if ``None`` defaults to
        ``~/gemmr_data``
    fetch : bool
        if True and data is not found in ``data_home`` attempts to retrieve it
        from repository

    Returns
    -------
    table : pd.DataFrame
        metaanalysis table
    """
    synthanad_home = _check_data_home(data_home)
    fname = 'metaanalysis/metaanalysis.xlsx'
    path = os.path.join(synthanad_home, fname)
    _check_outcome_data_exists(fname, path, fetch=fetch)
    return pd.read_excel(path, header=3)


def print_ds_stats(ds):
    """Print outcome dataset statistics.

    Parameters
    ----------
    ds : xr.Dataset
        outcome dataset
    """
    if 'mode' in ds.dims:
        print('n_modes\t\t', len(ds.mode))
        ds = ds.sel(mode=0)

    for d in ['rep', 'perm', 'bs', 'loo']:
        if d in ds.dims:
            print('n_{}\t\t'.format(d), len(ds[d]))

    print('n_per_ftr\t', ds.n_per_ftr.values)
    print('r\t\t', ds.r.values)
    print('px\t\t', ds.px.values)

    axPlusay = ds.ax + ds.ay
    print('ax+ay range\t({:.2f}, {:.2f})'.format(
        float(axPlusay.min()), float(axPlusay.max())))

    mask = np.isfinite(ds.py)
    if ((ds.py == ds.px) | (~ mask)).all():
        print('py\t\t== px')
    else:
        print('py\t\t!= px')

    print()
    print(ds.between_assocs.sel(n_per_ftr=ds.n_per_ftr[0], rep=0, drop=True)
          .count('Sigma_id').rename('n_Sigmas'))
    print()

    if 'power' in ds:
        print('power\t\tcalculated')
    else:
        print('power\t\tnot calculated')


def generate_example_dataset(model, px=5, py=5, ax=0, ay=0, r_between=0.3,
                             n=1000, random_state=0):
    """Convenience function returning an example dataset for use with CCA or
    PLS.

    Parameters
    ----------
    model : "cca" or "pls"
        model for which example data is returned
    px : int
        number of features in dataset `X`
    py : int
        number of features in dataset `Y`
    ax : float < 0
        prinicpal component spectrum decay constant for `X`
    ay : float < 0
        prinicpal component spectrum decay constant for `Y`
    r_between : float between 0 and 1
        assumed true correlation between weighted composites of `X` and `Y`
    n : int
        number of samples to be returned
    random_state : None, int or random-number-generator instance
        for random number generator initialization

    Returns
    -------
    X : np.ndarray (n_samples, n_features)
        dataset `X`
    Y : np.ndarray (n_samples, n_features)
        dataset `Y`
    """

    Sigma = setup_model(
        model, random_state=random_state,
        px=px, py=py, qx=0.9, qy=0.9,
        m=1, c1x=1, c1y=1, ax=ax, ay=ay, r_between=r_between, a_between=-1,
        max_n_sigma_trials=10000, expl_var_ratio_thr=1. / 2,
        cx=None, cy=None, verbose=False
    )
    X, Y = generate_data(Sigma, px, n, random_state=0)

    return X, Y
