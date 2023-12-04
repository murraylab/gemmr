"""Functionality to load datasets."""

import warnings

import numpy as np
import xarray as xr
import pandas as pd
import pkg_resources
import atexit

from ..generative_model import GEMMR

import os
_default_data_home = os.path.expanduser('~/gemmr_data/gemmr_latest')


_remote_urls = {
    'synthanad_cca_cca.nc': 'https://osf.io/5cvxt/download',
    'sweep_cca_cca_random_sum+-2+0_wOtherModel.nc': 'https://osf.io/h8yqe/download',
    'sweep_cca_cca_random_sum+-3+-2_wOtherModel.nc': 'https://osf.io/pcsvx/download',
    'sweep_cca_cca_random_sum+-2+-2.nc': 'https://osf.io/kgqzx/download',
    'sweep_pls_pls_random_sum+-2+0_wOtherModel.nc': 'https://osf.io/jdrn7/download',
    'sweep_pls_pls_random_sum+-3+-2_wOtherModel.nc': 'https://osf.io/wa8f6/download',
    'sweep_pls_pls_random_sum+-2+-2.nc': 'https://osf.io/r27kc/download',
    'sweep_cca_cca_random_sum+-3+0_wOtherModel.nc': 'https://osf.io/m2eg3/download',
    'sweep_pls_pls_random_sum+-3+0_wOtherModel.nc': 'https://osf.io/ntuch/download',
}


__all__ = [
    'set_data_home',
    'load_outcomes',
    'load_metaanalysis_outcomes',
    'load_other_outcomes',
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


def load_outcomes(dsid, model=None, data_home=None, fetch=True,
                  add_prefix=False):
    """Load previously generated outcome data.

    Loads the file ``{data_home}/{dsid}.nc``

    Parameters
    ----------
    dsid : str
        Descriptor for dataset to load.
    model : str
        ''`cca`'' or ''`pls`''
    data_home : None or str
        folder in which outcome data resides, if ``None`` defaults to
        ``~/gemmr_data``
    fetch : bool
        if True and data is not found in ``data_home`` attempts to retrieve it
        from repository
    add_prefix: bool
        if ``True``, data variables in dataset are prefixed with `model` unless
        they already start with `other_model` where `other_model` is set to
        "cca" if ``model == "pls"``, or to "pls" if ``model == "cca"``.

    Returns
    -------
    data : xr.Dataset
        outcome dataset
    """

    if model is None:
        if 'cca' in dsid:
            if 'pls' in dsid:
                raise ValueError("Couldn't auto-detect model, "
                                 "need to set it explicitly")
            model = 'cca'
        elif 'pls' in dsid:
            if 'cca' in dsid:
                raise ValueError("Couldn't auto-detect model, "
                                 "need to set it explicitly")
        else:
            raise ValueError("Couldn't auto-detect model, "
                             "need to set it explicitly")

    synthanad_home = _check_data_home(data_home)
    print(f"Loading data from subfolder '{synthanad_home.rsplit('/', 1)[1]}'")
    fname = f'{dsid}.nc'
    path = os.path.join(synthanad_home, fname)
    _check_outcome_data_exists(fname, path, fetch=fetch)
    ds = xr.open_dataset(path)

    if add_prefix:

        if model == 'cca':
            other_model = 'pls'
        elif model == 'pls':
            other_model = 'cca'
        else:
            raise ValueError(f"model must be 'cca' or 'pls', not {model}")

        for v in ds.data_vars:
            if v in ['py']:
                continue
            elif not v.startswith(other_model):
                ds = ds.rename(**{v: f'{model}_{v}'})

    return ds


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
    fname = os.path.join(synthanad_home, 'litana', f'{px}_{py}_{n}_0.nc')
    path = os.path.join(synthanad_home, fname)
    if not os.path.exists(path):
        raise FileNotFoundError("File ({}) not found. Please download it "
                                "from https://osf.io/8expj/".format(fname))
    return xr.open_dataset(path)


def load_other_outcomes(fname, data_home=None, fetch=True):
    """Load previously generated data.

    Parameters
    ----------
    fname : str
        Name of file, must be compatible with ``xr.load_dataset``.
    data_home : None or str
        folder in which outcome data resides, if ``None`` defaults to
        ``~/gemmr_data``
    fetch : bool
        if True and data is not found in ``data_home`` attempts to retrieve it
        from repository

    Returns
    -------
    data : xr.Dataset
        loaded dataset
    """
    data_home = _check_data_home(data_home)
    path = os.path.join(data_home, fname)
    _check_outcome_data_exists(fname, path, fetch=fetch)
    ds = xr.open_dataset(path)
    return ds


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
        raise FileNotFoundError(f"Couldn't find data file: {path}")


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
    fname = 'litana/metaanalysis.xlsx'
    path = os.path.join(synthanad_home, fname)
    _check_outcome_data_exists(fname, path, fetch=fetch)
    return pd.read_excel(path, header=3)


def print_ds_stats(ds, prefix=''):
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

    axPlusay = ds[f'{prefix}ax'] + ds[f'{prefix}ay']
    print('ax+ay range\t({:.2f}, {:.2f})'.format(
        float(axPlusay.min()), float(axPlusay.max())))

    mask = np.isfinite(ds.py)
    if ((ds.py == ds.px) | (~ mask)).all():
        print('py\t\t== px')
    else:
        print('py\t\t!= px')

    print()
    print(
        ds[f'{prefix}between_assocs'].sel(n_per_ftr=ds.n_per_ftr[0], rep=0,
                                          drop=True)
          .count('Sigma_id').rename('n_Sigmas')
    )
    print()

    if f'{prefix}power' in ds:
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

    gm = GEMMR(
        model, random_state=random_state,
        wx=px, wy=py,
        ax=ax, ay=ay, r_between=r_between,
        max_n_sigma_trials=10000, expl_var_ratio_thr=1. / 2
    )
    X, Y = gm.generate_data(n)

    return X, Y
