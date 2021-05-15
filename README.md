[![Build Status](https://travis-ci.com/murraylab/gemmr.svg?branch=master)](https://travis-ci.com/murraylab/gemmr)
[![codecov](https://codecov.io/gh/murraylab/gemmr/branch/master/graph/badge.svg)](https://codecov.io/gh/murraylab/gemmr)
[![Documentation Status](https://readthedocs.org/projects/gemmr/badge/?version=latest)](https://gemmr.readthedocs.io/en/latest/?badge=latest)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gemmr)
![PyPI](https://img.shields.io/pypi/v/gemmr)

gemmr - Generative Modeling of Multivariate Relationships
=========================================================

*gemmr* calculates required sample sizes for Canonical Correlation Analysis (CCA) and
Partial Least Squares (PLS). In addition, it can generate synthetic datasets for use 
with CCA and PLS, and provides functionality to run and examine CCA and PLS analyses.
It also provides a Python wrapper for *PMA*, a sparse CCA implementation.

Hardware requirements
---------------------

GEMMR runs on standard hardware. To thoroughly sweep through parameters of the generative model a high-performance-computing (HPC) environment is recommended.

Dependencies
------------

  * numpy
  * scipy
  * pandas
  * xarray
  * netcdf4
  * scikit-learn
  * statsmodels
  * joblib
  * tqdm

Some functions have additional dependencies that need to be installed separately if they are used:
  * holoviews
  * rpy2
      
The repository also contains an ``environment.yml`` file specifying a conda-environment with specific versions of all dependencies. We have tested the code with this environment. To instantiate the environment run
```
>>> conda env create -f environment.yml
```
      
Installation
------------

The easiest way to install *gemmr* is with `pip`:
```
pip install gemmr
```
 
Alternatively, to install and use the most current code:
```
git clone https://github.com/murraylab/gemmr.git
cd gemmr
python setup.py install
```

Installation of *gemmr* itself (without potentially required dependencies) should take only a few seconds.

Documentation
-------------
 
Extensive documentation can be found [here](https://gemmr.readthedocs.io/en/latest/).

The documentation contains
   * Demonstration of the *gemmr*'s functionality, including exptected outputs (all of which should execute quickly)
   * Juyter notebooks detailing generation of the figures for the accompanying manuscripts
   * API reference

To generate the documentation from source, install *gemmr* as described above and make sure you also have the following dependencies installed:
   * ipython
   * matplotlib
   * sphinx
   * nbsphinx
   * sphinx_rtd_theme
and run (in the `doc` subfolder):
```
make html
```
and open `doc/_build/html/index.html`  .

Citation
--------
If you're using *gemmr* in a publication, please cite [Helmer et al. (2020)](https://www.biorxiv.org/content/10.1101/2020.08.25.265546v1)
