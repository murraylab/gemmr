[![Build Status](https://travis-ci.com/murraylab/gemmr.svg?branch=master)](https://travis-ci.com/murraylab/gemmr)
[![codecov](https://codecov.io/gh/murraylab/gemmr/branch/master/graph/badge.svg)](https://codecov.io/gh/murraylab/gemmr)
[![Documentation Status](https://readthedocs.org/projects/mdhelmer-demo/badge/?version=latest)](https://mdhelmer-demo.readthedocs.io/en/latest/?badge=latest)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gemmr)
![PyPI](https://img.shields.io/pypi/v/gemmr)

gemmr - Generative Modeling of Multivariate Relationships
=========================================================

*gemmr* calculates required sample sizes for Canonical Correlation Analysis (CCA) and
Partial Least Squares (PLS). In addition, it can generate synthetic datasets for use 
with CCA and PLS, and provides functionality to run and examine CCA and PLS analyses.
It also provides a Python wrapper for *PMA*, a sparse CCA implementation.

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
      
Installation
------------

The easiest way to install *gemmr* is with `pip`:
```
pip install gemmr
```
 
Alternatively, to install and use the most current code:
```
git clone https://github.com/mdhelmer/gemmr.git
cd gemmr
python setup.py install
```

Documentation
-------------
 
Extensive documentation can be found [here](https://mdhelmer-demo.readthedocs.io/en/latest/).

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
If you're using *gemmr* in a publication, please cite **TODO**
