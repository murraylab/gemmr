.. image:: https://travis-ci.com/murraylab/gemmr.svg?branch=master
    :target: https://travis-ci.com/murraylab/gemmr
.. image:: https://codecov.io/gh/murraylab/gemmr/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/murraylab/gemmr
.. image:: https://readthedocs.org/projects/mdhelmer-demo/badge/?version=latest
    :target: https://mdhelmer-demo.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://img.shields.io/pypi/v/gemmr
    :alt: PyPI
.. image:: https://img.shields.io/pypi/pyversions/gemmr
    :alt: PyPI - Python Version

|

Welcome to gemmr's documentation!
==================================

*gemmr* (short for *GEnerative Modeling of Multivariate Relationships*)
calculates sample sizes required to run a canonical correlation
analysis (CCA) or partial least squares (PLS). To that end, it also provides

    - scikit-learn_-style estimators for CCA and PLS,
    - a scikit-learn-style Python wrapper for PMA_, a sparse CCA implementation
    - functionality to generate synthetic data for use with CCA and PLS, and
    - code to determine their parameter dependencies.

.. _scikit-learn: https://scikit-learn.org/stable/index.html
.. _PMA: https://cran.r-project.org/web/packages/PMA/index.html

See also
--------

- source code repository on `github`_

.. _github: https://github.com/mdhelmer/gemmr

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   overview
   installation
   quickstart

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: User Guide

   how_it_works
   sample_size_calculation
   model_param_ana
   analyses_from_paper

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reference

   api
   private_api
   genindex
