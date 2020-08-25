.. _installation:

Installation
============

Dependencies
------------

- Python (3.6 or later)
- [numpy]_
- [scipy]_
- [pandas]_
- [xarray]_
- netcdf4
- [scikit-learn]_
- [statsmodels]_
- joblib
- tqdm

Optional dependencies
---------------------

These need to be installed separately to use some functionality:

- holoviews (for some plotting functions)
- rpy2 (for sparse CCA)

Instructions
------------

*gemmr* can be installed with ``pip``::

	$ pip install gemmr
 
Alternatively, the most current version can be obtained from github::

	$ git clone https://github.com/murraylab/gemmr.git
	$ cd gemmr
	$ python setup.py install

Tests
-----

Unit tests can be run with ``pytest`` from the root directory::

    $ pytest

References
----------
.. [numpy] van der Walt S. *et al.*, "The NumPy Array: A Structure for Efficient Numerical Computation", Computing in Science & Engineering, 13, 22-30, 2011. DOI: 10.1109/MCSE.2011.37. https://numpy.org
.. [scipy] Virtanen P. *et al.*, "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python", Nature Methods, 2020. DOI:10.1038/s41592-019-0686-2. https://scipy.org/
.. [pandas] McKinney W. "Data structures for statistical computing in python", Proceedings of the 9th Python in Science Conference, Volume 445, 2010. https://pandas.pydata.org/
.. [xarray] Hoyer, S. & Hamman, J. "xarray: N-D labeled Arrays and Datasets in Python", Journal of Open Research Software. 5(1), p.10. 2017. DOI: 10.5334/jors.148. http://xarray.pydata.org/
.. [scikit-learn] Buitinck *et al.*, "API design for machine learning software: experiences from the scikit-learn project", ECML PKDD Workshop: Languages for Data Mining and Machine Learning, 2013. https://scikit-learn.org/
.. [statsmodels] Seabold, Skipper, and Josef Perktold. “statsmodels: Econometric and statistical modeling with python.” Proceedings of the 9th Python in Science Conference, 2010. https://www.statsmodels.org
