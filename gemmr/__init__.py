"""Generative Modeling for Multivariate Relationships"""

from .sample_size.linear_model import cca_sample_size, pls_sample_size
from .sample_size.univariate_correlations import pearson_sample_size

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
