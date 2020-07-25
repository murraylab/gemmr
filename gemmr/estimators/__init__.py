"""Estimators implementing models with which data are analyzed."""

from .vanilla import *
try:
    from .r_estimators import *
except ModuleNotFoundError:
    pass
from .helpers import SingularMatrixError
