from numpy.testing import assert_raises

from gemmr.data import *
import gemmr.data.loaders

from gemmr.data.loaders import _check_data_home, load_outcomes, \
    _check_outcome_data_exists, load_metaanalysis, generate_example_dataset


def test_set_data_home():
    original_data_home = gemmr.data.loaders._default_data_home
    set_data_home('DUMMY')
    assert gemmr.data.loaders._default_data_home == 'DUMMY'
    # restore
    set_data_home(original_data_home)
    assert gemmr.data.loaders._default_data_home == original_data_home


def test__check_data_home():
    path = 'path'
    assert _check_data_home(path) == path
    assert _check_data_home(None) == gemmr.data.loaders._default_data_home
    assert _check_data_home(path, 'subfolder').startswith(path)


def test_load_outcomes():
    # Nothing to test?
    pass


def test__check_outcome_data_exists():
    assert_raises(FileNotFoundError, _check_outcome_data_exists, 'fname',
                  'path/that/most/definitely/does/not/exist/fname',
                  fetch=False)


def test_load_metaanalysis():
    # Nothing to test?
    pass


def test_generate_example_dataset():
    # Nothing to test?
    pass
