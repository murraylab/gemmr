import numpy as np
from numpy.testing import assert_raises
from gemmr.sample_size.univariate_correlations import *
from gemmr.sample_size.univariate_correlations import _cohen_pearson_sample_size


def test_pearson_sample_size():
    rs = (.1, .3, .5)
    res = pearson_sample_size(rs)
    for r in rs:
        assert r in res
        assert res[r] > 0
    assert_raises(ValueError, pearson_sample_size, criterion='NOTACRITERION')

def test__cohen_pearson_sample_size():

    power_table_01 = [
        (56, .2, .20),
        (92, .3, .73),
        (58, .4, .80),
        (52, .5, .94),
        (19, .6, .69),
        (20, .7, .91),
        (10, .8, .76),
        (8, .9, .88)
    ]
    power_table_05 = [
        (96, .1, .25),
        (80, .2, .56),
        (58, .3, .75),
        (52, .4, .91),
        (29, .5, .89),
        (18, .6, .87),
        (11, .7, .83),
        (8, .8, .85),
        (8, .9, .97),
    ]
    power_table_10 = [
        (44, .1, .26),
        (42, .2, .50),
        (37, .3, .71),
        (34, .4, .87),
        (26, .5, .92),
        (9, .6, .72),
        (11, .7, .91),
        (10, .8, .97),
    ]
    power_tables = {
        .01: power_table_01,
        .05: power_table_05,
        .10: power_table_10
    }
    for alpha, power_table in power_tables.items():
        for n, r, power in power_table:
            n_predicted = _cohen_pearson_sample_size(r, alpha, 1 - power)
            print(n_predicted, n)
            assert np.abs(n_predicted - n) <= 1
