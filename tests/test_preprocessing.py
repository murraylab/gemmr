import numpy as np
import pandas as pd

from numpy.testing import assert_raises

from gemmr.data.preprocessing import *
from gemmr.data.preprocessing import prepare_confounders, preproc_sm, \
    preproc_fc, _check_features, _smith_feature_names


def _mk_fc():
    np.random.seed(0)
    X = np.random.normal(size=(33, 8))
    return X


def _mk_sm():
    np.random.seed(0)
    n = 33
    p = 20
    feature_names = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10',]
    cols = ['fMRI_3T_ReconVrs',
            'Weight', 'Height', 'BPSystolic', 'BPDiastolic', 'HbA1C',
            'FS_BrainSeg_Vol', 'FS_IntraCranial_Vol',
            'movement_AbsoluteRMS_mean', 'movement_RelativeRMS_mean',] + feature_names
    sm = pd.DataFrame(np.random.normal(size=(n, p)), columns=cols)
    sm['fMRI_3T_ReconVrs'] = ['r177', 'r177 r227', 'r227'] * 11
    return sm, feature_names


def test_prepare_confounders():
    sm = _mk_sm()[0]
    prepare_confounders(
        sm,
        confounders=('movement_AbsoluteRMS_mean', 'movement_RelativeRMS_mean'),
        hcp_confounders=True, hcp_confounder_software_version=True,
        squared_confounders=True, )
    assert_raises(ValueError, prepare_confounders, sm,
                  confounders=['NOTACONFOUNDER'])


def test_preproc_sm():
    sm, ftr_names = _mk_sm()
    confounders = prepare_confounders(sm)
    uu2, uu2_white, S4_raw, S4_deconfounded, feature_names = preproc_sm(sm, confounders,
               feature_names=ftr_names, final_deconfound=False)
    assert set(feature_names) == set(ftr_names)
    assert np.allclose(uu2_white.var(0), 1)


def test_preproc_fc():
    sm = _mk_sm()[0]
    confounders = prepare_confounders(sm)
    X = _mk_fc()
    uu1, uu1_white, vv1, N4 = preproc_fc(X, confounders)
    assert np.allclose(uu1_white.var(0), 1)


def test_preproc_smith():
    fc = _mk_fc()
    sm, feature_names = _mk_sm()
    res = preproc_smith(fc, sm, feature_names=feature_names,
                        final_sm_deconfound=False,
                        hcp_data_dict_correct_pct_to_t=False)

    assert 'X' in res
    assert 'Y' in res
    assert 'X_whitened' in res
    assert 'Y_whitened' in res
    assert 'Y_raw' in res
    assert 'feature_names' in res

    res = preproc_smith(fc, sm, feature_names=feature_names,
                        final_sm_deconfound=False,
                        hcp_data_dict_correct_pct_to_t=True)

    assert 'X' in res
    assert 'Y' in res
    assert 'X_whitened' in res
    assert 'Y_whitened' in res
    assert 'Y_raw' in res
    assert 'feature_names' in res


def test__check_features():
    checked_features = _check_features(feature_names=None, available_feature_names=None,
                                       hcp_data_dict_correct_pct_to_t=False)
    assert np.all(np.sort(checked_features) == np.sort(_smith_feature_names))
