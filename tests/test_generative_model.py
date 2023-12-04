import numpy as np
from scipy.spatial.distance import cosine as cosdist

from numpy.testing import assert_raises, assert_allclose, assert_equal, assert_warns, assert_array_almost_equal

from unittest.mock import patch, create_autospec

from sklearn.utils import check_random_state

import gemmr.generative_model
from gemmr.generative_model import *
from gemmr.estimators import SVDCCA, SVDPLS


def _align(x, y):
    for i in range(x.shape[1]):
        if np.dot(x[:, i], y[:, i]) < 0:
            x[:, i] *= -1
    return x


def _test_jcov(model, estr, rs, decimal):
    for px, py in [(2, 4), (16, 8), (32, 32)]:
        for r_between in rs:
            for ax, ay in [(0, 0), (-.5, -1.), (-.25, -.75)]:
                gemmr = GEMMR(model=model, wx=px, wy=py, r_between=r_between, ax=ax, ay=ay)
                if model == 'cca':
                    JointCovarianceModel = JointCovarianceModelCCA
                elif model == 'pls':
                    JointCovarianceModel = JointCovarianceModelPLS
                else:
                    raise ValueError(f'Invalid model: {model}')
                jcov = JointCovarianceModel.from_jcov_model(gemmr, random_state=0)

                assert_allclose(gemmr.true_corrs_, jcov.true_corrs_, atol=0.01, rtol=1e-6)

                _align(jcov.x_weights_, gemmr.x_weights_)
                _align(jcov.y_weights_, gemmr.y_weights_)
                assert_array_almost_equal(gemmr.x_weights_, jcov.x_weights_, decimal=decimal)
                assert_array_almost_equal(gemmr.y_weights_, jcov.y_weights_, decimal=decimal)


def test_JointCovarianceModel():
    _test_jcov('cca', SVDCCA(n_components=1), [.3], 1)
    _test_jcov('cca', SVDCCA(n_components=1), [.5, .7, .9], 2)
    _test_jcov('pls', SVDPLS(n_components=1), [.3, .5], 1)
    _test_jcov('pls', SVDPLS(n_components=1), [.7, .9], 2)


def test_GEMMR():
    assert_raises(TypeError, GEMMR, 'cca')
    assert_raises(TypeError, GEMMR, 'cca', wx=5)
    assert_raises(TypeError, GEMMR, 'cca', wy=7)
    assert_raises(gemmr.generative_model.base.WeightNotFoundError, GEMMR, 'cca', wx=4, wy=5, max_n_sigma_trials=0)
    assert_raises(ValueError, GEMMR, 'BLA')
    assert_raises(ValueError, GEMMR, 'pls', wx=4, wy=5, ax=2)
    assert_raises(ValueError, GEMMR, 'cca', wx=4, wy=5, ay=.2)
    assert_raises(ValueError, GEMMR, 'cca', wx=4, wy=5, r_between=-.1)
    assert_raises(ValueError, GEMMR, 'pls', wx=4, wy=5, r_between=1.1)

    r_between = 0.3
    Sigma = GEMMR('cca', wx=2, wy=2, ax=0, ay=0, r_between=r_between).Sigma_
    SigmaXY = Sigma[:2, 2:]
    r_hat = np.linalg.svd(SigmaXY, full_matrices=False, compute_uv=False)
    assert_array_almost_equal(r_hat, [r_between, 0])


def test_generative_model_class():
    def _test_gm(gm):
        assert hasattr(gm, 'n_components')
        assert hasattr(gm, 'px')
        assert hasattr(gm, 'py')
        assert hasattr(gm, 'ax')
        assert hasattr(gm, 'ay')
        assert hasattr(gm, 'random_state')
        assert hasattr(gm, 'Sigma_')
        assert hasattr(gm, 'true_assocs_')
        assert hasattr(gm, 'true_corrs_')
        assert hasattr(gm, 'x_weights_')
        assert hasattr(gm, 'y_weights_')
        assert hasattr(gm, 'generate_data')

    gemmr = GEMMR('cca', wx=4, wy=5)
    _test_gm(gemmr)
    _test_gm(JointCovarianceModelCCA.from_jcov_model(gemmr))
    _test_gm(JointCovarianceModelPLS.from_jcov_model(gemmr))


def _test_jcov_from_other_jcov(gemmr, estr, Jcov, n_per_ftr=1024):
    n = (gemmr.px + gemmr.py) * n_per_ftr
    X, Y = gemmr.generate_data(n)
    estr.fit(X, Y)

    jcov = Jcov.from_jcov_model(gemmr)

    dissim = 1 - min(
        np.abs(1 - cosdist(estr.x_rotations_[:, 0], jcov.x_weights_[:, 0])),
        np.abs(1 - cosdist(estr.y_rotations_[:, 0], jcov.y_weights_[:, 0])),
    )
    assert dissim < 0.03


def _test_jcov_from_same_jcov(gemmr, Jcov):
    jcov = Jcov.from_jcov_model(gemmr)

    assert np.isclose(gemmr.px, jcov.px)
    assert np.isclose(gemmr.py, jcov.py)
    assert np.isclose(gemmr.n_components, jcov.n_components)
    assert np.isclose(gemmr.ax, jcov.ax)
    assert np.isclose(gemmr.ay, jcov.ay)

    assert np.allclose(gemmr.Sigma_, jcov.Sigma_)

    dissim = 1 - min(
        np.abs(1 - cosdist(gemmr.x_weights_[:, 0], jcov.x_weights_[:, 0])),
        np.abs(1 - cosdist(gemmr.y_weights_[:, 0], jcov.y_weights_[:, 0])),
    )
    assert dissim < 0.001

    assert np.allclose(gemmr.true_assocs_, jcov.true_assocs_)
    assert np.allclose(gemmr.true_corrs_, jcov.true_corrs_)


def test_jcov_from_jcov():
    cca = SVDCCA()
    pls = SVDPLS()
    for px in [2, 4, 8, 16, 32]:
        for py in [px, px + 4]:
            for ax in [-1.5, -1, -.5]:
                for ay in [-1.5, -1, -.5]:
                    for r in [.3, .5]:
                        gemmr = GEMMR('cca', wx=px, wy=py, ax=ax, ay=ay,
                                      r_between=r)
                        _test_jcov_from_same_jcov(gemmr,
                                                  JointCovarianceModelCCA)
                        _test_jcov_from_other_jcov(gemmr, pls,
                                                   JointCovarianceModelPLS)

                        gemmr = GEMMR('pls', wx=px, wy=py, ax=ax, ay=ay,
                                      r_between=r, expl_var_ratio_thr=.4)
                        _test_jcov_from_same_jcov(gemmr,
                                                  JointCovarianceModelPLS)
                        _test_jcov_from_other_jcov(gemmr, cca,
                                                   JointCovarianceModelCCA)


def test__mk_Sigmaxy():
    pass  # Nothing to test?


def test__variance_explained_by_latent_modes():
    pass  # Nothing to test?


def test_generate_data():
    px, py = 3, 2
    n = 4
    Sigma = np.ones((px+py, px+py))
    X, Y = generate_data(Sigma, px, n, random_state=42)
    assert len(X) == len(Y)
    assert X.shape[1] == px
    assert Y.shape[1] == py


def _test_generated_data_consistency_with_model(model):
    estr = dict(cca=SVDCCA(), pls=SVDPLS())[model]
    for px in [2, 4, 32]:
        py = px + 2
        for r_between in [.7, .5, .3, .2]:
            for ax in [0, -.5, -1]:
                gm = GEMMR(model, wx=px, wy=py, r_between=r_between,
                           ax=ax, ay=ax, random_state=0)
                n_per_ftr = 1024
                for random_state in range(2):
                    X, Y = gm.generate_data(n=(px + py) * n_per_ftr,
                                            random_state=random_state)
                    estr.fit(X, Y)
                    assert_allclose(estr.corrs_[0], r_between,
                                    rtol=1e-2, atol=0.05)


def test_generated_data_consistency_with_model():
    for model in ['cca', 'pls']:
        _test_generated_data_consistency_with_model(model)
