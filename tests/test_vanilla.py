import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose, assert_raises

from scipy.stats import pearsonr

from sklearn.base import clone
from sklearn.utils import check_random_state

from gemmr.generative_model import GEMMR
from gemmr.estimators import SVDPLS, SVDCCA, NIPALSPLS, NIPALSCCA, SingularMatrixError
from gemmr.estimators.vanilla import _S_invsqrt
from gemmr.estimators.helpers import cov_transform_scorer, pearson_transform_scorer
from testtools import assert_array_almost_equal_up_to_sign


def _test_estrs_equal(model, estr1, estr2, n=1000):

    rng = check_random_state(42)

    n_components = min(estr1.n_components, estr2.n_components)
    assert(n_components == 1)

    for px, py, ax, ay, r_latent in [
        (1, 3, -.25, -.3, .3),
        (3, 1, -.15, -.23, .3),
        (16, 3, -1.5, -.5, .3),
        (4, 9, -.1, -.2, .9),
    ]:

        print('pxy', px, py)

        gm = GEMMR(model, random_state=rng, wx=px, wy=py, ax=ax, ay=ay,
                   r_between=r_latent)
        X, Y = gm.generate_data(n, random_state=rng)
        Xtest, Ytest = gm.generate_data(n, random_state=rng)

        if py == 1:
            Y = Y[:, 0]
            Ytest = Ytest[:, 0]

        estr1.fit(X, Y)
        if (Y.ndim == 1):
            assert estr1.y_rotations_.ndim == 1
            assert estr1.y_scores_.ndim == 1
            
        estr2.fit(X, Y)
        if (Y.ndim == 1):
            assert estr2.y_rotations_.ndim == 1
            assert estr2.y_scores_.ndim == 1

        # check corrs
        assert_allclose(estr1.assocs_[:n_components],
                                  estr2.assocs_[:n_components],
                                  err_msg="corrs failed: {} vs {}".format(estr1, estr2))

        # check rotations
        assert_array_almost_equal_up_to_sign(estr1.x_rotations_,
                                             estr2.x_rotations_,
                                             err_msg="X rotations failed: {} vs {}".format(estr1, estr2))
        assert_array_almost_equal_up_to_sign(estr1.y_rotations_,
                                             estr2.y_rotations_,
                                             err_msg="Y rotations failed: {} vs {}".format(estr1, estr2))

        # check scores
        assert_array_almost_equal_up_to_sign(estr1.x_scores_,
                                             estr2.x_scores_,
                                             err_msg="X scores failed: {} vs {}".format(estr1, estr2))
        assert_array_almost_equal_up_to_sign(estr1.y_scores_,
                                             estr2.y_scores_,
                                             err_msg="Y scores failed: {} vs {}".format(estr1, estr2))

        XtestT1, YtestT1 = estr1.transform(Xtest, Ytest)
        XtestT2, YtestT2 = estr2.transform(Xtest, Ytest)

        # check transformed scores
        assert_array_almost_equal_up_to_sign(XtestT1,
                                             XtestT2,
                                             err_msg="transformed X scores failed: {} vs {}".format(estr1, estr2))
        assert_array_almost_equal_up_to_sign(YtestT1,
                                             YtestT2,
                                             err_msg="transformed Y scores failed: {} vs {}".format(estr1, estr2))

        _test_fit_transform(X, Y, estr1)
        _test_fit_transform(X, Y, estr2)


def _test_fit_transform(X, Y, estr):
    new_estr = clone(estr)
    Xt, Yt = new_estr.fit_transform(X, Y)
    # check assocs
    assert_allclose(estr.assocs_, new_estr.assocs_)
    # check rotations
    assert_array_almost_equal_up_to_sign(estr.x_rotations_,
                                         new_estr.x_rotations_,
                                         err_msg="X rotations failed: {} vs {}".format(estr, new_estr))
    assert_array_almost_equal_up_to_sign(estr.y_rotations_,
                                         new_estr.y_rotations_,
                                         err_msg="Y rotations failed: {} vs {}".format(estr, new_estr))
    # check scores
    assert_array_almost_equal_up_to_sign(estr.x_scores_,
                                         new_estr.x_scores_,
                                         err_msg="X scores failed: {} vs {}".format(estr, new_estr))
    assert_array_almost_equal_up_to_sign(estr.y_scores_,
                                         new_estr.y_scores_,
                                         err_msg="Y scores failed: {} vs {}".format(estr, new_estr))


def test_svdpls_nipalspls():
    svdpls = SVDPLS(n_components=1, scale=False, std_ddof=1)
    nipalspls = NIPALSPLS(n_components=1, scale=False, tol=1e-12)
    _test_estrs_equal('pls', svdpls, nipalspls)

    svdpls.set_params(covariance='NOT_A_COVARIANCE')
    assert_raises(ValueError, svdpls.fit, np.arange(6).reshape(3,2), np.arange(6).reshape(3,2))


def test_svdcca_nipalscca():
    svdcca = SVDCCA(n_components=1, scale=False, std_ddof=1, normalize_weights=True, cov_out_of_bounds='nan')
    nipalscca = NIPALSCCA(n_components=1, scale=False, tol=1e-12)
    _test_estrs_equal('cca', svdcca, nipalscca)

    svdcca.set_params(covariance='NOT_A_COVARIANCE')
    X = np.arange(6).reshape(3,2)
    assert_raises(ValueError, svdcca.fit, X, X)
    svdcca.set_params(covariance='empirical')

    svdcca.X_whitener_ = svdcca.Y_whitener_ = np.eye(2)

    svdcca.set_params(cov_out_of_bounds='raise')
    assert_raises(ValueError, svdcca._postprocess, X, X, np.eye(2), np.eye(2), np.arange(2,4))

    svdcca.set_params(cov_out_of_bounds='ignore')
    U, V, s = svdcca._postprocess(X, X, np.eye(2), np.eye(2), np.arange(2,4))
    assert np.all(np.isfinite(U))
    assert np.all(np.isfinite(V))
    assert np.all(np.isfinite(s))
    assert np.all(np.isfinite(svdcca.corrs_))

    svdcca.set_params(cov_out_of_bounds='NOT_A_VALUE')
    assert_raises(ValueError, svdcca._postprocess, X, X, np.eye(2), np.eye(2), np.arange(2, 4))

    svdcca.set_params(cov_out_of_bounds='nan')
    U, V, s = svdcca._postprocess(X, X, np.eye(2), np.eye(2), np.arange(2,4))
    assert np.all(np.isnan(U))
    assert np.all(np.isnan(V))
    assert np.all(np.isnan(s))
    assert np.all(np.isnan(svdcca.corrs_))

def test_S_invsqrt():

    m = np.zeros((10, 3))
    assert_raises(SingularMatrixError, _S_invsqrt, m, if_singular='raise')
    assert_raises(AssertionError, _S_invsqrt, m, if_singular='warn')  # assert statement at end of function comes into play
    assert_raises(ValueError, _S_invsqrt, m, if_singular='alwekjr')

    np.random.seed(0)
    U = np.linalg.qr(np.random.normal(size=(10, 10)))[0]
    V = np.linalg.qr(np.random.normal(size=(2, 2)))[0]
    s = np.vstack([
        np.diag(np.array([.9, .2])),
        np.zeros((8, 2))
        ])
    X = U.dot(s).dot(V.T)
    ddof = 0
    S = 1/(len(X)-ddof)*np.dot(X.T, X)

    invsqrt = _S_invsqrt(X, if_singular='raise')
    assert_array_almost_equal(
        invsqrt.dot(1 / (len(X) - ddof) * np.dot(X.T, X)).dot(invsqrt),
        np.eye(X.shape[1])
    )


class MockTransformer():
    def __init__(self, Xt, Yt):
        self.Xt = Xt
        self.Yt = Yt
    def transform(self, X, Y):
        return self.Xt, self.Yt


def test_transform_scorers():
    np.random.seed(0)
    n_ftrs = 2
    Xt = np.random.normal(size=(10, n_ftrs))
    Yt = np.random.normal(size=(10, n_ftrs+1))

    transformer = MockTransformer(Xt, Yt)

    cov_scores = np.array([cov_transform_scorer(transformer, None, None, ftr=i) for i in range(n_ftrs)])
    true_cov_scores = np.array([np.cov(Xt[:, i], Yt[:, i])[0, 1] for i in range(n_ftrs)])
    assert_allclose(cov_scores, true_cov_scores)

    corr_scores = np.array([pearson_transform_scorer(transformer, None, None, ftr=i) for i in range(n_ftrs)])
    true_corr_scores = np.array([pearsonr(Xt[:, i], Yt[:, i])[0] for i in range(n_ftrs)])
    assert_allclose(corr_scores, true_corr_scores)

    # Y is 1d
    n_ftrs = 1
    Xt = np.random.normal(size=(8, 2))
    Yt = np.random.normal(size=(8))

    transformer = MockTransformer(Xt, Yt)

    cov_scores = np.array([cov_transform_scorer(transformer, None, None, ftr=i) for i in range(n_ftrs)])
    true_cov_scores = np.array([np.cov(Xt[:, i], Yt)[0, 1] for i in range(n_ftrs)])
    assert_allclose(cov_scores, true_cov_scores)

    corr_scores = np.array([pearson_transform_scorer(transformer, None, None, ftr=i) for i in range(n_ftrs)])
    true_corr_scores = np.array([pearsonr(Xt[:, i], Yt)[0] for i in range(n_ftrs)])
    assert_allclose(corr_scores, true_corr_scores)


def test_xy_reversed():

    rng = check_random_state(42)

    for px, py, ax, ay, r_latent, n in [
        (1, 3, -.25, -.3, .3, 16),
        (3, 1, -.15, -.23, .3, 32),
        (16, 3, -1.5, -.5, .3, 64),
        (4, 9, -.1, -.2, .9, 128),
    ]:

        for model, estr in [
            ('cca', SVDCCA(n_components=1, scale=False, std_ddof=1,
                           normalize_weights=True, cov_out_of_bounds='nan')),
            ('pls', SVDPLS(n_components=1, scale=False, std_ddof=1)),
        ]:

            gm = GEMMR(model, random_state=rng, wx=px, wy=py, ax=ax,
                       ay=ay, r_between=r_latent)
            X, Y = gm.generate_data(n, random_state=rng)

            estr1 = clone(estr).fit(X, Y)
            estr2 = clone(estr).fit(Y, X)

            n_components = min(estr1.n_components, estr2.n_components)
            assert (n_components == 1)

            # check corrs
            assert_allclose(estr1.assocs_[:n_components],
                            estr2.assocs_[:n_components],
                            err_msg="corrs failed: {} vs {}".format(estr1, estr2))

            # check rotations
            assert_array_almost_equal_up_to_sign(estr1.x_rotations_,
                                                 estr2.y_rotations_,
                                                 err_msg="X / Y rotations failed: {} vs {}".format(
                                                     estr1, estr2))
            assert_array_almost_equal_up_to_sign(estr1.y_rotations_,
                                                 estr2.x_rotations_,
                                                 err_msg="Y / X rotations failed: {} vs {}".format(
                                                     estr1, estr2))

            # check scores
            assert_array_almost_equal_up_to_sign(estr1.x_scores_,
                                                 estr2.y_scores_,
                                                 err_msg="X / Y scores failed: {} vs {}".format(
                                                     estr1, estr2))
            assert_array_almost_equal_up_to_sign(estr1.y_scores_,
                                                 estr2.x_scores_,
                                                 err_msg="Y / X scores failed: {} vs {}".format(
                                                     estr1, estr2))
