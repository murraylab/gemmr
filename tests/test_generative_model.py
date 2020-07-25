import numpy as np

from numpy.testing import assert_raises, assert_allclose, assert_equal, assert_warns, assert_array_almost_equal

from unittest.mock import patch, create_autospec

from sklearn.utils import check_random_state

import gemmr.generative_model
from gemmr.generative_model import *
from gemmr.generative_model import _check_subspace_dimension, _mk_Sigmaxy, \
    _find_latent_mode_vectors_pc1, _find_latent_mode_vectors_opti, _find_latent_mode_vectors_qr, \
    _find_latent_mode_vectors, \
    _generate_random_dominant_subspace_rotations, _generate_dominant_subspace_rotations_from_opti, \
    _add_lowvariance_subspace_components, _add_lowvariance_subspace_component_1dim, _Sigmaxy_negative_min_eval, \
    _assemble_Sigmaxy_pls, _assemble_Sigmaxy_cca, calc_schur_complement


def test_setup_model():

    assert_raises(ValueError, setup_model, 'cca', max_n_sigma_trials=0)
    assert_raises(ValueError, setup_model, 'cca',  c1x=2)
    assert_raises(ValueError, setup_model, 'cca',  c1y=.5)
    assert_raises(ValueError, setup_model, 'BLA')
    assert_raises(ValueError, setup_model, 'cca', a_between=.2)
    assert_raises(ValueError, setup_model, 'pls', ax=2)
    assert_raises(ValueError, setup_model, 'cca', ay=.2)
    assert_raises(ValueError, setup_model, 'cca', r_between=-.1)
    assert_raises(ValueError, setup_model, 'pls', r_between=1.1)
    assert_warns(UserWarning, setup_model, 'cca', a_between=0)
    assert_raises(ValueError, setup_model, 'cca',  px=3, cx=[1,2])
    assert_raises(ValueError, setup_model, 'cca',  py=2, cy=[1,2,3])
    assert_raises(ValueError, setup_model, 'cca',  px=4, py=4, qx=2, qy=3, m=3)

    res = setup_model('cca', px=2, py=2, return_full=True)
    assert len(res) == 12
    res = setup_model('pls', px=2, py=2, return_full=True)
    assert len(res) == 12

    r_between = 0.3
    Sigma = setup_model('cca', px=2, py=2, ax=0, ay=0, r_between=r_between, m=1, return_full=False)
    SigmaXY = Sigma[:2, 2:]
    r_hat = np.linalg.svd(SigmaXY, full_matrices=False, compute_uv=False)
    assert_array_almost_equal(r_hat, [r_between, 0])


def test__check_subspace_dimension():
    Sigmaxx = np.diag([3, 2, 1])
    px = Sigmaxx.shape[1]
    assert _check_subspace_dimension(Sigmaxx, None) == px
    assert _check_subspace_dimension(Sigmaxx, 'all') == px
    assert _check_subspace_dimension(Sigmaxx, 'force_1') == 1
    assert_raises(ValueError, _check_subspace_dimension, Sigmaxx, '2')
    assert_raises(ValueError, _check_subspace_dimension, Sigmaxx, str(px+1))
    assert_raises(ValueError, _check_subspace_dimension, Sigmaxx, 'a;lkj')
    assert_raises(ValueError, _check_subspace_dimension, Sigmaxx, px + .1)
    assert_raises(ValueError, _check_subspace_dimension, Sigmaxx, px + 1)
    assert_raises(ValueError, _check_subspace_dimension, Sigmaxx, -.1)
    assert _check_subspace_dimension(Sigmaxx, 2./3) == int(2./3 * px)
    assert _check_subspace_dimension(Sigmaxx, 1./px) == 2
    assert_raises(ValueError, _check_subspace_dimension, Sigmaxx, 1)
    assert _check_subspace_dimension(Sigmaxx, 2) == 2
    assert _check_subspace_dimension(Sigmaxx, 2.0) == 2
    assert_raises(ValueError, _check_subspace_dimension, Sigmaxx, 1.1)


def test__mk_Sigmaxy():
    pass  # Nothing to test?


mocked__add_lowvariance_subspace_components = create_autospec(
    gemmr.generative_model._add_lowvariance_subspace_components,
    return_value=(np.eye(4)[:, [0]], np.eye(5)[:, [0]])
)


@patch('gemmr.generative_model._add_lowvariance_subspace_components', side_effect=mocked__add_lowvariance_subspace_components)
def test__find_latent_mode_vectors_pc1(mock__add_lowvariance_subspace_components):
    px, py = 4, 5
    qx, qy = 4, 3
    m = 1
    Sigmaxx = np.eye(px)
    Sigmayy = np.eye(py)
    U = np.eye(px)
    V = np.eye(py)
    assemble_Sigmaxy = _assemble_Sigmaxy_pls
    expl_var_ratio_thr = 1./2
    max_n_sigma_trials = 1
    rng = check_random_state(0)
    true_corrs = np.array([1./2,])

    Sigmaxy, Sigmaxy_svals, U_, V_, latent_expl_var_ratios_x, latent_expl_var_ratios_y, min_eval, true_corrs, latent_mode_vector_algo = \
            _find_latent_mode_vectors_pc1(Sigmaxx, Sigmayy, U, V, assemble_Sigmaxy, expl_var_ratio_thr, m, max_n_sigma_trials, qx, qy, rng, true_corrs, verbose=True)
    assert mock__add_lowvariance_subspace_components.call_count > 0
    assert_array_almost_equal(U_[:, 0], U[:, 0])
    assert_array_almost_equal(V_[:, 0], V[:, 0])

    m = 2
    true_corrs = np.array([1./2, 1./4])
    assert_raises(ValueError, _find_latent_mode_vectors_pc1, Sigmaxx, Sigmayy, U, V, assemble_Sigmaxy,
                  expl_var_ratio_thr, m, max_n_sigma_trials, qx, qy, rng, true_corrs, verbose=True)


def test__find_latent_mode_vectors_opti():
    px, py = 4, 5
    qx, qy = 4, 3
    m = 1
    Sigmaxx = np.eye(px)
    Sigmayy = np.eye(py)
    U = np.eye(px)
    V = np.eye(py)
    assemble_Sigmaxy = _assemble_Sigmaxy_pls
    expl_var_ratio_thr = 1. / 2
    max_n_sigma_trials = 1
    rng = check_random_state(0)
    true_corrs = np.array([1. / 2, ])

    assert_raises(NotImplementedError, _find_latent_mode_vectors_opti, Sigmaxx, Sigmayy, U, V, assemble_Sigmaxy, expl_var_ratio_thr, m=2,
                                      max_n_sigma_trials=max_n_sigma_trials, qx=qx, qy=qy, rng=rng, true_corrs=true_corrs, verbose=True)

    Sigmaxy, Sigmaxy_svals, U_, V_, latent_expl_var_ratios_x, latent_expl_var_ratios_y, min_eval, true_corrs, latent_mode_vector_algo = \
        _find_latent_mode_vectors_opti(Sigmaxx, Sigmayy, U, V, assemble_Sigmaxy, expl_var_ratio_thr, m,
                                      max_n_sigma_trials, qx, qy, rng, true_corrs, verbose=True)
    assert U_.shape == (px, m)
    assert V_.shape == (py, m)


def test__find_latent_mode_vectors_qr():
    pass  # nothing to test?


def test__find_latent_mode_vectors():
     pass  # nothing to test?


def test__generate_random_dominant_subspace_rotations():
    px, py = 4, 5
    qx, qy = 4, 3
    m = 2
    U = np.eye(px)
    V = np.eye(py)
    rng = check_random_state(0)
    uvrots = None
    U_dominant, V_dominant = _generate_random_dominant_subspace_rotations(U, V, m, qx, qy, rng, uvrots)
    assert U_dominant.shape == (px, m)
    assert V_dominant.shape == (py, m)


def test__generate_dominant_subspace_rotations_from_opti():
    px, py = 4, 5
    qx, qy = 4, 3
    m = 2
    U = np.eye(px)
    V = np.eye(py)
    rng = check_random_state(0)
    uvrots = [(None, np.arange(qx), np.arange(qy))]
    U_dominant, V_dominant = _generate_dominant_subspace_rotations_from_opti(U, V, m, qx, qy, rng, uvrots)

    true_U_dominant = np.dot(U[:, :qx], uvrots[0][1].reshape(-1, 1))[:, :m]
    true_V_dominant = np.dot(V[:, :qy], uvrots[0][2].reshape(-1, 1))[:, :m]

    assert_array_almost_equal(U_dominant, true_U_dominant)
    assert_array_almost_equal(V_dominant, true_V_dominant)


def test__add_lowvariance_subspace_components():
    px, py = 4, 5
    qx, qy = 4, 3
    m = 2
    U = np.eye(px)
    V = np.eye(py)
    U_dominant = U[:, :m]
    V_dominant = V[:, :m]
    rng = check_random_state(0)
    min_weight = 0.5

    assert_raises(AssertionError, _add_lowvariance_subspace_components, U, U_dominant, V, V_dominant, m, qx, qy, rng, -.1)
    assert_raises(AssertionError, _add_lowvariance_subspace_components, U, U_dominant, V, V_dominant, m, qx, qy, rng, 1.01)

    U_, V_ = _add_lowvariance_subspace_components(U, U_dominant, V, V_dominant, m, qx, qy, rng, min_weight)
    assert U_.shape == U_dominant.shape
    assert V_.shape == V_dominant.shape


def test__add_lowvariance_subspace_component_1dim():

    px, py = 2, 3
    qx, qy = 2, 2
    m = 1
    U = np.eye(px)
    V = np.eye(py)
    U_dominant = U[:, :m]
    V_dominant = V[:, :m]
    rng = check_random_state(0)
    min_weight = 0.5

    # qx == px
    U_ = _add_lowvariance_subspace_component_1dim(U, U_dominant, m, min_weight, qx, rng)
    assert np.allclose(U_, U_dominant)

    # qy != py
    V_ = _add_lowvariance_subspace_component_1dim(V, V_dominant, m, min_weight, qy, rng)
    assert V_.shape == V_dominant.shape
    assert not np.allclose(V_, V_dominant)
    assert np.allclose(np.linalg.norm(V_, axis=0), 1)

    ### m = 2
    px, py = 2, 3
    qx, qy = 2, 2
    m = 2
    U = np.eye(px)
    V = np.eye(py)
    U_dominant = U[:, :m]
    V_dominant = V[:, :m]
    rng = check_random_state(0)
    min_weight = 0.5

    # qx == px
    U_ = _add_lowvariance_subspace_component_1dim(U, U_dominant, m, min_weight, qx, rng)
    assert np.allclose(U_, U_dominant)

    # qy != py
    V_ = _add_lowvariance_subspace_component_1dim(V, V_dominant, m, min_weight, qy, rng)
    assert V_.shape == V_dominant.shape
    assert not np.allclose(V_, V_dominant)
    assert np.allclose(np.linalg.norm(V_, axis=0), 1)


def test__variance_explained_by_latent_modes():
    pass  # Nothing to test?


def test__Sigmaxy_negative_min_eval():
    m = 1
    global_true_corrs = np.array([1./2,])
    px, py = 2, 3
    global_Sigmaxx = np.eye(px)
    global_Sigmayy = np.eye(py)
    U = np.eye(px)
    V = np.eye(py)

    qx, qy = px, py
    urot = np.ones(px)/np.sqrt(qx)
    vrot = np.ones(py)/np.sqrt(qy)
    uvrot = np.hstack([urot, vrot])

    def assemble_Sigmaxy_test(Sigmaxx, Sigmayy, U_, V_, m, true_corrs):
        assert_array_almost_equal(Sigmaxx, global_Sigmaxx)
        assert_array_almost_equal(Sigmayy, global_Sigmayy)
        assert_allclose(true_corrs, global_true_corrs)
        assert_array_almost_equal(U_, np.dot(U, urot).reshape(-1, 1))
        assert_array_almost_equal(V_, np.dot(V, vrot).reshape(-1, 1))
        return 1, 2, 3, 4, 5, 6

    uvrots = []

    assert_raises(AssertionError, _Sigmaxy_negative_min_eval, uvrot, assemble_Sigmaxy_test, global_Sigmaxx, global_Sigmayy, U, V, m=2, qx=px, qy=py,
        rng=None, true_corrs=global_true_corrs, uvrots=uvrots, min_eval_thr=1e-5)
    assert_raises(AssertionError, _Sigmaxy_negative_min_eval, uvrot, assemble_Sigmaxy_test, global_Sigmaxx,
                  global_Sigmayy, U, V, m=1, qx=px, qy=py,
                  rng=None, true_corrs=global_true_corrs, uvrots=uvrots, min_eval_thr=-1e-5)
    assert_raises(AssertionError, _Sigmaxy_negative_min_eval, uvrot[:px], assemble_Sigmaxy_test, global_Sigmaxx,
                  global_Sigmayy, U, V, m=1, qx=px, qy=py,
                  rng=None, true_corrs=global_true_corrs, uvrots=uvrots, min_eval_thr=1e-5)
    assert_raises(AssertionError, _Sigmaxy_negative_min_eval, np.r_[uvrot, uvrot], assemble_Sigmaxy_test, global_Sigmaxx,
                  global_Sigmayy, U, V, m=1, qx=px, qy=py,
                  rng=None, true_corrs=global_true_corrs, uvrots=uvrots, min_eval_thr=1e-5)

    _Sigmaxy_negative_min_eval(uvrot, assemble_Sigmaxy_test, global_Sigmaxx, global_Sigmayy, U, V, m, qx, qy,
        rng=None, true_corrs=global_true_corrs, uvrots=uvrots, min_eval_thr=1e-5)
    assert uvrots[0][0] == 5
    assert_array_almost_equal(uvrots[0][1], urot)
    assert_array_almost_equal(uvrots[0][2], vrot)

    uvrots = []
    _Sigmaxy_negative_min_eval(uvrot, assemble_Sigmaxy_test, global_Sigmaxx, global_Sigmayy, U, V, m, qx, qy,
                               rng=None, true_corrs=global_true_corrs, uvrots=uvrots, min_eval_thr=5 + .1)
    assert len(uvrots) == 0



def test_assemble_Sigmaxy_pls():
    px, py = 2, 3
    Sigmaxx = np.eye(px)
    Sigmayy = np.eye(py)
    U_ = np.eye(px)
    V_ = np.eye(py)
    true_corrs = np.array([1./4, 1./2,])
    m = 2

    Sigmaxy, Sigmaxy_svals, U_out, V_out, min_eval, true_corrs_out = _assemble_Sigmaxy_pls(Sigmaxx, Sigmayy, U_, V_, m, true_corrs)
    assert Sigmaxy.shape == (px, py)
    assert np.all(Sigmaxy_svals == true_corrs_out)

    order = np.argsort(true_corrs)[::-1]
    assert np.all(true_corrs[order] == true_corrs_out)
    assert_array_almost_equal(U_out, U_[:, order])
    assert_array_almost_equal(V_out, V_[:, order])
    assert min_eval > 0

    np.random.seed(0)
    U_ = np.linalg.qr(np.random.normal(size=(px, px)))[0]
    V_ = np.linalg.qr(np.random.normal(size=(py, py)))[0][:, :px]

    Sigmaxy, Sigmaxy_svals, U_out, V_out, min_eval, true_corrs_out = _assemble_Sigmaxy_pls(Sigmaxx, Sigmayy, U_, V_, m, true_corrs)
    assert Sigmaxy.shape == (px, py)
    assert np.all(Sigmaxy_svals != true_corrs_out)

    order = np.argsort(true_corrs)[::-1]
    assert np.all(true_corrs[order] == true_corrs_out)
    assert_array_almost_equal(U_out, U_[:, order])
    assert_array_almost_equal(V_out, V_[:, order])
    assert min_eval > 0


def test__assemble_Sigmaxy_cca():
    px, py = 2, 3
    Sigmaxx = np.eye(px)
    Sigmayy = np.eye(py)
    U_ = np.eye(px)
    V_ = np.eye(py)[:, :2]
    true_corrs = np.array([1./4, 1./2,])
    m = 2
    Sigmaxy, Sigmaxy_svals, U_out, V_out, min_eval, true_corrs_out = _assemble_Sigmaxy_cca(Sigmaxx, Sigmayy, U_, V_, m, true_corrs)
    assert Sigmaxy.shape == (px, py)
    assert np.all(true_corrs == true_corrs_out)
    assert np.all(Sigmaxy_svals == true_corrs_out)
    assert_array_almost_equal(U_out, U_)
    assert_array_almost_equal(V_out, V_)
    assert min_eval > 0

    Sigmaxx = np.diag(np.arange(1, 3)[::-1])
    Sigmayy = np.diag(np.arange(1, 4)[::-1])
    assert_raises(NotImplementedError, _assemble_Sigmaxy_cca, Sigmaxx, Sigmayy, U_, V_, m, true_corrs)

    U_ = U_[:, :1]
    V_ = V_[:, :1]
    true_corrs = true_corrs[:1]
    m = 1
    Sigmaxy, Sigmaxy_svals, U_out, V_out, min_eval, true_corrs_out = _assemble_Sigmaxy_cca(Sigmaxx, Sigmayy, U_, V_, m, true_corrs)
    assert Sigmaxy.shape == (px, py)
    assert np.all(true_corrs == true_corrs_out)
    assert np.all(Sigmaxy_svals == true_corrs_out)
    assert_array_almost_equal(U_out, U_)
    assert_array_almost_equal(V_out, V_)
    assert min_eval > 0


def test_calc_schur_complement():
    A = np.eye(3)
    B = np.zeros((3, 2))
    B[0, 0] = B[1, 1] = 1
    C = B.T
    D = np.eye(2)
    M = np.vstack([
        np.hstack([A, B]),
        np.hstack([C, D])
    ])
    sc1 = calc_schur_complement(A, B, C, D, kind='A')
    sc2 = calc_schur_complement(M, A.shape[1], kind='A')
    sc_true = A - B.dot(np.linalg.inv(D)).dot(C)
    assert_allclose(sc1, sc2)
    assert_allclose(sc1, sc_true)
    assert_raises(ValueError, calc_schur_complement, A, 1, kind='WRONG_KIND')
    assert_raises(ValueError, calc_schur_complement, A, 'no_int')


def test_generate_data():
    px, py = 3, 2
    n = 4
    Sigma = np.ones((px+py, px+py))
    X, Y = generate_data(Sigma, px, n, random_state=42)
    assert len(X) == len(Y)
    assert X.shape[1] == px
    assert Y.shape[1] == py