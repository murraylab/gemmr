import numpy as np
from scipy.stats import pearsonr

from numpy.testing import assert_warns, assert_raises, assert_array_almost_equal, assert_allclose

try:
    from gemmr.estimators import SparsePLS
    from gemmr.estimators.r_estimators import _check_penalty, _check_penalty_cv, _fit_and_score_spls, \
        _select_best_penalties
except ImportError:
    pass
else:  # run tests only if rpy2 is available and SparsePLS can be imported

    def test_SparsePLS():

        np.random.seed(0)

        scca = SparsePLS(n_components=1)
        assert_raises(ValueError, scca.fit, np.arange(20).reshape(10, 2), np.arange(30).reshape(10, 3))
        assert_raises(ValueError, scca.fit, np.arange(30).reshape(10, 3), np.arange(10).reshape(10, 1))

        assert_raises(ValueError, scca.fit, np.arange(30).reshape(10, 3), np.arange(10))

        scca = SparsePLS(n_components=20)
        assert_raises(ValueError, scca.fit, np.arange(30).reshape(10, 3), np.arange(30).reshape(10, 3))

        scca = SparsePLS(n_components=2)
        assert_warns(UserWarning, scca.fit, np.random.uniform(size=(10, 3)), np.random.uniform(size=(10, 2)))

        ###

        n = 10000
        corr_signal = np.cos(np.arange(n)).reshape(-1, 1)
        X = np.c_[corr_signal, corr_signal, corr_signal, np.random.normal(size=(n, 3))]
        Y = np.c_[np.random.normal(size=(n, 1)), corr_signal, corr_signal, corr_signal, np.random.normal(size=(n, 1))]

        scca = SparsePLS(n_components=1, scale=False, optimize_penalties='NOT_AN_OPTION')
        assert_raises(ValueError, scca.fit, X, Y)

        # maximum penalty
        scca = SparsePLS(n_components=1, scale=False, optimize_penalties=False, penaltyxs=0, penaltyys=0)
        _test_SparsePLS_fit_results(scca, X, Y)
        assert_allclose(scca.x_rotations_[[3, 4, 5]], 0)
        assert_allclose(scca.y_rotations_[[0, 4]], 0)
        max_penalties_covs = scca.covs_

        # no penalty
        scca.set_params(penaltyxs=1, penaltyys=1)
        _test_SparsePLS_fit_results(scca, X, Y)
        assert np.all(scca.x_rotations_[[3, 4, 5]] != 0)
        assert np.all(scca.y_rotations_[[0, 4]] != 0)
        no_penalties_covs = scca.covs_
        assert max_penalties_covs[0] <= no_penalties_covs[0]

        # cv
        scca = SparsePLS(n_components=1, scale=False, optimize_penalties='cv', penaltyxs=[0, 1], penaltyys=[0, 1], verbose=True)
        scca.fit(X, Y)


    def _test_SparsePLS_fit_results(scca, X, Y):

        scca.fit(X, Y)

        Xt, Yt = scca.transform(X, Y)
        assert_array_almost_equal(Xt, scca.x_scores_)
        assert_array_almost_equal(Yt, scca.y_scores_)
        assert_allclose(scca.covs_, scca.assocs_)
        assert_allclose([pearsonr(Xt[:, mode], Yt[:, mode])[0] for mode in range(scca.n_components_)], scca.corrs_)
        assert_allclose([np.cov(Xt[:, mode], Yt[:, mode])[0, 1] for mode in range(scca.n_components_)], scca.covs_)

        Xt2, Yt2 = scca.fit_transform(X, Y)
        assert_array_almost_equal(Xt2, Xt)
        assert_array_almost_equal(Yt2, Yt)

        print('covs', np.cov(Xt[:, 0], Yt[:, 0])[0,1])

        print(scca.corrs_)
        print(scca.x_rotations_)
        print(scca.y_rotations_)


    def test__check_penalty():
        assert _check_penalty(None) == 1
        assert _check_penalty([1./2]) == 1./2
        assert_raises(ValueError, _check_penalty, [1, 2])
        assert _check_penalty(1./4) == 1./4
        assert_raises(ValueError, _check_penalty, -.1)
        assert_raises(ValueError, _check_penalty, [1.1])


    def test__check_penalty_cv():
        penalties = _check_penalty_cv(None)
        assert np.all(penalties >= 0) & np.all(penalties <= 1)
        assert _check_penalty_cv(1./2) == [1./2]
        assert _check_penalty_cv([1./4, 1./8]) == [1./4, 1./8]
        assert_raises(ValueError, _check_penalty_cv, [1./2, -.1])
        assert_raises(ValueError, _check_penalty_cv, [1./4, 1.1])


    def test__fit_and_score_scca():
        def fitfun(X, Y, penaltyx, penaltyy):
            class MockRres:
                def rx2(self, which):
                    if which == 'u':
                        return np.eye(2)[:, [0]]
                    elif which == 'v':
                        return np.eye(2)[:, [1]]
                    else:
                        raise ValueError()
            return MockRres()
        np.random.seed(0)
        n = 10000
        data_vec = np.cos(np.arange(n)).reshape(-1, 1)
        noise_vec = np.random.normal(size=(n, 1))
        X = np.c_[data_vec, noise_vec]
        Y = np.c_[noise_vec, data_vec]
        score = _fit_and_score_spls(fitfun, X, Y, slice(None), slice(None), penalties=(None, None))
        assert np.isclose(score, 1)


    def test__select_best_penalties():
        penalty_candidates = np.array([
            [.3, .1],
            [.1, .3],
            [.5, .5]
        ])
        assert np.all(_select_best_penalties(penalty_candidates) ==[.3, .1])


    # def test_SparsePLS_parallel():
    #     np.random.seed(0)
    #
    #     n = 10000
    #     corr_signal = np.cos(np.arange(n)).reshape(-1, 1)
    #     X = np.c_[corr_signal, corr_signal, corr_signal, np.random.normal(size=(n, 3))]
    #     Y = np.c_[np.random.normal(size=(n, 1)), corr_signal, corr_signal, corr_signal, np.random.normal(size=(n, 1))]
    #
    #     # maximum penalty
    #     scca = SparsePLS(n_components=1, scale=False, optimize_penalties='cv', penaltyxs=[0, 1], penaltyys=[0, 1],
    #                      n_jobs=-1, verbose=True)
    #     scca.fit(X, Y)
