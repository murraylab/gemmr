"""Setup of model covariance matrices and generation of data.
"""

import numbers
import warnings
from functools import partial

import numpy as np
import scipy.spatial
from sklearn.utils import check_random_state


from ..util import check_positive_definite, pc_spectrum_decay_constant
from ..model_selection import n_components_to_explain_variance
from .other import sympsd_inv_sqrt

__all__ = ['JointCovarianceModel',
           'JointCovarianceModelCCA', 'JointCovarianceModelPLS',
           'JointCovarianceBlocks', 'JointCovarianceDataGenerator', 'GEMMR',
           'CCAgm', 'PLSgm', 'generate_data']


class SubspaceDim1To2Warning(Warning):
    pass


warnings.simplefilter('always', SubspaceDim1To2Warning)


class JointCovarianceModel:

    def __init__(self, estr, X, Y, m=1, random_state=0):

        if (X.ndim != 2) or (Y.ndim != 2):
            raise ValueError('X and Y must be 2-dimensional')

        self.px = X.shape[1]
        self.py = Y.shape[1]
        self.random_state = random_state

        if m is None:
            m = min(px, py)
        self.m = m

        XY = np.hstack([X, Y])
        self.Sigma_ = np.dot(XY.T, XY) / len(XY - 1)

        self.estr = estr
        self.estr.fit(X, Y)
        self.U_latent_ = estr.x_rotations_[:, :m]
        self.V_latent_ = estr.y_rotations_[:, :m]
        self.true_corrs_ = estr.corrs_[:m]

        # --- add some additonal attributes for compatibility with GEMMR ---
        Sigmaxx = self.Sigma_[:self.px, :self.px]
        Sigmayy = self.Sigma_[self.px:, self.px:]
        Sigmaxy = self.Sigma_[:self.px, self.py:]
        self.Sigmaxy_svals_ = np.linalg.svd(Sigmaxy, compute_uv=False)[:m]

        self.ax = pc_spectrum_decay_constant(
            pc_spectrum=np.linalg.eigvalsh(Sigmaxx)[::-1],
            expl_var_ratios=(.99,)
        )[0]
        self.ay = pc_spectrum_decay_constant(
            pc_spectrum=np.linalg.eigvalsh(Sigmayy)[::-1],
            expl_var_ratios=(.99,)
        )[0]

        expl_var_x, tot_var_x = _calc_explained_variance(Sigmaxx,
                                                         self.U_latent_)
        self.latent_expl_var_ratios_x_ = (expl_var_x / tot_var_x)[:m]

        expl_var_y, tot_var_y = _calc_explained_variance(Sigmayy,
                                                         self.V_latent_)
        self.latent_expl_var_ratios_y_ = (expl_var_y / tot_var_y)[:m]

        self.latent_mode_vector_algo_ = 'data'

    @classmethod
    def from_gemmr(cls, estr, gemmr, n_per_ftr='auto', m=1, random_state=None):
        if n_per_ftr == 'auto':
            n_per_ftr = 4096
        else:
            n_per_ftr = int(n_per_ftr)
        n = (gemmr.px + gemmr.py) * n_per_ftr
        X, Y = gemmr.generate_data(n, random_state=random_state)
        return cls(estr, X, Y, m=m, random_state=random_state)

    @classmethod
    def from_gemmr_factory(cls, estr):
        return partial(cls.from_gemmr, estr)

    def generate_data(self, n, random_state=None):
        if random_state is None:
            random_state = self.random_state
        return generate_data(self.Sigma_, self.px, n, random_state)


def _triu(M, k=1):
    """extract upper diagonal from FC matrix
    """

    if M.ndim != 2:
        raise ValueError('2-dim array required, got {} dims'.format(M.ndim))

    res = M[np.triu_indices_from(M, k=k)]
    assert len(res) > 0
    return res


def is_diagonal(M):
    if np.allclose(_triu(M), 0) and np.allclose(_triu(M.T), 0):
        return True
    else:
        return False


class JointCovarianceModelBase:
    def __init__(self, Sigma, px, n_components=1,
                 ax=None, ay=None, random_state=0):

        self.px = px
        self.py = len(Sigma) - px
        self.n_components = n_components
        ## I think that's wrong - why should it only work with m==1?
        # if self.m > 1:
        #     raise ValueError(f"This algorithm only works with m == 1, "
        #                      f"got {self.m} modes.")

        self.random_state = random_state

        self.Sigma_ = Sigma
        Sigmaxx = Sigma[:px, :px]
        Sigmayy = Sigma[px:, px:]
        Sigmaxy = Sigma[:px, px:]

        self._init_latent_modes(Sigmaxx, Sigmaxy, Sigmayy, n_components)

        if ax is None:
            self.ax = pc_spectrum_decay_constant(
                pc_spectrum=np.linalg.eigvalsh(Sigmaxx)[::-1],
                expl_var_ratios=(.99,)
            )[0]
        else:
            self.ax = float(ax)

        if ay is None:
            self.ay = pc_spectrum_decay_constant(
                pc_spectrum=np.linalg.eigvalsh(Sigmayy)[::-1],
                expl_var_ratios=(.99,)
            )[0]
        else:
            self.ay = float(ay)

        # expl_var_x, tot_var_x = _calc_explained_variance(Sigmaxx,
        #                                                  self.U_latent_)
        # self.latent_expl_var_ratios_x_ = (expl_var_x / tot_var_x)[:self.m]
        #
        # expl_var_y, tot_var_y = _calc_explained_variance(Sigmayy,
        #                                                  self.V_latent_)
        # self.latent_expl_var_ratios_y_ = (expl_var_y / tot_var_y)[:self.m]
        #
        # self.latent_mode_vector_algo_ = 'p2c_'

    @classmethod
    def from_jcov_model(cls, jcov, n_components=1, random_state=None):
        return cls(
            jcov.Sigma_, jcov.px, n_components=n_components,
            ax=jcov.ax, ay=jcov.ay,
            random_state=random_state
        )

    def generate_data(self, n, random_state=None, split=True):
        if random_state is None:
            random_state = self.random_state
        return generate_data(self.Sigma_, self.px, n, random_state,
                             split=split)


class JointCovarianceModelCCA(JointCovarianceModelBase):
    def _init_latent_modes(self, Sigmaxx, Sigmaxy, Sigmayy, n_components):
        # WHY DID I REQUIRE SYMMETRIC MATRICES?
        #if not (is_diagonal(Sigmaxx) and is_diagonal(Sigmayy)):
        #    raise ValueError('Sigmaxx and Sigmayy must be diagonal matrices')

        #Sigmaxx_invsqrt = np.diag(1. / np.sqrt(np.diag(Sigmaxx)))
        #Sigmayy_invsqrt = np.diag(1. / np.sqrt(np.diag(Sigmayy)))
        Sigmaxx_invsqrt = sympsd_inv_sqrt(Sigmaxx)
        Sigmayy_invsqrt = sympsd_inv_sqrt(Sigmayy)
        K = Sigmaxx_invsqrt @ Sigmaxy @ Sigmayy_invsqrt
        U, s, Vh = np.linalg.svd(K)

        self.true_assocs_ = self.true_corrs_ = s[:n_components]

        self.x_weights_ = Sigmaxx_invsqrt @ U[:, :n_components]
        self.x_weights_ /= np.linalg.norm(self.x_weights_, axis=0, keepdims=True)

        self.y_weights_ = Sigmayy_invsqrt @ ((Vh.T)[:, :n_components])
        self.y_weights_ /= np.linalg.norm(self.y_weights_, axis=0, keepdims=True)


class JointCovarianceModelPLS(JointCovarianceModelBase):
    def _init_latent_modes(self, Sigmaxx, Sigmaxy, Sigmayy, n_components):
        U, s, Vh = np.linalg.svd(Sigmaxy)
        V = Vh.T
        self.x_weights_ = U[:, :n_components]
        self.true_assocs_ = self.true_covs_ = s[:n_components]
        self.y_weights_ = V[:, :n_components]

        # true corrs are unknown, set attribute anyway for compatibility
        x_vars = np.diag(self.x_weights_.T @ Sigmaxx @ self.x_weights_)
        y_vars = np.diag(self.y_weights_.T @ Sigmayy @ self.y_weights_)
        self.true_corrs_ = self.true_covs_ / np.sqrt(x_vars * y_vars)


class JointCovarianceBlocks:
    @property
    def Sigmaxy_(self):
        return self.Sigma_[:self.px, self.px:]

    @property
    def Sigmaxx_(self):
        return self.Sigma_[:self.px, :self.px]

    @property
    def Sigmayy_(self):
        return self.Sigma_[self.px:, self.px:]


class JointCovarianceDataGenerator:
    def generate_data(self, n, random_state=None):
        if random_state is None:
            random_state = self.random_state
        return generate_data(self.Sigma_, self.px, n, random_state)


class CovNotPositiveDefiniteError(Exception):
    pass


class WeightNotFoundError(Exception):
    pass


def get_random_unit_vector(p, rng):
    x = rng.normal(size=p)
    return x / np.linalg.norm(x)


def is_weight_expl_var_sufficient(w, S, expl_var_ratio_thr):
    expl_var = (w.T @ S @ w)[0, 0]
    # print(expl_var, .5 * np.diag(S).mean())
    if expl_var > expl_var_ratio_thr * np.diag(S).mean():
        return True
    else:
        return False


class _GEMMR:
    def __init__(self, wx, wy, ax=-1., ay=-1., r_between=0.3,
                 max_n_sigma_trials=100000, expl_var_ratio_thr=0.5,
                 coordinate_system='pc', weight_coordinate_system='pc',
                 random_state=0):

        self.ax = float(ax)
        self.ay = float(ay)
        
        if (self.ax > 0) or (self.ay > 0):
            raise ValueError(f"ax and ay must be <= 0 (got {self.ax} and {self.ay})")

        if isinstance(r_between, numbers.Number):
            r_between = np.array([r_between])
        self.r_between = r_between
        if np.any(self.r_between < 0) or np.any(self.r_between > 1):
            raise ValueError(f"r_between must be >= 0 and <= 1, got {self.r_between}")

        self.expl_var_ratio_thr = expl_var_ratio_thr

        self.random_state = random_state
        rng = check_random_state(self.random_state)

        if isinstance(wx, numbers.Integral) and \
                isinstance(wy, numbers.Integral):
            # px, py are the dimensions for X and Y

            self.px = wx
            self.py = wy
            self.n_components = 1
            self._mk_withinset_covs()

            self.Rx, self.Ry = self._check_coordinate_system(coordinate_system,
                                                             rng)

            # generate weight vectors
            for i in range(max_n_sigma_trials):
                wx = get_random_unit_vector(self.px, rng).reshape(-1, 1)
                wy = get_random_unit_vector(self.py, rng).reshape(-1, 1)

                if not (is_weight_expl_var_sufficient(wx, self.Sigmaxx_,
                                                      self.expl_var_ratio_thr) and
                        is_weight_expl_var_sufficient(wy, self.Sigmayy_,
                                                      self.expl_var_ratio_thr)):
                    continue  # try again
                # else: weights explain enough variance

                self._mk_Sxy(wx, wy)

                try:
                    self._mk_S()
                except WeightNotFoundError:
                    continue  # try again
                else:  # success
                    break

            else:
                # no success
                raise WeightNotFoundError()

            # now wx, wy are the weight vectors, store them
            self.x_weights_ = wx
            self.y_weights_ = wy

        elif (wx.ndim == 2) and (wy.ndim == 2):
            # wx, ay are weight vectors

            if wx.shape != wy.shape:
                raise ValueError("wx and wy must have same shape")
            if (wx.shape[1] > 1):  # r_between must have same shape
                if not (self.r_between.shape == (wx.shape[1],)):
                    raise ValueError('r_between must be ndarray with same '
                                     'length as wx.shape[1]')

            self.px = wx.shape[0]
            self.py = wy.shape[0]
            self.n_components = wx.shape[1]
            self._mk_withinset_covs()

            self.Rx, self.Ry = self._check_coordinate_system(coordinate_system,
                                                             rng)

            # potentially normalize weight vectors
            wx = wx / np.linalg.norm(wx, axis=0, keepdims=True)
            wy = wy / np.linalg.norm(wy, axis=0, keepdims=True)

            if weight_coordinate_system == 'pc':
                pass  # use weights as given
            elif weight_coordinate_system == 'data':
                # for constructing Sxy, rotate weights into PC coordinate
                # system
                wx = self.Rx @ wx  # or Rx?
                wy = self.Ry @ wy  # or Ry?
            else:
                raise ValueError(f'Invalid weight_coordinate_system '
                                 f'({weight_coordinate_system}')

            self._mk_Sxy(wx, wy)
            self._mk_S()

            # store weight vectors
            self.x_weights_ = wx
            self.y_weights_ = wy

        else:
            raise ValueError('Invalid wx/wy')

        self._rotate_into_coordinate_system(rng)

    def _check_coordinate_system(self, coordinate_system, rng):
        if coordinate_system == 'pc':
            return np.eye(self.px), np.eye(self.py)
        elif coordinate_system == 'random':
            Rx = np.linalg.qr(rng.normal(size=(self.px, self.px)))[0]
            Ry = np.linalg.qr(rng.normal(size=(self.py, self.py)))[0]
            return Rx, Ry
        elif ((len(coordinate_system) == 2) and
              (coordinate_system[0].shape == (self.px, self.px)) and
              (coordinate_system[1].shape == (self.py, self.py))):
            np.allclose(Qx.T @ Qx, np.eye(len(Qx)))
            Rx, Ry = coordinate_system
            if not (np.allclose(Rx.T @ Rx, np.eye(len(Rx))) and
                    np.allclose(Ry.T @ Ry, np.eye(len(Ry)))):
                raise ValueError('coordinate system not unitary')
        else:
            raise ValueError('Invlaid coordinate_system')

    def _mk_withinset_covs(self):
        self.Sigmaxx_ = np.diag(np.arange(1, self.px + 1) ** self.ax)
        self.Sigmayy_ = np.diag(np.arange(1, self.py + 1) ** self.ay)

    def _mk_S(self):
        self.Sigma_ = np.vstack([
            np.hstack([self.Sigmaxx_, self.Sigmaxy_]),
            np.hstack([self.Sigmaxy_.T, self.Sigmayy_])
        ])
        if not (np.linalg.eigvalsh(self.Sigma_) > 0).all():
            raise WeightNotFoundError()

    def _rotate_into_coordinate_system(self, rng):
        # Qx = np.linalg.qr(rng.normal(size=(self.px, self.px)))[0]
        # Qy = np.linalg.qr(rng.normal(size=(self.py, self.py)))[0]
        # assert np.allclose(Qx.T @ Qx, np.eye(len(Qx)))
        # assert np.allclose(Qy.T @ Qy, np.eye(len(Qy)))

        self.Sigmaxx_ = self.Rx.T @ self.Sigmaxx_ @ self.Rx
        self.Sigmaxy_ = self.Rx.T @ self.Sigmaxy_ @ self.Ry
        self.Sigmayy_ = self.Ry.T @ self.Sigmayy_ @ self.Ry
        self._mk_S()

        self.x_weights_ = self.Rx.T @ self.x_weights_  # Qx.T = Qx^{-1}
        self.y_weights_ = self.Ry.T @ self.y_weights_  # Qy.T = Qy^{-1}

    def generate_data(self, n, random_state=None):
        rng = check_random_state(random_state)
        XY = rng.multivariate_normal(
            np.zeros(self.px + self.py), self.Sigma_, size=n)
        X, Y = XY[:, :self.px], XY[:, self.px:]
        return X, Y

    def select_features(self, x_features, y_features):
        x_features = np.asarray(x_features)
        y_features = np.asarray(y_features)
        ftrs = np.r_[x_features, self.px + y_features]
        new_Sigma = self.Sigma_[ftrs][:, ftrs]
        new_px = len(x_features)
        return self._construct_gm_from_jcov(new_Sigma, new_px)


class PLSgm(_GEMMR):
    def _mk_Sxy(self, wx, wy):
        xstds = np.array([np.sqrt(wx[:, m].T @ self.Sigmaxx_ @ wx[:, m])
                          for m in range(wx.shape[1])])
        ystds = np.array([np.sqrt(wy[:, m].T @ self.Sigmayy_ @ wy[:, m])
                          for m in range(wy.shape[1])])
        self.true_assocs_ = self.true_covs_ = self.r_between * xstds * ystds
        self.true_corrs_ = self.r_between
        self.Sigmaxy_ = wx @ np.diag(self.true_covs_) @ wy.T

    def _construct_gm_from_jcov(self, Sigma, px):
        return JointCovarianceModelPLS(Sigma, px, self.n_components)


class CCAgm(_GEMMR):
    def _mk_Sxy(self, wx, wy):
        if wx.shape[1] != 1:
            raise NotImplementedError('Only 1 mode supported for CCA weights')
        self.true_assocs_ = self.true_corrs_ = self.r_between
        self.Sigmaxy_ = self.Sigmaxx_ @ wx @ np.diag(
            self.true_corrs_) @ wy.T @ self.Sigmayy_ / (
                                np.sqrt(wx.T @ self.Sigmaxx_ @ wx)[0, 0] *
                                np.sqrt(wy.T @ self.Sigmayy_ @ wy)[0, 0]
                        )

    def _construct_gm_from_jcov(self, Sigma, px):
        return JointCovarianceModelCCA(Sigma, px, self.n_components)

def GEMMR(model, *args, **kwargs):
    r"""Generate a joint covariance matrix for `X` and `Y`.

    Within-set principal component spectra are set to follow power laws with
    decay constants `ax` and `ay` for `X` and `Y`, respectively.

    For generation of the between-set covariance matrix :math:`\Sigma_{XY}`
    :func:`_mk_Sigmaxy` is called, see there for details.

    Parameters
    ----------
    model: "pls" or "cca"
        whether to return a covariance matrix for CCA or PLS
    random_state : None or int or a random number generator instance
        For reproducibility, a random number generator is instantiated and all
        random numbers are drawn from that
    px : int
        number of features in `X`
    py : int
        number of features in `Y`
    ax : float
        should usually be <= 0. Eigenvalues of within-modality covariance for
        `X` are assumed to follow a power-law with this exponent
    ay : float
        should usually be <= 0. Eigenvalues of within-modality covariance for
        `X` are assumed to follow a power-law with this exponent
    r_between : float between 0 and 1
        cross-modality correlation the latent mode vectors should have
    max_n_sigma_trials : int >= 1
        number of times an attempt is made to find suitable latent mode
        vectors. See :func:`_mk_Sigmaxy` for details.
    expl_var_ratio_thr : float
        threshold for required within-modality variance along latent mode
        vectors
    coordinate_system : "pc", "random", or vector of length 2
        see `_check_coordinate_system`
    """

    if model == 'cca':
        GM = CCAgm
    elif model == 'pls':
        GM = PLSgm
    else:
        raise ValueError(f'Invalid model: {model}')
    return GM(*args, **kwargs)


def generate_data(Sigma, px, n, random_state=42, split=True):
    r"""Generate synthetic data for a given model.

    The assumed model is a multivariate normal distribution with covariance
    matrix ``Sigma``. ``n`` samples are drawn and from this distribution and
    returned.

    Parameters
    ----------
    Sigma : np.ndarray (total number of features in `X` and `Y`, total number
        of features in `X` and `Y`) joint covariance matrix of model
    px : int
        number of features in dataset `X` (number of features in `Y` is
        inferred from size of ``Sigma``
    n : int
        number of samples to return
    random_state : ``None``, int or random number generator instance
        used to generate random numbers

    Returns
    -------
    X : np.ndarray (n_samples, n_features)
        dataset `X`
    Y : np.ndarray (n_samples, n_features)
        dataset `Y`
    """
    rng = check_random_state(random_state)
    XY = rng.multivariate_normal(
        np.zeros(len(Sigma)), Sigma, size=(n,), check_valid='raise')
    if not split:
        return XY
    else:
        X, Y = XY[:, :px], XY[:, px:]
        return X, Y
