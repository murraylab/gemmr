import numpy as np


def sympsd_inv_sqrt(M):
    assert np.allclose(M, M.T)
    assert np.all(np.linalg.eigvalsh(M) > 0)

    evals, evecs = np.linalg.eigh(M)
    evals = np.where(evals > 1e-15, 1 / np.sqrt(evals), 0)
    Minvsqrt = evecs.dot(np.diag(evals)).dot(evecs.T)
    assert np.allclose(
        Minvsqrt.dot(M).dot(Minvsqrt),
        np.eye(len(M))
    )
    assert np.allclose(Minvsqrt @ Minvsqrt @ M, np.eye(len(M)))
    assert np.allclose(M @ Minvsqrt @ Minvsqrt, np.eye(len(M)))

    return Minvsqrt