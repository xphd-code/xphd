import numpy as np


def exciton_X_matelem(Q, q, Akq, Ak, em, kBZ, contribution="b",
                      diagonal_only=False, ktree=None):
    """(nmodes, n_out, n_in). em is (nm, nk, ns, fin, init)."""
    a = Akq.reshape(Akq.shape[0], Akq.shape[1], -1)
    b = Ak.reshape(Ak.shape[0], Ak.shape[1], -1)
    ge = np.einsum("akx,mk,bkx->mab", a.conj(), em[:, :, 0, 0, 1], b)
    gh = np.einsum("akx,mk,bkx->mab", a.conj(), em[:, :, 0, 2, 3], b)
    if contribution == "b":
        return ge + gh
    if contribution == "e":
        return ge
    if contribution == "h":
        return gh
    raise ValueError(contribution)
