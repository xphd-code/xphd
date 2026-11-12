"""Mirror parity of near-degenerate multiplets."""
from __future__ import annotations

import numpy as np


def energy_groups(E, tol):
    """Group labels for energies E (any order): neighbours in sorted order closer than tol share
    a group."""
    E = np.asarray(E, float)
    order = np.argsort(E)
    lab = np.empty(len(E), int)
    g = 0
    for i, j in enumerate(order):
        if i > 0 and E[j] - E[order[i - 1]] >= tol:
            g += 1
        lab[j] = g
    return lab


def sigma_blocks(A, sh_op, kQ, groups, rep_matrix_Q):
    """Block-diagonal sigma_h matrix of the states in A, one block per group."""
    Dc, Dv, kmap = sh_op[-3], sh_op[-2], sh_op[-1]
    n = len(groups)
    M = np.zeros((n, n), complex)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        M[np.ix_(idx, idx)] = rep_matrix_Q(A, Dc, Dv, kmap, kQ, list(idx))
    return M


def time_reversed(ylat):
    """For every full-zone point: is it reached from its irreducible parent by an operation
    containing time reversal?"""
    sidx = np.asarray(ylat.symmetry_indexes, dtype=int)
    nsym = len(ylat.sym_car)
    return sidx >= nsym / (1 + int(np.rint(ylat.time_rev)))


def split_by_parity(G, M, groups, tol=0.5):
    """Split |G|^2 by the parity eigenstates of each multiplet."""
    G = np.atleast_2d(G)
    n = M.shape[0]
    mu = np.zeros(n)
    W = np.zeros((n, G.shape[0]))
    pos = 0
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        Mg = M[np.ix_(idx, idx)]
        Mh = 0.5 * (Mg + Mg.conj().T)
        w, V = np.linalg.eigh(Mh)
        amp = V.conj().T @ G[:, idx].T
        mu[pos:pos + len(idx)] = w
        W[pos:pos + len(idx)] = np.abs(amp) ** 2
        pos += len(idx)
    return mu, W
