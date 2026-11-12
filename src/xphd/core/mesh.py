"""Uniform 2D q-meshes: indexing, real-space duals, and shells."""
from __future__ import annotations

import numpy as np

__all__ = ["mesh_indices", "to_grid", "blockwise", "ws_vectors",
           "hex_norm", "hex_dist", "shells", "shift_grid"]


def mesh_indices(q_red, n1: int, n2: int, tol: float = 1e-4):
    """Reduced q-points -> integer mesh indices, asserting exact coverage."""
    q = np.mod(np.asarray(q_red, float), 1.0)
    fi, fj = q[:, 0] * n1, q[:, 1] * n2
    i = np.rint(fi).astype(int) % n1
    j = np.rint(fj).astype(int) % n2
    err = max(np.abs(fi - np.rint(fi)).max(), np.abs(fj - np.rint(fj)).max())
    if err > tol:
        raise ValueError(
            f"q-points are not on a {n1}x{n2} mesh (max residual {err:.2e} "
            f"in index units, tolerance {tol:.0e}); are they in reduced "
            f"coordinates, and is the mesh size right?")
    flat = i * n2 + j
    if len(np.unique(flat)) != n1 * n2 or len(flat) != n1 * n2:
        raise ValueError(
            f"mesh not covered exactly once: {len(np.unique(flat))} unique of "
            f"{n1*n2} expected, {len(flat)} points given. Unfold the IBZ first.")
    return i, j


def to_grid(flat, i, j, n1: int, n2: int):
    """Scatter a (nq, ...) array onto an (n1, n2, ...) mesh."""
    flat = np.asarray(flat)
    out = np.zeros((n1, n2) + flat.shape[1:], dtype=flat.dtype)
    out[i, j] = flat
    return out


def blockwise(a, r: int):
    """Piecewise-constant upsampling by NEAREST coarse point, with wrap."""
    a = np.asarray(a)
    if r == 1:
        return a
    n1, n2 = a.shape[:2]
    i = np.rint(np.arange(n1 * r) / r).astype(int) % n1
    j = np.rint(np.arange(n2 * r) / r).astype(int) % n2
    return a[np.ix_(i, j)]


def ws_vectors(n1: int, n2: int, a1, a2):
    """Cartesian R vectors of the Born-von Karman lattice, folded into the Wigner-Seitz cell of
    the supercell (minimum image over all 9 images)."""
    a1 = np.asarray(a1, float)[:2]
    a2 = np.asarray(a2, float)[:2]
    m1 = np.arange(n1)[:, None] * np.ones((1, n2), int)
    m2 = np.ones((n1, 1), int) * np.arange(n2)[None, :]

    best_n = np.full((n1, n2), np.inf)
    best_v = np.zeros((n1, n2, 2))
    for p in (-1, 0, 1):
        for s in (-1, 0, 1):
            v = (m1 + p * n1)[..., None] * a1 + (m2 + s * n2)[..., None] * a2
            nrm = np.linalg.norm(v, axis=-1)
            better = nrm < best_n - 1e-10
            best_n = np.where(better, nrm, best_n)
            best_v = np.where(better[..., None], v, best_v)
    return best_v, best_n


def hex_norm(q_red):
    """|q| in units of |b|, minimum image, in the hexagonal reciprocal metric of a QE / yambo
    hexagonal cell: b1, b2 at 60 degrees, so |q|^2 = q1^2 + q2^2 + q1 q2 (and K = (1/3,
    1/3))."""
    f = ((np.asarray(q_red, float)[..., :2] + 0.5) % 1.0) - 0.5
    best = np.full(f.shape[:-1], np.inf)
    for m1 in (-1.0, 0.0, 1.0):
        for m2 in (-1.0, 0.0, 1.0):
            a, b = f[..., 0] + m1, f[..., 1] + m2
            best = np.minimum(best, a * a + b * b + a * b)
    return np.sqrt(best)


def hex_dist(a, b):
    """Minimum-image distance between reduced points a and b (broadcast over the leading axes),
    in units of |b|; see hex_norm."""
    return hex_norm(np.asarray(a, float)[..., :2] - np.asarray(b, float)[..., :2])


def shells(r, tol: float = 1e-6):
    """Group q-points into stars of equal |q|."""
    r = np.asarray(r, float)
    out, seen = [], np.zeros(len(r), bool)
    for k in np.argsort(r):
        if seen[k]:
            continue
        m = np.abs(r - r[k]) < tol
        seen |= m
        out.append((float(r[k]), np.where(m)[0]))
    return out


def shift_grid(field, iQ: int, jQ: int):
    """E(Q+q) from E(q) by rolling with the integer mesh index of Q."""
    return np.roll(np.roll(np.asarray(field), -iQ, axis=0), -jQ, axis=1)
