"""Point-group symmetry: stars, unfolding, and violation metrics."""
from __future__ import annotations

import numpy as np

__all__ = ["reduced_ops", "load_ops", "star_map", "unfold",
           "violation", "full_mesh", "irreducible", "expand"]


def reduced_ops(lat, tol=1e-4):
    """Point-group operations as matrices in reduced coordinates."""
    ops, seen = [], set()
    for s in lat.sym_car:
        R = lat.lat @ s @ np.linalg.inv(lat.lat)
        Ri = np.rint(R)
        if np.abs(R - Ri).max() > tol:
            raise SystemExit(
                f"reduced-coordinate rotation is not integer:\n{R}\n"
                "the lattice convention differs from the one assumed here")
        key = tuple(Ri.ravel().astype(int))
        if key not in seen:
            seen.add(key)
            ops.append(Ri)
    return ops


def load_ops(savepath="SAVE"):
    """Read the point group from a yambo ns.db1."""
    try:
        from yambopy import YamboLatticeDB
    except ImportError:
        raise SystemExit("load_ops needs yambopy; pass operations directly "
                         "if it is not installed")
    return reduced_ops(YamboLatticeDB.from_db_file(f"{savepath}/ns.db1"))


def _wrap(x):
    return x - np.rint(x)


def star_map(Q_red, ops, tol=1e-4):
    """Cluster Q-points into symmetry stars."""
    Q = np.asarray(Q_red, float)
    owner = np.full(len(Q), -1, int)
    reps = []
    for i, q in enumerate(Q):
        if owner[i] != -1:
            continue
        reps.append(i)
        owner[i] = i
        for R in ops:
            d = _wrap(Q - (R @ q)[None, :])
            j = int(np.argmin(np.linalg.norm(d, axis=1)))
            if np.linalg.norm(d[j]) < tol:
                owner[j] = i
    return np.array(reps), owner


def unfold(values, Q_red, ops, tol=1e-4, verbose=True):
    """Replace every point by its star representative's value."""
    v = np.asarray(values, float)
    reps, owner = star_map(Q_red, ops, tol)
    out = v[owner]
    if verbose:
        sizes = np.bincount(owner)[reps]
        rel = np.abs(v - out) / np.maximum(np.abs(out), 1e-30)
        print(f"  {len(reps)} stars over {len(v)} points "
              f"(sizes {sizes.min()}-{sizes.max()})")
        print(f"  max shift from unfolding: {rel.max()*100:.2f}%  "
              f"(this is the triangulation noise being removed)")
    return out


def violation(values, Q_red, ops, tol=1e-4):
    """max relative spread of `values` within a star."""
    v = np.asarray(values, float)
    reps, owner = star_map(Q_red, ops, tol)
    worst = 0.0
    for r in reps:
        m = owner == r
        s = v[m]
        if s.size > 1 and np.abs(s).max() > 0:
            worst = max(worst, float((s.max() - s.min())
                                     / max(abs(s).max(), 1e-30)))
    return worst


def full_mesh(n1, n2=None):
    """Reduced coordinates of a full Gamma-centred (n1 x n2) mesh, row-major."""
    n2 = n2 or n1
    i, j = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
    return np.stack([i.ravel() / n1, j.ravel() / n2,
                     np.zeros(n1 * n2)], 1)


def irreducible(n1, ops, n2=None, verbose=True):
    """One representative per star of an (n1 x n2) mesh."""
    Q = full_mesh(n1, n2)
    reps, owner = star_map(Q, ops)
    if verbose:
        sizes = np.bincount(owner)[reps]
        print(f"  {len(reps)} stars over {len(Q)} points "
              f"(sizes {sizes.min()}-{sizes.max()}), "
              f"{len(Q)/len(reps):.1f}x saving")
    return np.sort(reps), Q


def expand(values, Q_computed, Q_full, ops, tol=1e-4, verbose=True):
    """Expand a field computed on a SUBSET of Q onto the full list."""
    v = np.asarray(values)
    Qc = np.asarray(Q_computed, float)
    Qf = np.asarray(Q_full, float)

    src = np.full(len(Qf), -1, int)
    for k, qc in enumerate(Qc):
        for R in ops:
            d = _wrap(Qf - (R @ qc)[None, :])
            hit = np.linalg.norm(d, axis=1) < tol
            src[hit & (src < 0)] = k

    missing = int((src < 0).sum())
    if missing:
        raise SystemExit(
            f"{missing} of {len(Qf)} points are in no star of the computed "
            f"set. The subset does not cover the irreducible wedge -- check "
            f"it against xphd.symmetry.irreducible().")
    if verbose:
        print(f"  expanded {len(Qc)} computed -> {len(Qf)} points "
              f"({len(Qf)/max(len(Qc),1):.1f}x)")
    return v[src]
