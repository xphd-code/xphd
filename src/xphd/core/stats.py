"""Occupations, degeneracy grouping, and envelope statistics."""
from __future__ import annotations

import numpy as np

KB_EV = 8.617333262145e-5
HBAR_EV_PS = 6.582119569e-4

__all__ = ["KB_EV", "HBAR_EV_PS", "bose", "degeneracy_groups",
           "envelope", "shell_maxima"]


def bose(w, kT: float, wmin: float = 1e-5):
    """Bose-Einstein occupation; modes below wmin (eV) return 0."""
    w = np.asarray(w, float)
    n = np.zeros_like(w)
    m = w > wmin
    n[m] = 1.0 / np.expm1(np.clip(w[m] / kT, 0.0, 200.0))
    return n


def degeneracy_groups(E, tol: float = 2e-3):
    """Indices of degenerate multiplets in a sorted energy list (eV)."""
    E = np.asarray(E, float)
    groups, cur = [], [0]
    for k in range(1, len(E)):
        if abs(E[k] - E[cur[0]]) < tol:
            cur.append(k)
        else:
            groups.append(cur)
            cur = [k]
    groups.append(cur)
    return groups


def envelope(w):
    """sqrt(sum |w|^2) over all trailing axes of an (n1, n2, ...) array."""
    w = np.asarray(w)
    return np.sqrt(np.sum(np.abs(w) ** 2, axis=tuple(range(2, w.ndim))))


def shell_maxima(r_norm, values, nbins: int = 60, drop_zero: bool = True):
    """Max of `values` in each radial shell."""
    r = np.asarray(r_norm).ravel()
    v = np.asarray(values).ravel()
    if drop_zero:
        keep = r > 0
        r, v = r[keep], v[keep]
    if r.size == 0:
        return np.array([]), np.array([])
    edges = np.linspace(0.0, r.max() * (1 + 1e-9), nbins + 1)
    which = np.clip(np.digitize(r, edges) - 1, 0, nbins - 1)
    c, p = [], []
    for b in range(nbins):
        s = which == b
        if np.any(s):
            c.append(r[s].mean())
            p.append(v[s].max())
    return np.array(c), np.array(p)
