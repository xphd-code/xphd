"""Analytic linear-triangle ("2D tetrahedron") Brillouin-zone integration."""
from __future__ import annotations

import numpy as np

__all__ = ["delta_weights_2d", "triangle_weights"]


def _tri_weights(e, area, flat_width):
    """Vertex weights for  int_T f(q) delta(D(q)) dA / Omega_BZ  ~  sum_i w_i f(q_i)."""
    e = np.asarray(e, dtype=float)
    order = np.argsort(e, axis=-1, kind="stable")
    es = np.take_along_axis(e, order, axis=-1)
    e1, e2, e3 = es[:, 0], es[:, 1], es[:, 2]

    d21 = e2 - e1
    d31 = e3 - e1
    d32 = e3 - e2

    w = np.zeros_like(es)

    m = (e1 <= 0.0) & (0.0 < e2) & (d21 > flat_width) & (d31 > flat_width)
    if m.any():
        a = (-e1[m]) / d21[m]
        b = (-e1[m]) / d31[m]
        Dt = 2.0 * area * (-e1[m]) / (d21[m] * d31[m])
        w[m, 0] = Dt * (2.0 - a - b) * 0.5
        w[m, 1] = Dt * a * 0.5
        w[m, 2] = Dt * b * 0.5

    m = (e2 <= 0.0) & (0.0 <= e3) & (d32 > flat_width) & (d31 > flat_width)
    if m.any():
        c = e3[m] / d31[m]
        d = e3[m] / d32[m]
        Dt = 2.0 * area * e3[m] / (d32[m] * d31[m])
        w[m, 0] = Dt * c * 0.5
        w[m, 1] = Dt * d * 0.5
        w[m, 2] = Dt * (2.0 - c - d) * 0.5

    m = d31 <= flat_width
    if m.any():
        eav = (e1[m] + e2[m] + e3[m]) / 3.0
        lor = (1.0 / np.pi) * flat_width / (eav ** 2 + flat_width ** 2)
        w[m, :] = (area * lor / 3.0)[:, None]

    out = np.empty_like(w)
    np.put_along_axis(out, order, w, axis=-1)
    return out


def delta_weights_2d(D, flat_width=1e-4, diag="anti"):
    """Integration weights on a periodic 2D mesh for  (1/Omega) int d2q f(q) delta(D(q))."""
    D = np.asarray(D, dtype=float)
    n1, n2 = D.shape
    N = n1 * n2
    idx = np.arange(N).reshape(n1, n2)

    def corners(a):
        a00 = a
        a10 = np.roll(a, -1, axis=0)
        a01 = np.roll(a, -1, axis=1)
        a11 = np.roll(np.roll(a, -1, axis=0), -1, axis=1)
        return a00, a10, a01, a11

    d00, d10, d01, d11 = corners(D)
    i00, i10, i01, i11 = corners(idx)

    if diag == "anti":
        tA = (d00, d10, d01);  iA = (i00, i10, i01)
        tB = (d10, d11, d01);  iB = (i10, i11, i01)
    elif diag == "main":
        tA = (d00, d10, d11);  iA = (i00, i10, i11)
        tB = (d00, d11, d01);  iB = (i00, i11, i01)
    else:
        raise ValueError("diag must be 'anti' or 'main'")

    Dv = np.concatenate([np.stack(tA, -1).reshape(-1, 3),
                         np.stack(tB, -1).reshape(-1, 3)], axis=0)
    Iv = np.concatenate([np.stack(iA, -1).reshape(-1, 3),
                         np.stack(iB, -1).reshape(-1, 3)], axis=0)

    area = 1.0 / (2.0 * N)
    w = _tri_weights(Dv, area, flat_width)

    W = np.bincount(Iv.ravel(), weights=w.ravel(), minlength=N)
    return W.reshape(n1, n2)


def triangle_weights(vertex_values, area: float = 1.0, flat_width: float = 0.0):
    """Weights for explicit triangles, targeting D = 0."""
    v = np.asarray(vertex_values, float)
    if v.shape[0] != 3:
        raise ValueError(f"first axis must be 3, got {v.shape}")
    tail = v.shape[1:]
    flat = np.moveaxis(v, 0, -1).reshape(-1, 3)
    w = _tri_weights(flat, area, flat_width)
    return np.moveaxis(w.reshape(tail + (3,)), -1, 0)
