"""Zero-padded Fourier interpolation on a periodic mesh."""
from __future__ import annotations

import numpy as np

__all__ = ["fourier_refine", "ringing_report", "fine_band", "local_quadratic",
           "hex_cells", "rhombic_cells"]


def fourier_refine(f, r: int):
    """Refine a real periodic field by a factor r per axis."""
    f = np.asarray(f, float)
    if r == 1:
        return f.copy()
    n1, n2 = f.shape[:2]
    extra = f.shape[2:]
    N1, N2 = n1 * r, n2 * r

    F = np.fft.fftshift(np.fft.fft2(f, axes=(0, 1)), axes=(0, 1))
    if n1 % 2 == 0:
        F[0] = F[0] * 0.5
    if n2 % 2 == 0:
        F[:, 0] = F[:, 0] * 0.5

    G = np.zeros((N1, N2) + extra, complex)
    o1, o2 = (N1 - n1) // 2, (N2 - n2) // 2
    G[o1:o1 + n1, o2:o2 + n2] = F
    if n1 % 2 == 0:
        G[o1 + n1, o2:o2 + n2] = F[0]
    if n2 % 2 == 0:
        G[o1:o1 + n1, o2 + n2] = F[:, 0]
    if n1 % 2 == 0 and n2 % 2 == 0:
        G[o1 + n1, o2 + n2] = F[0, 0]

    out = np.fft.ifft2(np.fft.ifftshift(G, axes=(0, 1)), axes=(0, 1))
    return out.real * (N1 * N2) / (n1 * n2)


def ringing_report(f, r: int):
    """Quantify Gibbs overshoot of a Fourier refinement."""
    f = np.asarray(f, float)
    F = fourier_refine(f, r)
    rng = float(f.max() - f.min())
    over = float(max(F.max() - f.max(), f.min() - F.min()))
    node = float(np.abs(F[::r, ::r] - f).max()) if r > 1 else 0.0
    return dict(overshoot=over, overshoot_frac=over / rng if rng else 0.0,
                data_range=rng, node_error=node, refined=F)


def rhombic_cells(n1: int, n2: int, r: int):
    """Each fine point's coarse cell by rounding each reduced coordinate."""
    N1, N2 = n1 * r, n2 * r
    ci = np.rint(np.arange(N1) / r).astype(int)
    cj = np.rint(np.arange(N2) / r).astype(int)
    I, J = np.meshgrid(np.arange(N1), np.arange(N2), indexing="ij")
    CI, CJ = np.meshgrid(ci, cj, indexing="ij")
    idx = ((CI % n1) * n2 + CJ % n2).reshape(-1, 1)
    di = (I - CI * r).reshape(-1, 1)
    dj = (J - CJ * r).reshape(-1, 1)
    return idx, np.ones(idx.shape), di, dj


def hex_cells(n1: int, n2: int, r: int, tol: float = 1e-9):
    """Each fine point's coarse cell as the Wigner-Seitz (Voronoi) hexagon of the coarse lattice
    in the 60-degree basis (|q|^2 = q1^2 + q2^2 + q1 q2)"""
    N1, N2 = n1 * r, n2 * r
    I, J = np.meshgrid(np.arange(N1), np.arange(N2), indexing="ij")
    I, J = I.ravel(), J.ravel()
    a0, b0 = I // r, J // r
    cand = [(a0, b0), (a0 + 1, b0), (a0, b0 + 1), (a0 + 1, b0 + 1)]
    d = []
    for a, b in cand:
        u = (I - a * r) / r
        v = (J - b * r) / r
        d.append(u * u + v * v + u * v)
    d = np.stack(d, 1)
    dmin = d.min(axis=1, keepdims=True)
    tie = d <= dmin + tol
    nt = tie.sum(axis=1)
    if nt.max() > 3:
        raise RuntimeError("hex_cells: more than three equidistant coarse points")
    idx = np.zeros((len(I), 3), int)
    w = np.zeros((len(I), 3))
    di = np.zeros((len(I), 3), int)
    dj = np.zeros((len(I), 3), int)
    slot = np.zeros(len(I), int)
    for c, (a, b) in enumerate(cand):
        m = tie[:, c]
        s_ = slot[m]
        rows = np.nonzero(m)[0]
        idx[rows, s_] = (a[m] % n1) * n2 + b[m] % n2
        w[rows, s_] = 1.0 / nt[m]
        di[rows, s_] = I[m] - a[m] * r
        dj[rows, s_] = J[m] - b[m] * r
        slot[m] += 1
    return idx, w, di, dj


def local_quadratic(f, r: int, cells: str = "rhombic"):
    """Refine a periodic field by a factor r per axis with a LOCAL quadratic around every coarse
    point, applied across that point's cell."""
    fc = np.asarray(f, float)
    if r == 1:
        return fc.copy()
    n1, n2 = fc.shape[:2]
    if n1 < 3 or n2 < 3:
        raise ValueError("local_quadratic needs at least a 3 x 3 coarse mesh")
    N1, N2 = n1 * r, n2 * r

    def R(di, dj):
        return np.roll(fc, (-di, -dj), axis=(0, 1))

    h1, h2 = 1.0 / n1, 1.0 / n2
    S1 = R(1, 0) + R(-1, 0) - 2 * fc
    S2 = R(0, 1) + R(0, -1) - 2 * fc
    S3 = R(1, -1) + R(-1, 1) - 2 * fc
    A = S1 / (2 * h1 * h1)
    D = S2 / (2 * h2 * h2)
    B = (S1 + S2 - S3) / (2 * h1 * h2)
    d1 = R(1, 0) - R(-1, 0)
    d2 = R(0, 1) - R(0, -1)
    d3 = R(1, -1) - R(-1, 1)
    gx = (2 * d1 + d2 + d3) / 6 / h1
    gy = (d1 + 2 * d2 - d3) / 6 / h2
    if cells == "hex":
        idx, w, di, dj = hex_cells(n1, n2, r)
        k = fc.ndim - 2
        flat = lambda a: a.reshape((n1 * n2,) + a.shape[2:])
        F0, GX, GY, AA, BB, DD = (flat(x) for x in (fc, gx, gy, A, B, D))
        out = 0.0
        for t in range(3):
            U = (di[:, t] / N1).reshape((-1,) + (1,) * k)
            V = (dj[:, t] / N2).reshape((-1,) + (1,) * k)
            c = idx[:, t]
            wt = w[:, t].reshape((-1,) + (1,) * k)
            out = out + wt * (F0[c] + GX[c] * U + GY[c] * V + AA[c] * U * U
                              + BB[c] * U * V + DD[c] * V * V)
        return out.reshape((N1, N2) + fc.shape[2:])
    if cells != "rhombic":
        raise ValueError(f"cells must be 'rhombic' or 'hex', not {cells!r}")
    ci = np.rint(np.arange(N1) / r).astype(int)
    cj = np.rint(np.arange(N2) / r).astype(int)
    u = (np.arange(N1) - ci * r) / N1
    v = (np.arange(N2) - cj * r) / N2
    I, J = np.meshgrid(ci % n1, cj % n2, indexing="ij")
    U, V = np.meshgrid(u, v, indexing="ij")
    k = fc.ndim - 2
    U = U.reshape(U.shape + (1,) * k)
    V = V.reshape(V.shape + (1,) * k)
    return (fc[I, J] + gx[I, J] * U + gy[I, J] * V + A[I, J] * U * U
            + B[I, J] * U * V + D[I, J] * V * V)


def fine_band(f, r: int, method: str = "local", cells: str = "rhombic"):
    """Exciton energies on the fine mesh. method = "local" (default): local quadratics, which do
    not ring; "fourier": the zero-padded Fourier interpolant used before, to reproduce
    earlier numbers. cells: which coarse point's quadratic a fine point uses (local only; see
    local_quadratic)."""
    if method == "local":
        return local_quadratic(f, r, cells)
    if method == "fourier":
        return fourier_refine(f, r)
    raise ValueError(f"method must be 'local' or 'fourier', not {method!r}")
