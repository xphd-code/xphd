"""Full exciton-phonon self-energy matrix and the absorption lineshape."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core.interp import fine_band, fourier_refine, hex_cells, rhombic_cells
from .core.stats import KB_EV, bose
from .tetra import delta_weights_2d

__all__ = ["SigmaMatrix", "sigma_matrix", "epsilon2", "read_dipoles",
           "lineshape_metrics", "peak_metrics", "bright_window", "onshell",
           "symmetry_report", "parity_at_q", "interp_uniform", "final_width_sc",
           "final_width_gain"]


def _kk(omega, y):
    """selfenergy.kramers_kronig by FFT: the same sums (subtracted principal value, its regular
    integrand's own value at w' = w, plus the analytic log of the constant part), O(n log n)
    instead of O(n^2). y."""
    w = np.asarray(omega, float)
    y = np.asarray(y, float)
    n = w.size
    y2 = y.reshape(-1, n)
    k = np.arange(-(n - 1), n, dtype=float)
    c = np.zeros_like(k)
    c[k != 0] = 1.0 / k[k != 0]
    L = 1 << int(np.ceil(np.log2(3 * n)))
    z = np.fft.irfft(np.fft.rfft(y2, L, axis=1) * np.fft.rfft(c, L)[None], L, axis=1)
    s1 = -z[:, n - 1: 2 * n - 1]
    Hn = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, n))])
    i = np.arange(n)
    s2 = Hn[n - 1 - i] - Hn[i]
    out = (s1 - y2 * s2[None]) / np.pi
    out[:, 1:-1] += (y2[:, 2:] - y2[:, :-2]) / (2.0 * np.pi)
    lo, hi = w - w[0], w[-1] - w
    m = (lo > 0) & (hi > 0)
    out[:, m] += y2[:, m] * np.log(hi[m] / lo[m])[None] / np.pi
    return out.reshape(y.shape)


def _lorentz_hat(x, h, eta):
    """The weight a node at offset x carries in a Lorentzian smoothing of half-width eta: the
    integral of the Lorentzian against the node's linear hat function (width h), so the
    smoothing of the piecewise-linear function the grid represents is exact, and eta -> 0
    returns the function itself."""
    def G(u):
        return (u * np.arctan(u / eta) - 0.5 * eta * np.log(u * u + eta * eta)) / np.pi
    return (G(x + h) - 2.0 * G(x) + G(x - h)) / h


def _smooth(omega, D, eta):
    """D (n, ...) convolved along its first axis with a Lorentzian of half-width eta, on the
    grid (outside it D is taken as zero)."""
    w = np.asarray(omega, float)
    n = w.size
    h = w[1] - w[0]
    K = _lorentz_hat(np.arange(-(n - 1), n) * h, h, eta)
    flat = D.reshape(n, -1)
    L = 1 << int(np.ceil(np.log2(3 * n)))
    fk = np.fft.rfft(K, L)[:, None]
    def conv(a):
        return np.fft.irfft(np.fft.rfft(a, L, axis=0) * fk, L, axis=0)[n - 1: 2 * n - 1]
    out = conv(flat.real) + (1j * conv(flat.imag) if np.iscomplexobj(flat) else 0.0)
    return out.reshape(D.shape)


def interp_uniform(x0, Y, x):
    """Y (n, ...) given on the uniform grid x0, at the points x, linearly."""
    x0 = np.asarray(x0, float)
    Y = np.asarray(Y)
    x = np.asarray(x, float)
    n = len(x0)
    h = x0[1] - x0[0]
    u = np.clip((x - x0[0]) / h, 0.0, n - 1.0)
    i0 = np.minimum(np.floor(u).astype(int), n - 2)
    f = (u - i0).reshape((-1,) + (1,) * (Y.ndim - 1))
    return (1.0 - f) * Y[i0] + f * Y[i0 + 1]


@dataclass
class SigmaMatrix:
    """Sigma_{S lambda}(omega), shape (n_omega, nexc, nexc), in eV."""
    omega: np.ndarray
    sigma: np.ndarray
    E_bare: np.ndarray
    T: float
    meta: dict = field(default_factory=dict)

    def diagonal_only(self):
        """The same object with off-diagonal elements removed."""
        s = np.zeros_like(self.sigma)
        d = np.arange(self.sigma.shape[1])
        s[:, d, d] = self.sigma[:, d, d]
        return SigmaMatrix(self.omega, s, self.E_bare, self.T,
                           {**self.meta, "diagonal_only": True})

    def at(self, w):
        """Sigma at the frequencies w (inside the grid), interpolated."""
        w = np.asarray(w, float)
        if w.min() < self.omega[0] - 1e-12 or w.max() > self.omega[-1] + 1e-12:
            raise ValueError("frequencies outside the self-energy grid")
        return interp_uniform(self.omega, self.sigma, w)

    def with_final_width(self, eta):
        """Sigma(w + i eta): every final state given the half-width eta (eV)."""
        if not eta or eta <= 0:
            return self
        S = self.sigma
        Sd = np.conj(np.swapaxes(S, 1, 2))
        D = (Sd - S) / 2j
        D = 0.5 * (D + np.conj(np.swapaxes(D, 1, 2)))
        De = _smooth(self.omega, D, float(eta))
        n = len(self.omega)
        flat = De.reshape(n, -1).T
        H = (_kk(self.omega, -flat.real) + 1j * _kk(self.omega, -flat.imag)).T.reshape(S.shape)
        return SigmaMatrix(self.omega, H - 1j * De, self.E_bare, self.T,
                           {**self.meta, "final_width": float(eta)})

    def gamma_diag(self):
        """-2 Im Sigma_SS at each state's own bare energy (FWHM, eV), with the interpolation
        `at` uses, so it equals onshell(..., diagonal=True)."""
        E = np.asarray(self.E_bare, float)
        inside = (E >= self.omega[0]) & (E <= self.omega[-1])
        out = np.full(len(E), np.nan)
        if inside.any():
            S = self.at(E[inside])
            k = np.nonzero(inside)[0]
            out[inside] = -2.0 * S[np.arange(len(k)), k, k].imag
        return out

    def report(self):
        ne = self.sigma.shape[1]
        d = np.arange(ne)
        on = self.gamma_diag()
        offd = np.abs(self.sigma).copy()
        offd[:, d, d] = 0.0
        dia = np.abs(self.sigma[:, d, d])
        r = offd.max(axis=(1, 2)) / np.maximum(dia.max(axis=1), 1e-300)
        print(f"  T = {self.T:g} K")
        print(f"  {'state':>6} {'E (eV)':>9} {'Gamma (meV)':>13} {'tau (ps)':>11}")
        for k in d:
            print(f"  {k+1:6d} {self.E_bare[k]:9.4f} {on[k]*1e3:13.3f} "
                  f"{6.582119569e-4/max(on[k],1e-30):11.4f}")
        print(f"\n  max |off-diagonal| / max |diagonal| = {r.max():.3f}")


def parity_at_q(path, archive, e_tol=2e-3):
    """(n1*n2, ne) sigma_h parity of the final states of a Q = 0 archive at every coarse q (rows
    a*n2 + b), from a parity.npz (Q_red, chi, E), checked against the archive's final-state
    energies E_m."""
    n1, n2, ne = archive.n1, archive.n2, archive.nexc
    d = np.load(path)
    pq = np.mod(np.asarray(d["Q_red"], float)[:, :2], 1.0)
    chi = np.real(np.asarray(d["chi"]))
    E = np.asarray(d["E"], float)
    if chi.shape[1] < ne:
        raise SystemExit(f"{path}: {chi.shape[1]} states per Q, the archive {ne}")
    idx = (np.rint(pq[:, 0] * n1).astype(int) % n1) * n2 + np.rint(pq[:, 1] * n2).astype(int) % n2
    look = np.full(n1 * n2, -1)
    look[idx] = np.arange(len(idx))
    if (look < 0).any():
        raise SystemExit(f"{path} does not cover the {n1}x{n2} mesh")
    Em = np.asarray(archive.grid("E_m"), float).reshape(n1 * n2, ne)
    out = chi[look, :ne]
    worst = float(np.abs(E[look, :ne] - Em).max())
    if worst > e_tol:
        raise SystemExit(f"{path}: exciton energies differ from the archive's by "
                         f"{worst * 1e3:.2f} meV -- not the same states")
    return out


def sigma_matrix(archive, omega, T, hw_fine=None, refine=1, flat_width=None,
                 acoustic_cut=1e-4, band_interp="local", acoustic_model="none",
                 acoustic_scaling="none", parity=None, prune=True, method="sweep",
                 cells="hex", verbose=True):
    """Full Sigma_{S lambda}(omega) at the archive's Q (= 0)."""
    if not archive.has("G"):
        raise ValueError("needs the COMPLEX G_grid; |G|^2 alone cannot give "
                         "the off-diagonal elements")
    if acoustic_model not in ("none", "shell"):
        raise ValueError(f"acoustic_model must be 'none' or 'shell', not {acoustic_model!r}")
    if acoustic_scaling not in ("none", "omega"):
        raise ValueError(f"acoustic_scaling must be 'none' or 'omega', not {acoustic_scaling!r}")
    if acoustic_model == "shell" and parity is None:
        raise ValueError("acoustic_model='shell' needs the final states' parities")
    from .bte import _parity_rank_map, gamma_acoustic
    from .core.mesh import hex_norm

    scalar = np.ndim(T) == 0
    Ts = np.atleast_1d(np.asarray(T, float))
    kTs = [max(KB_EV * t, 1e-12) for t in Ts]
    nT = len(Ts)

    n1, n2 = archive.n1, archive.n2
    r = int(refine)
    N1, N2 = n1 * r, n2 * r
    ne, nmod = archive.nexc, archive.nmod
    fw = flat_width if flat_width is not None else 2.4e-3 / N1

    G = np.asarray(archive.grid("G")).reshape(n1 * n2, nmod, ne, ne)
    E_m_c = archive.grid("E_m")
    hw_c = np.asarray(archive.grid("hw"), float)
    E_m = fine_band(E_m_c, r, band_interp, cells=cells)
    hw = (np.asarray(hw_fine, float).reshape(N1, N2, nmod)
          if hw_fine is not None else fourier_refine(hw_c, r))

    if cells == "hex":
        cidx, cw, _, _ = hex_cells(n1, n2, r)
    elif cells == "rhombic":
        cidx, cw, _, _ = rhombic_cells(n1, n2, r)
    else:
        raise ValueError(f"cells must be 'hex' or 'rhombic', not {cells!r}")
    ncell = n1 * n2
    gfrac = (cw * (cidx == 0)).sum(axis=1)
    hwc_flat = hw_c.reshape(ncell, nmod)

    dead0, res = gamma_acoustic(hwc_flat[0], acoustic_cut)
    dead = set(int(x) for x in dead0)
    live = [hw[:, :, nu] > acoustic_cut for nu in range(nmod)]

    scale = []
    for nu in range(nmod):
        s_ = np.ones(cidx.shape)
        if nu in dead:
            if acoustic_scaling == "omega":
                wc = hwc_flat[cidx, nu]
                s_ = np.where(wc > 0, hw[:, :, nu].ravel()[:, None] / np.maximum(wc, 1e-300),
                              1.0)
            s_[cidx == 0] = 0.0
        scale.append(s_)

    ref = {}
    if acoustic_model == "shell" and r > 1 and len(dead0):
        a_, b_ = np.divmod(np.arange(ncell), n2)
        qn = hex_norm(np.stack([a_ / n1, b_ / n2, np.zeros(ncell)], 1))
        r1 = qn[qn > 1e-12].min()
        s1 = np.nonzero(np.abs(qn - r1) < 1e-6 * r1)[0]
        chi = np.asarray(parity, float)
        for nu in dead0:
            acc = np.zeros((ne, ne, ne), complex)
            cnt = np.zeros(ne)
            for s in s1:
                mp = _parity_rank_map(chi[0], chi[s])
                ok = mp >= 0
                gm = G[s, nu][:, mp[ok]]
                acc[:, :, ok] += np.einsum("sn,ln->sln", gm.conj(), gm) * hwc_flat[s, nu]
                cnt[ok] += 1
            wm = max(float(hwc_flat[s1, nu].mean()), 1e-12)
            P = acc / np.maximum(cnt, 1)[None, None, :]
            ref[int(nu)] = P / wm ** (2 if acoustic_scaling == "omega" else 1)
        if verbose:
            print(f"  acoustic branches at q = 0: {[int(x) + 1 for x in dead0]}"
                  + (f" (sum-rule residue {res * 1e3:.3f} meV)" if res else "")
                  + f"; cell filled from the first shell ({len(s1)} points, "
                    f"|q| = {r1:.4f} |b|), carried down "
                  + ("as w" if acoustic_scaling == "omega" else "constant"))
    elif verbose and len(dead0):
        print(f"  acoustic branches at q = 0: {[int(x) + 1 for x in dead0]}; the q = 0 cell "
              f"gets no acoustic coupling (acoustic_model 'none')"
              + ("; scaled as w elsewhere" if acoustic_scaling == "omega" else ""))
    gw_fac = {nu: (hw[:, :, nu].ravel() if acoustic_scaling == "omega"
                   else np.ones(N1 * N2)) for nu in ref}

    omega = np.asarray(omega, float)
    if len(omega) > 1 and not np.allclose(np.diff(omega), omega[1] - omega[0], rtol=1e-6):
        raise ValueError("omega must be uniform (Kramers-Kronig)")
    Nb = [[bose(hw[:, :, nu], kT).ravel() for nu in range(nmod)] for kT in kTs]
    pad = 1000 * fw
    if verbose:
        print(f"Sigma matrix: {ne} states, {nmod} branches, {len(omega)} frequencies, "
              f"fine {N1}x{N2}, T = {', '.join(f'{t:g}' for t in Ts)} K")
    args = (omega, E_m, hw, G, Nb, live, scale, ref, gw_fac, cidx, cw, gfrac, ncell,
            fw, pad, prune, nT, verbose)
    Dw = _spectral_sweep(*args) if method == "sweep" else _spectral_loop(*args)

    out = []
    d_ = np.arange(ne)
    for t in range(nT):
        flat = Dw[t].reshape(len(omega), -1).T
        H = (_kk(omega, -flat.real)
             + 1j * _kk(omega, -flat.imag)).T.reshape(len(omega), ne, ne)
        sig = H - 1j * Dw[t]
        edge = max(np.abs(sig[0].imag[d_, d_]).max(),
                   np.abs(sig[-1].imag[d_, d_]).max()) \
            / max(np.abs(sig.imag[:, d_, d_]).max(), 1e-300)
        if verbose and t == 0:
            need_lo = float(E_m_c.min() - hw_c.max())
            need_hi = float(E_m_c.max() + hw_c.max())
            print(f"\n  |Im Sigma| at the grid edges: {edge:.2%} of its maximum")
            if edge > 0.05:
                print(f"    truncated -- Kramers-Kronig, and therefore every peak")
                print(f"    position, is unreliable. The grid should span at least")
                print(f"    {need_lo:.3f} .. {need_hi:.3f} eV; it spans "
                      f"{omega[0]:.3f} .. {omega[-1]:.3f}.")
        out.append(SigmaMatrix(omega, sig, np.asarray(archive.E_n, float), float(Ts[t]),
                               dict(refine=r, flat_width=fw, acoustic_cut=acoustic_cut,
                                    acoustic_model=acoustic_model,
                                    acoustic_scaling=acoustic_scaling,
                                    acoustic_branches=[int(x) for x in dead0],
                                    cells=cells,
                                    edge=edge)))
    if verbose:
        print()
        for sm in out:
            sm.report()
    return out[0] if scalar else out


def _spectral_loop(omega, E_m, hw, G, Nb, live, scale, ref, gw_fac, cidx, cw, gfrac,
                   ncell, fw, pad, prune, nT, verbose):
    """D_t(w) by integrating the whole fine mesh at every frequency (reference)."""
    nmod, ne = G.shape[1], G.shape[2]
    Dw = np.zeros((nT, len(omega), ne, ne), complex)
    rng = {}
    for nu in range(nmod):
        for n in range(ne):
            Xe = E_m[:, :, n] + hw[:, :, nu]
            Xa = E_m[:, :, n] - hw[:, :, nu]
            rng[nu, n] = ((Xe.min() - pad, Xe.max() + pad), (Xa.min() - pad, Xa.max() + pad))
    for k, w in enumerate(omega):
        for nu in range(nmod):
            if not live[nu].any():
                continue
            wq, ok = hw[:, :, nu], live[nu]
            for n in range(ne):
                dE = w - E_m[:, :, n]
                for emis, (lo, hi) in zip((True, False), rng[nu, n]):
                    if prune and (w < lo or w > hi):
                        continue
                    W = (delta_weights_2d(dE - wq if emis else dE + wq, fw) * ok).ravel()
                    if not W.any():
                        continue
                    occ = np.stack([Nb[t][nu] + 1.0 if emis else Nb[t][nu] for t in range(nT)])
                    Wt = W[None, :] * occ
                    ws = np.stack([sum(np.bincount(cidx[:, sl],
                                                   weights=Wt[t] * cw[:, sl] * scale[nu][:, sl],
                                                   minlength=ncell)
                                       for sl in range(cidx.shape[1])) for t in range(nT)])
                    g = G[:, nu, :, n]
                    Dw[:, k] += np.pi * np.einsum("tc,cs,cl->tsl", ws, g.conj(), g)
                    if nu in ref:
                        wg = Wt @ (gw_fac[nu] * gfrac)
                        Dw[:, k] += np.pi * wg[:, None, None] * ref[nu][None, :, :, n]
        if verbose and len(omega) >= 10 and (k + 1) % (len(omega) // 10) == 0:
            print(f"  {k+1}/{len(omega)}", end="\r", flush=True)
    return Dw


def _spectral_sweep(omega, E_m, hw, G, Nb, live, scale, ref, gw_fac, cidx, cw, gfrac,
                    ncell, fw, pad, prune, nT, verbose):
    """D_t(w) on the uniform grid omega, triangle by triangle."""
    from .tetra import _tri_weights
    N1, N2, nmod = hw.shape
    ne = G.shape[2]
    nw = len(omega)
    w0 = float(omega[0])
    h = float(omega[1] - omega[0]) if nw > 1 else 1.0
    Dw = np.zeros((nT, nw, ne, ne), complex)
    idx = np.arange(N1 * N2).reshape(N1, N2)
    i10 = np.roll(idx, -1, axis=0)
    i01 = np.roll(idx, -1, axis=1)
    i11 = np.roll(i10, -1, axis=1)
    Iv = np.concatenate([np.stack((idx, i10, i01), -1).reshape(-1, 3),
                         np.stack((i10, i11, i01), -1).reshape(-1, 3)])
    ntri = len(Iv)
    area = 1.0 / (2.0 * N1 * N2)
    nsl = cidx.shape[1]
    for nu in range(nmod):
        if not live[nu].any():
            continue
        okf = live[nu].ravel().astype(float)
        sc = scale[nu]
        hwf = hw[:, :, nu].ravel()
        for n in range(ne):
            g = G[:, nu, :, n]
            prod = (g.conj()[:, :, None] * g[:, None, :]).reshape(len(g), ne * ne)
            En = E_m[:, :, n].ravel()
            for emis in (True, False):
                Xv = (En + hwf if emis else En - hwf)[Iv]
                lo_t, hi_t = Xv.min(axis=1), Xv.max(axis=1)
                flat = (hi_t - lo_t) <= fw
                if flat.any():
                    mid = Xv[flat].mean(axis=1)
                    if prune:
                        lo_t[flat], hi_t[flat] = mid - pad, mid + pad
                    else:
                        lo_t[flat], hi_t[flat] = w0 - h, omega[-1] + h
                k0 = np.floor((lo_t - w0) / h).astype(np.int64)
                k1 = np.ceil((hi_t - w0) / h).astype(np.int64)
                k0 = np.clip(k0, 0, nw - 1)
                k1 = np.clip(k1, -1, nw - 1)
                cnt = np.where((hi_t >= w0 - h) & (lo_t <= omega[-1] + h), k1 - k0 + 1, 0)
                cnt = np.clip(cnt, 0, None)
                P = int(cnt.sum())
                if P == 0:
                    continue
                tri = np.repeat(np.arange(ntri), cnt)
                kk = k0[tri] + (np.arange(P) - np.repeat(np.cumsum(cnt) - cnt, cnt))
                wv = _tri_weights(omega[kk][:, None] - Xv[tri], area, fw)
                if not wv.any():
                    continue
                pts = Iv[tri]
                base = wv * okf[pts]
                key = np.concatenate([(kk[:, None] * ncell + cidx[pts, sl]).ravel()
                                      for sl in range(nsl)])
                bsl = [(base * cw[pts, sl] * sc[pts, sl]) for sl in range(nsl)]
                ws = np.empty((nT, nw * ncell))
                for t in range(nT):
                    occ = (Nb[t][nu] + 1.0 if emis else Nb[t][nu])[pts]
                    ws[t] = np.bincount(key, weights=np.concatenate(
                        [(b_ * occ).ravel() for b_ in bsl]), minlength=nw * ncell)
                ws = ws.reshape(nT * nw, ncell)
                Dw += np.pi * ((ws @ prod.real) + 1j * (ws @ prod.imag)).reshape(nT, nw, ne, ne)
                if nu in ref:
                    gm = gfrac[pts]
                    if gm.any():
                        bg = base * gw_fac[nu][pts] * gm
                        wg = np.empty((nT, nw))
                        for t in range(nT):
                            occ = Nb[t][nu] + 1.0 if emis else Nb[t][nu]
                            wg[t] = np.bincount(kk, weights=(bg * occ[pts]).sum(axis=1),
                                                minlength=nw)
                        Dw += np.pi * wg[:, :, None, None] * ref[nu][None, None, :, :, n]
        if verbose:
            print(f"  branch {nu + 1}/{nmod}", end="\r", flush=True)
    return Dw


def epsilon2(sm, dipoles, eta=1e-3, omega=None):
    """Absorption spectrum from the self-energy matrix."""
    d = np.asarray(dipoles, complex)
    ne = sm.sigma.shape[1]
    if d.size != ne:
        raise ValueError(f"{d.size} dipoles for {ne} states")
    w_ = sm.omega if omega is None else np.asarray(omega, float)
    S = sm.sigma if omega is None else sm.at(w_)
    I = np.eye(ne)
    E = np.diag(sm.E_bare)
    M = (w_[:, None, None] + 1j * eta) * I[None] - E[None] - S
    x = np.linalg.solve(M, np.broadcast_to(d, (len(w_), ne))[..., None])[..., 0]
    return -np.imag(x @ d.conj())


def _find_bsdb(path, name="ndb.BS_diago_Q1"):
    """The BS_diago file itself."""
    import os
    from pathlib import Path
    p = Path(path)
    if p.is_file():
        return str(p)
    if not p.is_dir():
        raise SystemExit(f"{path}: no such file or folder")
    hits = sorted(str(h) for h in p.rglob(name))
    if len(hits) == 1:
        print(f"  dipoles from {hits[0]}")
        return hits[0]
    if not hits:
        raise SystemExit(f"{path} is a folder with no {name} in it; give the "
                         f"file of the BSE the archive was built from")
    raise SystemExit(f"{path} holds {len(hits)} copies of {name}; give the one "
                     f"the archive was built from:" + "".join(f"\n  {h}" for h in hits))


def read_dipoles(bsdb, neigs=None, pol=0):
    """Exciton dipole amplitudes from a yambo BS_diago database."""
    from netCDF4 import Dataset
    bsdb = _find_bsdb(bsdb)
    with Dataset(bsdb) as f:
        keys = list(f.variables)
        lk = "BS_left_Residuals" if "BS_left_Residuals" in keys else "BS_Residuals"
        L = np.asarray(f[lk][:], float)
        R = (np.asarray(f["BS_right_Residuals"][:], float)
             if "BS_right_Residuals" in keys else L)
    Lc, Rc = L[..., 0] + 1j * L[..., 1], R[..., 0] + 1j * R[..., 1]
    if Lc.ndim == 2:
        Lc, Rc = Lc[pol], Rc[pol]
    amp = np.sqrt(np.abs(Lc * Rc.conj()))
    d = amp * np.exp(1j * np.angle(Lc))
    return d[:neigs] if neigs else d


def lineshape_metrics(omega, y, frac=0.5):
    """Peak, width and asymmetry of a spectrum, by interpolation."""
    w = np.asarray(omega, float)
    y = np.asarray(y, float)
    k = int(np.argmax(y))
    if 0 < k < len(w) - 1:
        y0, y1, y2 = y[k-1], y[k], y[k+1]
        den = y0 - 2 * y1 + y2
        pk = w[k] + 0.5 * (w[1] - w[0]) * (y0 - y2) / den if den != 0 else w[k]
    else:
        pk = w[k]

    h = y.max() * frac
    def cross(lo, hi):
        if y[lo] == y[hi]:
            return w[lo]
        t = (h - y[lo]) / (y[hi] - y[lo])
        return w[lo] + t * (w[hi] - w[lo])
    l = r = np.nan
    for i in range(k, 0, -1):
        if y[i-1] < h <= y[i]:
            l = cross(i-1, i); break
    for i in range(k, len(w) - 1):
        if y[i+1] < h <= y[i]:
            r = cross(i+1, i); break
    fw = (r - l) if np.isfinite(l) and np.isfinite(r) else np.nan
    asym = ((r - pk) - (pk - l)) if np.isfinite(fw) else np.nan
    return dict(peak=pk, fwhm=fw, asym=asym, left=l, right=r)


def bright_window(E_bare, dipoles, omega, frac=1e-2, gap=0.02, below=0.25):
    """(lo, hi, members): the window around the lowest bright multiplet."""
    E = np.asarray(E_bare, float)
    a2 = np.abs(np.asarray(dipoles)) ** 2
    br = np.nonzero(a2 >= frac * a2.max())[0]
    br = br[np.argsort(E[br])]
    first = [int(br[0])]
    for k in br[1:]:
        if E[k] - E[first[0]] < gap:
            first.append(int(k))
    Ea = float(E[first].mean())
    nxt = [k for k in br if E[k] - E[first[0]] >= gap]
    hi = Ea + (0.5 * (float(E[nxt[0]]) - Ea) if nxt else 0.15)
    lo = Ea - below
    return max(lo, float(omega[0])), min(hi, float(omega[-1])), first


def peak_metrics(sm, dipoles, window, eta=1e-5, n=4001, max_zoom=6, resolve=200):
    """lineshape_metrics of eps_2 in `window`, zooming onto the peak until the FWHM spans at
    least `resolve` grid steps."""
    lo, hi = window
    x0 = np.linspace(lo, hi, n)
    y0 = epsilon2(sm, dipoles, eta, omega=x0)
    m = lineshape_metrics(x0, y0)
    x_lo, x_hi = lo, hi
    for _ in range(max_zoom):
        step = (x_hi - x_lo) / (n - 1)
        if not np.isfinite(m["fwhm"]) or m["fwhm"] >= resolve * step:
            break
        half = 8.0 * m["fwhm"]
        x_lo, x_hi = max(lo, m["peak"] - half), min(hi, m["peak"] + half)
        x = np.linspace(x_lo, x_hi, n)
        m = lineshape_metrics(x, epsilon2(sm, dipoles, eta, omega=x))
    return m, x0, y0


def onshell(sm, members, diagonal=False):
    """Width and shift of a bright multiplet evaluated AT its bare energy."""
    mem = list(members)
    E_A = float(np.mean(np.asarray(sm.E_bare, float)[mem]))
    S = sm.at([E_A])[0]
    if diagonal:
        S = np.diag(np.diag(S))
    lam, V = np.linalg.eig(np.diag(np.asarray(sm.E_bare, float)) + S)
    wt = (np.abs(V[mem, :]) ** 2).sum(0) / (np.abs(V) ** 2).sum(0)
    pick = np.argsort(wt)[::-1][:len(mem)]
    return dict(fwhm=float(np.mean(-2.0 * lam[pick].imag)),
                shift=float(np.mean(lam[pick].real) - E_A),
                weight=float(np.mean(wt[pick])), E_A=E_A)


def _width_of_eta(sm, members):
    """Gamma_A(eta): the diagonal A width at E_A with final-state half-width eta, from D_AA at
    the two nodes around E_A (interpolated as `onshell`)."""
    w = np.asarray(sm.omega, float)
    n = len(w)
    h = w[1] - w[0]
    mem = list(members)
    E_A = float(np.mean(np.asarray(sm.E_bare, float)[mem]))
    u = float(np.clip((E_A - w[0]) / h, 0.0, n - 1.0))
    i0 = min(int(np.floor(u)), n - 2)
    f = u - i0
    d = np.mean([-sm.sigma[:, m, m].imag for m in mem], axis=0)

    def gam(eta):
        if eta <= 0:
            a, b = d[i0], d[i0 + 1]
        else:
            a = _lorentz_hat(w[i0] - w, h, eta) @ d
            b = _lorentz_hat(w[i0 + 1] - w, h, eta) @ d
        return 2.0 * ((1.0 - f) * a + f * b)

    return gam


def final_width_gain(sm, members, eta, rel=0.02):
    """d Gamma_A / d eta at eta: how much the A width grows with the width of the final states."""
    h = float(sm.omega[1] - sm.omega[0])
    if eta <= 0.01 * h:
        return float("nan")
    gam = _width_of_eta(sm, members)
    return float((gam(eta * (1 + rel)) - gam(eta * (1 - rel))) / (2 * rel * eta))


def final_width_sc(sm, members, tol=1e-4, maxit=200):
    """(eta, iterations, converged): the final-state half-width that equals half the A width it
    produces, eta = -Im Sigma_AA,eta(E_A) (the diagonal at the mean bare energy of `members`,
    interpolated as `onshell` does)."""
    gam = _width_of_eta(sm, members)
    eta = 0.5 * gam(0.0)
    if eta <= 0:
        return 0.0, 0, True
    for it in range(1, maxit + 1):
        new = 0.5 * gam(eta)
        if it > maxit // 2:
            new = 0.5 * (new + eta)
        if abs(new - eta) <= tol * max(eta, 1e-12):
            return float(new), it, True
        eta = new
    return float(eta), maxit, False


def symmetry_report(sm, chi0=None, deg_tol=1e-4, verbose=True, top=3):
    """How far the computed Sigma(Q = 0) is from what the point group demands."""
    S = sm.sigma
    ne = S.shape[1]
    d = np.arange(ne)
    scale = max(float(np.abs(S[:, d, d]).max()), 1e-300)
    out = {}
    c = None
    if chi0 is not None:
        c = np.asarray(chi0, float)[:ne]
        mask = (np.outer(c, c) < -0.81)
        out["parity"] = float(np.abs(S[:, mask]).max()) / scale if mask.any() else 0.0
    E = np.asarray(sm.E_bare, float)
    order = np.argsort(E)
    groups, cur = [], [int(order[0])]
    for k in order[1:]:
        if E[k] - E[cur[0]] < deg_tol:
            cur.append(int(k))
        else:
            groups.append(cur)
            cur = [int(k)]
    groups.append(cur)
    worst = 0.0
    per = []
    for g in groups:
        if len(g) < 2:
            continue
        B = S[:, g][:, :, g]
        tr = np.trace(B, axis1=1, axis2=2) / len(g)
        dev = B - tr[:, None, None] * np.eye(len(g))[None]
        v = float(np.abs(dev).max())
        per.append((g, v))
        worst = max(worst, v)
    out["multiplet"] = worst / scale
    out["groups"] = [g for g in groups if len(g) > 1]
    out["per_multiplet"] = per
    off = np.abs(S).max(axis=0)
    off[d, d] = 0.0
    iu = np.triu_indices(ne, 1)
    vals = off[iu]
    pick = np.argsort(vals)[::-1][:top]
    out["pairs"] = [(int(iu[0][k]), int(iu[1][k]), float(vals[k])) for k in pick]
    if verbose:
        print(f"  symmetry (T = {sm.T:g} K; largest diagonal |Sigma| {scale * 1e3:.2f} meV):")
        if "parity" in out:
            print(f"    |Sigma| between opposite sigma_h parities: "
                  f"{out['parity']:.2e} of it (0 by symmetry)")
        print(f"    degenerate blocks differ from c*1 by {out['multiplet']:.2e} of it "
              f"(0 by Schur; {len(out['groups'])} multiplets)")
        for g, v in per:
            print(f"      states {[k + 1 for k in g]} at {E[g[0]]:.4f} eV: "
                  f"{v * 1e3:.3f} meV")
        print("    largest off-diagonal elements (max over w):")
        for a_, b_, v in out["pairs"]:
            par = "" if c is None else f"  parities {c[a_]:+.0f}/{c[b_]:+.0f}"
            print(f"      {a_ + 1:>3} {b_ + 1:>3}  {E[a_]:.4f}/{E[b_]:.4f} eV  "
                  f"{v * 1e3:8.3f} meV{par}")
    return out
