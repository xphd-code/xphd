"""Time- and temperature-resolved ARPES from exciton populations."""
from __future__ import annotations

import numpy as np

__all__ = ["arpes_map", "exciton_arpes", "star_kmaps", "star_assignment",
           "valence_energies", "qp_corrections", "check_qp_alignment",
           "populations",
           "path_indices", "fine_path", "grid_lookup", "interp_matrix",
           "vbm_points", "dispersion_on_path", "hole_points", "hole_band_energy",
           "curves_on_path",
           "exciton_spectra", "boltzmann_weights", "geometry_factor",
           "photoelectron_momentum", "lorentz_convolve",
           "carrier_occupations", "valence_photoemission", "cell_area_nm2",
           "parse_times", "box_points", "occupied_dos", "exph_fwhm", "LorentzBins"]

HBAR2_2M = 3.8099821161


def photoelectron_momentum(hnu, work_function=4.5, binding=0.0):
    """|kappa| in 1/Ang for a photoelectron of the given kinetic energy."""
    ek = max(hnu - work_function - binding, 1e-6)
    return float(np.sqrt(ek / HBAR2_2M))


def geometry_factor(kpar, pol, hnu, work_function=4.5, binding=0.0):
    """|eps . kappa|^2 for each in-plane k."""
    kpar = np.atleast_2d(np.asarray(kpar, float))[:, :2]
    kmag = photoelectron_momentum(hnu, work_function, binding)
    kz2 = np.maximum(kmag ** 2 - (kpar ** 2).sum(1), 0.0)
    kap = np.column_stack([kpar, np.sqrt(kz2)])
    e = np.asarray(pol, float)
    e = e / max(np.linalg.norm(e), 1e-30)
    return np.abs(kap @ e) ** 2


def boltzmann_weights(E, T, mu=None):
    """Normalised Boltzmann occupations for a list of exciton energies."""
    E = np.asarray(E, float)
    kT = 8.617333262e-5 * float(T)
    w = np.exp(-(E - E.min()) / max(kT, 1e-12))
    return w / max(w.sum(), 1e-300)


def arpes_map(A, E_exc, eps_v, kmap_Q, N, omega, sigma=0.05,
              mat=None, verbose=True):
    """I(k, E) for one set of exciton occupations."""
    A = np.asarray(A)
    nQ, nS, nk, nv, nc = A.shape
    omega = np.asarray(omega, float)
    I = np.zeros((nk, len(omega)))
    W = np.abs(A) ** 2
    if mat is not None:
        W = W * np.asarray(mat)[None, None, :, None, :]
    inv = 1.0 / (2.0 * sigma ** 2)

    tot = float(np.asarray(N).sum())
    if verbose:
        print(f"   {nQ} Q-points, {nS} states, {nk} k, {len(omega)} energies")
        print(f"   total occupation {tot:.4e}"
              + ("" if tot > 0 else "   <-- nothing populated"))

    for iQ in range(nQ):
        nQ_ = np.asarray(N)[iQ]
        if nQ_.sum() <= 0:
            continue
        ev = eps_v[kmap_Q[iQ]]
        for S in range(nS):
            if nQ_[S] <= 0:
                continue
            w = W[iQ, S].sum(axis=2)
            pos = E_exc[iQ, S] + ev
            d = omega[None, None, :] - pos[:, :, None]
            I += nQ_[S] * np.einsum("kv,kve->ke", w, np.exp(-d ** 2 * inv))
    return I / max(I.max(), 1e-300)


HA2EV = 27.211386245988


def exciton_arpes(A, E, eps_v, kplusQ, N, omega, sigma):
    """Photoemission from the excitons at ONE Q, indexed by photoelectron p."""
    A = np.asarray(A)
    nS, nk = A.shape[0], A.shape[1]
    omega = np.asarray(omega, float)
    inv = 1.0 / (2.0 * sigma ** 2)
    W = np.sum(np.abs(A) ** 2, axis=3)
    out = np.zeros((nk, len(omega)))
    for S in range(nS):
        if N[S] <= 0:
            continue
        pos = E[S] + np.asarray(eps_v)[kplusQ]
        g = np.exp(-(omega[None, None, :] - pos[:, :, None]) ** 2 * inv)
        out += N[S] * np.einsum("kv,kve->ke", W[S], g)
    return out


def star_kmaps(save, kf):
    """(matrix, kmap) for every operation, spatial and time-reversed."""
    from .symmetry import load_ops
    R_all = np.asarray(load_ops(save))
    ops = []
    for R in R_all[: len(R_all) // 2]:
        for M in (R, -R):
            d = kf[:, None, :] - (kf @ M.T)[None, :, :]
            d -= np.rint(d)
            dist = np.linalg.norm(d, axis=-1)
            km = np.argmin(dist, axis=1)
            if (dist[np.arange(len(kf)), km].max() < 1e-5
                    and len(np.unique(km)) == len(kf)):
                ops.append((M, km))
    return ops


def star_assignment(kf, kidx, first, ops):
    """For each full-zone point i, the kmap of an operation carrying its irreducible
    representative Q_parent onto Q_i."""
    out = []
    for i in range(len(kf)):
        Qj = kf[first[kidx[i]]]
        for M, km in ops:
            dq = Qj @ M.T - kf[i]
            if np.abs(dq - np.rint(dq)).max() < 1e-5:
                out.append(km)
                break
        else:
            raise SystemExit(f"no operation maps the representative of "
                             f"point {i} onto it; the star is incomplete")
    return out


def qp_corrections(path, n_ibz, vbands):
    """(E_QP - E_KS, E_KS), eV, each (n_ibz, nv), for the 0-based `vbands`."""
    import os
    from yambopy import YamboQPDB
    folder, name = os.path.split(str(path))
    qp = YamboQPDB.from_db(filename=name, folder=folder or ".")
    if qp.spin:
        raise SystemExit(f"{path}: spin-polarised QP databases are not "
                         f"supported")
    kmin, kmax = int(qp.min_kpoint), int(qp.max_kpoint)
    bmin, bmax = int(qp.min_band), int(qp.max_band)
    b1 = np.asarray(vbands, dtype=int) + 1
    if kmin != 1 or kmax < n_ibz:
        raise SystemExit(f"{path} covers k-points {kmin}-{kmax}; the BSE "
                         f"needs all {n_ibz} irreducible points (QPkrange)")
    if b1.min() < bmin or b1.max() > bmax:
        raise SystemExit(f"{path} covers bands {bmin}-{bmax}; the BSE "
                         f"valence bands are {list(b1)} (QPkrange)")
    have = set(zip(np.asarray(qp.kpoint_index, int).tolist(),
                   np.asarray(qp.band_index, int).tolist()))
    missing = [(k, int(b)) for k in range(1, n_ibz + 1) for b in b1
               if (k, int(b)) not in have]
    if missing:
        raise SystemExit(f"{path} has no QP energy for {len(missing)} of the "
                         f"{n_ibz * len(b1)} (k, band) pairs the BSE uses, "
                         f"e.g. (k, band) = {missing[:5]} (QPkrange)")
    e0 = np.asarray(qp.eigenvalues_dft, float)[:n_ibz][:, b1 - bmin]
    e1 = np.asarray(qp.eigenvalues_qp, float)[:n_ibz][:, b1 - bmin]
    return e1 - e0, e0


def check_qp_alignment(ev, e0, path="ndb.QP", tol=5e-3):
    """The KS energies of ns.db1 and of the QP database must differ by ONE constant -- yambo's
    choice of zero -- at every (k, band)."""
    d = np.asarray(ev, float) - np.asarray(e0, float)
    off = float(np.median(d))
    worst = float(np.abs(d - off).max())
    if worst > tol:
        k, b = np.unravel_index(np.argmax(np.abs(d - off)), d.shape)
        raise SystemExit(
            f"{path}: its Kohn-Sham energies do not match ns.db1 point by "
            f"point (off by {worst * 1e3:.1f} meV at irreducible k {k + 1}, "
            f"valence band {b + 1} of the BSE). The GW was run on another "
            f"SAVE or k ordering; use the ndb.QP made on this SAVE.")
    return off


def valence_energies(save, ylat, vbands, shift=0.0, qp_db=None):
    """Valence energies on the full zone, eV, for the given 0-based bands."""
    from netCDF4 import Dataset
    with Dataset(f"{save}/ns.db1") as d:
        ev = np.asarray(d.variables["EIGENVALUES"][:], float)
    if ev.ndim == 3:
        ev = ev[0]
    kibz = np.asarray(ylat.kmap)[:, 0]
    n_ibz = int(kibz.max()) + 1
    if ev.shape[0] != n_ibz and ev.shape[1] == n_ibz:
        ev = ev.T
    if ev.shape[0] != n_ibz:
        raise SystemExit(f"EIGENVALUES {ev.shape} does not match {n_ibz} "
                         f"irreducible k-points")
    ev = ev[:, vbands] * HA2EV
    if qp_db:
        dE, e0 = qp_corrections(qp_db, n_ibz, vbands)
        off = check_qp_alignment(ev, e0, qp_db)
        top = np.unravel_index(np.argmax(ev), ev.shape)
        print(f"   QP corrections from {qp_db}: {dE.min():+.3f} .. "
              f"{dE.max():+.3f} eV, {dE[top]:+.3f} eV at the valence maximum; "
              f"its KS energies match ns.db1 at every point (yambo's zero "
              f"{off:+.4f} eV)")
        ev = ev + dE
    return ev[kibz] + shift


_HEX_B = np.array([[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])


def _path_vertices(labels, B):
    """Reduced coordinates of the labels, each segment taken the short way."""
    from .excsym import HIGH_SYMMETRY
    V = [np.array(HIGH_SYMMETRY[l], float) for l in labels]
    shifts = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)], float)
    for n in range(1, len(V)):
        cand = V[n] + shifts
        seg = (cand - V[n - 1]) @ B
        d = np.linalg.norm(seg, axis=1)
        d0 = float(np.linalg.norm((V[n] - V[n - 1]) @ B))
        tol = 1e-9 * max(d0, 1.0)
        if d.min() >= d0 - tol:
            continue
        tie = np.nonzero(d <= d.min() + tol)[0]
        if n >= 2 and len(tie) > 1:
            prev = (V[n - 1] - V[n - 2]) @ B
            cos = seg[tie] @ prev / (d[tie] * max(np.linalg.norm(prev), 1e-300))
            V[n] = cand[tie[int(np.argmax(cos))]]
        else:
            V[n] = cand[tie[0]]
    return V


def fine_path(labels, rlat, npts=600):
    """Points along a path of high-symmetry labels, evenly spaced in length."""
    from .excsym import HIGH_SYMMETRY
    B = np.asarray(rlat, float)[:2, :2]
    V = _path_vertices(labels, B)
    L = [float(np.linalg.norm((b - a) @ B)) for a, b in zip(V[:-1], V[1:])]
    if "G" in labels and "M" in labels and "K" in labels:
        gm = np.linalg.norm(np.array(HIGH_SYMMETRY["M"]) @ B)
        gk = np.linalg.norm(np.array(HIGH_SYMMETRY["K"]) @ B)
        if abs(gk / gm - 2 / np.sqrt(3)) > 1e-3:
            print(f"   WARNING: |GK|/|GM| = {gk / gm:.4f}, not 2/sqrt(3): the "
                  f"reduced basis is not the 60-degree one K = (1/3, 1/3) "
                  f"assumes")
    total = sum(L)
    pts, x, xt = [], [], [0.0]
    for a, b, l in zip(V[:-1], V[1:], L):
        m = max(int(round(npts * l / max(total, 1e-300))), 2)
        s = np.linspace(0.0, 1.0, m, endpoint=False)
        pts.append(a + s[:, None] * (b - a))
        x.append(xt[-1] + s * l)
        xt.append(xt[-1] + l)
    pts.append(V[-1][None, :])
    x.append(np.array([xt[-1]]))
    pts = np.concatenate(pts)
    return (np.column_stack([pts, np.zeros(len(pts))]), np.concatenate(x),
            np.array(xt))


def grid_lookup(kf):
    """(n1, n2, table) for a regular full mesh: table[i1, i2] is the index in kf of the point
    (i1/n1, i2/n2)."""
    k = np.mod(np.asarray(kf, float)[:, :2], 1.0)
    k[np.abs(k - 1.0) < 1e-6] = 0.0
    n1 = len(np.unique(np.round(k[:, 0], 5)))
    n2 = len(np.unique(np.round(k[:, 1], 5)))
    if n1 * n2 != len(k):
        raise SystemExit(f"{len(k)} k-points are not a regular {n1}x{n2} mesh")
    i1 = np.rint(k[:, 0] * n1).astype(int) % n1
    i2 = np.rint(k[:, 1] * n2).astype(int) % n2
    off = np.abs(np.column_stack([i1 / n1, i2 / n2]) - k)
    off = np.minimum(off, 1.0 - off)
    if off.max() > 1e-5:
        raise SystemExit("the k-points are not on a regular mesh")
    table = -np.ones((n1, n2), dtype=int)
    table[i1, i2] = np.arange(len(k))
    if (table < 0).any():
        raise SystemExit("the k-point mesh has holes or duplicates")
    return n1, n2, table


def _keys(t, a=-0.5):
    """Cubic convolution kernel (Keys 1981): exact at the nodes, partition of unity, continuous
    first derivative."""
    t = np.abs(t)
    return np.where(t <= 1.0, (a + 2) * t ** 3 - (a + 3) * t ** 2 + 1.0,
                    np.where(t < 2.0,
                             a * t ** 3 - 5 * a * t ** 2 + 8 * a * t - 4 * a,
                             0.0))


def interp_matrix(kf, pts):
    """(npts, nk) weights of periodic bicubic interpolation from the regular mesh kf onto the
    reduced points pts."""
    n1, n2, table = grid_lookup(kf)
    p = np.asarray(pts, float)[:, :2]
    u, v = p[:, 0] * n1, p[:, 1] * n2
    i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - i0, v - j0
    M = np.zeros((len(p), len(table.ravel())))
    rows = np.arange(len(p))
    for di in (-1, 0, 1, 2):
        wu = _keys(fu - di)
        for dj in (-1, 0, 1, 2):
            idx = table[(i0 + di) % n1, (j0 + dj) % n2]
            np.add.at(M, (rows, idx), wu * _keys(fv - dj))
    return M


def vbm_points(eps_v, tol=1e-3):
    """Mesh indices where the valence band reaches its maximum (within tol eV)"""
    top = np.asarray(eps_v, float).max(axis=1)
    return np.where(top >= top.max() - tol)[0]


def dispersion_on_path(E1, kf, pts, holes):
    """The exciton band E1(Q) seen along the path, with the hole at each valence maximum."""
    pts = np.asarray(pts, float)
    out = []
    for h in holes:
        Q = np.mod(pts[:, :2] - np.asarray(kf, float)[h, :2][None, :], 1.0)
        out.append(interp_matrix(kf, Q) @ np.asarray(E1, float))
    return np.array(out)


def hole_band_energy(E_ibz, kidx, smap, shifts, Walls, bands, region, frac=0.3):
    """(nk,) lowest exciton energy at every full-zone Q among the states whose HOLE sits near a
    point h in the valence band(s) `bands` (columns of the BSE window; a degenerate pair
    together); NaN where none of the states kept does."""
    n_ibz, nS = E_ibz.shape
    nk = len(kidx)
    H = []
    for j in range(n_ibz):
        Hj = np.zeros((nS, nk))
        for S in range(nS):
            np.add.at(Hj[S], shifts[j],
                      np.asarray(Walls[j][S], float)[:, list(bands)].sum(axis=1))
        H.append(Hj)
    region = np.asarray(region, int)
    out = np.full(nk, np.nan)
    for i in range(nk):
        j = int(kidx[i])
        w = H[j][:, np.asarray(smap[i])[region]].sum(axis=1)
        ok = np.nonzero(w >= frac)[0]
        if len(ok):
            out[i] = float(np.min(np.asarray(E_ibz[j], float)[ok]))
    return out


def curves_on_path(E_k, kf, pts, holes, offsets):
    """E_k[n](p - h_n) + offsets[n] along the path, one row per hole; NaN where the
    interpolation reaches a point without a value (no state found)."""
    pts = np.asarray(pts, float)
    out = []
    for E, h, off in zip(E_k, holes, offsets):
        Q = np.mod(pts[:, :2] - np.asarray(kf, float)[h, :2][None, :], 1.0)
        W = interp_matrix(kf, Q)
        E = np.asarray(E, float)
        good = np.isfinite(E)
        val = W @ np.where(good, E, 0.0)
        bad = np.abs(W) @ (~good).astype(float) > 1e-12
        out.append(np.where(bad, np.nan, val + off))
    return np.array(out)


def hole_points(kf, eps_v, labels, nbands=1, radius=0.12):
    """[(name, mesh index, band columns, offset eV, local maximum?, region)] for each
    high-symmetry label and each of its `nbands` highest valence levels at h (bands within 1
    meV of each other there -- a Kramers or spin-degenerate pair -- count as one level and
    are kept together). offset = eps_v(h, level) - eps_VBM."""
    from .core.mesh import hex_dist
    from .excsym import HIGH_SYMMETRY
    kf = np.asarray(kf, float)
    eps_v = np.asarray(eps_v, float)
    vbm = float(eps_v.max())
    d1 = None
    out = []
    for lab in labels:
        if lab not in HIGH_SYMMETRY:
            raise SystemExit(f"--holes {lab}: not one of {sorted(HIGH_SYMMETRY)}")
        dist = hex_dist(kf[:, :2], np.array(HIGH_SYMMETRY[lab])[None, :])
        h = int(np.argmin(dist))
        if dist[h] > 1e-6:
            raise SystemExit(f"--holes {lab}: not a point of the {len(kf)}-point mesh")
        dd = hex_dist(kf[:, :2], kf[h, :2][None, :])
        if d1 is None:
            d1 = float(np.min(dd[dd > 1e-9]))
        nbr = np.nonzero(np.abs(dd - d1) < 1e-6)[0]
        region = np.nonzero(dd <= radius + 1e-9)[0]
        order = np.argsort(eps_v[h])[::-1]
        levels = []
        for b in order:
            if levels and abs(eps_v[h, b] - eps_v[h, levels[-1][0]]) < 1e-3:
                levels[-1].append(int(b))
            else:
                levels.append([int(b)])
        for r_, cols in enumerate(levels[:nbands]):
            e = float(eps_v[h, cols].max())
            is_max = bool(e > eps_v[np.ix_(nbr, cols)].max() + 1e-6)
            name = lab if r_ == 0 else f"{lab} (v-{r_})"
            out.append((name, h, cols, e - vbm, is_max, region))
    return out


def exciton_spectra(W, H, members, omega, sigma, interp=None):
    """Photoemission of ONE exciton (one irreducible Q, one state) and of all its images over
    the star of Q."""
    omega = np.asarray(omega, float)
    inv = 1.0 / (2.0 * sigma ** 2)
    one = np.zeros((W.shape[0], len(omega)))
    _add_gaussians(one, W, H, omega, inv)
    I_mesh = np.zeros_like(one)
    I_path = None if interp is None else np.zeros((interp.shape[0],
                                                   len(omega)))
    for n, km in members:
        if n <= 0:
            continue
        I_mesh += n * one[km]
        if interp is not None:
            Wp = np.clip(interp @ W[km], 0.0, None)
            Hp = interp @ H[km]
            _add_gaussians(I_path, n * Wp, Hp, omega, inv)
    return I_mesh, I_path


def _add_gaussians(out, W, H, omega, inv, reach=8.0):
    """out[p] += sum_v W[p, v] exp(-(omega - H[p, v])^2 inv), laid down only within `reach`
    standard deviations of each centre (exp(-32) beyond 8)."""
    nE = len(omega)
    d = (omega[-1] - omega[0]) / max(nE - 1, 1)
    if nE < 2 or not np.allclose(np.diff(omega), d, rtol=1e-6, atol=1e-12):
        out += np.einsum("pv,pve->pe", W, np.exp(
            -(omega[None, None, :] - H[:, :, None]) ** 2 * inv))
        return
    half = int(np.ceil(reach / np.sqrt(2.0 * inv) / d))
    c = np.rint((H - omega[0]) / d).astype(int)
    idx = c[..., None] + np.arange(-half, half + 1)
    ok = (idx >= 0) & (idx < nE)
    idc = np.clip(idx, 0, nE - 1)
    val = W[..., None] * np.exp(-(omega[idc] - H[..., None]) ** 2 * inv)
    rows = np.broadcast_to(np.arange(out.shape[0])[:, None, None], idx.shape)
    flat = (rows * nE + idc)[ok]
    out += np.bincount(flat, weights=val[ok],
                       minlength=out.size).reshape(out.shape)


def _snapshot_states(d, kf, nS):
    """Row of kf, 0-based band, and kept mask of every state of a snapshot."""
    windowed = "Q_coarse_red" in d.files
    Qr = np.mod(np.asarray(d["Q_coarse_red"] if windowed else d["Q_red"], float), 1.0)
    band = np.asarray(d["band"]).astype(int)
    if "band_base" in d.files:
        band = band - int(d["band_base"])
    elif band.min() == 1:
        band = band - 1
    from .core.mesh import hex_dist
    idx = np.argmin(hex_dist(np.asarray(kf, float)[None, :, :2], Qr[:, None, :2]), axis=1)
    return idx, band, band < nS, windowed


DEG_TOL = 1e-4


def multiplets_by_k(E_k, tol=DEG_TOL):
    """{k: [index lists]} of the degenerate multiplets (more than one state) at each point of
    the zone, from the energies E_k (nk, nS); see bte.degenerate_multiplets."""
    from .bte import degenerate_multiplets
    out = {}
    if not tol or tol <= 0:
        return out
    E_k = np.asarray(E_k, float)
    for k in range(len(E_k)):
        gs = [g for g in degenerate_multiplets(E_k[k], tol) if len(g) > 1]
        if gs:
            out[k] = gs
    return out


def populations(kf, kidx, E_ibz, T=None, snaps=None, time=None,
                return_total=False, deg_tol=DEG_TOL):
    """N[i, S] on the full zone: thermal at T, or a transport snapshot, normalised to a sum of
    one."""
    nk, nS = len(kf), E_ibz.shape[1]
    N = np.zeros((nk, nS))
    if snaps is None:
        kT = 8.617333262e-5 * float(T)
        Ef = E_ibz[kidx]
        N = np.exp(-(Ef - Ef.min()) / max(kT, 1e-12))
        return (N / N.sum(), None) if return_total else N / N.sum()
    d = np.load(snaps)
    it = int(np.argmin(np.abs(d["t"] - time)))
    F = d["F"][it]
    idx, band, keep, windowed = _snapshot_states(d, kf, nS)
    np.add.at(N, (idx[keep], band[keep]), F[keep])
    man = multiplets_by_k(np.asarray(E_ibz, float)[kidx][:, :nS], deg_tol)
    for k, gs in man.items():
        for g in gs:
            N[k, g] = N[k, g].mean()
    print(f"   snapshot t = {d['t'][it]:.1f} fs: "
          f"{keep.sum()} of {len(F)} states within the {nS} kept per Q"
          + (" (refined states summed at their parent BSE-mesh point)" if windowed else "")
          + (f"; shared within {sum(len(g) for g in man.values())} degenerate multiplets "
             f"at {len(man)} k" if man else ""))
    tot = float(N.sum())
    if return_total:
        lost = float(F[~keep].sum())
        if lost > 1e-6 * max(float(F.sum()), 1e-300):
            print(f"   {lost / F.sum() * 100:.2f}% of the excitons sit in states "
                  f"beyond the {nS} kept per Q and are left out (raise --nstates)")
        if windowed and "coarse_mesh" in d.files and "mesh" in d.files:
            m, c = np.asarray(d["mesh"]).ravel(), np.asarray(d["coarse_mesh"]).ravel()
            tot *= float(c[0] * c[1]) / float(m[0] * m[1])
        return N / max(float(N.sum()), 1e-300), tot
    return N / max(tot, 1e-300)


def lorentz_convolve(I, omega, fwhm):
    """I (..., ne) convolved along the last axis with a normalised Lorentzian of full width fwhm
    (eV) on the uniform grid omega."""
    omega = np.asarray(omega, float)
    d = float(omega[1] - omega[0])
    if fwhm < 0.5 * d:
        return np.array(I, copy=True)
    n = len(omega)
    x = np.arange(-(n - 1), n) * d
    L = (fwhm / (2 * np.pi)) / (x ** 2 + (fwhm / 2) ** 2)
    L /= L.sum()
    from scipy.signal import fftconvolve
    shape = (1,) * (np.ndim(I) - 1) + (len(L),)
    return fftconvolve(I, L.reshape(shape), mode="same", axes=-1)


def exph_fwhm(kf, nS, snaps, time, mode="full", E_k=None, deg_tol=DEG_TOL):
    """(nk, nS) exciton-phonon linewidth (FWHM, eV) of every state, from a snapshot of `xphd
    bte`, placed as `populations` places the occupations."""
    d = np.load(snaps)
    key = "fwhm_exph_t" if mode == "full" else "fwhm_exph0"
    if key not in d.files:
        raise SystemExit(f"--eta-exph: {snaps} carries no {key}. Snapshots from xphd bte "
                         f"before 2.0.0.dev0 (22 Oct 2026) do not save the linewidths: "
                         f"rerun xphd bte with the same arguments")
    it = int(np.argmin(np.abs(d["t"] - time)))
    g = np.asarray(d[key][it] if mode == "full" else d[key], float)
    idx, band, keep, _ = _snapshot_states(d, kf, nS)
    at = (idx[keep], band[keep])
    G = np.zeros((len(kf), nS))
    cnt = np.zeros_like(G)
    np.add.at(cnt, at, 1.0)
    man = multiplets_by_k(E_k, deg_tol) if E_k is not None else {}
    if cnt.max() <= 1 and not man:
        G[at] = g[keep]
    else:
        w = np.asarray(d["F"][it], float)[keep]
        num, den, plain = np.zeros_like(G), np.zeros_like(G), np.zeros_like(G)
        np.add.at(num, at, w * g[keep])
        np.add.at(den, at, w)
        np.add.at(plain, at, g[keep])
        for k, gs in man.items():
            for grp in gs:
                for arr in (num, den, plain, cnt):
                    arr[k, grp] = arr[k, grp].sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            G = np.where(den > 0, num / np.where(den > 0, den, 1.0),
                         np.where(cnt > 0, plain / np.maximum(cnt, 1.0), 0.0))
    neg = int((G < 0).sum())
    return np.clip(G, 0.0, None), neg


def _axpy(w, x, y):
    """y += w x in place, without the temporary w * x (BLAS daxpy when the arrays are contiguous
    float64)."""
    try:
        from scipy.linalg.blas import daxpy
    except ImportError:
        daxpy = None
    xc = np.ascontiguousarray(x, dtype=float)
    if daxpy is not None and y.flags.c_contiguous and y.dtype == np.float64:
        yv = y.reshape(-1)
        res = daxpy(xc.reshape(-1), yv, a=float(w))
        if not np.shares_memory(res, y):
            y[...] = res.reshape(y.shape)
    else:
        y += w * xc


class LorentzBins:
    """Sum of spectra, each convolved along the energy with a Lorentzian of its own FWHM, at the
    cost of one convolution per width bin rather than one per spectrum."""

    def __init__(self, omega, shapes, ratio=1.1):
        omega = np.asarray(omega, float)
        self.omega = omega
        self.g0 = 0.5 * float(omega[1] - omega[0])
        self.lr = float(np.log(ratio))
        self.ratio = float(ratio)
        self.total = [np.zeros(sh) for sh in shapes]
        self.open = {}
        self.last = -np.inf
        self.nconv = 0

    def _flush(self, below):
        for b in sorted(k for k in self.open if k < below):
            g = self.g0 * self.ratio ** b
            for tot, x in zip(self.total, self.open.pop(b)):
                tot += lorentz_convolve(x, self.omega, g)
            self.nconv += 1

    def _put(self, b, w, parts):
        if w <= 0:
            return
        if b not in self.open:
            self.open[b] = [np.zeros_like(t) for t in self.total]
        for acc, x in zip(self.open[b], parts):
            _axpy(w, x, acc)

    def add(self, g, parts):
        g = float(g)
        if g < self.last * (1 - 1e-12):
            raise ValueError("LorentzBins: add the spectra in ascending width")
        self.last = g
        if g < self.g0:
            for tot, x in zip(self.total, parts):
                tot += x
            return
        x = np.log(g / self.g0) / self.lr
        b = int(np.floor(x + 1e-12))
        al = min(max(x - b, 0.0), 1.0)
        self._flush(b)
        self._put(b, 1.0 - al, parts)
        self._put(b + 1, al, parts)

    def finish(self):
        self._flush(np.inf)
        return self.total


def carrier_occupations(Wv, Wc, kplusQ, members, nk=None):
    """Hole and electron occupations left by ONE exciton state at one irreducible Q and by all
    its images over the star."""
    Wv, Wc = np.asarray(Wv, float), np.asarray(Wc, float)
    nk = Wv.shape[0] if nk is None else nk
    fh = np.zeros((nk, Wv.shape[1]))
    fe = np.zeros((nk, Wc.shape[1]))
    kplusQ = np.asarray(kplusQ, int)
    for n, km in members:
        if n <= 0:
            continue
        km = np.asarray(km, int)
        fwd = np.empty_like(km)
        fwd[km] = np.arange(len(km))
        fe += n * Wc[km]
        np.add.at(fh, fwd[kplusQ], n * Wv)
    return fh, fe


def valence_photoemission(eps, fh, omega, sigma, g_s=2.0):
    """(I, I0, dI), each (np, nE): the valence photoemission with the holes fh taken out, at
    equilibrium, and their difference (-sum_v fh G)."""
    eps = np.asarray(eps, float)
    fh = np.clip(np.asarray(fh, float), 0.0, g_s)
    omega = np.asarray(omega, float)
    inv = 1.0 / (2.0 * sigma ** 2)
    I0 = np.zeros((eps.shape[0], len(omega)))
    dep = np.zeros_like(I0)
    _add_gaussians(I0, np.full_like(eps, g_s), eps, omega, inv)
    _add_gaussians(dep, fh, eps, omega, inv)
    return I0 - dep, I0, -dep


def cell_area_nm2(ylat):
    """|a1 x a2| of the SAVE's lattice in nm^2 (yambo's lat is in bohr), or None if the lattice
    carries no vectors."""
    lat = getattr(ylat, "lat", None)
    if lat is None:
        return None
    lat = np.asarray(lat, float)
    return float(np.linalg.norm(np.cross(lat[0], lat[1]))) * 0.052917721067 ** 2


def parse_times(tokens, available):
    """Delays for --times: numbers, "all" (every snapshot), or a:b[:s] ranges (fs, step 1 by
    default)."""
    av = np.asarray(available, float)
    req = []
    for tok in tokens:
        tok = str(tok).strip()
        if tok == "all":
            req += list(av)
        elif ":" in tok:
            parts = [float(x) for x in tok.split(":")]
            if len(parts) not in (2, 3):
                raise ValueError(f"--times {tok!r}: use a:b or a:b:s")
            step = parts[2] if len(parts) == 3 else 1.0
            if step <= 0:
                raise ValueError(f"--times {tok!r}: the step must be positive")
            req += list(np.arange(parts[0], parts[1] + 0.5 * step, step))
        else:
            req.append(float(tok))
    out, seen = [], set()
    for t in req:
        i = int(np.argmin(np.abs(av - t)))
        if i not in seen:
            seen.add(i)
            out.append(float(av[i]))
    return out


def box_points(rlat, label, radius, n_across=21):
    """Reduced points (n, 3) of a square grid inside the disc of radius `radius` (1/Angstrom)
    about the high-symmetry point `label`."""
    from .excsym import HIGH_SYMMETRY
    B = 2.0 * np.pi * np.asarray(rlat, float)[:2, :2] / 0.529177210903
    c = np.asarray(HIGH_SYMMETRY[label], float) @ B
    g = np.linspace(-radius, radius, n_across)
    X, Y = np.meshgrid(g, g)
    keep = X ** 2 + Y ** 2 <= radius ** 2 * (1 + 1e-12)
    cart = np.column_stack([X[keep], Y[keep]]) + c
    red = cart @ np.linalg.inv(B)
    return np.column_stack([red, np.zeros(len(red))])


def occupied_dos(E, N, omega, sigma):
    """sum_i N_i g(w - E_i) with g a normalised Gaussian of width sigma: the occupied-exciton
    density of states on the grid omega (integral sum N)."""
    E = np.asarray(E, float).reshape(1, -1)
    N = np.asarray(N, float).reshape(1, -1)
    out = np.zeros((1, len(omega)))
    _add_gaussians(out, N, E, np.asarray(omega, float), 1.0 / (2.0 * sigma ** 2))
    return out[0] / (sigma * np.sqrt(2.0 * np.pi))


def path_indices(kf, labels, npts=120):
    """Nearest mesh point along a path of high-symmetry labels, each segment the short way
    (_path_vertices, hexagonal metric)."""
    pts = _path_vertices(labels, _HEX_B)
    seg, ticks, x = [], [0], 0
    for a0, a1 in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, npts, endpoint=False):
            seg.append(a0 + t * (a1 - a0))
        x += npts
        ticks.append(x)
    seg.append(pts[-1])
    seg = np.array(seg)
    from .core.mesh import hex_dist
    d = hex_dist(kf[None, :, :2], seg[:, None, :2])
    return np.argmin(d, axis=1), ticks


def main(argv=None):
    import argparse
    from .excsym import shift_map
    from .yambo import full_zone_kpoints, ibz_parents, lattice, load_excitons

    p = argparse.ArgumentParser(prog="xphd arpes",
                                description="exciton photoemission I(k, E)")
    p.add_argument("--save", default="../phonons/SAVE")
    p.add_argument("--bse-dir", default="../BSE/output_all")
    p.add_argument("--nstates", type=int, default=15,
                   help="exciton states per Q")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--temperature", type=float,
                   help="K; quasi-equilibrium Boltzmann occupations")
    g.add_argument("--snapshots",
                   help="snaps.npz from `xphd bte`; use with --time")
    p.add_argument("--time", type=float, default=0.0,
                   help="fs; the snapshot nearest this time is used")
    p.add_argument("--emin", type=float, default=None,
                   help="eV, on the valence energies' reference (default: automatic)")
    p.add_argument("--emax", type=float, default=None)
    p.add_argument("--ne", type=int, default=800)
    p.add_argument("--sigma", type=float, default=0.05,
                   help="eV, energy resolution")
    p.add_argument("--valence-shift", type=float, default=0.0,
                   help="eV added to the Kohn-Sham valence energies")
    p.add_argument("--valence-npy", default=None,
                   help="full-zone valence energies (nk, nv) in eV, used "
                        "instead of ns.db1")
    p.add_argument("--qp-db", default=None,
                   help="the GW ndb.QP: quasiparticle corrections added to "
                        "the Kohn-Sham valence energies, as the GW-BSE "
                        "exciton energies require")
    p.add_argument("--path", nargs="*", default=["G", "M", "K", "G"],
                   help="high-symmetry labels (G M K K' M' M'' Q Q'), each segment "
                        "taken the short way: M K G K' M is the straight line "
                        "through Gamma, as in valley-resolved figures")
    p.add_argument("--fine", type=int, default=600,
                   help="points on the path, evenly spaced in length, with "
                        "the envelopes and energies interpolated between "
                        "mesh points; 0 keeps the nearest-mesh-point path only")
    p.add_argument("--min-occ", type=float, default=1e-6,
                   help="skip excitons with occupation below this fraction "
                        "of the largest")
    p.add_argument("--lorentz-fwhm", type=float, default=0.0,
                   help="eV: convolve every line (excitons and valence band) with a "
                        "Lorentzian of this FWHM, the line shape of a damping as in "
                        "yambo; keep --sigma well below it (e.g. 0.005)")
    p.add_argument("--eta-exph", nargs="?", const="full", default=None,
                   choices=("full", "dilute"),
                   help="give every exciton line a Lorentzian of its own exciton-phonon "
                        "linewidth, from the snapshot (xphd bte saves it): 'full' (default) "
                        "with the stimulated factors at that delay, as Chen, Sangalli and "
                        "Bernardi (PRR 4, 043203) set their eta; 'dilute' the F -> 0 width. "
                        "On top of --sigma; --lorentz-fwhm adds to it")
    p.add_argument("--eta-scale", type=float, default=1.0,
                   help="multiplies the --eta-exph widths (default 1: FWHM = hbar/tau; "
                        "2 reads 'eta = Gamma' in 1/(w - E + i eta) as a half width)")
    p.add_argument("--deg-tol", type=float, default=DEG_TOL,
                   help="eV: with --snapshots, states of one Q closer than this share "
                        "their occupations (and --eta-exph widths) equally, since the "
                        "BSE solver's basis inside a degenerate multiplet is arbitrary "
                        "(0: off)")
    p.add_argument("--density", type=float, default=None,
                   help="exciton density, cm^-2, for --valence with --temperature")
    p.add_argument("--valence", action="store_true",
                   help="also write the photoemission of the valence band, with the "
                        "holes the excitons leave taken out of it (needs the number of "
                        "excitons: the snapshot's own, or --density with --temperature)")
    p.add_argument("--vb-window", type=float, nargs=2, default=(-3.0, 0.5),
                   metavar=("LO", "HI"),
                   help="eV relative to the valence maximum: the energy range of the "
                        "valence map (default -3 0.5)")
    p.add_argument("--vb-ne", type=int, default=400,
                   help="energies of the valence map")
    p.add_argument("--spin-deg", type=float, default=2.0,
                   help="electrons per valence state (2: spin-degenerate)")
    p.add_argument("--cell-area", type=float, default=None,
                   help="nm^2, for --valence with --temperature (default: from the SAVE)")
    p.add_argument("--times", nargs="+", default=None, metavar="T",
                   help="several delays in one run (with --snapshots): numbers, 'all', or "
                        "a:b[:s] ranges, fs. -o names one map per delay with {t}, e.g. "
                        "77K/arpes_{t}fs.npz")
    p.add_argument("--series", default=None, metavar="FILE",
                   help="with --times: the spectra against the delay (k-integrated, "
                        "in --box, occupied-exciton DOS); default arpes_series.npz "
                        "beside the maps")
    p.add_argument("--no-maps", action="store_true",
                   help="with --times: write the series only")
    p.add_argument("--holes", nargs="+", default=None, metavar="POINT",
                   help="also save the replica edge with the hole at these high-symmetry "
                        "points (G K K' M ...): E(Q) + eps_v(point) - eps_VBM along the "
                        "path, E(Q) the lowest exciton whose hole sits at the point; "
                        "plot_arpes.py draws them. Needs --fine > 0")
    p.add_argument("--hole-bands", type=int, default=1,
                   help="with --holes: how many valence levels at each point, from the "
                        "top (2 at K: both spin-split bands)")
    p.add_argument("--hole-radius", type=float, default=0.12,
                   help="with --holes: |b| units; the hole must sit this close to the point")
    p.add_argument("--hole-frac", type=float, default=0.3,
                   help="with --holes: the share of the hole weight that must sit there")
    p.add_argument("--box", nargs=2, action="append", default=None,
                   metavar=("POINT", "RADIUS"),
                   help="also integrate the photoemission over a disc of RADIUS "
                        "(1/Angstrom) about POINT (G, K, M, ...), e.g. --box G 0.12; repeat "
                        "for several")
    p.add_argument("-o", "--out", default="arpes.npz")
    a = p.parse_args(argv)
    if a.sigma <= 0:
        raise SystemExit("--sigma must be positive; for a Lorentzian line shape give "
                         "--lorentz-fwhm with a small --sigma under it")
    if a.lorentz_fwhm < 0:
        raise SystemExit("--lorentz-fwhm must be positive (or 0: none)")
    if a.eta_exph and not a.snapshots:
        raise SystemExit("--eta-exph takes the linewidths from a snapshot of xphd bte: "
                         "give --snapshots")
    if a.eta_scale <= 0:
        raise SystemExit("--eta-scale must be positive")

    ylat = lattice(a.save)
    kf = full_zone_kpoints(ylat)
    kidx, first = ibz_parents(ylat)
    n_ibz = len(first)
    print(f"{len(kf)} k-points, {n_ibz} irreducible")

    ops = star_kmaps(a.save, kf)
    smap = star_assignment(kf, kidx, first, ops)
    print(f"{len(ops)} operations (spatial and time-reversed); every "
          f"full-zone point reached from its representative")

    A_ibz, E_ibz, vb = [], [], None
    for j in range(n_ibz):
        exc, A, E = load_excitons(f"{a.bse_dir}/ndb.BS_diago_Q{j + 1}",
                                  ylat, a.nstates)
        A_ibz.append(A)
        E_ibz.append(E[: a.nstates])
        if vb is None:
            vb = np.array(sorted(set(np.asarray(exc.unique_vbands).ravel())),
                          dtype=int)
    E_ibz = np.array(E_ibz)

    if a.valence_npy:
        eps_v = np.load(a.valence_npy)
    else:
        eps_v = valence_energies(a.save, ylat, vb, a.valence_shift,
                                 qp_db=a.qp_db)
    print(f"valence bands (0-based) {list(vb)}: eps_v {eps_v.shape}, "
          f"{eps_v.min():.3f} .. {eps_v.max():.3f} eV")
    if eps_v.shape != (len(kf), A_ibz[0].shape[2]):
        raise SystemExit(f"eps_v {eps_v.shape} does not match "
                         f"({len(kf)}, {A_ibz[0].shape[2]})")

    import os
    times = [a.time]
    if a.times:
        if not a.snapshots:
            raise SystemExit("--times needs --snapshots")
        with np.load(a.snapshots) as z:
            avail = np.asarray(z["t"], float)
        try:
            times = parse_times(a.times, avail)
        except ValueError as e:
            raise SystemExit(str(e))
        print(f"--times: {len(times)} delays, {times[0]:g} .. {times[-1]:g} fs "
              f"(each the nearest snapshot)")
    many = len(times) > 1
    if many and not a.no_maps and "{t}" not in a.out:
        raise SystemExit("several delays: put {t} in -o for one map per delay "
                         "(e.g. -o 77K/arpes_{t}fs.npz), or give --no-maps")
    series_path = a.series
    if many and series_path is None:
        series_path = os.path.join(os.path.dirname(a.out.replace("{t}", "0")) or ".",
                                   "arpes_series.npz")

    def out_name(t):
        return a.out.replace("{t}", f"{t:g}")

    lo = float(E_ibz.min() + eps_v.min()) - 0.5
    hi = float(E_ibz.max() + eps_v.max()) + 0.5
    omega = np.linspace(a.emin if a.emin is not None else lo,
                        a.emax if a.emax is not None else hi, a.ne)
    omega_x = np.linspace(float(E_ibz.min()) - 0.3, float(E_ibz.max()) + 0.3, a.ne)
    E_full = E_ibz[kidx]
    dos_all = occupied_dos(E_full, np.full(E_full.shape, 1.0 / len(kf)), omega_x, a.sigma)

    from .excsym import HIGH_SYMMETRY as _HS
    bad = [l for l in a.path if l not in _HS]
    if bad or len(a.path) < 2:
        raise SystemExit(f"--path: {bad or a.path} -- give two or more of {sorted(_HS)}")
    interp = xf = xt = pts = None
    if a.fine > 0:
        pts, xf, xt = fine_path(a.path, np.asarray(ylat.rlat), a.fine)
        interp = interp_matrix(kf, pts)
        print(f"path {' '.join(a.path)}: {len(pts)} points, envelopes and "
              f"energies interpolated between mesh points")
    boxes = []
    for lab, rad in (a.box or []):
        from .excsym import HIGH_SYMMETRY
        if lab not in HIGH_SYMMETRY:
            raise SystemExit(f"--box {lab}: not one of {sorted(HIGH_SYMMETRY)}")
        bp = box_points(np.asarray(ylat.rlat), lab, float(rad))
        boxes.append((lab, float(rad), interp_matrix(kf, bp)))
        print(f"--box {lab} {float(rad):g} 1/A: {len(bp)} points inside the disc")

    pidx, ticks = path_indices(kf, a.path)
    vbm = float(eps_v.max())
    E_path = None if interp is None else interp @ E_full
    nv_, nc_ = A_ibz[0].shape[2], A_ibz[0].shape[3]
    T_map = a.temperature if a.temperature is not None else float(np.load(a.snapshots)["T"])
    inv = 1.0 / (2.0 * a.sigma ** 2)

    shifts, Walls = [], []
    for j in range(n_ibz):
        kplusQ, w = shift_map(kf, kf[first[j]])
        if kplusQ is None:
            raise SystemExit(f"IBZ {j}: k+Q leaves the grid ({w:.1e})")
        shifts.append(kplusQ)
        Walls.append(np.sum(np.abs(A_ibz[j][: a.nstates]) ** 2, axis=3))

    hole_out = {}
    if a.holes:
        if interp is None:
            raise SystemExit("--holes draws on the smooth path: needs --fine > 0")
        hp = hole_points(kf, eps_v, a.holes, a.hole_bands, a.hole_radius)
        E_h, names, rows = [], [], []
        print(f"hole curves (--holes): states whose hole holds >= {a.hole_frac:g} of its "
              f"weight within {a.hole_radius:g} |b| of the point")
        for name, h, cols, off, is_max, region in hp:
            Eh = hole_band_energy(E_ibz[:, : a.nstates], kidx, smap, shifts, Walls, cols,
                                  region, a.hole_frac)
            E_h.append(Eh)
            names.append(name)
            rows.append((h, cols, off, is_max, float(np.isfinite(Eh).mean())))
            print(f"   hole at {name:<10} k = {tuple(np.round(kf[h, :2], 4))}, band(s) "
                  f"{[int(vb[c]) + 1 for c in cols]}: {off * 1e3:+8.1f} meV from the VBM, "
                  + ("a local maximum (the curve is that replica's edge)" if is_max
                     else "NOT a local maximum (a guide only)")
                  + f"; a state found at {100 * rows[-1][4]:.0f}% of Q"
                  + ("  <-- few: raise --nstates" if rows[-1][4] < 0.5 else ""))
        curves = curves_on_path(E_h, kf, pts, [r_[0] for r_ in rows], [r_[2] for r_ in rows])
        hole_out = dict(hole_curves=curves, hole_labels=np.array(names),
                        hole_k=kf[[r_[0] for r_ in rows], :2],
                        hole_offset=np.array([r_[2] for r_ in rows]),
                        hole_local_max=np.array([r_[3] for r_ in rows]),
                        hole_found=np.array([r_[4] for r_ in rows]),
                        hole_frac=np.array(a.hole_frac), hole_radius=np.array(a.hole_radius))

    ser = dict(edc_zone=[], edc_box=[], dos_occ=[], E_mean=[], N_tot=[], fw_mean=[])
    for it, t in enumerate(times):
        verbose = it == 0 or not many
        N, N_snap = populations(kf, kidx, E_ibz, T=a.temperature, snaps=a.snapshots,
                                time=t, return_total=True, deg_tol=a.deg_tol)
        N_tot = None
        if a.valence:
            area = a.cell_area if a.cell_area is not None else cell_area_nm2(ylat)
            if a.snapshots:
                N_tot = N_snap
                src = "the snapshot's occupations"
            else:
                if a.density is None:
                    raise SystemExit("--valence with --temperature needs --density (cm^-2): "
                                     "thermal occupations carry no number of excitons")
                if area is None:
                    raise SystemExit("--valence: the SAVE carries no lattice vectors; give "
                                     "--cell-area (nm^2)")
                N_tot = a.density * 1e-14 * area * len(kf)
                src = f"--density {a.density:.3e} cm^-2"
            if verbose:
                msg = (f"valence band: {N_tot:.4e} excitons in the {len(kf)}-cell supercell "
                       f"({src})")
                if area is not None:
                    msg += (f", n = {N_tot / (len(kf) * area) * 1e14:.3e} cm^-2 "
                            f"(A_cell {area:.5f} nm^2)")
                print(msg)

        nmax = float(N.max())
        fh = np.zeros((len(kf), nv_))
        fe = np.zeros((len(kf), nc_))
        I = np.zeros((len(kf), len(omega)))
        I_fine = None if interp is None else np.zeros((len(pts), len(omega)))
        I_box = [np.zeros((B.shape[0], len(omega))) for _, _, B in boxes]
        G = None
        if a.eta_exph:
            G, nneg = exph_fwhm(kf, a.nstates, a.snapshots, t, a.eta_exph,
                                E_k=np.asarray(E_ibz, float)[kidx][:, :a.nstates],
                                deg_tol=a.deg_tol)
            G *= a.eta_scale
        used = skipped = 0
        tasks, spread = [], 0.0
        for j in range(n_ibz):
            members = np.where(kidx == j)[0]
            wj = N[members]
            if wj.sum() <= 0:
                continue
            for S in range(a.nstates):
                keep_m = [(int(i), float(n)) for i, n in zip(members, wj[:, S])
                          if n > a.min_occ * nmax]
                skipped += int(np.sum((wj[:, S] > 0)
                                      & (wj[:, S] <= a.min_occ * nmax)))
                if not keep_m:
                    continue
                used += len(keep_m)
                gam = 0.0
                if G is not None:
                    nn = np.array([n for _, n in keep_m])
                    gg = G[[i for i, _ in keep_m], S]
                    gam = float(nn @ gg / nn.sum())
                    if gam > 0:
                        spread = max(spread, float((gg.max() - gg.min()) / gam))
                tasks.append((gam, j, S, [(n, smap[i]) for i, n in keep_m]))
        if G is not None:
            tasks.sort(key=lambda x: x[0])
            shapes = [I.shape] + ([I_fine.shape] if I_fine is not None else []) \
                + [Ib.shape for Ib in I_box]
            acc = LorentzBins(omega, shapes)
        for gam, j, S, mem in tasks:
            kplusQ, Wall = shifts[j], Walls[j]
            H = E_ibz[j][S] + eps_v[kplusQ]
            if a.valence:
                Wc = np.sum(np.abs(A_ibz[j][S]) ** 2, axis=1)
                dh, de = carrier_occupations(Wall[S], Wc, kplusQ, mem, len(kf))
                fh += dh
                fe += de
            Im, Ip = exciton_spectra(Wall[S], H, mem, omega, a.sigma, interp)
            Ibs = [np.zeros_like(Ib) for Ib in I_box]
            for (_, _, B), Ib in zip(boxes, Ibs):
                for n, km in mem:
                    Wp = np.clip(B @ Wall[S][km], 0.0, None)
                    _add_gaussians(Ib, n * Wp, B @ H[km], omega, inv)
            if G is None:
                I += Im
                if Ip is not None:
                    I_fine += Ip
                for Ib, x in zip(I_box, Ibs):
                    Ib += x
            else:
                acc.add(gam, [Im] + ([Ip] if Ip is not None else []) + Ibs)
        fw_mean = np.nan
        if G is not None:
            tot = acc.finish()
            I = tot[0]
            if I_fine is not None:
                I_fine = tot[1]
            I_box = tot[1 + (I_fine is not None):]
            fw_mean = float((N * G).sum() / max(N.sum(), 1e-300))
            if verbose:
                print(f"exciton-phonon line widths (--eta-exph {a.eta_exph}"
                      + (f", x {a.eta_scale:g}" if a.eta_scale != 1 else "")
                      + f"): Lorentzian FWHM {G.min() * 1e3:.3g} .. {G.max() * 1e3:.3g} meV, "
                      f"occupation-weighted {fw_mean * 1e3:.3g} meV, on top of sigma "
                      f"{a.sigma * 1e3:.3g} meV ({acc.nconv} width bins)")
                if nneg:
                    print(f"   {nneg} negative widths (stimulated gain) clipped to 0")
                if spread > 1e-3:
                    print(f"   widths differ within a star by up to {spread * 100:.2g}%: "
                          f"each star takes the occupation-weighted mean")
        if verbose:
            print(f"{used} occupied excitons included, {skipped} below "
                  f"--min-occ {a.min_occ:g} of the largest skipped")
        widths = [a.lorentz_fwhm] if a.lorentz_fwhm > 0 else []
        if a.lorentz_fwhm > 0 and verbose:
            print(f"Lorentzian line shape: FWHM {a.lorentz_fwhm * 1e3:.1f} meV on every line "
                  f"(Gaussian --sigma {a.sigma * 1e3:.1f} meV under it)"
                  + (" (below half the energy step: no effect)"
                     if a.lorentz_fwhm < 0.5 * (omega[1] - omega[0]) else ""))
        for w_ in widths:
            I = lorentz_convolve(I, omega, w_)
            if I_fine is not None:
                I_fine = lorentz_convolve(I_fine, omega, w_)
            I_box = [lorentz_convolve(Ib, omega, w_) for Ib in I_box]

        edc_zone = I.mean(axis=0)
        edc_box = np.array([Ib.mean(axis=0) for Ib in I_box]) if boxes else np.zeros((0, len(omega)))
        dos_occ = occupied_dos(E_full, N, omega_x, a.sigma)
        E_mean = float((N * E_full).sum() / max(N.sum(), 1e-300))
        ser["edc_zone"].append(edc_zone)
        ser["edc_box"].append(edc_box)
        ser["dos_occ"].append(dos_occ)
        ser["E_mean"].append(E_mean)
        ser["N_tot"].append(np.nan if N_snap is None else N_snap)
        ser["fw_mean"].append(fw_mean)
        if many and not verbose:
            k = int(np.argmax(edc_zone))
            print(f"   t = {t:8.2f} fs: <E_X> = {E_mean:.4f} eV, k-integrated peak at "
                  f"{omega[k] - vbm:+.3f} eV from the VBM"
                  + ("" if G is None else f", <FWHM> {fw_mean * 1e3:.3g} meV"))
        if a.no_maps:
            continue

        norm = max(float(I.max()), 1e-300)
        I = I / norm
        out = dict(I=I, omega=omega, kf=kf, I_path=I[pidx],
                   path=np.array(a.path), ticks=np.array(ticks),
                   T=np.array(a.temperature if a.temperature else np.nan),
                   time=np.array(t), I_norm=np.array(norm),
                   vbm=np.array(vbm), E_exc_min=np.array(float(E_ibz.min())),
                   qp=np.array(bool(a.qp_db) or bool(a.valence_npy)),
                   sigma=np.array(a.sigma),
                   lorentz_fwhm=np.array(a.lorentz_fwhm),
                   edc_zone=edc_zone, edc_box=edc_box,
                   box_label=np.array([b[0] for b in boxes]),
                   box_radius=np.array([b[1] for b in boxes]),
                   omega_x=omega_x, dos_occ=dos_occ, dos_all=dos_all,
                   E_mean=np.array(E_mean),
                   eta_exph=np.array(a.eta_exph or ""), eta_scale=np.array(a.eta_scale),
                   fwhm_exph=np.zeros((0, 0)) if G is None else G,
                   fwhm_exph_mean=np.array(fw_mean))
        if I_fine is not None:
            out.update(I_fine=I_fine / norm, x_fine=xf, xt_fine=xt, k_fine=pts,
                       E_path=E_path, N_path=np.clip(interp @ N, 0.0, None))
            holes = vbm_points(eps_v)
            E1 = E_ibz.min(axis=1)[kidx]
            disp = dispersion_on_path(E1, kf, pts, holes)
            out.update(E_disp=disp.min(axis=0), E_disp_valleys=disp,
                       vbm_k=kf[holes, :2], **hole_out)
            if verbose:
                print(f"exciton band E1(Q) on the path, hole at the valence maximum "
                      f"(k = {[tuple(np.round(kf[h, :2], 4)) for h in holes]}): "
                      f"{disp.min():.3f} .. {disp.max():.3f} eV above the valence maximum; "
                      f"saved as E_disp (on the omega axis: E_disp + vbm)")
        if a.valence:
            fh *= N_tot
            fe *= N_tot
            omega_vb = np.linspace(vbm + a.vb_window[0], vbm + a.vb_window[1], a.vb_ne)
            Ivb, Ivb0, dIvb = valence_photoemission(eps_v[pidx], fh[pidx], omega_vb,
                                                    a.sigma, a.spin_deg)
            if a.lorentz_fwhm > 0:
                Ivb, Ivb0, dIvb = (lorentz_convolve(x, omega_vb, a.lorentz_fwhm)
                                   for x in (Ivb, Ivb0, dIvb))
            out.update(fh=fh, fe=fe, N_tot=np.array(N_tot), spin_deg=np.array(a.spin_deg),
                       omega_vb=omega_vb, I_vb_path=Ivb, I_vb0_path=Ivb0, dI_vb_path=dIvb,
                       I_abs_scale=np.array(norm * N_tot))
            if interp is not None:
                ep = interp @ eps_v
                hp = np.clip(interp @ fh, 0.0, None)
                Ivf, Iv0f, dIvf = valence_photoemission(ep, hp, omega_vb, a.sigma, a.spin_deg)
                if a.lorentz_fwhm > 0:
                    Ivf, Iv0f, dIvf = (lorentz_convolve(x, omega_vb, a.lorentz_fwhm)
                                       for x in (Ivf, Iv0f, dIvf))
                out.update(I_vb_fine=Ivf, I_vb0_fine=Iv0f, dI_vb_fine=dIvf)
            if verbose:
                kh, vh = np.unravel_index(np.argmax(fh), fh.shape)
                print(f"valence band: sum f_h = {fh.sum():.4e}, sum f_e = {fe.sum():.4e} "
                      f"(= the {N_tot:.4e} excitons, less any below --min-occ)")
                print(f"   largest hole occupation {fh.max():.3e} at k = {np.round(kf[kh, :2], 4)}, "
                      f"valence band {vb[vh] + 1}: the band there is emptied by "
                      f"{fh.max() / a.spin_deg * 100:.3f}%")
                rep = float(I.max()) * norm * N_tot
                print(f"   exciton replica peak / full valence peak = {rep / a.spin_deg:.3e} "
                      f"(same matrix element; I x I_abs_scale is on the valence map's scale)")
        path_out = out_name(t)
        if os.path.dirname(path_out):
            os.makedirs(os.path.dirname(path_out), exist_ok=True)
        np.savez(path_out, **out)
        if verbose:
            ip, ie = np.unravel_index(np.argmax(I), I.shape)
            print(f"\nwrote {path_out}: I {I.shape} (k, E), path {I[pidx].shape}"
                  + ("" if I_fine is None else f", fine path {I_fine.shape}"))
            print(f"   brightest feature: E = {omega[ie]:.3f} eV at k = "
                  f"{np.round(kf[ip, :2], 4)}; {omega[ie] - vbm:+.3f} eV from the "
                  f"valence maximum ({vbm:.3f} eV, saved as vbm)")

    if series_path:
        if os.path.dirname(series_path):
            os.makedirs(os.path.dirname(series_path), exist_ok=True)
        np.savez(series_path, times=np.array(times), omega=omega, omega_x=omega_x,
                 edc_zone_t=np.array(ser["edc_zone"]), edc_box_t=np.array(ser["edc_box"]),
                 dos_occ_t=np.array(ser["dos_occ"]), dos_all=dos_all,
                 E_mean_t=np.array(ser["E_mean"]), N_tot_t=np.array(ser["N_tot"]),
                 fwhm_exph_mean_t=np.array(ser["fw_mean"]),
                 eta_exph=np.array(a.eta_exph or ""), eta_scale=np.array(a.eta_scale),
                 box_label=np.array([b[0] for b in boxes]),
                 box_radius=np.array([b[1] for b in boxes]),
                 vbm=np.array(vbm), E_exc_min=np.array(float(E_ibz.min())),
                 T=np.array(T_map), sigma=np.array(a.sigma),
                 path=np.array(a.path),
                 maps=np.array([] if a.no_maps else [out_name(t) for t in times]))
        print(f"\nwrote {series_path}: {len(times)} delays, the k-integrated spectrum "
              f"(edc_zone_t), {len(boxes)} box(es) (edc_box_t), the occupied-exciton DOS "
              f"(dos_occ_t) and <E_X> (E_mean_t)")


if __name__ == "__main__":
    main()
