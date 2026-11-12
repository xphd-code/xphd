"""Real-time Boltzmann equation for exciton populations."""
from __future__ import annotations

from dataclasses import dataclass, field

import os

import numpy as np

from . import parallel
from .core.interp import fourier_refine
from .core.mesh import blockwise
from .core.stats import KB_EV
from .tetra import triangle_weights

HBAR_EV_FS = 0.6582119569509065

__all__ = ["RateMatrix", "equilibrium", "build_rates", "refined_out_rates",
           "rescale_rows", "rhs_factory", "jac_factory", "check_balance",
           "propagate", "thermal_target",
           "inject_optical", "inject_thermal", "save_snapshots", "exph_linewidths",
           "state_labels", "windowed_rates", "q0_states"]


def equilibrium(E, T, N_tot, stat="boltzmann"):
    """Stationary distribution at temperature T holding N_tot excitons."""
    E = np.asarray(E, float)
    kT = max(KB_EV * T, 1e-12)
    x = np.clip((E - E.min()) / kT, 1e-12, 300.0)

    if stat == "boltzmann":
        F = np.exp(-x)
        return F / F.sum() * N_tot, None
    if stat == "bose0":
        F = 1.0 / np.expm1(x)
        return F / F.sum() * N_tot, 0.0
    if stat != "bose":
        raise ValueError(f"unknown stat {stat!r}")

    from scipy.optimize import brentq

    def residual(mu):
        return np.sum(1.0 / np.expm1(
            np.clip((E - mu) / kT, 1e-12, 700.0))) - N_tot

    hi = E.min() - 1e-12
    lo = E.min() - 100.0 * kT
    while residual(lo) > 0:
        lo -= 100.0 * kT
    try:
        mu = brentq(residual, lo, hi)
    except ValueError:
        mu = lo
    return 1.0 / np.expm1(np.clip((E - mu) / kT, 1e-12, 700.0)), mu


def inject_optical(E_Q0, osc, N_tot, center=None, sigma=None, verbose=True):
    """Optically-created population at Q = 0."""
    E_Q0 = np.asarray(E_Q0, float)
    w = np.abs(np.asarray(osc, float)).copy()
    if w.size != E_Q0.size:
        if w.size > E_Q0.size:
            w = w[:E_Q0.size]
        else:
            raise SystemExit(
                f"osc_Q0 has {w.size} entries but the archive has "
                f"{E_Q0.size} states. Trimming the archives and writing "
                f"osc_Q0 must use the SAME window, or states beyond the "
                f"oscillator list would be silently treated as dark.")
    if center is not None and sigma:
        w *= np.exp(-0.5 * ((E_Q0 - center) / sigma) ** 2)
    if w.sum() <= 0:
        raise ValueError("no oscillator strength inside the pump window")
    F = N_tot * w / w.sum()
    if verbose:
        order = np.argsort(w)[::-1]
        share = w[order] / w.sum()
        print("  optical injection, brightest states:")
        for k in order[:4]:
            print(f"    state {k+1:2d}  E = {E_Q0[k]:.4f} eV  "
                  f"{w[k]/w.sum()*100:5.1f}%")
        if share[0] > 0.9:
            print("    one state carries >90%: this is effectively "
                  "single-state injection")
    return F


def inject_thermal(E, N_tot, hot, width):
    """Phenomenological hot distribution, Gaussian in energy across the BZ."""
    x = (np.asarray(E, float) - np.min(E)) * 1e3
    F = np.exp(-((x - hot) ** 2) / (2.0 * width ** 2))
    return N_tot * F / F.sum()


@dataclass
class RateMatrix:
    P: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    E: np.ndarray
    dims: tuple
    channels: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def out_rate(self):
        """sum_j P_ij, in 1/fs."""
        return self.P.sum(axis=1)


def _triangulate(n1, n2):
    i, j = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
    v0 = (i * n2 + j).ravel()
    v1 = (((i + 1) % n1) * n2 + j).ravel()
    v2 = (i * n2 + ((j + 1) % n2)).ravel()
    v3 = (((i + 1) % n1) * n2 + ((j + 1) % n2)).ravel()
    return np.vstack([np.column_stack([v0, v1, v2]),
                      np.column_stack([v1, v3, v2])])


def _bose_pair(dE_pair, kT):
    """N(|E_i - E_j|) for every pair (0 on the diagonal)."""
    w_pair = np.abs(dE_pair)
    N_pair = np.zeros_like(w_pair)
    m = w_pair > 1e-9
    N_pair[m] = 1.0 / np.expm1(np.clip(w_pair[m] / kT, 1e-12, 300.0))
    return N_pair


def _pool(A_em, A_ab, E, kT, bose, acc=None, verbose=False):
    """Detailed balance from the forward accumulators."""
    dE_pair = E[:, None] - E[None, :]
    N_pair = _bose_pair(dE_pair, kT)
    nm = 0 if acc is None else acc.shape[1]

    if bose == "crossing":
        ex = np.exp(np.clip(dE_pair / kT, -600.0, 600.0))
        Rf = A_em + A_ab
        P = 0.5 * (Rf + Rf.T * ex)
        R = P - P.T
        for c in range(nm):
            em_c, ab_c = acc[1, c].astype(float), acc[0, c].astype(float)
            acc[1, c] = 0.5 * (em_c + ab_c.T * ex)
            acc[0, c] = 0.5 * (ab_c + em_c.T * ex)
    else:
        B = 0.5 * (A_em + A_ab.T)
        B_tot = B + B.T
        fac = np.where(dE_pair > 0, 1.0 + N_pair, N_pair)
        P = B_tot * fac
        R = B_tot * np.sign(dE_pair)
        emis = dE_pair > 0
        for c in range(nm):
            Bn = 0.5 * (acc[1, c] + acc[0, c].T)
            Bt = (Bn + Bn.T).astype(float)
            acc[1, c] = np.where(emis, Bt * (1.0 + N_pair), 0.0)
            acc[0, c] = np.where(emis, 0.0, Bt * N_pair)
    if acc is not None and verbose:
        err = np.abs(acc.sum(axis=(0, 1)) - P).max()
        print(f"  channels sum to P: rel err "
              f"{err / max(abs(P).max(), 1e-300):.2e}")
    return P, R, acc


ASR_TOL = 3e-3


def gamma_acoustic(hw0, acoustic_cut):
    """Indices of the acoustic branches among the q = 0 frequencies hw0 (eV), and the ASR
    residue when it had to be recognised (else None)."""
    hw0 = np.asarray(hw0, float)
    dead = np.nonzero(hw0 <= acoustic_cut)[0]
    if len(dead) < 3 and len(hw0) >= 3:
        low = np.sort(np.argsort(hw0)[:3])
        res = float(np.abs(hw0[low]).max())
        if res < ASR_TOL:
            return low, res
    return dead, None


DEG_TOL = 1e-4


def degenerate_multiplets(E, tol=DEG_TOL, chi=None):
    """Degenerate multiplets of the states at one Q: index lists."""
    E = np.asarray(E, float)
    order = np.argsort(E, kind="stable")
    groups, cur = [], [int(order[0])]
    for k in order[1:]:
        if tol > 0 and abs(E[k] - E[cur[0]]) < tol:
            cur.append(int(k))
        else:
            groups.append(cur)
            cur = [int(k)]
    groups.append(cur)
    if chi is None:
        return groups
    chi = np.asarray(chi, float)
    out = []
    for g in groups:
        sg = [int(np.sign(chi[i])) if abs(chi[i]) > 0.5 else 0 for i in g]
        for v in (1, -1):
            sub = [i for i, s in zip(g, sg) if s == v]
            if sub:
                out.append(sub)
        out += [[i] for i, s in zip(g, sg) if s == 0]
    return out


def degeneracy_matrices(E_grid, tol=DEG_TOL, chi_grid=None):
    """(nQ, ne, ne) block-averaging matrices, one per Q."""
    E_grid = np.asarray(E_grid, float)
    nQ, ne = E_grid.shape
    D = np.zeros((nQ, ne, ne))
    nman, qs, spread, gap = 0, set(), 0.0, np.inf
    gamma = []
    for a in range(nQ):
        gs = degenerate_multiplets(E_grid[a], tol,
                                  None if chi_grid is None else chi_grid[a])
        for g in gs:
            D[a][np.ix_(g, g)] = 1.0 / len(g)
            if len(g) > 1:
                nman += 1
                qs.add(a)
                spread = max(spread, float(np.ptp(E_grid[a, g])))
                if a == 0:
                    gamma.append((sorted(g), float(E_grid[a, g].mean()),
                                  float(np.ptp(E_grid[a, g]))))
        Es = np.sort(E_grid[a])
        d = np.diff(Es)
        member = np.zeros(ne, int)
        for gi, g in enumerate(gs):
            member[g] = gi
        o = np.argsort(E_grid[a], kind="stable")
        apart = member[o][1:] != member[o][:-1]
        if apart.any():
            gap = min(gap, float(d[apart].min()))
    return D, dict(multiplets=nman, q_points=len(qs), spread=spread, gap=gap, gamma=gamma)


def degeneracy_average(g2, D_init, D_fin):
    """Couplings averaged over degenerate multiplets, at both ends."""
    return np.einsum("ab,qmbc,qdc->qmad", D_init, g2, D_fin, optimize=True)


def _deg_report(info, tol, chi):
    if info["multiplets"] == 0:
        return (f"  degenerate multiplets (--deg-tol {tol * 1e3:g} meV): none -- "
                f"couplings used state by state")
    at_g = "; ".join(f"states {'+'.join(str(i + 1) for i in g)} at {e:.5f} eV "
                     f"({w * 1e3:.3f} meV wide)" for g, e, w in info.get("gamma", []))
    return (f"  couplings averaged over degenerate multiplets (--deg-tol {tol * 1e3:g} meV"
            + (", split by parity" if chi is not None else ", by energy alone") + "): "
            f"{info['multiplets']} multiplets at {info['q_points']} Q, widest "
            f"{info['spread'] * 1e3:.3f} meV; nearest states kept apart "
            f"{info['gap'] * 1e3:.3f} meV"
            + (f"\n    at Gamma: {at_g}" if at_g else "\n    at Gamma: none"))


def build_rates(archives, T, acoustic_cut=1e-4, flat_width=0.0,
                channels=False, verbose=True, bose="node", deg_tol=DEG_TOL,
                parity=None):
    """Assemble the collision matrix from one archive per Q-point."""
    if bose not in ("node", "crossing"):
        raise ValueError(f"bose must be 'node' or 'crossing', not {bose!r}")
    first = next(iter(archives.values()))
    n1, n2 = first.n1, first.n2
    ne, nm = first.nexc, first.nmod
    Nq = n1 * n2
    nst = Nq * ne
    kT = max(KB_EV * T, 1e-12)
    if parity is not None:
        parity = np.asarray(parity, float)
        if parity.shape != (Nq, ne):
            raise ValueError(f"parity has shape {parity.shape}, expected ({Nq}, {ne})")

    q = np.mod(first.q_red, 1.0)
    qi = np.rint(q[:, 0] * n1).astype(int) % n1
    qj = np.rint(q[:, 1] * n2).astype(int) % n2

    if len(archives) < Nq:
        raise SystemExit(
            f"only {len(archives)} of {Nq} Q-points supplied. The rate matrix "
            f"connects states at different Q, so the missing points would "
            f"become zero-energy sinks with no out-rate and the run would "
            f"complete silently. Generate every Q.")

    E = np.zeros(nst)
    for (a, b), arc in archives.items():
        E[(a * n2 + b) * ne:(a * n2 + b) * ne + ne] = arc.E_n
    Dg = None
    if deg_tol and deg_tol > 0:
        Dg, info = degeneracy_matrices(E.reshape(Nq, ne), deg_tol, parity)
        if verbose:
            print(_deg_report(info, deg_tol, parity))
        if info["multiplets"] == 0:
            Dg = None

    tri = _triangulate(n1, n2)
    area = 1.0 / len(tri)
    pref = 2.0 * np.pi / HBAR_EV_FS

    g0 = None
    at0 = np.nonzero((qi == 0) & (qj == 0))[0]
    if len(at0):
        g0 = int(at0[0])
        ac0, res = gamma_acoustic(np.asarray(first._raw("hw"), float)[g0], acoustic_cut)
        if verbose and res is not None:
            print(f"  acoustic branches at q = 0: {[int(x) + 1 for x in ac0]}, with an ASR "
                  f"residue up to {res * 1e3:.3f} meV (above --acoustic-cut): treated as acoustic")

    A_em = np.zeros((nst, nst))
    A_ab = np.zeros((nst, nst))
    acc = Aem_nu = Aab_nu = None
    if channels:
        need = 2 * nm * nst * nst * 4 / 1e9
        if verbose:
            print(f"  channel matrices: {need:.2f} GB")
        acc = np.zeros((2, nm, nst, nst), np.float32)
        Aab_nu, Aem_nu = acc[0], acc[1]

    if verbose:
        print(f"assembling over {len(tri)} triangles, {nst} states")

    for (a, b), arc in archives.items():
        ta, tb = (a + qi) % n1, (b + qj) % n2
        target = ta * n2 + tb

        g2 = np.maximum(np.asarray(arc._raw("g2"), float), 0.0)
        if Dg is not None:
            g2 = degeneracy_average(g2, Dg[a * n2 + b], Dg[target])
        hw = np.asarray(arc._raw("hw"), float)
        live = hw > acoustic_cut
        if g0 is not None:
            live[g0, ac0] = False
        N = np.zeros_like(hw)
        N[live] = 1.0 / np.expm1(np.clip(hw[live] / kT, 0.0, 200.0))
        g2 = np.where(live[:, :, None, None], g2, 0.0)

        E_fin = E[target[:, None] * ne + np.arange(ne)[None, :]]
        Et = np.transpose(E_fin[tri], (1, 0, 2))
        Wt = np.transpose(hw[tri], (1, 0, 2))
        Nt = np.transpose(N[tri], (1, 0, 2))
        gt = np.transpose(g2[tri], (1, 0, 2, 3, 4))
        nodes = np.transpose(
            target[tri][:, :, None] * ne + np.arange(ne)[None, None, :],
            (1, 0, 2))

        dE = arc.E_n[None, None, :, None] - Et[:, :, None, :]
        Wb = Wt[:, :, :, None, None]
        Wa = triangle_weights(dE[:, :, None, :, :] + Wb, area, flat_width)
        We = triangle_weights(dE[:, :, None, :, :] - Wb, area, flat_width)

        base = (a * n2 + b) * ne + np.arange(ne)
        cols = np.concatenate([nodes[v] for v in range(3)]).ravel()
        em_v, ab_v, emc_v, abc_v = [], [], [], []
        for v in range(3):
            wt = pref * gt[v]
            if bose == "crossing":
                Nv = Nt[v][:, :, None, None]
                we, wa = wt * We[v] * (1.0 + Nv), wt * Wa[v] * Nv
            else:
                we, wa = wt * We[v], wt * Wa[v]
            em_v.append(we.sum(axis=1))
            ab_v.append(wa.sum(axis=1))
            if Aem_nu is not None:
                emc_v.append(we)
                abc_v.append(wa)
        em_all = np.concatenate(em_v)
        ab_all = np.concatenate(ab_v)
        for n_ in range(ne):
            A_em[base[n_]] += np.bincount(cols, em_all[:, n_, :].ravel(),
                                          minlength=nst)
            A_ab[base[n_]] += np.bincount(cols, ab_all[:, n_, :].ravel(),
                                          minlength=nst)
        if Aem_nu is not None:
            emc = np.concatenate(emc_v)
            abc = np.concatenate(abc_v)
            for c in range(nm):
                for n_ in range(ne):
                    Aem_nu[c, base[n_]] += np.bincount(
                        cols, emc[:, c, n_, :].ravel(), minlength=nst)
                    Aab_nu[c, base[n_]] += np.bincount(
                        cols, abc[:, c, n_, :].ravel(), minlength=nst)

    P, R, chan = _pool(A_em, A_ab, E, kT, bose, acc, verbose)
    if verbose and bose == "crossing":
        print(f"  phonon occupations taken at the crossings (bose='crossing'); "
              f"pooling with the reverse process\n  changed the total rate by "
              f"{P.sum() / max(A_em.sum() + A_ab.sum(), 1e-300):.4f} x")
    elif verbose:
        dE_pair = E[:, None] - E[None, :]
        N_pair = _bose_pair(dE_pair, kT)
        raw = (A_em * np.where(dE_pair > 0, 1 + N_pair, 0)
               + A_ab * np.where(dE_pair > 0, 0, N_pair)).sum()
        ratio = P.sum() / max(raw, 1e-300)
        both = int(((A_em > 0) & (A_ab.T > 0)).sum())
        anyp = int(((A_em > 0) | (A_ab.T > 0)).sum())
        print(f"  total rate after symmetrisation: {ratio:.4f} x raw   "
              f"({100*both/max(anyp,1):.0f}% of pairs seen both ways)")
        if abs(ratio - 1) > 0.2:
            print("    Enforcing detailed balance has changed the TOTAL rate")
            print("    by more than 20%. Branching ratios are unaffected, but")
            print("    the absolute timescale is not reliable at this level.")
            print("    Use --refine to rescale each row against a properly")
            print("    integrated out-rate.")

    meta = dict(T=T, acoustic_cut=acoustic_cut, flat_width=flat_width, bose=bose,
                deg_tol=float(deg_tol or 0.0))
    if Dg is not None:
        meta["deg"] = Dg
    return RateMatrix(P, P.T.copy(), R, E, (n1, n2, ne), chan, meta=meta)


def _outrate_worker(args):
    """One archive's contribution to the refined out-rates."""
    path, cfg = args
    from .io.excph import ExcPhArchive
    arc = ExcPhArchive(path, mesh=cfg.get("mesh"))
    n1, n2 = arc.n1, arc.n2
    ne, nmod = arc.nexc, arc.nmod
    f = cfg["refine"]
    m1, m2 = n1 * f, n2 * f
    kT = max(KB_EV * cfg["T"], 1e-12)
    fw = cfg["flat_width"]

    tri = _triangulate(m1, m2)
    area = 1.0 / len(tri)
    pref = 2.0 * np.pi / HBAR_EV_FS

    Ef = fourier_refine(arc.grid("E_m"), f).reshape(m1 * m2, ne)
    if cfg.get("hw_fine"):
        hf = np.asarray(np.load(cfg["hw_fine"]), float).reshape(m1 * m2, nmod)
    else:
        hf = np.maximum(fourier_refine(arc.grid("hw"), f),
                        0.0).reshape(m1 * m2, nmod)
    if cfg["g2_mode"] == "complexfft":
        if not arc.has("G"):
            raise ValueError("complexfft needs G_grid")
        G = arc.grid("G")
        gf = np.clip(fourier_refine(G.real, f) ** 2
                     + fourier_refine(G.imag, f) ** 2, 0.0, None)
        gf = gf.reshape(m1 * m2, nmod, ne, ne)
    else:
        gf = blockwise(np.maximum(arc.grid("g2"), 0.0),
                       f).reshape(m1 * m2, nmod, ne, ne)

    live = hf > cfg["acoustic_cut"]
    N = np.zeros_like(hf)
    N[live] = 1.0 / np.expm1(np.clip(hf[live] / kT, 0.0, 200.0))
    gf = np.where(live[:, :, None, None], gf, 0.0)

    chunk = int(cfg.get("chunk") or len(tri))
    acc = np.zeros(ne)
    for s0 in range(0, len(tri), chunk):
        tc = tri[s0:s0 + chunk]
        Et = np.transpose(Ef[tc], (1, 0, 2))
        Wt = np.transpose(hf[tc], (1, 0, 2))
        Nt = np.transpose(N[tc], (1, 0, 2))
        gt = np.transpose(gf[tc], (1, 0, 2, 3, 4))
        dE = arc.E_n[None, None, :, None] - Et[:, :, None, :]
        Wb = Wt[:, :, :, None, None]
        Wa = triangle_weights(dE[:, :, None, :, :] + Wb, area, fw)
        We = triangle_weights(dE[:, :, None, :, :] - Wb, area, fw)
        for v in range(3):
            Nv = Nt[v][:, :, None, None]
            acc += (pref * gt[v] * (Wa[v] * Nv + We[v] * (1.0 + Nv))
                    ).sum(axis=(0, 1, 3))
    return arc.q_index(), acc


def outrate_chunk(nmod, ne, mem_gb=None):
    """Triangles per chunk so one worker stays inside `mem_gb` GB."""
    if mem_gb is None:
        mem_gb = float(os.environ.get("XPHD_TASK_MEM_GB", "2"))
    per_tri = 3 * 10 * nmod * ne * ne * 8
    return max(1024, int(mem_gb * 1e9 / per_tri))


def refined_out_rates(archives, T, refine=None, acoustic_cut=1e-4,
                      flat_width=None, g2_mode="const", hw_fine=None,
                      workers=1, verbose=True, checkpoint=None, mem_gb=None):
    """Out-rates sum_j P_ij integrated on a refined q-mesh."""
    first = next(iter(archives.values()))
    n1, n2 = first.n1, first.n2
    ne, nmod = first.nexc, first.nmod

    if hw_fine is not None:
        h = np.load(hw_fine, mmap_mode="r")
        N = int(round(np.sqrt(h.shape[0]))) if h.ndim == 2 else h.shape[0]
        if N % n1:
            raise SystemExit(f"hw_fine is {N}x{N}, not a multiple of the "
                             f"source mesh {n1}")
        f = N // n1
        if refine is not None and int(refine) != f:
            raise SystemExit(f"hw_fine implies refine={f} but refine="
                             f"{refine} was given; they must agree")
    else:
        f = int(refine or 1)

    if flat_width is None:
        flat_width = 2.4e-3 / (n1 * f)

    ntri = 2 * (n1 * f) * (n2 * f)
    chunk = outrate_chunk(nmod, ne, mem_gb)
    cfg = dict(T=T, refine=f, acoustic_cut=acoustic_cut,
               flat_width=flat_width, g2_mode=g2_mode, mesh=(n1, n2),
               hw_fine=str(hw_fine) if hw_fine is not None else None,
               chunk=chunk if chunk < ntri else None)
    paths = [arc.path for arc in archives.values()]

    if verbose and parallel.is_root():
        print(f"refining out-rates on {n1*f}x{n2*f} "
              f"({ntri} triangles) x {len(paths)} Q-points"
              + (f", in chunks of {chunk} triangles" if chunk < ntri else "")
              + ("\n  phonons from matdyn (matching the linewidths)"
                 if hw_fine is not None else
                 "\n  phonons Fourier-interpolated -- pass hw_fine= to use "
                 "the same\n  matdyn frequencies as the linewidth "
                 "calculation"))

    keys = None
    if checkpoint is not None:
        tag = parallel.config_key("outrate", {k: v for k, v in cfg.items()
                                              if k != "chunk"})
        keys = [f"out-{os.path.basename(str(p))}-{tag}" for p in paths]

    res = parallel.pmap(_outrate_worker, [(p, cfg) for p in paths],
                        workers=workers, checkpoint=checkpoint, keys=keys,
                        desc="out-rates", verbose=verbose, to_all=True)

    out = np.zeros(n1 * n2 * ne)
    for (a, b), acc in res:
        out[(a * n2 + b) * ne:(a * n2 + b) * ne + ne] = acc
    return out


def rescale_rows(rm, out_ref, floor_frac=1e-4, verbose=True):
    """Bring sum_j P_ij toward the refined out-rate WITHOUT breaking balance."""
    out_ref = np.asarray(out_ref, float)
    if rm.meta.get("deg") is not None:
        D = rm.meta["deg"]
        out_ref = np.einsum("aij,aj->ai", D, out_ref.reshape(D.shape[0], -1)).ravel()
    out = rm.P.sum(axis=1)
    med = np.median(out[out > 0]) if np.any(out > 0) else 0.0
    good = out > max(1e-30, floor_frac * med)
    s = np.ones_like(out)
    s[good] = out_ref[good] / out[good]

    S = np.sqrt(np.outer(s, s))
    P = rm.P * S
    R = rm.R * S

    if verbose:
        if (~good).sum():
            print(f"  {(~good).sum()} rows left unscaled (below "
                  f"{floor_frac:g} x median)")
        print(f"  scale: median {np.median(s[good]):.4f}  "
              f"10-90 {np.percentile(s[good],10):.4f}-"
              f"{np.percentile(s[good],90):.4f}")
        hit = P.sum(axis=1)[good] / np.maximum(out_ref[good], 1e-300)
        print(f"  out-rates reach {np.median(hit):.3f} of target "
              f"(10-90 {np.percentile(hit,10):.3f}-"
              f"{np.percentile(hit,90):.3f})")
        print("    symmetric scaling preserves P_ij/P_ji = exp(dE/kT), so the")
        print("    equilibrium is untouched; the row sums land on target only")
        print("    to the extent that s varies slowly between coupled states.")

    return RateMatrix(P, P.T.copy(), 0.5 * (R - R.T), rm.E, rm.dims,
                      rm.channels, {**rm.meta, "row_scale": s}), s


def _fine_factor(n1, refine=None, hw_fine=None):
    """(f, hw_fine array on the n1*f mesh or None, subsampling step)."""
    if hw_fine is None:
        return int(refine or 1), None, 1
    h = np.asarray(np.load(hw_fine) if isinstance(hw_fine, (str, os.PathLike))
                   else hw_fine, float)
    N = int(round(np.sqrt(h.shape[0]))) if h.ndim == 2 else h.shape[0]
    if N % n1:
        raise SystemExit(f"hw_fine is {N}x{N}, not a multiple of the source "
                         f"mesh {n1}")
    F = N // n1
    f = F if refine in (None, 0, 1) else int(refine)
    if F % f:
        raise SystemExit(f"hw_fine is {N}x{N} (refine {F} on {n1}x{n1}); "
                         f"refine {f} must divide {F}")
    sub = F // f
    h = h.reshape(N, N, -1)[::sub, ::sub]
    return f, np.ascontiguousarray(h.reshape(N // sub * (N // sub), -1)), sub


def _parity_rank_map(c_from, c_to):
    """For each band at one coarse point, the band at another that is the same state: same
    sigma_h parity, same rank among the bands of that parity (energy order). -1 where the
    parity is not clean or the other point has too few bands of it."""
    c_from, c_to = np.asarray(c_from, float), np.asarray(c_to, float)
    out = np.full(len(c_from), -1)
    for sgn in (1.0, -1.0):
        src = np.nonzero(sgn * c_from > 0.9)[0]
        dst = np.nonzero(sgn * c_to > 0.9)[0]
        k = min(len(src), len(dst))
        out[src[:k]] = dst[:k]
    return out


def parity_grid(path, archives, e_tol=2e-3):
    """(n1*n2, ne) sigma_h parity of the archives' states from a parity.npz (Q_red, chi, E),
    rows a*n2 + b, checked against the archive energies."""
    first = next(iter(archives.values()))
    n1, n2, ne = first.n1, first.n2, first.nexc
    d = np.load(path)
    pq = np.mod(np.asarray(d["Q_red"], float)[:, :2], 1.0)
    chi = np.real(np.asarray(d["chi"]))
    E = np.asarray(d["E"], float)
    if chi.shape[1] < ne:
        raise SystemExit(f"{path}: {chi.shape[1]} states per Q, the archives {ne}")
    idx = (np.rint(pq[:, 0] * n1).astype(int) % n1) * n2 + np.rint(pq[:, 1] * n2).astype(int) % n2
    look = np.full(n1 * n2, -1)
    look[idx] = np.arange(len(idx))
    if (look < 0).any():
        raise SystemExit(f"{path} does not cover the {n1}x{n2} mesh")
    out = np.zeros((n1 * n2, ne))
    worst = 0.0
    for (a, b), arc in archives.items():
        r = look[a * n2 + b]
        out[a * n2 + b] = chi[r, :ne]
        worst = max(worst, float(np.abs(E[r, :ne] - arc.E_n).max()))
    if worst > e_tol:
        raise SystemExit(f"{path}: exciton energies differ from the archives' by "
                         f"{worst * 1e3:.2f} meV -- not the same states")
    return out


def _window_worker(args):
    """Rows of the windowed rate matrix for some fine-mesh Q of one archive."""
    path, fine_q, cfg = args
    from .io.excph import ExcPhArchive
    arc = ExcPhArchive(path, mesh=cfg["mesh"])
    n1, n2 = cfg["mesh"]
    f = cfg["refine"]
    m1, m2 = n1 * f, n2 * f
    ne, nm = arc.nexc, arc.nmod
    kT = max(KB_EV * cfg["T"], 1e-12)
    fw = cfg["flat_width"]
    crossing = cfg["bose"] == "crossing"
    Ef, wmap, tri = cfg["Ef"], cfg["wmap"], cfg["tri"]
    nw, mfin = cfg["nw"], cfg["mfin"]
    area = 1.0 / (2 * m1 * m2)
    pref = 2.0 * np.pi / HBAR_EV_FS

    if cfg["hw"] is not None:
        hf = cfg["hw"]
    else:
        hf = np.maximum(fourier_refine(arc.grid("hw"), f),
                        0.0).reshape(m1 * m2, nm)
    live = hf > cfg["acoustic_cut"]
    N = np.zeros_like(hf)
    N[live] = 1.0 / np.expm1(np.clip(hf[live] / kT, 0.0, 200.0))

    g2c = np.maximum(np.asarray(arc.grid("g2"), float),
                     0.0).reshape(n1 * n2, nm, ne, ne)
    Qa = np.rint(np.mod(arc.Q_red[:2], 1.0) * [n1, n2]).astype(int)
    P0 = (Qa[0] % n1) * n2 + Qa[1] % n2
    if cfg.get("deg") is not None:
        si_, sj_ = np.divmod(np.arange(n1 * n2), n2)
        tgt_all = ((Qa[0] + si_) % n1) * n2 + (Qa[1] + sj_) % n2
        g2c = degeneracy_average(g2c, cfg["deg"][P0], cfg["deg"][tgt_all])
    table = g2c[..., mfin]
    ci = np.rint(np.arange(m1) / f).astype(int) % n1
    cj = np.rint(np.arange(m2) / f).astype(int) % n2
    hw_c = np.asarray(arc.grid("hw"), float).reshape(n1 * n2, nm)
    dead0, _ = gamma_acoustic(hw_c[0], cfg["acoustic_cut"])
    mode = cfg.get("acoustic_scaling", "none")
    scale_ac = mode in ("omega", "power") and len(dead0)
    npow = float(cfg.get("acoustic_power") or 0.0)
    qnf = qnc = None
    if mode == "power" and len(dead0):
        from .core.mesh import hex_norm
        fi_, fj_ = np.divmod(np.arange(m1 * m2), m2)
        qnf = hex_norm(np.stack([fi_ / m1, fj_ / m2, np.zeros(m1 * m2)], 1))
        ci_, cj_ = np.divmod(np.arange(n1 * n2), n2)
        qnc = hex_norm(np.stack([ci_ / n1, cj_ / n2, np.zeros(n1 * n2)], 1))
    A = C = None
    if cfg["acoustic_model"] == "q2" and f > 1:
        from .core.mesh import hex_norm
        from .linewidth import fit_acoustic
        qn = np.zeros((n1, n2))
        qn[arc._i, arc._j] = hex_norm(arc.q_red)
        A, C, _ = fit_acoustic(g2c.reshape(n1, n2, nm, ne, ne), arc.grid("hw"), qn,
                               cfg["n_acoustic"], cfg["n_shell"],
                               verbose=False, model="q2")
        na = min(cfg["n_acoustic"], nm)
        A, C = A[:na][..., mfin], C[:na][..., mfin]
        fi_, fj_ = np.divmod(np.arange(m1 * m2), m2)
        q2f = hex_norm(np.stack([fi_ / m1, fj_ / m2, np.zeros(m1 * m2)],
                                1)) ** 2
    ref1 = None
    if cfg["acoustic_model"] == "shell" and f > 1 and len(dead0):
        from .core.mesh import hex_norm
        qn = np.zeros((n1, n2))
        qn[arc._i, arc._j] = hex_norm(arc.q_red)
        r1 = qn[qn > 1e-12].min()
        s1 = np.nonzero((np.abs(qn - r1) < 1e-6 * r1).ravel())[0]
        chi = cfg["parity"]
        full = g2c
        acc = np.zeros((len(dead0), ne, ne))
        cnt = np.zeros(ne)
        for s in s1:
            si, sj = divmod(int(s), n2)
            tgt = ((Qa[0] + si) % n1) * n2 + (Qa[1] + sj) % n2
            mp = _parity_rank_map(chi[P0], chi[tgt])
            ok = mp >= 0
            acc[:, :, ok] += (full[s][dead0][:, :, mp[ok]]
                              * hw_c[s, dead0][:, None, None])
            cnt[ok] += 1
        wm = np.maximum(hw_c[s1][:, dead0].mean(axis=0), 1e-12)
        S = acc / np.maximum(cnt, 1)[None, None, :]
        if mode == "power":
            ref1 = (S / r1 ** npow)[..., mfin]
        else:
            ref1 = (S / wm[:, None, None] ** (2 if scale_ac else 1))[..., mfin]

    fQi, fQj = np.divmod(tri, m2)
    cols_all = wmap[tri][:, :, mfin]
    chunk = int(cfg.get("chunk") or len(tri))
    nrow = int((wmap[fine_q] >= 0).sum())
    rows = np.zeros(nrow, int)
    Rem = np.zeros((nrow, nw))
    Rab = np.zeros((nrow, nw))
    out = np.zeros(nrow)
    Cem = Cab = None
    if cfg["channels"]:
        Cem = np.zeros((nm, nrow, nw), np.float32)
        Cab = np.zeros((nm, nrow, nw), np.float32)

    r0 = 0
    for Qf in fine_q:
        sel = np.nonzero(wmap[Qf] >= 0)[0]
        ns = len(sel)
        if ns == 0:
            continue
        rr = slice(r0, r0 + ns)
        rows[rr] = wmap[Qf, sel]
        Qi, Qj = divmod(int(Qf), m2)
        pa, pb = ci[Qi], cj[Qj]
        En = Ef[Qf, sel]
        for s0 in range(0, len(tri), chunk):
            sl = slice(s0, s0 + chunk)
            qv = ((fQi[sl] - Qi) % m1) * m2 + (fQj[sl] - Qj) % m2
            Et = np.transpose(Ef[tri[sl]][:, :, mfin], (1, 0, 2))
            Wt = np.transpose(hf[qv], (1, 0, 2))
            Nt = np.transpose(N[qv], (1, 0, 2))
            Lt = np.transpose(live[qv], (1, 0, 2))
            qc = ((ci[fQi[sl]] - pa) % n1) * n2 + (cj[fQj[sl]] - pb) % n2
            g = table[qc][:, :, :, sel]
            if scale_ac:
                wc = hw_c[qc][:, :, dead0]
                wf = hf[qv][:, :, dead0]
                if mode == "power":
                    qcn = qnc[qc]
                    qr = np.where(qcn > 0, qnf[qv] / np.maximum(qcn, 1e-300), 0.0)[:, :, None]
                    r = np.where((wc > 0) & (wf > 0),
                                 qr ** npow * wc / np.maximum(wf, 1e-300), 0.0)
                else:
                    r = np.where(wc > 0, wf / np.maximum(wc, 1e-300), 1.0)
                g[:, :, dead0] *= r[:, :, :, None, None]
            same = qc == 0
            if same.any():
                gs = g[same]
                if len(dead0):
                    gs[:, dead0] = 0.0
                if ref1 is not None:
                    gs[:, dead0] = ref1[None, :, sel]
                    if mode == "power":
                        wq = hf[qv[same]][:, dead0]
                        fac = np.where(wq > 0, qnf[qv[same]][:, None] ** npow
                                       / np.maximum(wq, 1e-300), 0.0)
                        gs[:, dead0] *= fac[:, :, None, None]
                    elif scale_ac:
                        gs[:, dead0] *= hf[qv[same]][:, dead0][:, :, None, None]
                elif A is not None:
                    qs = qv[same]
                    gs[:, :na] = ((A[None, :, sel] + C[None, :, sel]
                                   * q2f[qs][:, None, None, None])
                                  / np.maximum(2.0 * hf[qs, :na],
                                               1e-12)[:, :, None, None])
                g[same] = gs
            gt = np.transpose(g, (1, 0, 2, 3, 4)) * Lt[:, :, :, None, None]
            dE = En[None, None, :, None] - Et[:, :, None, :]
            Wb = Wt[:, :, :, None, None]
            Wa = triangle_weights(dE[:, :, None, :, :] + Wb, area, fw)
            We = triangle_weights(dE[:, :, None, :, :] - Wb, area, fw)
            cv = np.transpose(cols_all[sl], (1, 0, 2))
            for v in range(3):
                wt = pref * gt[v]
                Nv = Nt[v][:, :, None, None]
                we_x, wa_x = wt * We[v] * (1.0 + Nv), wt * Wa[v] * Nv
                we, wa = (we_x, wa_x) if crossing else (wt * We[v], wt * Wa[v])
                ok = cv[v] >= 0
                col = cv[v][ok]
                out[rr] += ((we_x + wa_x) * ok[:, None, None, :]).sum(
                    axis=(0, 1, 3))
                es, as_ = we.sum(axis=1), wa.sum(axis=1)
                for s in range(ns):
                    Rem[r0 + s] += np.bincount(col, es[:, s, :][ok],
                                               minlength=nw)
                    Rab[r0 + s] += np.bincount(col, as_[:, s, :][ok],
                                               minlength=nw)
                    if Cem is not None:
                        for c in range(nm):
                            Cem[c, r0 + s] += np.bincount(
                                col, we[:, c, s, :][ok], minlength=nw)
                            Cab[c, r0 + s] += np.bincount(
                                col, wa[:, c, s, :][ok], minlength=nw)
        r0 += ns
    return rows, Rem, Rab, out, Cem, Cab


def windowed_rates(archives, T, window, refine=None, hw_fine=None,
                   acoustic_cut=1e-4, flat_width=None, bose="node",
                   band_interp="local", acoustic_model="none", n_shell=3,
                   n_acoustic=3, channels=False, workers=1, checkpoint=None,
                   mem_gb=None, q_per_task=16, out_ref=None, plan_only=False,
                   acoustic_scaling="none", parity=None, acoustic_power=None,
                   deg_tol=DEG_TOL, verbose=True):
    """Rate matrix on the REFINED mesh, for the states within `window` (eV) of the lowest
    exciton energy."""
    if bose not in ("node", "crossing"):
        raise ValueError(f"bose must be 'node' or 'crossing', not {bose!r}")
    if acoustic_scaling not in ("none", "omega", "power"):
        raise ValueError(f"acoustic_scaling must be 'none', 'omega' or 'power', "
                         f"not {acoustic_scaling!r}")
    if acoustic_scaling == "power":
        if acoustic_power is None or not np.isfinite(float(acoustic_power)):
            raise ValueError("acoustic_scaling='power' needs acoustic_power (the n "
                             "of |M|^2 ~ q^n)")
        acoustic_power = float(acoustic_power)
    if acoustic_model not in ("none", "q2", "shell"):
        raise ValueError(f"acoustic_model must be 'none', 'q2' or 'shell', "
                         f"not {acoustic_model!r}")
    from .core.interp import fine_band
    root = parallel.is_root()
    first = next(iter(archives.values()))
    n1, n2 = first.n1, first.n2
    ne, nm = first.nexc, first.nmod
    if len(archives) < n1 * n2:
        raise SystemExit(f"only {len(archives)} of {n1 * n2} Q-points "
                         f"supplied; the window needs every Q")
    f, hw, sub = _fine_factor(n1, refine, hw_fine)
    m1, m2 = n1 * f, n2 * f
    if acoustic_model == "shell" and parity is None:
        raise ValueError("acoustic_model='shell' needs the state parities "
                         "(parity=, e.g. bte.parity_grid('parity.npz', archives))")
    if parity is not None:
        parity = np.asarray(parity, float)
        if parity.shape != (n1 * n2, ne):
            raise ValueError(f"parity has shape {parity.shape}, expected "
                             f"({n1 * n2}, {ne})")
    if flat_width is None:
        flat_width = 2.4e-3 / (n1 * f)
    kT = max(KB_EV * T, 1e-12)

    Ec = np.zeros((n1, n2, ne))
    for (a, b), arc in archives.items():
        Ec[a, b] = arc.E_n
    Dg, deg_info = None, None
    if deg_tol and deg_tol > 0:
        Dg, deg_info = degeneracy_matrices(Ec.reshape(n1 * n2, ne), deg_tol, parity)
        if deg_info["multiplets"] == 0:
            Dg = None
    Ef = fine_band(Ec, f, band_interp).reshape(m1 * m2, ne)
    Emin = Ef.min()
    inwin = (Ef - Emin) <= window
    nw = int(inwin.sum())
    wmap = np.full(inwin.shape, -1)
    wmap[inwin] = np.arange(nw)
    fq, band = np.nonzero(inwin)
    mfin = np.nonzero(inwin.any(axis=0))[0]
    boltz = np.exp(-(Ef - Emin) / kT)
    lost = float(boltz[~inwin].sum() / boltz.sum())

    tri_all = _triangulate(m1, m2)
    qwin = inwin.any(axis=1)
    tri = tri_all[qwin[tri_all].any(axis=1)]

    ci = np.rint(np.arange(m1) / f).astype(int) % n1
    cj = np.rint(np.arange(m2) / f).astype(int) % n2
    fi, fj = np.divmod(np.arange(m1 * m2), m2)
    cQ = ci[fi] * n2 + cj[fj]
    on_node = (fi % f == 0) & (fj % f == 0)

    if verbose and root:
        lay = (f"{n1}x{n2} -> {m1}x{m2} (refine {f}"
               + (f", hw_fine subsampled every {sub}" if sub > 1 else "") + ")")
        print(f"windowed rate matrix on {lay}, window {window * 1e3:.0f} meV "
              f"above the minimum")
        print(f"  {nw} states kept of {m1 * m2 * ne} at {int(qwin.sum())} fine Q "
              f"(final bands {[int(m) + 1 for m in mfin]}); dense matrix "
              f"{nw * nw * 8 / 1e9:.2f} GB")
        print(f"  Boltzmann weight outside the window at {T:g} K: {lost:.2e}"
              + ("   <-- raise the window" if lost > 1e-3 else ""))
        if hw is not None:
            nac = min(3, nm)
            neg = (hw[:, :nac] < 0).any(axis=1)
            if neg.any():
                from .core.mesh import hex_norm
                fi_, fj_ = np.divmod(np.arange(m1 * m2), m2)
                rr = hex_norm(np.stack([fi_ / m1, fj_ / m2, np.zeros(m1 * m2)], 1)) * n1
                print(f"  WARNING hw_fine: {int(neg.sum())} fine q with an imaginary acoustic "
                      f"branch (down to {hw[:, :nac].min() * 1e3:.3f} meV), out to "
                      f"{rr[neg].max():.2f} first-shell radii -- dropped as dead phonons,\n"
                      f"    i.e. a hidden acoustic cut. Missing rotational sum rule: "
                      f"matdyn asr='all', or scripts/patch_za.py")
        print(f"  {len(tri)} of {len(tri_all)} fine triangles touch the window; "
              f"energies: {band_interp}, couplings: between parent coarse points "
              f"(acoustic scaling {acoustic_scaling}), "
              f"acoustic model {acoustic_model}, bose {bose}")
        if f > 1:
            print("  pairs with the same parent: acoustic coupling "
                  + ("ZERO -- below the first shell the acoustic channel acts only\n"
                     "    across parent boundaries; --acoustic-model shell --parity fills the cell"
                     if acoustic_model == "none" else
                     ("first shell between the same states (parity, rank), "
                      + ("carried down as w" if acoustic_scaling == "omega" else
                         f"carried down as |M|^2 ~ q^{acoustic_power:g}, g^2 = |M|^2/2w"
                         if acoustic_scaling == "power" else "held constant"))
                     if acoustic_model == "shell" else
                     f"C q^2 / 2w fitted on {n_shell} shell(s) by band index"
                     "\n    (unsafe where bands of opposite parity reorder; use 'shell')"))
        if deg_info is not None:
            print(_deg_report(deg_info, deg_tol, parity))
        ac0, res = gamma_acoustic(np.asarray(first.grid("hw"), float)[0, 0], acoustic_cut)
        if res is not None:
            print(f"  acoustic branches at q = 0: {[int(x) + 1 for x in ac0]}, with an ASR "
                  f"residue up to {res * 1e3:.3f} meV (above --acoustic-cut): treated as acoustic")
        elif len(ac0) < 3 and f > 1:
            print(f"  WARNING: only {len(ac0)} branch(es) at or below --acoustic-cut at q = 0 "
                  f"and no ASR-like triplet within {ASR_TOL * 1e3:g} meV: the cell around "
                  f"q = 0 has no acoustic treatment for the others")
        if channels:
            print(f"  channel matrices: {4 * nm * nw * nw * 4 / 1e9:.2f} GB "
                  f"during assembly")
        Eg = Ef.reshape(m1, m2, ne)
        steps = []
        for ax in (0, 1):
            d = np.abs(np.roll(Eg, -1, axis=ax) - Eg).reshape(m1 * m2, ne)
            both = inwin & np.roll(inwin.reshape(m1, m2, ne), -1,
                                   axis=ax).reshape(m1 * m2, ne)
            steps.append(d[both])
        steps = np.concatenate(steps)
        if len(steps):
            print(f"  energy step between neighbouring kept states: median "
                  f"{np.median(steps) * 1e3:.2f} meV, 90% {np.percentile(steps, 90) * 1e3:.2f} "
                  f"meV (kT = {kT * 1e3:.2f} meV)"
                  + ("   <-- refine further" if np.percentile(steps, 90) > kT else ""))
    if plan_only:
        return None

    paths = {k: arc.path for k, arc in archives.items()}
    tasks, cost = [], []
    chunk = outrate_chunk(nm, ne, mem_gb)
    cfg = dict(T=T, refine=f, mesh=(n1, n2), acoustic_cut=acoustic_cut,
               flat_width=flat_width, bose=bose, Ef=Ef, wmap=wmap, tri=tri,
               nw=nw, mfin=mfin, hw=hw, acoustic_model=acoustic_model,
               n_shell=n_shell, n_acoustic=n_acoustic, channels=channels,
               acoustic_scaling=acoustic_scaling, parity=parity,
               acoustic_power=acoustic_power, deg=Dg,
               deg_tol=float(deg_tol or 0.0) if Dg is not None else 0.0,
               chunk=chunk if chunk < len(tri) else None)
    for (a, b) in sorted(paths):
        mine = np.nonzero(qwin & (cQ == a * n2 + b))[0]
        for k in range(0, len(mine), q_per_task):
            part = mine[k:k + q_per_task]
            tasks.append((paths[(a, b)], part, cfg))
            cost.append(int(inwin[part].sum()))
    keys = None
    if checkpoint is not None:
        tag = parallel.config_key("window", {k: v for k, v in cfg.items()
                                             if k not in ("chunk", "Ef", "wmap",
                                                          "tri", "hw", "mfin",
                                                          "deg")},
                                  float(window), band_interp,
                                  str(hw_fine), float(Ef.sum()))
        keys = [f"win-{os.path.basename(str(t[0]))}-{int(t[1][0])}-{tag}"
                for t in tasks]
    res = parallel.pmap(_window_worker, tasks, workers=workers, cost=cost,
                        checkpoint=checkpoint, keys=keys, desc="window rows",
                        verbose=verbose)
    if not root:
        return None

    A_em = np.zeros((nw, nw))
    A_ab = np.zeros((nw, nw))
    out = np.zeros(nw)
    acc = np.zeros((2, nm, nw, nw), np.float32) if channels else None
    for rows, Rem, Rab, o, Cem, Cab in res:
        A_em[rows] += Rem
        A_ab[rows] += Rab
        out[rows] += o
        if acc is not None:
            acc[1][:, rows] += Cem
            acc[0][:, rows] += Cab
    del res
    E = Ef[fq, band]
    P, R, chan = _pool(A_em, A_ab, E, kT, bose, acc, verbose)
    del A_em, A_ab

    Q_red = np.stack([fi[fq] / m1, fj[fq] / m2, np.zeros(nw)], 1)
    Qc = cQ[fq]
    Q_coarse = np.stack([(Qc // n2) / n1, (Qc % n2) / n2, np.zeros(nw)], 1)
    node = on_node[fq]
    if verbose:
        prow = P.sum(axis=1)
        print(f"  out-rate (detailed-balance matrix) / forward out-rate: "
              f"median {np.median(prow / np.maximum(out, 1e-300)):.3f} "
              f"(1 when the mesh resolves the crossings)")
    if out_ref is not None and verbose:
        ref = np.asarray(out_ref, float)[Qc * ne + band]
        deep = node & ((E - Emin) <= window - 0.05) & (ref > 0)
        edge = node & ((E - Emin) > window - 0.05) & (ref > 0)
        for tag, m in (("deeper than window - 50 meV", deep),
                       ("within 50 meV of the edge", edge)):
            if m.any():
                r_ = out[m] / ref[m]
                print(f"  forward out-rate / refined out-rate, {int(m.sum())} "
                      f"coarse-node states {tag}: median {np.median(r_):.4f}, "
                      f"10-90% {np.percentile(r_, 10):.4f}-"
                      f"{np.percentile(r_, 90):.4f}")
        if band_interp != "fourier":
            print("    (the refined out-rates use Fourier-refined energies; "
                  "band_interp='fourier' compares like with like)")

    meta = dict(T=T, acoustic_cut=acoustic_cut, flat_width=flat_width,
                bose=bose, window=float(window), refine=f,
                coarse_mesh=(n1, n2), band_interp=band_interp,
                acoustic_model=acoustic_model, outside_weight=lost,
                acoustic_scaling=acoustic_scaling, acoustic_power=acoustic_power,
                deg_tol=float(deg_tol or 0.0),
                Q_red=Q_red, band=band, Q_coarse_red=Q_coarse, on_node=node,
                fine_q=fq, forward_out=out)
    return RateMatrix(P, P.T.copy(), R, E, (m1, m2, ne), chan, meta=meta)


def q0_states(rm):
    """Index of the state at Q = 0 for each band, -1 where it is not kept."""
    n1, n2, ne = rm.dims
    if "fine_q" not in rm.meta:
        return np.arange(ne)
    idx = np.full(ne, -1)
    fq, band = rm.meta["fine_q"], rm.meta["band"]
    sel = np.nonzero(fq == 0)[0]
    idx[band[sel]] = sel
    return idx


def thermal_target(E, F):
    """The Boltzmann distribution with the number and the energy of F."""
    e = np.asarray(E, float) - float(np.min(E))
    F = np.asarray(F, float)
    N = float(F.sum())
    if N <= 0:
        z = np.zeros_like(e)
        return z, z, 0.0, 0.0, np.inf
    u = float(F @ e) / N
    span = float(e.max())

    def stats(b):
        x = -b * e
        x -= x.max()
        w = np.exp(x)
        p = w / w.sum()
        return p, float(p @ e)

    big = 1e7
    if span <= 0 or u <= 1e-12 * max(span, 1e-300):
        b = big
    elif u >= span * (1 - 1e-12):
        b = -big
    else:
        from scipy.optimize import brentq

        def f(b_):
            return stats(b_)[1] - u
        if f(0.0) > 0:
            lo, hi = 0.0, 1.0
            while f(hi) > 0 and hi < big:
                lo, hi = hi, hi * 4
        else:
            lo, hi = -1.0, 0.0
            while f(lo) < 0 and lo > -big:
                lo, hi = lo * 4, lo
        b = brentq(f, lo, hi, xtol=1e-14, rtol=1e-13, maxiter=200) \
            if np.sign(f(lo)) != np.sign(f(hi)) else (hi if abs(f(hi)) < abs(f(lo)) else lo)
    p, _ = stats(b)
    var = float(p @ (e - u) ** 2)
    return N * p, p, u, var, b


def rhs_factory(rm, P=None, Q=None, R=None):
    """dF/dt as a function of F."""
    P = rm.P if P is None else P
    Q = rm.Q if Q is None else Q
    R = rm.R if R is None else R
    Prow = np.asarray(P.sum(axis=1)).ravel()

    def rhs(t, F):
        r = -(Prow * F - Q @ F + F * (R @ F))
        return r
    return rhs


def jac_factory(rm, P=None, Q=None, R=None):
    """Analytic Jacobian of `rhs_factory`"""
    P = rm.P if P is None else P
    Q = rm.Q if Q is None else Q
    R = rm.R if R is None else R
    Prow = np.asarray(P.sum(axis=1)).ravel()
    try:
        import scipy.sparse as sp
        sparse = sp.issparse(Q)
    except ImportError:
        sparse = False

    if sparse:
        def jac(t, F):
            J = Q - sp.diags(Prow + R @ F) - sp.diags(F) @ R
            return J.tocsc()
    else:
        idx = np.diag_indices(len(Prow))

        def jac(t, F):
            J = Q - F[:, None] * R
            J[idx] -= Prow + R @ F
            return J
    return jac


def _as_sparse(rm, matrix, verbose):
    """Decide dense vs sparse storage for the propagation."""
    n = len(rm.E)
    nnz = int(np.count_nonzero(rm.P))
    dens = nnz / max(n * n, 1)
    use = matrix == "sparse" or (matrix == "auto" and n > 2000 and dens < 0.1)
    if verbose:
        print(f"  rate matrix: {n} states, {dens*100:.1f}% non-zero -> "
              f"{'sparse' if use else 'dense'} propagation")
    if not use:
        return None
    import scipy.sparse as sp
    return sp.csr_matrix(rm.P), sp.csr_matrix(rm.Q), sp.csr_matrix(rm.R)


def check_balance(rm, N_tot, verbose=True):
    """Symmetry and stationarity residuals."""
    T = rm.meta.get("T", 300.0)
    s1 = np.abs(rm.P - rm.Q.T).max() / max(rm.P.max(), 1e-300)
    s2 = np.abs(rm.R + rm.R.T).max() / max(np.abs(rm.R).max(), 1e-300)
    rhs = rhs_factory(rm)
    Prow = rm.P.sum(axis=1)

    out = {"P_minus_QT": s1, "R_plus_RT": s2}
    if verbose:
        print(f"  |P - Q^T| {s1:.3e}   |R + R^T| {s2:.3e}")
        print(f"\n  {'distribution':<20} {'residual':>12} {'weighted':>12}")
    for tag in ("boltzmann", "bose0", "bose"):
        try:
            F, _ = equilibrium(rm.E, T, N_tot, tag)
        except Exception:
            continue
        F = np.where(np.isfinite(F), F, 0.0)
        if F.sum() <= 0:
            continue
        r = rhs(0.0, F)
        sc = max(np.abs(Prow * F).max(), 1e-300)
        wt = F / F.sum()
        rel = np.abs(r) / np.maximum(np.abs(Prow * F), 1e-300)
        out[tag] = float(np.max(np.abs(r)) / sc)
        if verbose:
            print(f"  {tag:<20} {out[tag]:12.3e} {np.sum(wt*rel):12.3e}")
    if verbose:
        print("  This operator keeps the stimulated (1+F) factors, for which")
        print("  the exact stationary state is BOSE-EINSTEIN at any chemical")
        print("  potential -- read that row. Boltzmann is stationary only in")
        print("  the linear limit F << 1, so its residual measures how far")
        print("  the injected density is from that limit.")
    return out


def propagate(rm, F0, t_end=10000.0, nt=12, method="BDF",
              rtol=1e-7, atol=1e-12, jac="analytic", matrix="dense",
              verbose=False):
    """Integrate the BTE from F0 and return the solve_ivp solution."""
    from scipy.integrate import solve_ivp
    mats = _as_sparse(rm, matrix, verbose) if matrix != "dense" else None
    P, Q, R = mats if mats is not None else (None, None, None)
    kw = {}
    if jac == "analytic" and method in ("BDF", "Radau", "LSODA"):
        kw["jac"] = jac_factory(rm, P, Q, R)
    elif jac not in ("analytic", "fd"):
        raise ValueError(f"jac must be 'analytic' or 'fd', not {jac!r}")
    sol = solve_ivp(rhs_factory(rm, P, Q, R), (0.0, t_end),
                    np.asarray(F0, float),
                    t_eval=np.linspace(0.0, t_end, nt), method=method,
                    rtol=rtol, atol=atol, **kw)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    if verbose:
        print(f"  solver: {sol.nfev} rhs evaluations, {sol.njev} Jacobians, "
              f"{sol.nlu} LU factorisations")
    return sol


def state_labels(rm):
    """(Q_red, band) for every state, from the flat index (a*n2+b)*ne + l, or as stored by
    windowed_rates."""
    if "Q_red" in rm.meta:
        return np.asarray(rm.meta["Q_red"]), np.asarray(rm.meta["band"])
    n1, n2, ne = rm.dims
    i = np.arange(len(rm.E))
    q, band = i // ne, i % ne
    a, b = q // n2, q % n2
    return np.stack([a / n1, b / n2, np.zeros(len(i))], 1), band


_SNAPSHOT_BLOCK = 1e7


def save_snapshots(rm, sol, path, channels=True, verbose=True, extra=None):
    """Write populations over time in the layout the plotting scripts read."""
    Q_red, band = state_labels(rm)
    out = dict(t=sol.t, F=sol.y.T, E=rm.E, Q_red=Q_red, band=band,
               band_base=np.array(0),
               mesh=np.array([rm.dims[0], rm.dims[1], 1]),
               T=rm.meta.get("T", np.nan))

    if "Q_coarse_red" in rm.meta:
        out.update(Q_coarse_red=rm.meta["Q_coarse_red"],
                   coarse_mesh=np.array([*rm.meta["coarse_mesh"], 1]),
                   window=rm.meta["window"], refine=rm.meta["refine"],
                   on_node=rm.meta["on_node"], bose=rm.meta.get("bose", ""))

    fw_t, fw0 = exph_linewidths(rm, sol.y)
    out.update(fwhm_exph_t=fw_t.T.astype(np.float32), fwhm_exph0=fw0.astype(np.float32))

    if channels and rm.channels is not None:
        nch, nmq = rm.channels.shape[:2]
        flat = rm.channels.reshape(nch * nmq, *rm.channels.shape[2:])
        Y = np.asarray(sol.y, float)
        nst, nt = Y.shape
        dm = np.zeros((nt, nst), np.int16)
        ie = np.zeros((nt, nst), bool)
        df = np.zeros((nt, nst), np.float32)
        step = max(1, min(nt, int(_SNAPSHOT_BLOCK // max(nst, 1))))
        for k0 in range(0, nt, step):
            Yb = Y[:, k0:k0 + step]
            tot = np.zeros(Yb.shape)
            top = np.full(Yb.shape, -np.inf)
            best = np.zeros(Yb.shape, np.int32)
            for ci, c in enumerate(flat):
                infl = c.T @ Yb
                tot += infl
                up = infl > top
                top[up] = infl[up]
                best[up] = ci
            sl = slice(k0, k0 + Yb.shape[1])
            dm[sl] = (best % nmq).T.astype(np.int16)
            ie[sl] = ((best // nmq) == 1).T
            with np.errstate(invalid="ignore", divide="ignore"):
                df[sl] = np.where(tot > 0, top / np.maximum(tot, 1e-300), 0.0).T
        rows = np.stack([c.sum(axis=1) for c in flat])
        fl = (Y.T @ rows.T).reshape(nt, nch, nmq).astype(np.float32)
        out.update(dom_mode_t=dm, is_emission_t=ie, dom_frac_t=df, flux_t=fl)
        if verbose:
            print(f"  channel decomposition saved: {nch} x {nmq} channels, "
                  f"{nt} times")

    if extra:
        out.update(extra)
    np.savez(path, **out)
    if verbose:
        x = (rm.E - rm.E.min()) * 1e3
        N = sol.y.sum(axis=0)
        print(f"  wrote {path}: {len(sol.t)} snapshots, {len(rm.E)} states")
        print(f"    <E>-Emin: {(sol.y[:,0]@x)/N[0]:.2f} -> "
              f"{(sol.y[:,-1]@x)/N[-1]:.2f} meV   "
              f"N/N0 drift {abs(N[-1]/N[0]-1):.2e}")
        wf = [float((fw_t[:, k] @ sol.y[:, k]) / max(N[k], 1e-300)) for k in (0, -1)]
        print(f"    exciton-phonon linewidths (FWHM) saved as fwhm_exph_t / fwhm_exph0: "
              f"occupation-weighted {wf[0] * 1e3:.2f} -> {wf[1] * 1e3:.2f} meV")
        neg = int((fw_t < 0).sum())
        if neg:
            print(f"    {neg} (state, time) entries negative: stimulated scattering INTO "
                  f"the state outweighs its decay (N - F < 0 for absorption); "
                  f"xphd arpes clips them to 0")
    return out


def exph_linewidths(rm, F):
    """Exciton-phonon linewidth (FWHM, eV) of every state for occupations F."""
    Prow = np.asarray(rm.P.sum(axis=1)).ravel()
    F = np.asarray(F, float)
    RF = rm.R @ F
    full = HBAR_EV_FS * ((Prow if F.ndim == 1 else Prow[:, None]) + RF)
    return full, HBAR_EV_FS * Prow
