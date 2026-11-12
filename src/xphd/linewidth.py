"""Exciton linewidths from the on-shell exciton-phonon Fan-Migdal self-energy."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np

from .core.interp import fine_band, fourier_refine, ringing_report
from .core.mesh import blockwise, hex_norm
from .core.stats import HBAR_EV_PS, KB_EV, bose, degeneracy_groups
from .tetra import delta_weights_2d

__all__ = ["LinewidthResult", "linewidth_one", "compute",
           "Interference", "interference", "fit_acoustic"]

FLAT_WIDTH_SCALE = 2.4e-3


@dataclass
class LinewidthResult:
    """Per-state linewidths in eV, indexed [temperature, state]."""
    T: np.ndarray
    total: np.ndarray
    emission: np.ndarray
    absorption: np.ndarray
    E_n: np.ndarray
    groups: list
    meta: dict = field(default_factory=dict)
    mode_em: np.ndarray | None = None
    mode_ab: np.ndarray | None = None

    def dominant_mode(self, it=0):
        """(mode index, is_emission, fraction) per state."""
        if self.mode_em is None:
            return None
        tot = self.mode_em[it] + self.mode_ab[it]
        best = np.argmax(tot, axis=1)
        k = np.arange(len(best))
        frac = tot[k, best] / np.maximum(tot.sum(axis=1), 1e-300)
        return best, self.mode_em[it][k, best] > self.mode_ab[it][k, best], frac

    def tau_ps(self):
        with np.errstate(divide="ignore"):
            return HBAR_EV_PS / self.total

    def by_multiplet(self, it: int = 0):
        """[(label, total, emission, absorption, tau_ps), ...] in meV / ps."""
        rows = []
        for g in self.groups:
            t = float(self.total[it, g].mean())
            e = float(self.emission[it, g].mean())
            a = float(self.absorption[it, g].mean())
            lab = "{" + ",".join(str(k + 1) for k in g) + "}"
            rows.append((lab, t * 1e3, e * 1e3, a * 1e3,
                         HBAR_EV_PS / t if t > 0 else np.inf))
        return rows

    def report(self):
        for it, T in enumerate(self.T):
            print(f"\nT = {T:6.1f} K   [FWHM, meV]")
            print("     states |    total |  emission | absorption |    tau (ps)")
            for lab, t, e, a, tau in self.by_multiplet(it):
                print(f"  {lab:>9s} | {t:8.3f} | {e:9.3f} | {a:10.3f} "
                      f"| {tau:11.4f}")

    def mode_report(self, it: int = 0):
        """Per-branch linewidth for each multiplet, FWHM in meV."""
        if self.mode_em is None:
            print("   (no per-branch data in this result)")
            return
        em, ab = self.mode_em[it], self.mode_ab[it]
        nmod = em.shape[1]
        print(f"\nT = {self.T[it]:6.1f} K   per-branch linewidth [FWHM, meV]"
              f"  (branches energy-ordered)")
        print("   " + "states".rjust(9)
              + "".join(f"{f'nu={m+1}':>10}" for m in range(nmod))
              + "     total")
        worst = 0.0
        for g in self.groups:
            lab = "{" + ",".join(str(k + 1) for k in g) + "}"
            tot = (em[g] + ab[g]).mean(axis=0) * 1e3
            ref = float(self.total[it, g].mean()) * 1e3
            worst = max(worst, abs(tot.sum() - ref) / max(ref, 1e-12))
            print(f"   {lab:>9}" + "".join(f"{x:10.3f}" for x in tot)
                  + f"{tot.sum():10.3f}")
        print("\n   emission share by branch [%]  (- where the branch "
              "contributes < 0.1% of the multiplet)")
        for g in self.groups:
            lab = "{" + ",".join(str(k + 1) for k in g) + "}"
            e = em[g].mean(axis=0); t = (em[g] + ab[g]).mean(axis=0)
            tt = max(t.sum(), 1e-300)
            cells = [f"{100*e[m]/t[m]:10.1f}" if t[m] > 1e-3 * tt
                     else f"{'-':>10}" for m in range(nmod)]
            print(f"   {lab:>9}" + "".join(cells))
        print(f"\n   branch sum vs multiplet total: {worst:.1e} relative"
              + ("" if worst < 1e-6 else "   <-- INCONSISTENT"))

    def save_modes(self, path):
        """Per-branch linewidths as a long table: one row per (T, state, branch), FWHM in meV."""
        if self.mode_em is None:
            raise ValueError("no per-branch data in this result")
        path = str(path)
        sep = "," if path.lower().endswith(".csv") else " "
        m = self.meta
        L = ["# exciton-phonon linewidth by phonon branch (FWHM, meV)",
             "# branches are energy-ordered at each q; near Gamma 1-3 are "
             "acoustic",
             f"# acoustic_model {m.get('acoustic_model', '?')}"
             + (f" (n_shell {m.get('n_shell')})"
                if m.get('acoustic_model') not in (None, 'none') else "")
             + f"   band_interp {m.get('band_interp', '?')}",
             "# bare exciton energies (eV): "
             + " ".join(f"{v:.4f}" for v in self.E_n),
             sep.join(["#T(K)", "state", "branch", "emission", "absorption",
                       "total"])]
        for it, T in enumerate(self.T):
            for n in range(self.mode_em.shape[1]):
                for m in range(self.mode_em.shape[2]):
                    e = self.mode_em[it, n, m] * 1e3
                    a = self.mode_ab[it, n, m] * 1e3
                    L.append(sep.join([f"{T:.1f}", str(n + 1), str(m + 1),
                                       f"{e:.6f}", f"{a:.6f}",
                                       f"{e + a:.6f}"]))
        with open(path, "w") as f:
            f.write("\n".join(L) + "\n")

    def save(self, path):
        """Write results."""
        path = str(path)
        if not path.lower().endswith((".txt", ".dat", ".csv")):
            extra = {}
            if self.mode_em is not None:
                extra = dict(mode_em=self.mode_em, mode_ab=self.mode_ab)
            np.savez(path, T=self.T, total=self.total,
                     emission=self.emission, absorption=self.absorption,
                     E_n=self.E_n, **extra,
                     **{k: np.asarray(v) for k, v in self.meta.items()})
            return

        m, ne = self.meta, len(self.E_n)
        sep = "," if path.lower().endswith(".csv") else ""
        L = ["# exciton-phonon linewidths (FWHM, meV)",
             f"# source mesh {m.get('mesh')}  refine {m.get('refine')}  "
             f"scheme {m.get('scheme')}",
             f"# acoustic_cut {m.get('acoustic_cut', 0)*1e3:.3f} meV   "
             f"flat_width {m.get('flat_width', 0)*1e6:.3f} ueV",
             f"# acoustic_model {m.get('acoustic_model', '?')}"
             + (f" (n_shell {m.get('n_shell')})"
                if m.get('acoustic_model') not in (None, 'none') else "")
             + f"   band_interp {m.get('band_interp', '?')}",
             f"# bare exciton energies (eV): "
             + " ".join(f"{v:.4f}" for v in self.E_n),
             "# tau (ps) = 6.582119569e-4 / Gamma (eV)",
             "#",
             "# columns: T(K), then Gamma_n for n = 1.." + str(ne)
             + ", then emission_n, then absorption_n"]
        w = 12
        hdr = ("#" + "T(K)".rjust(w - 1)
               + "".join(f"G_{n+1}".rjust(w) for n in range(ne))
               + "".join(f"em_{n+1}".rjust(w) for n in range(ne))
               + "".join(f"ab_{n+1}".rjust(w) for n in range(ne)))
        L.append(sep.join(hdr.split()) if sep else hdr)
        for it, T in enumerate(self.T):
            row = ([f"{T:{w}.1f}"]
                   + [f"{v*1e3:{w}.4f}" for v in self.total[it]]
                   + [f"{v*1e3:{w}.4f}" for v in self.emission[it]]
                   + [f"{v*1e3:{w}.4f}" for v in self.absorption[it]])
            L.append(sep.join(x.strip() for x in row) if sep else "".join(row))

        L += ["", "# multiplet averages (degenerate states combined)",
              "# " + "  ".join("{" + ",".join(str(k+1) for k in g) + "}"
                               for g in self.groups)]
        hdr = ("#" + "T(K)".rjust(w - 1)
               + "".join(("{" + ",".join(str(k+1) for k in g) + "}").rjust(w)
                         for g in self.groups))
        L.append(sep.join(hdr.split()) if sep else hdr)
        for it, T in enumerate(self.T):
            row = [f"{T:{w}.1f}"] + [f"{self.total[it, g].mean()*1e3:{w}.4f}"
                                     for g in self.groups]
            L.append(sep.join(x.strip() for x in row) if sep else "".join(row))
        pathlib.Path(path).write_text("\n".join(L) + "\n")


def fit_acoustic(g2, hw, q_norm, n_acoustic=3, n_shell=3, verbose=True,
                 model="auto"):
    """Small-q form of the acoustic coupling, from the innermost shells."""
    if model not in ("q2", "auto"):
        raise ValueError(f"model must be 'q2' or 'auto', not {model!r}")
    need = 1 if model == "q2" else 3
    if n_shell < need:
        raise SystemExit(
            f"--n-shell {n_shell}: the {model} fit needs at least {need} "
            f"shell{'s' if need > 1 else ''}"
            + ("; --acoustic-model q2 works from one" if model == "auto"
               else ""))
    g2 = np.asarray(g2, float)
    hw = np.asarray(hw, float)
    r = np.asarray(q_norm, float)

    flat = r.ravel()
    pos = np.sort(flat[flat > 1e-12])
    edges = [pos[0]]
    for v in pos[1:]:
        if v > edges[-1] * (1 + 1e-6):
            edges.append(v)
    if len(edges) < n_shell:
        raise SystemExit(f"--n-shell {n_shell}: the source mesh has only "
                         f"{len(edges)} shells")
    edges = edges[:n_shell]

    masks = [np.abs(r - e) < e * 1e-6 for e in edges]
    x = np.array([e ** 2 for e in edges])

    A = np.zeros(g2.shape[2:])
    C = np.zeros(g2.shape[2:])
    info = {}
    n_use = min(n_acoustic, A.shape[0])

    for nu in range(n_use):
        Y = np.array([(g2[:, :, nu][m].sum(axis=(1, 2))
                       * 2.0 * hw[:, :, nu][m]).mean() for m in masks])
        Ci = Y / x
        keep_A, A_b, sA = False, 0.0, float("nan")
        if model == "auto":
            try:
                coef, cov = np.polyfit(x, Y, 1, cov=True)
                sA = float(np.sqrt(abs(cov[1, 1])))
            except Exception:
                coef, sA = np.polyfit(x, Y, 1), float("inf")
            C_joint, A_joint = float(coef[0]), float(coef[1])
            keep_A = bool(np.isfinite(sA) and A_joint > 0
                          and A_joint > 2.0 * sA)

        Yc = np.stack([(g2[:, :, nu][m] * 2.0 * hw[:, :, nu][m][:, None, None]
                        ).mean(axis=0) for m in masks])
        sh = Yc.shape[1:]
        if keep_A:
            cf = np.polyfit(x, Yc.reshape(len(x), -1), 1)
            C[nu] = np.clip(cf[0], 0.0, None).reshape(sh)
            A[nu] = np.clip(cf[1], 0.0, None).reshape(sh)
            A_b, C_b = A_joint, C_joint
        else:
            C[nu] = np.clip((Yc / x[:, None, None]).mean(axis=0), 0.0, None)
            A_b, C_b = 0.0, float(Ci.mean())
        fit = A_b + C_b * x
        resid = (float(np.sqrt(np.mean((Y - fit) ** 2))
                       / max(abs(Y).max(), 1e-300))
                 if len(x) > 1 else float("nan"))
        info[nu] = dict(A=A_b, C=C_b, sigma_A=sA, piezo=keep_A, resid=resid,
                        C_shell=Ci)

    if verbose:
        print(f"  acoustic fit, model {model}: "
              f"{sum(int(m.sum()) for m in masks)} points in {len(edges)} "
              f"shell{'s' if len(edges) > 1 else ''}, |q| = {edges[0]:.4f} "
              f".. {edges[-1]:.4f}")
        print(f"    {'branch':>7} {'A':>12} {'C':>12} {'form':>10} "
              f"{'resid':>7}   C per shell, inner -> outer")
        for nu in range(n_use):
            d = info[nu]
            rs = "-" if not np.isfinite(d["resid"]) else f"{d['resid']:.1%}"
            print(f"    {nu+1:7d} {d['A']:12.4e} {d['C']:12.4e} "
                  f"{'A + C q^2' if d['piezo'] else 'C q^2':>10} {rs:>7}   "
                  + " ".join(f"{c:.3e}" for c in d["C_shell"]))
        if any(info[nu]["piezo"] for nu in info):
            kept = [nu + 1 for nu in info if info[nu]["piezo"]]
            print(f"    WARNING a constant term A was kept for branch(es) "
                  f"{kept}. For a NEUTRAL")
            print("    exciton the long-range piezoelectric field cancels "
                  "between electron and")
            print("    hole as q -> 0, like the Froehlich term, so this is "
                  "the fit absorbing the")
            print("    bend of the exciton form factor, not piezoelectricity. "
                  "Being log-divergent,")
            print("    it can inflate the result many times over. Use "
                  "--acoustic-model q2.")
        dom = max(info, key=lambda nu: info[nu]["C"]) if info else None
        if dom is not None and len(edges) > 1:
            ci = info[dom]["C_shell"]
            if ci[0] > 0 and ci[-1] < 0.8 * ci[0]:
                print(f"    branch {dom + 1}: C falls {100 * (1 - ci[-1] / ci[0]):.0f}% "
                      f"from the inner to the outer shell -- the")
                print("    form factor is bending the data. If the trend "
                      "continues inward, the q -> 0")
                print("    coupling exceeds the mean used here and the "
                      "result is a LOWER bound.")
                print("    --n-shell 1 uses the innermost shell alone.")
        finite = [info[nu]["resid"] for nu in info
                  if np.isfinite(info[nu]["resid"])]
        if finite and max(finite) > 0.15:
            print(f"    residual up to {max(finite):.0%}: the form does not "
                  f"describe these shells;\n    treat the extrapolation as "
                  f"an estimate, not a value.")
    return A, C, info


def linewidth_one(E_n, E_m, w_ph, g2_coarse, r, T, flat_width,
                  diag="anti", fwhm=True, acoustic_cut=1e-4,
                  acoustic_A=None, acoustic_C=None, gamma_cell=None,
                  q2_fine=None, n_acoustic=3, state=None, per_mode=False):
    """Gamma for one initial state at one temperature."""
    scalar = np.ndim(T) == 0
    Ts = np.atleast_1d(np.asarray(T, float))
    kTs = [max(KB_EV * t, 1e-12) for t in Ts]
    nT = len(Ts)
    nmod, nexc = g2_coarse.shape[2], g2_coarse.shape[3]
    emis = np.zeros(nT)
    absr = np.zeros(nT)
    pm_e = np.zeros((nT, nmod))
    pm_a = np.zeros((nT, nmod))

    for nu in range(nmod):
        wq = w_ph[:, :, nu]
        live = wq > acoustic_cut
        if not live.any():
            continue
        Ns = [bose(wq, kT) for kT in kTs]
        for m in range(nexc):
            gc = g2_coarse[:, :, nu, m]
            if not np.any(gc):
                continue
            gg = blockwise(gc, r)
            if acoustic_C is not None and nu < n_acoustic:
                idx = (nu, state, m) if state is not None else (nu, m)
                mdl = (acoustic_A[idx] + acoustic_C[idx] * q2_fine) \
                    / np.maximum(2.0 * wq, 1e-12)
                gg = np.where(gamma_cell, mdl, gg)
            gg = gg * live
            dE = E_n - E_m[:, :, m]
            We = delta_weights_2d(dE - wq, flat_width, diag)
            Wa = delta_weights_2d(dE + wq, flat_width, diag)
            for k in range(nT):
                e_ = float(np.sum(We * gg * (Ns[k] + 1.0)))
                a_ = float(np.sum(Wa * gg * Ns[k]))
                emis[k] += e_
                absr[k] += a_
                pm_e[k, nu] += e_
                pm_a[k, nu] += a_

    pref = (2.0 if fwhm else 1.0) * np.pi
    if scalar:
        emis, absr, pm_e, pm_a = (float(emis[0]), float(absr[0]),
                                  pm_e[0], pm_a[0])
    if per_mode:
        return (pref * (emis + absr), pref * emis, pref * absr,
                pref * pm_e, pref * pm_a)
    return pref * (emis + absr), pref * emis, pref * absr


def compute(archive, temperatures, hw_fine=None, refine=1, scheme="const",
            flat_width=None, acoustic_cut=1e-4, fwhm=True,
            deg_tol=2e-3, states=None, acoustic_model="none",
            n_acoustic=3, n_shell=3, band_interp="local", verbose=True):
    """Linewidths for every state in `archive` at each temperature."""
    n1, n2 = archive.n1, archive.n2
    r = int(refine)
    N1, N2 = n1 * r, n2 * r
    fw = flat_width if flat_width is not None else FLAT_WIDTH_SCALE / max(N1, 1)

    g2 = archive.grid("g2")
    E_m_c = archive.grid("E_m")
    hw_c = archive.grid("hw")
    E_n = archive.E_n
    nmod, nexc = archive.nmod, archive.nexc
    sel = range(nexc) if states is None else list(states)

    if verbose:
        print(f"{archive!r}")
        print(f"source {n1}x{n2} -> fine {N1}x{N2} (r={r}), scheme={scheme}")
        print(f"flat_width {fw*1e6:.3f} ueV, acoustic_cut {acoustic_cut*1e3:.3f} meV")

    E_m = fine_band(E_m_c, r, band_interp)
    meta = {}
    if r > 1 and verbose and band_interp == "local":
        print("  exciton energies: local quadratic around each mesh point "
              "(nodes exact, no ringing)")
    if r > 1 and verbose and band_interp == "fourier":
        rep = ringing_report(E_m_c, r)
        meta["overshoot_eV"] = rep["overshoot"]
        print(f"  E_m nodes reproduced to {rep['node_error']:.2e} eV")
        print(f"  Gibbs overshoot {rep['overshoot']*1e3:.2f} meV "
              f"({rep['overshoot_frac']*100:.1f}% of bandwidth)")
        if rep["overshoot"] > 0.5 * hw_c.max():
            print("    comparable to the phonon energies -- this can move "
                  "where a delta surface crosses")

    if hw_fine is not None:
        hw = np.asarray(hw_fine, float)
        if hw.ndim == 2:
            hw = hw.reshape(N1, N2, nmod)
        if hw.shape != (N1, N2, nmod):
            raise ValueError(f"hw_fine is {hw.shape}, expected {(N1,N2,nmod)}")
        if verbose:
            dev = np.abs(hw[::r, ::r] - hw_c)
            rest = dev.copy()
            rest[0, 0, :3] = 0.0
            print(f"  matdyn vs hw_grid: Gamma acoustic "
                  f"{dev[0,0,:3].max()*1e3:.4f} meV, elsewhere "
                  f"{rest.max()*1e3:.4f} meV")
    else:
        if verbose:
            print("  [warn] no hw_fine; Fourier-refining hw_grid instead. "
                  "The acoustic branches are non-analytic at Gamma.")
        hw = fourier_refine(hw_c, r)

    r_eff = r
    if scheme == "fft" and r > 1:
        if verbose:
            print("  [note] Fourier-refining |G|^2; rings at the symmetry "
                  "nodes and can go negative. Clipped at zero.")
        g2 = np.clip(fourier_refine(g2, r), 0.0, None)
        r_eff = 1
    elif scheme != "const":
        raise ValueError(f"unknown scheme {scheme!r}")

    aA = aC = gcell = q2f = None
    if acoustic_model in ("q2", "auto"):
        qn_grid = np.zeros((n1, n2))
        qn_grid[archive._i, archive._j] = hex_norm(archive.q_red)
        aA, aC, _ = fit_acoustic(g2, hw_c, qn_grid, n_acoustic, n_shell,
                                 verbose, model=acoustic_model)
        ii = np.rint(np.arange(N1) / r).astype(int) % n1
        jj = np.rint(np.arange(N2) / r).astype(int) % n2
        gcell = np.outer(ii == 0, jj == 0)
        fi, fj = np.meshgrid(np.arange(N1) / N1, np.arange(N2) / N2,
                             indexing="ij")
        q2f = hex_norm(np.stack([fi.ravel(), fj.ravel(),
                                 np.zeros(N1 * N2)], 1)).reshape(N1, N2) ** 2
        if verbose:
            print(f"  Gamma cell covers {int(gcell.sum())} fine points, "
                  f"|q| < {np.sqrt(q2f[gcell].max()):.4f}")
    elif acoustic_model != "none":
        raise ValueError(f"unknown acoustic_model {acoustic_model!r}")

    groups = [g for g in degeneracy_groups(E_n, deg_tol)
              if all(k in sel for k in g)]

    T = np.atleast_1d(np.asarray(temperatures, float))
    tot = np.zeros((len(T), nexc))
    em = np.zeros_like(tot)
    ab = np.zeros_like(tot)
    m_e = np.zeros((len(T), nexc, nmod))
    m_a = np.zeros((len(T), nexc, nmod))
    for n in sel:
        (tot[:, n], em[:, n], ab[:, n],
         m_e[:, n], m_a[:, n]) = linewidth_one(
            E_n[n], E_m, hw, g2[:, :, :, n, :], r_eff, T, fw,
            fwhm=fwhm, acoustic_cut=acoustic_cut, acoustic_A=aA,
            acoustic_C=aC, gamma_cell=gcell, q2_fine=q2f,
            n_acoustic=n_acoustic, state=n, per_mode=True)

    meta.update(mesh=(n1, n2), refine=r, scheme=scheme, flat_width=fw,
                acoustic_cut=acoustic_cut, acoustic_model=acoustic_model,
                n_shell=n_shell, band_interp=band_interp)
    res = LinewidthResult(T, tot, em, ab, E_n, groups, meta, m_e, m_a)
    if verbose:
        res.report()
    return res


@dataclass
class Interference:
    """Gamma split into electron-only, hole-only, and the cross term."""
    T: float
    total: np.ndarray
    e_only: np.ndarray
    h_only: np.ndarray
    E_n: np.ndarray
    groups: list
    per_mode: np.ndarray | None = None
    optical_frac: np.ndarray | None = None

    @property
    def cross(self):
        return self.total - self.e_only - self.h_only

    def report(self):
        c = self.cross
        inc = self.e_only + self.h_only
        print(f"T = {self.T:.1f} K   [FWHM, meV]")
        head = (f"   {'states':>9} {'total':>9} {'e only':>9} {'h only':>9} "
                f"{'interf':>9} {'% of e+h':>9} {'h/e':>8}")
        if self.optical_frac is not None:
            head += f" {'% optical':>10}"
        print(head)
        for k, g in enumerate(self.groups):
            t, e, h = (self.total[g].mean(), self.e_only[g].mean(),
                       self.h_only[g].mean())
            x, i = c[g].mean(), inc[g].mean()
            lab = "{" + ",".join(str(m + 1) for m in g) + "}"
            row = (f"   {lab:>9} {t*1e3:9.3f} {e*1e3:9.3f} {h*1e3:9.3f} "
                   f"{x*1e3:+9.3f} {100*x/max(abs(i),1e-300):+8.1f}% "
                   f"{h/max(e,1e-300):8.1f}")
            if self.optical_frac is not None:
                row += f" {100*self.optical_frac[g].mean():9.1f}%"
            print(row)
        print("\n   negative interference = destructive: the electron and hole")
        print("   couplings partly cancel because the exciton is neutral. Two")
        print("   mechanisms produce it:")
        print("     - acoustic deformation potential: both sublattices move in")
        print("       phase, so similar electron and hole potentials cancel;")
        print("     - Frohlich coupling of the polar LO mode: the long-range")
        print("       field couples to charge, and electron and hole carry")
        print("       opposite charge, so it cancels as q*a_exc -> 0. This is")
        print("       often the LARGEST destructive term, and it is optical.")
        print("   positive = constructive: the couplings add, as for non-polar")
        print("   optical modes, or for the Frohlich term at larger q where the")
        print("   cancellation lifts. The sign of the LO term can therefore")
        print("   differ between multiplets; `xphd frohlich` resolves it by q.")

    def mode_table(self):
        """Per-branch cross term, meV."""
        if self.per_mode is None:
            return
        pm = self.per_mode
        cross = pm[0] - pm[1] - pm[2]
        nmod = cross.shape[0]
        print(f"\n   interference by branch [meV]  (branches energy-ordered: "
              f"1-3 are acoustic only near Gamma)")
        print("   " + "states".rjust(9)
              + "".join(f"{f'nu={m+1}':>10}" for m in range(nmod)))
        for g in self.groups:
            lab = "{" + ",".join(str(m + 1) for m in g) + "}"
            v = cross[:, g].mean(axis=1) * 1e3
            print(f"   {lab:>9}" + "".join(f"{x:+10.3f}" for x in v))
        print("   a sign that holds per branch, not just per multiplet, is what")
        print("   rules out a coincidence across a handful of points.")


def interference(archive, T, hw_fine=None, refine=1, flat_width=None,
                 acoustic_cut=1e-4, fwhm=True, deg_tol=2e-3,
                 per_mode=True, n_acoustic=3, band_interp="local",
                 verbose=True):
    """Decompose Gamma into electron, hole and interference contributions."""
    if not (archive.has("Ge") and archive.has("Gh")):
        raise ValueError(
            "archive has no Ge_grid/Gh_grid. Regenerate with the updated "
            "generate_excph.py -- without the separate amplitudes the "
            "decomposition is not defined, and a run that silently falls back "
            "to the total gives the degenerate tot == e == h output.")

    n1, n2 = archive.n1, archive.n2
    r = int(refine)
    fw = flat_width if flat_width is not None else FLAT_WIDTH_SCALE / max(n1*r, 1)

    Ge = archive.grid("Ge")
    Gh = archive.grid("Gh")
    g2 = {"tot": np.abs(Ge + Gh) ** 2,
          "e": np.abs(Ge) ** 2,
          "h": np.abs(Gh) ** 2}

    E_m = fine_band(archive.grid("E_m"), r, band_interp)
    hw_c = archive.grid("hw")
    if hw_fine is not None:
        hw = np.asarray(hw_fine, float)
        if hw.ndim == 2:
            hw = hw.reshape(n1 * r, n2 * r, archive.nmod)
    else:
        hw = fourier_refine(hw_c, r)

    E_n = archive.E_n
    nexc, nmod = archive.nexc, archive.nmod
    out = {k: np.zeros(nexc) for k in g2}
    pm = np.zeros((3, nmod, nexc)) if per_mode else None
    opt = np.zeros(nexc)

    for n in range(nexc):
        for ki, key in enumerate(("tot", "e", "h")):
            out[key][n] = linewidth_one(E_n[n], E_m, hw, g2[key][:, :, :, n, :],
                                        r, T, fw, fwhm=fwhm,
                                        acoustic_cut=acoustic_cut)[0]
            if per_mode:
                for nu in range(nmod):
                    sub = np.zeros_like(g2[key][:, :, :, n, :])
                    sub[:, :, nu] = g2[key][:, :, nu, n, :]
                    pm[ki, nu, n] = linewidth_one(
                        E_n[n], E_m, hw, sub, r, T, fw, fwhm=fwhm,
                        acoustic_cut=acoustic_cut)[0]
        if per_mode and out["tot"][n] > 0:
            opt[n] = pm[0, n_acoustic:, n].sum() / out["tot"][n]

    groups = degeneracy_groups(E_n, deg_tol)
    res = Interference(T, out["tot"], out["e"], out["h"], E_n, groups,
                       pm, opt if per_mode else None)
    if verbose:
        res.report()
        res.mode_table()
    return res
