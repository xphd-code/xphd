"""Command-line interface."""
from __future__ import annotations

import argparse
import os

import numpy as np


def _version():
    """Package version, without importing the package eagerly."""
    try:
        from importlib.metadata import version
        return version("xphd")
    except Exception:
        try:
            from . import __version__
            return __version__
        except Exception:
            return "unknown"


def _archive(a):
    from .io.excph import ExcPhArchive
    return ExcPhArchive(a.npz, mesh=getattr(a, "mesh", None))


def _hex(a):
    A = a.hex
    return np.array([A, 0.0]), np.array([-A / 2, A * np.sqrt(3) / 2])


def _refine_from(a, arc):
    """Refinement factor: --hw-fine wins, then --fine, then --refine."""
    hw = np.load(a.hw_fine) if getattr(a, "hw_fine", None) else None
    r = getattr(a, "refine", None) or 1
    if hw is not None:
        N = int(round(np.sqrt(hw.shape[0]))) if hw.ndim == 2 else hw.shape[0]
        if N % arc.n1:
            raise SystemExit(f"fine mesh {N} is not a multiple of the source "
                             f"mesh {arc.n1}")
        r = N // arc.n1
    elif getattr(a, "fine", None):
        if a.fine < arc.n1 or a.fine % arc.n1:
            raise SystemExit(f"--fine {a.fine} is not a multiple of the source mesh "
                             f"{arc.n1}")
        r = a.fine // arc.n1
    return hw, r


def cmd_check(a):
    _archive(a).check()


def cmd_matdyn(a):
    from .io.matdyn import compare_to_archive, read_freq
    _, ev = read_freq(a.freq, nbnd=a.nbnd, nks=a.fine ** 2)
    print(f"  range {ev.min()*1e3:.4f} .. {ev.max()*1e3:.3f} meV "
          f"({int((ev < -1e-9).sum())} negative)")
    if a.npz:
        compare_to_archive(ev, a.fine, _archive(a), q_order=a.q_order)
    if a.out:
        np.save(a.out, ev)
        print(f"  wrote {a.out}  {ev.shape}  eV")


def cmd_linewidth(a):
    from .linewidth import compute
    arc = _archive(a)
    hw, r = _refine_from(a, arc)
    res = compute(arc, a.T, hw_fine=hw, refine=r, scheme=a.scheme,
                  flat_width=a.flat_width, acoustic_cut=a.acoustic_cut,
                  acoustic_model=a.acoustic_model, n_shell=a.n_shell,
                  band_interp=a.band_interp)
    if a.acoustic_model == "none":
        mins = []
        for g in res.groups:
            em = res.emission[:, g].mean(axis=1)
            tot = res.total[:, g].mean(axis=1)
            if np.all(em <= 1e-2 * np.maximum(tot, 1e-300)):
                mins.append("{" + ",".join(str(k + 1) for k in g) + "}")
        if mins:
            print(f"\n  WARNING {', '.join(mins)} emit(s) nothing: a band "
                  f"minimum, whose only channel\n  is small-q acoustic "
                  f"absorption -- dropped by --acoustic-model none.\n  Its "
                  f"width here is not physical; rerun with --acoustic-model q2.")
    if a.per_mode:
        for it in range(len(res.T)):
            res.mode_report(it)
    if a.out:
        res.save(a.out)
        print(f"\nwrote {a.out}")
        if a.per_mode:
            import os
            stem, ext = os.path.splitext(a.out)
            if ext.lower() in (".txt", ".dat", ".csv"):
                mp = f"{stem}_modes{ext}"
                res.save_modes(mp)
                print(f"wrote {mp}")
            else:
                print("   (per-branch data is in the .npz as mode_em, mode_ab)")


def cmd_sweep(a):
    from . import parallel
    from .sweep import sweep
    from .symmetry import load_ops, unfold, violation
    fld = sweep(a.pattern, a.T, hw_fine=a.hw_fine, fine=a.fine,
                refine=a.refine, scheme=a.scheme, flat_width=a.flat_width,
                acoustic_cut=a.acoustic_cut, mesh=a.mesh, workers=a.workers,
                acoustic_model=a.acoustic_model, n_shell=a.n_shell,
                band_interp=a.band_interp, checkpoint=a.checkpoint)
    if not parallel.is_root():
        return
    fld.report()
    if a.unfold:
        ops = load_ops(a.unfold)
        print(f"\ntrace violation before: "
              f"{violation(fld.trace(0), fld.Q_red, ops)*100:.2f}%")
        for it in range(len(fld.T)):
            fld.LW[it] = unfold(fld.LW[it], fld.Q_red, ops, verbose=(it == 0))
            if fld.mode_em is not None:
                fld.mode_em[it] = unfold(fld.mode_em[it], fld.Q_red, ops,
                                         verbose=False)
                fld.mode_ab[it] = unfold(fld.mode_ab[it], fld.Q_red, ops,
                                         verbose=False)
        print(f"after:  {violation(fld.trace(0), fld.Q_red, ops)*100:.2f}%")
    fld.save(a.out, legacy=a.legacy)
    print(f"\nwrote {a.out}")


def cmd_ibz(a):
    """List the Q-points that actually need generating."""
    try:
        from .yambo import ibz_parents, lattice
        ylat = lattice(a.savepath)
    except Exception as e:
        ylat = None
        why = e
    if ylat is not None:
        kidx, first = ibz_parents(ylat)
        print(f"  {len(kidx)} full-zone points, {len(first)} irreducible")
        print("\narguments for `xphd generate` (0-based, lattice ordering):")
        print(" ".join(str(int(k)) for k in first))
        print("\nthe archives these produce:")
        print(" ".join(f"Q{int(k) + 1:04d}" for k in first))
        print("\nThe linewidth sweep needs only these; the real-time transport "
              "needs\nall {0} (`generate 0 .. {1}`), since a missing Q becomes "
              "a zero-energy\nsink that silently absorbs population."
              .format(len(kidx), len(kidx) - 1))
        return

    from .symmetry import irreducible, load_ops
    if a.mesh is None:
        raise SystemExit(f"could not read the lattice ({why}), and the "
                         f"row-major fallback needs --mesh N N")
    ops = load_ops(a.savepath)
    n1, n2 = a.mesh
    idx, Q = irreducible(n1, ops, n2)
    print(f"  [warn] could not read the lattice ({why}); falling back to a "
          f"row-major\n  construction, which matches `xphd generate` only if "
          f"the lattice orders\n  its k-points row-major. Check against "
          f"--iq-map before generating.")
    print(f"\n  {len(ops)} point-group operations, {len(idx)} irreducible")
    print("\nrow-major indices (0-based):")
    print(" ".join(str(int(k)) for k in idx))
    if a.iq_map:
        qr = np.mod(np.asarray(np.load(a.iq_map)["q_red"], float), 1.0)[:, :2]
        d = Q[idx][:, None, :2] - qr[None, :, :]
        d -= np.rint(d)
        dist = np.linalg.norm(d, axis=-1)
        m = np.argmin(dist, axis=1)
        if dist[np.arange(len(idx)), m].max() > 1e-5:
            print("\n  [warn] some points are not in the archive's q_red")
        else:
            print(f"\nthe same points in the q-ordering of {a.iq_map} "
                  f"(0-based):")
            print(" ".join(str(int(v)) for v in sorted(m)))


def cmd_unfold(a):
    from .sweep import LinewidthField
    from .symmetry import expand, full_mesh, load_ops, unfold, violation
    fld = LinewidthField.load(a.field)
    ops = load_ops(a.savepath)
    n1, n2 = (a.mesh if a.mesh
              else np.asarray(fld.meta["mesh"]).ravel()[:2].astype(int))
    Qf = full_mesh(int(n1), int(n2))

    if len(fld.Q_red) < len(Qf):
        print(f"partial set: {len(fld.Q_red)} computed, {len(Qf)} wanted")
        LW = np.stack([expand(fld.LW[it], fld.Q_red, Qf, ops,
                              verbose=(it == 0)) for it in range(len(fld.T))])
        E_n = expand(fld.E_n, fld.Q_red, Qf, ops, verbose=False)
        me = ma = None
        if fld.mode_em is not None:
            me = np.stack([expand(fld.mode_em[it], fld.Q_red, Qf, ops, verbose=False)
                           for it in range(len(fld.T))])
            ma = np.stack([expand(fld.mode_ab[it], fld.Q_red, Qf, ops, verbose=False)
                           for it in range(len(fld.T))])
        fld = LinewidthField(Qf, fld.T, LW, E_n, fld.meta, me, ma)
    else:
        print(f"trace violation before: "
              f"{violation(fld.trace(0), fld.Q_red, ops)*100:.2f}%")
        for it in range(len(fld.T)):
            fld.LW[it] = unfold(fld.LW[it], fld.Q_red, ops, verbose=(it == 0))
            if fld.mode_em is not None:
                fld.mode_em[it] = unfold(fld.mode_em[it], fld.Q_red, ops,
                                         verbose=False)
                fld.mode_ab[it] = unfold(fld.mode_ab[it], fld.Q_red, ops,
                                         verbose=False)
    print(f"trace violation after: "
          f"{violation(fld.trace(0), fld.Q_red, ops)*100:.2f}%")
    fld.save(a.out, legacy=a.legacy)
    print(f"wrote {a.out}  ({len(fld.Q_red)} points)"
          + ("  [+ Qx/Qy/LW for old plotting scripts]" if a.legacy else ""))


def cmd_selfenergy(a):
    from .selfenergy import self_energy
    arc = _archive(a)
    hw, r = _refine_from(a, arc)
    se = self_energy(arc, a.state - 1, a.T, hw_fine=hw, refine=r,
                     window=(a.window or None), n_omega=a.n_omega,
                     acoustic_cut=a.acoustic_cut, band_interp=a.band_interp)
    if a.out:
        import numpy as _np
        _np.savez(a.out, omega=se.omega, im=se.im, re=se.re,
                  spectral=se.spectral(), E_bare=se.E_bare, T=se.T)
        print(f"\nwrote {a.out}")


def cmd_lineshape(a):
    import numpy as _np
    from .lineshape import (bright_window, epsilon2, final_width_gain, final_width_sc,
                            onshell, parity_at_q, peak_metrics, read_dipoles,
                            sigma_matrix, symmetry_report)
    arc = _archive(a)
    hw, r = _refine_from(a, arc)
    lo = float(arc.grid("E_m").min() - arc.grid("hw").max()) - a.pad
    hi = float(arc.grid("E_m").max() + arc.grid("hw").max()) + a.pad
    w = _np.linspace(lo, hi, a.n_omega)
    step = w[1] - w[0]
    print(f"self-energy grid {lo:.3f} .. {hi:.3f} eV, {a.n_omega} points, "
          f"step {step * 1e3:.3f} meV; fine-mesh cells: {a.cells}")
    if step > 1e-3:
        print("  [warn] Sigma is interpolated between these points; above ~1 meV its "
              "structure on the\n  phonon and kT scales is undersampled -- raise --n-omega")
    par = parity_at_q(a.parity, arc) if a.parity else None
    if a.acoustic_model == "shell" and par is None:
        raise SystemExit("--acoustic-model shell needs --parity (exciton_parity.npz)")
    fw_mode = str(a.final_width).strip().lower()
    if fw_mode != "sc":
        try:
            fw_value = float(fw_mode)
        except ValueError:
            raise SystemExit(f"--final-width {a.final_width!r}: 'sc' or a half-width in eV")
        if fw_value < 0:
            raise SystemExit("--final-width must be >= 0")
    Ts = [float(t) for t in a.T]
    sms = sigma_matrix(arc, w, Ts, hw_fine=hw, refine=r,
                       acoustic_cut=a.acoustic_cut, band_interp=a.band_interp,
                       acoustic_model=a.acoustic_model,
                       acoustic_scaling=a.acoustic_scaling, parity=par,
                       cells=a.cells)
    sym = symmetry_report(sms[-1], None if par is None else par[0])
    out = dict(omega=w, T=_np.array(Ts), E_bare=sms[0].E_bare,
               sigma=_np.stack([s_.sigma for s_ in sms]),
               gamma_diag=_np.stack([s_.gamma_diag() for s_ in sms]),
               acoustic_model=a.acoustic_model, acoustic_scaling=a.acoustic_scaling,
               acoustic_cut=a.acoustic_cut, refine=r, eta=a.eta, cells=a.cells,
               sym_parity=sym.get("parity", _np.nan), sym_multiplet=sym["multiplet"])
    if a.bsdb or a.dipoles:
        d = (_np.load(a.dipoles)[: arc.nexc] if a.dipoles
             else read_dipoles(a.bsdb, neigs=arc.nexc))
        if len(d) < arc.nexc:
            raise SystemExit(f"{len(d)} dipoles for {arc.nexc} states")
        lo_w, hi_w, members = bright_window(sms[0].E_bare, d, w)
        if a.window:
            lo_w, hi_w = a.window
        E_A = float(_np.mean(sms[0].E_bare[members]))
        print(f"\n  lowest bright multiplet: states {[m + 1 for m in members]} at "
              f"{E_A:.4f} eV; peak search in {lo_w:.3f} .. {hi_w:.3f} eV")
        xs = _np.linspace(lo_w, hi_w, a.n_plot)
        keys = ("fwhm_full", "fwhm_diag", "peak_full", "peak_diag", "asym_full",
                "asym_diag", "gamma_A", "gamma_A_full", "shift_A", "shift_A_full",
                "weight_A_full", "gamma_A_0", "gamma_A_full_0", "final_width",
                "final_width_gain")
        res = {k: [] for k in keys}
        e_full, e_diag = [], []
        from .lineshape import SigmaMatrix
        rc, rcf, nit = [], [], []
        eye = _np.eye(arc.nexc)[None]
        for sm in sms:
            of0, od0 = onshell(sm, members), onshell(sm, members, diagonal=True)
            if fw_mode == "sc":
                eta_f, it, ok = final_width_sc(sm, members)
                nit.append(it)
                if not ok:
                    print(f"  [warn] T = {sm.T:g} K: the final-state width did not "
                          f"converge in {it} iterations (eta = {eta_f * 1e3:.3f} meV)")
            else:
                eta_f = fw_value
            sw = sm.with_final_width(eta_f)
            of, od = onshell(sw, members), onshell(sw, members, diagonal=True)
            s0 = od["shift"] if a.recenter else 0.0
            s1 = of["shift"] if a.recenter else 0.0
            rc.append(s0)
            rcf.append(s1)
            smr = SigmaMatrix(sw.omega, sw.sigma - s1 * eye, sw.E_bare, sw.T, sw.meta)
            dg = SigmaMatrix(sw.omega, sw.sigma - s0 * eye, sw.E_bare, sw.T,
                             sw.meta).diagonal_only()
            mf, _, _ = peak_metrics(smr, d, (lo_w, hi_w), eta=a.eta)
            md, _, _ = peak_metrics(dg, d, (lo_w, hi_w), eta=a.eta)
            e_full.append(epsilon2(smr, d, a.eta, omega=xs))
            e_diag.append(epsilon2(dg, d, a.eta, omega=xs))
            for k, v in (("fwhm_full", mf["fwhm"]), ("fwhm_diag", md["fwhm"]),
                         ("peak_full", mf["peak"]), ("peak_diag", md["peak"]),
                         ("asym_full", mf["asym"]), ("asym_diag", md["asym"]),
                         ("gamma_A", od["fwhm"]), ("gamma_A_full", of["fwhm"]),
                         ("shift_A", od["shift"]), ("shift_A_full", of["shift"]),
                         ("weight_A_full", of["weight"]),
                         ("gamma_A_0", od0["fwhm"]), ("gamma_A_full_0", of0["fwhm"]),
                         ("final_width", eta_f),
                         ("final_width_gain", final_width_gain(sm, members, eta_f))):
                res[k].append(v)
        res = {k: _np.array(v) for k, v in res.items()}
        if fw_mode == "sc":
            print(f"\n  final states: half-width eta self-consistent with the A width "
                  f"({min(nit)}-{max(nit)} iterations)")
        elif fw_value > 0:
            print(f"\n  final states: half-width eta = {fw_value * 1e3:.3f} meV")
        print(f"\n  {'':>7} {'':>7} {'--- at w = E_A (Chan) ---':>28}  "
              f"{'---- FWHM of eps_2 ----':>28}")
        print(f"  {'T (K)':>7} {'eta':>7} {'diag':>8} {'full':>8} {'ratio':>6}   "
              f"{'diag':>8} {'full':>8} {'ratio':>6}  {'pk-E_A':>7} {'shift':>8} "
              f"{'asym':>7}")
        print(f"  {'':>7} {'(meV)':>7} {'(meV)':>8} {'(meV)':>8} {'':>6}   "
              f"{'(meV)':>8} {'(meV)':>8} {'':>6}  {'(meV)':>7} {'(meV)':>8} "
              f"{'(meV)':>7}")
        for i, T in enumerate(Ts):
            r1 = (res["gamma_A_full"][i] / res["gamma_A"][i]
                  if res["gamma_A"][i] > 0 else _np.nan)
            r2 = (res["fwhm_full"][i] / res["fwhm_diag"][i]
                  if res["fwhm_diag"][i] > 0 else _np.nan)
            print(f"  {T:7.1f} {res['final_width'][i] * 1e3:7.3f} "
                  f"{res['gamma_A'][i] * 1e3:8.3f} "
                  f"{res['gamma_A_full'][i] * 1e3:8.3f} {r1:6.3f}   "
                  f"{res['fwhm_diag'][i] * 1e3:8.3f} {res['fwhm_full'][i] * 1e3:8.3f} "
                  f"{r2:6.3f}  {(res['peak_full'][i] - E_A) * 1e3:+7.3f} "
                  f"{res['shift_A_full'][i] * 1e3:+8.2f} "
                  f"{res['asym_full'][i] * 1e3:+7.3f}")
        print("  at w = E_A: -2 Im of the A eigenvalue(s) of E + Sigma(E_A), with the "
              "diagonal of\n  Sigma only or the full matrix (Chan's definition). FWHM of "
              "eps_2: the absorption\n  line" + (" with the A line's own static shift "
              "(full or diagonal) taken off the\n  whole matrix (every exciton "
              "renormalised together; --no-recenter keeps it);\n  pk-E_A: where the "
              "full line then peaks" if a.recenter else
              ", where Re Sigma puts it (final states unshifted); pk-E_A: its peak")
              + ". eta: final-state half-width\n  (--final-width). shift: Re of the full "
              "A eigenvalue(s) - E_A.")
        fin = res["fwhm_diag"][_np.isfinite(res["fwhm_diag"])]
        if fin.size and 2 * a.eta > 0.05 * fin.min():
            print(f"  [warn] --eta {a.eta:g} eV adds {2 * a.eta * 1e3:.3f} meV to every "
                  f"FWHM, over 5% of the narrowest; lower it")
        if (res["final_width"] > 0).any():
            g_ = res["final_width_gain"]
            print("  final-width gain dGamma_A/deta: "
                  + " ".join("-" if not _np.isfinite(x) else f"{x:.2f}" for x in g_)
                  + "\n    (well below 1: the width is scattering at E_A, the final width "
                    "only resolves the\n    band edge; near 2: the final states' Lorentzian "
                    "tails feed the width)")
            if (_np.nan_to_num(g_) > 1.0).any():
                print(f"  [warn] gain above 1 at T = "
                      f"{', '.join(f'{t:g}' for t, x in zip(Ts, g_) if x > 1.0)} K: the "
                      f"self-consistent width there is\n    mostly the final states' tails; "
                      f"compare --final-width 0 and a fixed --final-width")
        pinned = [T for i, T in enumerate(Ts)
                  if _np.isfinite(res["fwhm_diag"][i])
                  and res["fwhm_diag"][i] < 0.5 * res["gamma_A"][i]]
        if pinned:
            print(f"  [warn] at T = {', '.join(f'{t:g}' for t in pinned)} K the eps_2 line is "
                  f"under half its on-shell width:\n    the A line sits at the bottom "
                  f"of its own band, where Im Sigma steps up within a\n    fraction of a "
                  f"meV, and the line locks below the step. "
                  + ("Give the final states\n    their width: --final-width sc."
                     if fw_mode != "sc" else "Raise --n-omega."))
        out.update(recenter=bool(a.recenter), recenter_shift=_np.array(rc),
                   recenter_shift_full=_np.array(rcf), final_width_mode=fw_mode)
        out.update(dipoles=d, window=_np.array([lo_w, hi_w]), bright=_np.array(members),
                   E_A=E_A, x_plot=xs, eps2_full=_np.array(e_full),
                   eps2_diag=_np.array(e_diag), **res)
    if a.out:
        _np.savez(a.out, **out)
        print(f"\nwrote {a.out}")


def _effective_refine(a, n1):
    """The refinement factor a run actually uses: from --hw-fine (unless an explicit --refine >
    1 subsamples it), else --refine."""
    if a.hw_fine:
        h = np.load(a.hw_fine, mmap_mode="r")
        N = int(round(np.sqrt(h.shape[0]))) if h.ndim == 2 else int(h.shape[0])
        F = N // int(n1)
        return F if (a.refine in (None, 0, 1)) else int(a.refine)
    return int(a.refine or 1)


def _outrate_meta(a, arcs):
    """What the refined out-rates depend on; stored with them and checked when they are read
    back, so a file from another run is never used."""
    first = next(iter(arcs.values()))
    return dict(T=float(a.T), refine=int(a.refine or 0),
                f=_effective_refine(a, first.n1),
                hw_fine=os.path.abspath(a.hw_fine) if a.hw_fine else "",
                acoustic_cut=float(a.acoustic_cut), scheme=a.scheme,
                mesh=[int(first.n1), int(first.n2)], nexc=int(first.nexc),
                nQ=len(arcs))


def _outrate_mismatch(got, meta, a):
    """Settings in which a --outrates file differs from this run."""
    diff = {k: (got.get(k), meta[k]) for k in ("T", "acoustic_cut", "scheme", "mesh",
                                               "nexc", "nQ") if got.get(k) != meta[k]}
    if a.hw_fine or (a.refine and a.refine > 1):
        if "f" in got:
            if got["f"] != meta["f"]:
                diff["refine (effective)"] = (got["f"], meta["f"])
        else:
            if got.get("refine") != meta["refine"]:
                diff["refine"] = (got.get("refine"), meta["refine"])
            if os.path.basename(got.get("hw_fine", "")) != os.path.basename(meta["hw_fine"]):
                diff["hw_fine"] = (got.get("hw_fine"), meta["hw_fine"])
    return diff


def cmd_bte(a):
    """Assemble the collision matrix, inject, propagate, write snapshots."""
    import json
    import numpy as _np
    from . import bte, parallel
    from .io.excph import load_archives

    root = parallel.is_root()
    if a.nt < 2 and not (getattr(a, "plan", False) or a.refine_only):
        raise SystemExit(f"--nt {a.nt}: the snapshot times are linspace(0, --t-end, --nt), "
                         f"so --nt 1 keeps only t = 0 although the solver still integrates "
                         f"to --t-end. Use --nt 2 or more (--nt 3001 with --t-end 300000: "
                         f"every 100 fs)")
    arcs = load_archives(a.pattern, mesh=a.mesh, verbose=root)
    refine_on = bool(a.refine and a.refine > 1) or bool(a.hw_fine)
    bose = a.bose or ("crossing" if a.window is not None else "node")

    ref = None
    meta = _outrate_meta(a, arcs)
    if a.window is not None:
        if a.refine_only or a.save_outrates:
            raise SystemExit("--window builds the matrix on the refined mesh and needs "
                             "no out-rates: drop --refine-only / --save-outrates")
        if not refine_on:
            raise SystemExit("--window needs the refined mesh: --hw-fine and/or --refine > 1")
        if a.outrates and root:
            d = _np.load(a.outrates)
            got = json.loads(str(d["meta"]))
            if got.get("mesh") != meta["mesh"] or got.get("nexc") != meta["nexc"]:
                raise SystemExit(f"{a.outrates} is for another mesh or state count")
            ref = _np.asarray(d["out"], float)
            print(f"out-rates of {a.outrates} (refine {got.get('refine')}, "
                  f"T {got.get('T')} K) read for comparison only")
    elif a.outrates:
        if root:
            d = _np.load(a.outrates)
            got = json.loads(str(d["meta"]))
            diff = _outrate_mismatch(got, meta, a)
            if diff:
                raise SystemExit(f"{a.outrates} was computed with different "
                                 f"settings: {diff}")
            ref = _np.asarray(d["out"], float)
            print(f"out-rates read from {a.outrates} (refine "
                  f"{got.get('f', got.get('refine'))}, hw_fine "
                  f"{os.path.basename(got.get('hw_fine', '')) or 'none'})")
    elif refine_on:
        ref = bte.refined_out_rates(
            arcs, a.T, None if (a.hw_fine and a.refine in (None, 1))
            else a.refine, acoustic_cut=a.acoustic_cut, g2_mode=a.scheme,
            hw_fine=a.hw_fine, workers=a.workers, checkpoint=a.checkpoint,
            mem_gb=a.mem_per_task)
        if root and a.save_outrates:
            _np.savez(a.save_outrates, out=ref, meta=json.dumps(meta))
            print(f"wrote {a.save_outrates}")
    elif a.refine_only:
        raise SystemExit("--refine-only needs --refine > 1 or --hw-fine")

    rm = None
    par = bte.parity_grid(a.parity, arcs) if a.parity else None
    if a.window is not None:
        rm = bte.windowed_rates(
            arcs, a.T, a.window, refine=a.refine, hw_fine=a.hw_fine,
            acoustic_cut=a.acoustic_cut, flat_width=a.flat_width, bose=bose,
            band_interp=a.band_interp, acoustic_model=a.acoustic_model,
            n_shell=a.n_shell, channels=a.channels, workers=a.workers,
            checkpoint=a.checkpoint, mem_gb=a.mem_per_task, out_ref=ref,
            plan_only=a.plan, acoustic_scaling=a.acoustic_scaling,
            acoustic_power=a.acoustic_power, parity=par, deg_tol=a.deg_tol,
            verbose=root)
        if a.plan:
            return

    if a.refine_only or not root:
        return

    if a.threads:
        parallel.set_blas_threads(a.threads)
        avail = parallel.cpu_count()
        if a.threads > avail:
            print(f"  [warn] --threads {a.threads} but this process may use "
                  f"only {avail} cores (CPU binding). Launch with "
                  f"--bind-to none / --cpu-bind=none, or run the "
                  f"propagation as its own single-process step.")
    if rm is None:
        rm = bte.build_rates(arcs, T=a.T, acoustic_cut=a.acoustic_cut,
                             flat_width=a.flat_width or 0.0,
                             channels=a.channels, bose=bose,
                             deg_tol=a.deg_tol, parity=par)
        if ref is not None:
            rm, _ = bte.rescale_rows(rm, ref)

    print()
    bte.check_balance(rm, N_tot=a.density * arcs[(0, 0)].nexc)

    n1, n2, ne = rm.dims
    q0 = bte.q0_states(rm)
    F0 = _np.zeros(len(rm.E))
    if a.inject == "optical":
        arc0 = arcs[(0, 0)]
        if not arc0.has("osc_Q0"):
            raise SystemExit(
                "optical injection needs osc_Q0 in the Q=0 archive. Add the "
                "exciton oscillator strengths in generate_excph.py; only "
                "RELATIVE values matter.")
        F_q0 = bte.inject_optical(arc0.E_n, _np.asarray(arc0._raw("osc_Q0")),
                                  a.density * ne, center=a.pump_center,
                                  sigma=a.pump_sigma)
        kept = q0 >= 0
        F0[q0[kept]] = F_q0[kept]
        if F_q0[~kept].sum() > 1e-6 * F_q0.sum():
            print(f"  [warn] {F_q0[~kept].sum() / F_q0.sum():.1%} of the pumped "
                  f"excitons are in Q=0 states outside the window and are dropped")
    elif a.inject == "thermal":
        F0 = bte.inject_thermal(rm.E, a.density * len(rm.E), a.hot, a.width)
    else:
        sel = range(ne) if not a.state else [k - 1 for k in a.state]
        for k in sel:
            if q0[k] < 0:
                raise SystemExit(f"Q=0 state {k + 1} is outside the window")
            F0[q0[k]] = a.density
        print(f"  uniform injection into Q=0 states {[k+1 for k in sel]}")

    x = (rm.E - rm.E.min()) * 1e3
    print(f"  N_tot = {F0.sum():.4e}, <E>-Emin = {(F0 @ x)/F0.sum():.2f} meV")

    extra = {}

    sol = bte.propagate(rm, F0, t_end=a.t_end, nt=a.nt, rtol=a.rtol,
                        atol=a.atol, jac=a.jac, matrix=a.matrix, verbose=True)
    N = sol.y.sum(axis=0)
    head = f"\n  {'t (fs)':>10} {'<E>-Emin meV':>14} {'N/N0':>10} {'min F':>11}"
    print(head)
    for k, t in enumerate(sol.t):
        row = (f"  {t:10.1f} {(sol.y[:,k] @ x)/N[k]:14.3f} "
               f"{N[k]/N[0]:10.6f} {sol.y[:,k].min():11.3e}")
        print(row)
    worst = float(sol.y.min())
    if worst < 0:
        drift = float(abs(N[-1] / N[0] - 1))
        rel = abs(worst) / max(a.density, 1e-300)
        print(f"\n  most negative occupation {worst:.2e} "
              f"({rel:.1e} of the injected density), N drift {drift:.1e}")
        if abs(worst) < 10 * 1e-12 and drift < 1e-6:
            print("    below the integrator's absolute tolerance and "
                  "conservation holds:\n    this is round-off in states whose "
                  "equilibrium occupation is far\n    smaller than atol. It "
                  "does not affect the dynamics. Lower atol\n    only if you "
                  "need those states quantitatively.")
        else:
            print("    LARGER than the tolerance, or conservation has "
                  "drifted: the\n    quadratic term may have destabilised. "
                  "Lower --density and check\n    whether the cooling curve "
                  "is unchanged.")

    if a.snapshots:
        print()
        bte.save_snapshots(rm, sol, a.snapshots, channels=a.channels, extra=extra)


def cmd_hpc_info(a):
    """Report what xphd sees: ranks, nodes, cores, BLAS threads, memory."""
    from . import parallel
    if a.threads:
        parallel.set_blas_threads(a.threads)
    lines = parallel.describe()
    if parallel.is_root():
        print("\n".join(lines))
        if parallel.size() == 1 and parallel._launched_ntasks() > 1:
            print("[warn] launched as several tasks but MPI is not active")


DELEGATED = {
    "dmats":           ("xphd.generate.dmats",   "main",
                        "rotation matrices Dmats.npy; band window from the BSE"),
    "generate":        ("xphd.generate.archive", "main",
                        "one exciton-phonon archive for a given Q"),
    "check-archive":   ("xphd.generate.check",   "main",
                        "validate ONE archive before generating the rest"),
    "verify-archives": ("xphd.generate.verify",  "main",
                        "confirm the full set of archives is complete"),
    "parity":          ("xphd.parity",           "main",
                        "sigma_h parity of every exciton across the zone"),
    "irreps":          ("xphd.irreps",           "main",
                        "irrep labels at the high-symmetry points"),
    "selection-rule":  ("xphd.selection",        "main",
                        "coupling split into symmetry-allowed and -forbidden"),
    "dipoles":         ("xphd.dipoles",          "main",
                        "exciton dipoles for emission, validated against yambo"),
    "chirality":       ("xphd.chirality",        "main",
                        "helicity and valley polarisation of a degenerate pair"),
    "arpes":           ("xphd.arpes",            "main",
                        "exciton photoemission I(k,E), thermal or time-resolved"),
    "band-parity":     ("xphd.band_parity",      "main",
                        "sigma_h parity of the electronic bands, path and zone"),
    "mode-labels":     ("xphd.mode_labels",      "main",
                        "phonon branch characters on the archive mesh"),
    "matdyn-split":    ("xphd.matdyn_split",     "main_split",
                        "split a long q-list into chunks for matdyn.x"),
    "matdyn-merge":    ("xphd.matdyn_split",     "main_merge",
                        "merge chunked matdyn.modes and .freq outputs"),
    "modes-check":     ("xphd.matdyn_split",     "main_check",
                        "verify a merged matdyn.modes before using it"),
}


MPI_AWARE = {"sweep", "bte", "generate", "hpc-info"}


def _delegate(argv):
    """Run a DELEGATED command if argv names one; return True if it did."""
    if not argv or argv[0] not in DELEGATED:
        return False
    import importlib
    mod, fn, _ = DELEGATED[argv[0]]
    getattr(importlib.import_module(mod), fn)(argv[1:])
    return True


def main(argv=None):
    p = argparse.ArgumentParser(prog="xphd", description=__doc__)
    p.add_argument("--version", action="version", version=f"xphd {_version()}")
    p.add_argument("--no-report", action="store_true",
                   help="no report after the run (also XPHD_REPORT=0); by default "
                        "a report of the command, inputs, parameters, outputs and log is "
                        "written: <output>_report.pdf with plots of the results on "
                        "Windows and macOS, <output>_report.txt on Linux (XPHD_REPORT="
                        "pdf, txt or both to choose)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def archive_arg(sp):
        sp.add_argument("npz")
        sp.add_argument("--mesh", type=int, nargs=2, default=None)
        return sp

    archive_arg(sub.add_parser("check", help="archive integrity checks")) \
        .set_defaults(func=cmd_check)

    s = sub.add_parser("matdyn", help="read/validate matdyn frequencies")
    s.add_argument("freq")
    s.add_argument("--fine", type=int, default=360)
    s.add_argument("--nbnd", type=int, default=None)
    s.add_argument("--npz", default=None)
    s.add_argument("--mesh", type=int, nargs=2, default=None)
    s.add_argument("--q-order", choices=("rowmajor", "npz"), default="rowmajor")
    s.add_argument("-o", "--out", default=None)
    s.set_defaults(func=cmd_matdyn)

    s = archive_arg(sub.add_parser("linewidth", help="exciton linewidths"))
    s.add_argument("--hw-fine", default=None)
    s.add_argument("--fine", type=int, default=None)
    s.add_argument("--refine", type=int, default=1)
    s.add_argument("--scheme", choices=("const", "fft"), default="const")
    s.add_argument("--T", type=float, nargs="+", default=[300.0])
    s.add_argument("--flat-width", type=float, default=None)
    s.add_argument("--acoustic-cut", type=float, default=1e-4)
    s.add_argument("--acoustic-model", choices=("none", "q2", "auto"),
                   default="none",
                   help="acoustic coupling inside the cell around q = 0, where "
                        "the constant-coupling value is zero by the sum rule. "
                        "none: leave it zero, which drops small-q acoustic "
                        "scattering -- the ONLY channel at a band minimum. "
                        "q2: |G|^2 * 2w = C q^2, the form for a neutral "
                        "exciton; use this. auto: also keep a constant A where "
                        "it is above 2 sigma -- right for free carriers, but "
                        "for an exciton a kept A is a fitting artefact and is "
                        "flagged")
    s.add_argument("--n-shell", type=int, default=3,
                   help="innermost shells of the source mesh the fit uses; "
                        "at least 1 for q2, 3 for auto. The per-shell C it "
                        "prints shows whether the result depends on the choice")
    s.add_argument("--band-interp", choices=("local", "fourier"),
                   default="local",
                   help="how the exciton energies are refined onto the fine "
                        "mesh. local (default): a quadratic around every mesh "
                        "point from its six neighbours, which cannot ring. "
                        "fourier: the global interpolant, which rings from "
                        "every band crossing -- at GaN's K minimum it made the "
                        "band 34%% too steep. Reproduces earlier results")
    s.add_argument("-o", "--out", default=None)
    s.add_argument("--per-mode", action="store_true",
                   help="also print linewidths by phonon branch; with a text "
                        "-o, write <stem>_modes.txt as well")
    s.set_defaults(func=cmd_linewidth)

    s = sub.add_parser("sweep", help="linewidths at every Q in the zone")
    s.add_argument("pattern", help="glob, e.g. 'GI_ExcPh_Q*.npz'")
    s.add_argument("--mesh", type=int, nargs=2, default=None)
    s.add_argument("--hw-fine", default=None)
    s.add_argument("--fine", type=int, default=None)
    s.add_argument("--refine", type=int, default=None)
    s.add_argument("--scheme", choices=("const", "fft"), default="const")
    s.add_argument("--T", type=float, nargs="+", default=[77.0])
    s.add_argument("--flat-width", type=float, default=None)
    s.add_argument("--acoustic-cut", type=float, default=1e-4)
    s.add_argument("--workers", type=int, default=1,
                   help="local processes (per MPI rank under mpirun/srun)")
    s.add_argument("--checkpoint", metavar="DIR", default=None,
                   help="save each finished Q in DIR; a rerun skips them")
    s.add_argument("--unfold", metavar="SAVE", default=None)
    s.add_argument("--legacy", action="store_true",
                   help="also write Qx, Qy and a 2D LW (nexc, nq) in meV, as "
                        "aggregate.py did, for existing plotting scripts")
    s.add_argument("-o", "--out", default="lw_field.npz")
    s.add_argument("--acoustic-model", choices=("none", "q2", "auto"),
                   default="none",
                   help="small-q acoustic coupling, as in xphd linewidth; "
                        "use q2, and the same --n-shell as the single-Q runs")
    s.add_argument("--n-shell", type=int, default=3)
    s.add_argument("--band-interp", choices=("local", "fourier"),
                   default="local",
                   help="how the exciton energies are refined onto the fine "
                        "mesh. local (default): a quadratic around every mesh "
                        "point from its six neighbours, which cannot ring. "
                        "fourier: the global interpolant, which rings from "
                        "every band crossing -- at GaN's K minimum it made the "
                        "band 34%% too steep. Reproduces earlier results")
    s.set_defaults(func=cmd_sweep)

    s = archive_arg(sub.add_parser(
        "selfenergy", help="Sigma(omega), quasiparticle energy, spectral function"))
    s.add_argument("--state", type=int, default=1)
    s.add_argument("--T", type=float, default=77.0)
    s.add_argument("--hw-fine", default=None)
    s.add_argument("--refine", type=int, default=1,
                   help="the Kramers-Kronig transform integrates over omega "
                        "and does not need the on-shell resolution; a coarser "
                        "refinement here is usually enough")
    s.add_argument("--window", type=float, default=0.0,
                   help="eV, half-width about the bare energy. 0 (default) "
                        "spans the full support of Im Sigma instead, which is "
                        "what Kramers-Kronig needs: Im Sigma is one-sided for "
                        "states near the bottom of the band, so a "
                        "symmetric window cuts through its maximum.")
    s.add_argument("--n-omega", type=int, default=241)
    s.add_argument("--acoustic-cut", type=float, default=1e-4)
    s.add_argument("-o", "--out", default=None)
    s.add_argument("--band-interp", choices=("local", "fourier"),
                   default="local",
                   help="exciton-energy refinement, as in xphd linewidth")
    s.set_defaults(func=cmd_selfenergy)

    s = archive_arg(sub.add_parser(
        "lineshape", help="full Sigma matrix and the absorption lineshape"))
    s.add_argument("--T", type=float, nargs="+", default=[77.0],
                   help="K; several temperatures share one pass over the grid")
    s.add_argument("--hw-fine", default=None)
    s.add_argument("--refine", type=int, default=1)
    s.add_argument("--n-omega", type=int, default=1201,
                   help="points of the uniform self-energy grid (Kramers-Kronig); "
                        "keep its step <= ~1 meV")
    s.add_argument("--pad", type=float, default=0.15,
                   help="eV beyond the support of Im Sigma. The grid already "
                        "spans the final-state range plus the largest phonon; "
                        "this is extra margin.")
    s.add_argument("--acoustic-cut", type=float, default=1e-4)
    s.add_argument("--acoustic-model", choices=("none", "shell"), default="none",
                   help="the q = 0 cell for the acoustic branches: 'shell' fills it from "
                        "the first shell between the same states (needs --parity), as "
                        "xphd bte --window does")
    s.add_argument("--acoustic-scaling", choices=("none", "omega"), default="none",
                   help="'omega' carries acoustic couplings to the fine q as g^2 ~ w, "
                        "as xphd bte --acoustic-scaling omega")
    s.add_argument("--parity", default=None,
                   help="exciton_parity.npz (sigma_h parity of every state at every Q), "
                        "for --acoustic-model shell and the symmetry report")
    s.add_argument("--bsdb", default=None,
                   help="ndb.BS_diago_Q1 of the SAME diagonalisation as the archive, "
                        "for the dipoles and hence eps_2")
    s.add_argument("--cells", choices=("hex", "rhombic"), default="hex",
                   help="which coarse point a fine point takes its coupling and energy fit "
                        "from: its Wigner-Seitz hexagon (default; keeps the point-group "
                        "symmetry of Sigma) or the rounded reduced coordinates (the BTE's "
                        "and xphd linewidth's cells, not C3-invariant)")
    s.add_argument("--dipoles", default=None,
                   help="the dipoles as a .npy (complex, one per state) instead of "
                        "--bsdb, e.g. saved with lineshape.read_dipoles")
    s.add_argument("--eta", type=float, default=1e-5,
                   help="eV, added to every width; keep it far below the narrowest")
    s.add_argument("--window", type=float, nargs=2, default=None,
                   help="eV, where to look for the A peak (default: from the lowest "
                        "bright multiplet to half way to the next)")
    s.add_argument("--n-plot", type=int, default=4001,
                   help="points of the saved eps_2 curves")
    s.add_argument("--final-width", default="sc",
                   help="half-width given to every final state, Sigma(w) -> "
                        "Sigma(w + i eta): 'sc' (default) equal to half the A width it "
                        "produces (self-consistent), or a value in eV; 0 is one-shot. "
                        "The A line sits at the bottom of its own band, where the "
                        "acoustic channel makes Im Sigma step up within a fraction of a "
                        "meV: one-shot, the on-shell width depends on where E_A falls "
                        "on the frequency grid and the eps_2 line locks below the step")
    s.add_argument("--no-recenter", dest="recenter", action="store_false",
                   help="compute eps_2 with the A line where Re Sigma puts it; by default "
                        "the static A shift is taken off the whole matrix (all excitons "
                        "renormalised together)")
    s.add_argument("-o", "--out", default=None)
    s.add_argument("--band-interp", choices=("local", "fourier"),
                   default="local",
                   help="exciton-energy refinement, as in xphd linewidth")
    s.set_defaults(func=cmd_lineshape)

    s = sub.add_parser("bte", help="real-time Boltzmann transport")
    s.add_argument("pattern", help="glob, e.g. 'GI_ExcPh_Q*.npz'")
    s.add_argument("--mesh", type=int, nargs=2, default=None)
    s.add_argument("--T", type=float, default=300.0, help="lattice temperature")
    s.add_argument("--acoustic-cut", type=float, default=1e-4)
    s.add_argument("--flat-width", type=float, default=None,
                   help="eV; default 0 for the mesh matrix, 2.4e-3/(n f) with "
                        "--window (as the refined out-rates)")
    s.add_argument("--refine", type=int, default=1,
                   help="refine the OUT-RATE integral and rescale rows. The "
                        "matrix itself cannot be refined (each q is a state, "
                        "so its size grows as f^4) but the row sum can -- "
                        "or, with --window, the matrix on the refined mesh "
                        "for the bottom of the valleys only.")
    s.add_argument("--window", type=float, default=None, metavar="EV",
                   help="build the matrix on the REFINED mesh (--hw-fine / "
                        "--refine) for the states within EV of the lowest "
                        "exciton, instead of the mesh matrix rescaled to the "
                        "refined out-rates. For valleys too steep for the "
                        "source mesh. The Boltzmann weight left outside is "
                        "printed; --hw-fine may be finer than --refine "
                        "(subsampled exactly)")
    s.add_argument("--band-interp", choices=("local", "fourier"),
                   default="local",
                   help="exciton energies on the refined mesh, with --window "
                        "(as in xphd linewidth)")
    s.add_argument("--acoustic-model", choices=("none", "q2", "shell"), default="none",
                   help="with --window: acoustic coupling between fine states "
                        "with the same parent (the cell around q = 0). none: zero "
                        "(as the refined out-rates); q2: the fitted C q^2 form of "
                        "xphd linewidth (by band index: unsafe where bands of "
                        "opposite parity reorder); shell: the first-shell coupling "
                        "between the same states, matched by parity (needs "
                        "--parity), carried down by --acoustic-scaling")
    s.add_argument("--parity", default=None,
                   help="parity.npz of the exciton states (Q_red, chi, E): needed by "
                        "--window --acoustic-model shell; with it, degenerate "
                        "multiplets (--deg-tol) never join opposite parities")
    s.add_argument("--deg-tol", type=float, default=1e-4,
                   help="eV: states of one Q closer than this form a degenerate "
                        "multiplet, over which the couplings are averaged, so the "
                        "populations do not depend on the basis the BSE solver "
                        "picked inside it (0: off)")
    s.add_argument("--n-shell", type=int, default=3,
                   help="shells for --acoustic-model q2")
    s.add_argument("--acoustic-scaling", choices=("none", "omega", "power"), default="none",
                   help="with --window: how an acoustic branch's coarse coupling is "
                        "carried to the fine q. none: constant; omega: g^2 ~ w, the "
                        "long-wavelength law for strain and curvature coupling; power: "
                        "|M|^2 ~ q^n (--acoustic-power n) with the actual fine w. For a "
                        "soft quadratic branch (ZA) 'none' makes the rate follow "
                        "--acoustic-cut; see xphd.bte.windowed_rates")
    s.add_argument("--acoustic-power", type=float, default=None,
                   help="n of |M|^2 ~ q^n for --acoustic-scaling power (ZA: 2 tilt, "
                        "4 curvature; za_coupling_law.py measures it)")
    s.add_argument("--plan", action="store_true",
                   help="with --window: print the states kept, the memory, the "
                        "Boltzmann weight left out and the energy step against "
                        "kT, then stop (seconds; use it to pick --refine and "
                        "--window)")
    s.add_argument("--hw-fine", default=None,
                   help="matdyn frequencies on the refined mesh, as used by "
                        "the linewidths. Sets the refinement and makes the "
                        "two calculations use identical phonons.")
    s.add_argument("--scheme", choices=("const", "complexfft"), default="const")
    s.add_argument("--bose", choices=("node", "crossing"), default=None,
                   help="phonon occupation in the rate matrix: node = at the "
                        "energy difference of the two mesh states; crossing = "
                        "each phonon's own occupation where the delta crossing "
                        "happens, as the refined out-rates use. Default: node "
                        "for the mesh matrix (as before), crossing with "
                        "--window (see xphd.bte.windowed_rates)")
    s.add_argument("--inject", choices=("optical", "thermal", "uniform"),
                   default="optical")
    s.add_argument("--state", type=int, nargs="*", default=None,
                   help="1-based states for --inject uniform")
    s.add_argument("--pump-center", type=float, default=None, help="eV")
    s.add_argument("--pump-sigma", type=float, default=None, help="eV")
    s.add_argument("--hot", type=float, default=150.0,
                   help="meV above the minimum, for --inject thermal")
    s.add_argument("--width", type=float, default=25.0, help="meV")
    s.add_argument("--density", type=float, default=1e-3,
                   help="per-state occupation scale")
    s.add_argument("--t-end", type=float, default=10000.0, help="fs")
    s.add_argument("--nt", type=int, default=40,
                   help="number of saved snapshot times, linspace(0, --t-end, --nt) "
                        "(>= 2). Not the integration steps: the BDF solver picks "
                        "its own adaptive steps, set by --rtol/--atol")
    s.add_argument("--workers", type=int, default=1,
                   help="parallel processes for the out-rate refinement, "
                        "which dominates the cost at large --refine (per "
                        "MPI rank under mpirun/srun)")
    s.add_argument("--threads", type=int, default=None,
                   help="BLAS/OpenMP threads for assembly and propagation "
                        "on rank 0 (default: OMP_NUM_THREADS as set)")
    s.add_argument("--checkpoint", metavar="DIR", default=None,
                   help="save each refined Q in DIR; a rerun skips them")
    s.add_argument("--mem-per-task", type=float, default=None, metavar="GB",
                   help="memory bound for one refinement task; its triangles "
                        "are processed in chunks to stay inside it (default "
                        "XPHD_TASK_MEM_GB or 2)")
    s.add_argument("--refine-only", action="store_true",
                   help="stop after the refined out-rates (use with "
                        "--save-outrates under mpirun)")
    s.add_argument("--save-outrates", metavar="FILE", default=None)
    s.add_argument("--outrates", metavar="FILE", default=None,
                   help="read refined out-rates written by --save-outrates "
                        "instead of computing them; settings are checked")
    s.add_argument("--jac", choices=("analytic", "fd"), default="analytic",
                   help="Jacobian for the stiff solver. analytic: exact and "
                        "O(n^2); fd: finite differences, O(n^3), as before "
                        "1.1. Same trajectory within --rtol")
    s.add_argument("--matrix", choices=("dense", "sparse", "auto"),
                   default="dense",
                   help="storage for the propagation; sparse pays only when "
                        "most state pairs are uncoupled")
    s.add_argument("--rtol", type=float, default=1e-7)
    s.add_argument("--atol", type=float, default=1e-12)
    s.add_argument("--channels", action="store_true",
                   help="keep channel-resolved rates so the animation can "
                        "colour each state by the phonon mode filling it. "
                        "Costs 2*nmodes matrices of nstates^2 in float32.")
    s.add_argument("--snapshots", default=None)
    s.set_defaults(func=cmd_bte)

    s = sub.add_parser("hpc-info",
                       help="ranks, nodes, cores, BLAS threads and memory "
                            "as xphd sees them (run it under mpirun first)")
    s.add_argument("--threads", type=int, default=None)
    s.set_defaults(func=cmd_hpc_info)

    s = sub.add_parser("ibz", help="list the Q-points that need generating")
    s.add_argument("--savepath", default="SAVE",
                   help="directory holding ns.db1")
    s.add_argument("--mesh", type=int, nargs=2, default=None,
                   help="only for the fallback when yambopy is unavailable")
    s.add_argument("--iq-map", default=None,
                   help="fallback only: an archive whose q_red ordering to "
                        "report the points in")
    s.set_defaults(func=cmd_ibz)

    s = sub.add_parser("unfold",
                       help="symmetrise, or expand an IBZ set onto the full BZ")
    s.add_argument("field")
    s.add_argument("--savepath", default="SAVE")
    s.add_argument("--mesh", type=int, nargs=2, default=None)
    s.add_argument("--legacy", action="store_true",
                   help="also write Qx, Qy and a 2D LW (nexc, nq) in meV")
    s.add_argument("-o", "--out", default="lw_field_sym.npz")
    s.set_defaults(func=cmd_unfold)

    for name, (_, _, helptext) in DELEGATED.items():
        sub.add_parser(name, help=helptext, add_help=False)

    import sys
    import contextlib
    from . import parallel
    from . import report
    argv = sys.argv[1:] if argv is None else list(argv)
    argv, want_report = report.enabled(argv)

    parallel.silence_nonroot()
    if parallel.size() > 1 and argv and argv[0] not in MPI_AWARE:
        if parallel.is_root():
            print(f"[xphd] '{argv[0]}' is serial; running it on rank 0 only "
                  f"(MPI-parallel: {', '.join(sorted(MPI_AWARE))})")
        else:
            parallel.finish()
            return
    run = report.Run(argv) if (want_report and parallel.is_root()) else None
    try:
        if (argv and not ({"-h", "--help", "--version"} & set(argv))
                and parallel.is_root()):
            from .banner import banner
            text = banner(_version(), argv)
            if run is not None:
                run.banner = text
        with (run.capture() if run is not None else contextlib.nullcontext()):
            if not _delegate(argv):
                a = p.parse_args(argv)
                a.func(a)
    except BaseException as e:
        if run is not None and not run.usage_error(e):
            run.finish(e)
            _write_report(run)
        if parallel.size() > 1 and not (isinstance(e, SystemExit)
                                        and e.code in (0, None)):
            import traceback
            if not isinstance(e, SystemExit):
                traceback.print_exc()
            elif e.code not in (1,):
                sys.stderr.write(f"{e.code}\n")
            sys.stderr.write(f"[xphd] rank {parallel.rank()} failed; "
                             f"stopping all {parallel.size()} ranks\n")
            parallel.abort(1)
        raise
    if run is not None:
        run.finish()
        _write_report(run)
    parallel.finish()


def _write_report(run):
    try:
        paths = run.write(_version())
        print("\n" + "\n".join(f"report: {p}" for p in
                               ([paths] if isinstance(paths, str) else paths)))
    except Exception as e:
        print(f"\n[xphd] no report written: {e}")


if __name__ == "__main__":
    main()
