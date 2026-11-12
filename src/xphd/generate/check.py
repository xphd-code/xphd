#!/usr/bin/env python3
"""check_archive.py ================ Validate ONE exciton-phonon archive before launching the
rest."""
from __future__ import annotations

import sys

import numpy as np

REQUIRED = {
    "G_grid":     "complex amplitude; gauge and localisation diagnostics, lineshape",
    "g2_grid":    "|G|^2; linewidth, sweep, BTE, selection-rule test",
    "Ge_grid":    "electron part; interference analysis",
    "Gh_grid":    "hole part; interference analysis",
    "E_n_grid":   "initial exciton energies at Q",
    "E_m_grid":   "final exciton energies at Q+q",
    "hw_grid":    "phonon energies on the source mesh",
    "q_red":      "reduced q; mapping onto the mesh",
    "Q_red":      "exciton momentum; symmetry and parity maps",
    "mesh":       "mesh size; ExcPhArchive refuses the file without it",
}
USEFUL = {
    "elph_convention": "momentum convention of the e-ph elements",
    "bse_nv":          "valence bands in the window (--nv for symmetry scripts)",
    "bse_nc":          "conduction bands in the window",
    "bse_bands":       "the BSE band slice",
}


PH_TOL = 1e-4
OK_TOL, FAIL_TOL = 1e-2, 5e-2


def _groups(E, tol):
    """Index groups of E: neighbours in sorted order closer than tol share one."""
    order = np.argsort(E)
    out, cur = [], [int(order[0])]
    for a, b in zip(order[:-1], order[1:]):
        if E[b] - E[a] < tol:
            cur.append(int(b))
        else:
            out.append(np.array(cur))
            cur = [int(b)]
    out.append(np.array(cur))
    return out


def _multiplet_sums(g2q, pg, ig, fg):
    """|G|^2 summed over each (phonon group, initial group, final group)."""
    S = np.zeros((len(pg), len(ig), len(fg)))
    for a, p in enumerate(pg):
        gp = g2q[p].sum(axis=0)
        for b, i in enumerate(ig):
            row = gp[i].sum(axis=0)
            for c, f in enumerate(fg):
                S[a, b, c] = row[f].sum()
    return S


def symmetry_checks(arc, deg_tol, detail=False):
    """Do the couplings obey the crystal's symmetry?"""
    Q = np.asarray(arc.Q_red, float)[:2]
    if np.abs(2 * Q - np.rint(2 * Q)).max() > 1e-6:
        print("   skipped: Q is not a time-reversal-invariant point, where "
              "q and -q are related")
        return True
    i, j = arc._i, arc._j
    g2 = np.asarray(arc.grid("g2"))[i, j]
    hw = np.asarray(arc.grid("hw"))[i, j]
    Em = np.asarray(arc.grid("E_m"))[i, j]
    En = np.asarray(arc.E_n, float)
    qf = np.asarray(arc.q_red, float)[:, :2]
    nq = len(qf)
    if g2.shape[2] < 3 or g2.shape[3] < 3:
        print("   skipped: fewer than three states")
        return True

    def image(M):
        qi = qf @ np.asarray(M, float).T
        d = qi[:, None, :] - qf[None, :, :]
        d -= np.rint(d)
        dist = np.linalg.norm(d, axis=-1)
        k = np.argmin(dist, axis=1)
        return k, float(dist[np.arange(nq), k].max())

    ig = _groups(En, deg_tol)[:-1]
    tests = [("time reversal", "q -> -q", [-np.eye(2)])]
    c3 = []
    for name, M in (("60 deg basis", [[-1, -1], [1, 0]]),
                    ("120 deg basis", [[0, -1], [1, -1]])):
        k, dmax = image(M)
        c3.append((float(np.abs(Em[k] - Em).max()), name, M, dmax))
    c3.sort(key=lambda t: t[0])
    if c3[0][0] < 1e-4 and c3[0][3] < 1e-5:
        tests.append(("C3 rotation", f"q -> C3 q ({c3[0][1]})", [c3[0][2]]))
    else:
        print(f"   C3 rotation skipped: neither in-plane C3 matrix leaves the "
              f"exciton energies invariant (best {c3[0][0]:.1e} eV)")

    ok = True
    for name, desc, (M,) in tests:
        k, dmax = image(M)
        if dmax > 1e-5:
            print(f"   skipped {name}: the q-mesh is not mapped onto itself "
                  f"({dmax:.1e})")
            continue
        dE = max(float(np.abs(Em[k] - Em).max()), float(np.abs(hw[k] - hw).max()))
        num = den = worst = ref = 0.0
        ngrp = 0
        by_key, by_q = {}, np.zeros(nq)
        for q0 in range(nq):
            pg = _groups(hw[q0], PH_TOL)
            fg = _groups(Em[q0], deg_tol)[:-1]
            if not ig or not fg:
                continue
            Sa = _multiplet_sums(g2[q0], pg, ig, fg)
            Sb = _multiplet_sums(g2[k[q0]], pg, ig, fg)
            D = np.abs(Sa - Sb)
            num += D.sum()
            den += 0.5 * (Sa + Sb).sum()
            worst = max(worst, float(D.max()))
            ref = max(ref, float(Sa.max()))
            ngrp = max(ngrp, Sa.size)
            by_q[q0] = D.sum()
            if detail:
                for a_, p_ in enumerate(pg):
                    for b_, i_ in enumerate(ig):
                        for c_, f_ in enumerate(fg):
                            key = (tuple(int(x) + 1 for x in p_),
                                   tuple(int(x) + 1 for x in i_),
                                   tuple(int(x) + 1 for x in f_))
                            by_key[key] = by_key.get(key, 0.0) + D[a_, b_, c_]
        l1, linf = num / max(den, 1e-300), worst / max(ref, 1e-300)
        tag = "OK  " if l1 < OK_TOL else ("WARN" if l1 < FAIL_TOL else "FAIL")
        ok &= tag != "FAIL"
        print(f"   {tag} {name:<14} {desc:<34} L1 {l1:.1e}  max {linf:.1e}"
              f"   (energies differ by {dE:.0e} eV)")
        if detail and tag != "OK  ":
            tot = sum(by_key.values())
            print(f"        where the {name} mismatch sits (share of the total):")
            print("          branches        initial states   final states")
            for key, v in sorted(by_key.items(), key=lambda t: -t[1])[:8]:
                br, ini, fin = (",".join(map(str, x)) for x in key)
                print(f"          {br:<15} {ini:<16} {fin:<14} {100 * v / tot:5.1f}%")
            singles = {k[1][0] for k in sorted(by_key, key=lambda t: -by_key[t])[:6]
                       if len(k[1]) == 1}
            for st in sorted(singles):
                others = np.delete(En, st - 1)
                gap = float(np.abs(others - En[st - 1]).min())
                if gap < 0.03:
                    print(f"        state {st} has a partner {gap * 1e3:.1f} meV away: if the "
                          f"two are a doublet split by the\n        long-range exchange, "
                          f"rerun with --deg-tol above {gap * 1e3:.1f} meV")
            print("        worst q-points (reduced), their image, share:")
            for q0 in np.argsort(-by_q)[:6]:
                print(f"          {np.round(qf[q0], 4)}  ->  {np.round(qf[k[q0]], 4)}"
                      f"   {100 * by_q[q0] / by_q.sum():5.1f}%")
    print(f"   |G|^2 summed over multiplets closer than {deg_tol * 1e3:g} meV "
          f"({len(ig)} initial groups); OK below {OK_TOL:g}, FAIL above {FAIL_TOL:g}.")
    if not ok:
        print("   A mismatch means states at q and at its symmetry image are not\n"
              "   related as they must be. First rule out a doublet split by more\n"
              "   than --deg-tol (--detail says so); otherwise suspect the rotation\n"
              "   of the excitons from the irreducible Q (for spinors, its time-\n"
              "   reversal step and the D-matrices), not the coupling itself.")
    return ok


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="xphd check-archive",
                                description="Validate ONE archive before "
                                            "generating the rest.")
    p.add_argument("archive")
    p.add_argument("--detail", action="store_true",
                   help="attribute any symmetry mismatch to phonon branches, "
                        "initial and final states, and list the worst q-points")
    p.add_argument("--deg-tol", type=float, default=0.015,
                   help="eV; states closer than this form one multiplet in the "
                        "symmetry test (default 15 meV). At Q = 0 the BSE's "
                        "long-range exchange, taken along one direction, splits "
                        "each E' doublet into a longitudinal and a transverse "
                        "member -- by 10 meV for GaN's second bright pair -- and "
                        "each member alone is not C3-symmetric; only their sum "
                        "is. Grouping more states than strictly degenerate only "
                        "sums symmetric quantities, so a generous value is safe")
    a = p.parse_args(argv)
    path = a.archive
    d = np.load(path, allow_pickle=False)
    have = set(d.files)
    print(f"{path}\n")

    bad = 0
    print("  REQUIRED")
    for k, why in REQUIRED.items():
        ok = k in have
        bad += not ok
        shape = str(d[k].shape) if ok else "--"
        print(f"   {'OK ' if ok else 'MISSING'} {k:<11} {shape:<22} {why}")
    print("\n  USEFUL")
    for k, why in USEFUL.items():
        ok = k in have
        val = (str(d[k]) if d[k].ndim == 0 else str(d[k].ravel()[:4])) if ok else "--"
        print(f"   {'OK ' if ok else '-- '} {k:<16} {val:<14} {why}")

    if bad:
        raise SystemExit(f"\n  {bad} required field(s) missing. Fix the generator "
                         f"before running the remaining Q.")

    print("\n  CONSISTENCY")
    from ..io.excph import ExcPhArchive
    arc = ExcPhArchive(path)
    print(f"   mesh {arc.n1}x{arc.n2}, {arc.nmod} branches, {arc.nexc} states, "
          f"Q = {np.round(arc.Q_red, 4)}")
    G = np.asarray(d["G_grid"])
    g2 = np.asarray(d["g2_grid"])
    Ge, Gh = np.asarray(d["Ge_grid"]), np.asarray(d["Gh_grid"])

    def rel(a, b):
        return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))

    checks = [
        ("|G|^2 == g2", rel(np.abs(G) ** 2, g2), 1e-10),
        ("Ge + Gh == G", rel(Ge + Gh, G), 1e-5),
    ]
    Em = arc.grid("E_m")
    checks.append(("E_m(q=0) == E_n", float(np.abs(Em[0, 0] - arc.E_n).max()), 1e-6))
    ok_all = True
    for name, val, tol in checks:
        ok = val < tol
        ok_all &= ok
        print(f"   {'OK ' if ok else 'FAIL'} {name:<18} {val:.2e}  (tol {tol:.0e})")

    g2g = arc.grid("g2")
    print(f"   acoustic |G|^2 at Gamma: {g2g[0, 0, :3].max():.2e} eV^2  "
          f"(should vanish by the acoustic sum rule)")
    hw = arc.grid("hw")
    print(f"   phonons {hw.min()*1e3:.2f} .. {hw.max()*1e3:.2f} meV")

    print("\n  SYMMETRY")
    ok_all &= symmetry_checks(arc, a.deg_tol, a.detail)

    n = int(round(np.sqrt(len(arc.q_red))))
    print(f"\n   {len(arc.q_red)} q-points on a {n}x{n} mesh -> "
          f"generate iQ = 0 .. {n*n - 1} for the full set of {n*n} archives")
    print(f"   {'READY: launch the rest.' if ok_all else 'NOT READY: resolve the FAIL lines.'}")


if __name__ == "__main__":
    main()
