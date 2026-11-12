"""Irreducible-representation labels at the high-symmetry points."""
from __future__ import annotations

import argparse
import os

import numpy as np

from .excsym import (CLASSES, HIGH_SYMMETRY, assign_irreps, class_characters,
                     decompose_characters, degeneracy_groups, group_name,
                     rep_closure, rep_matrix_Q, shift_map, stabilizer)
from .spin import spin_report
from .yambo import (full_zone_kpoints, ibz_parents, lattice, load_dmats,
                    load_excitons, spatial_ops)


def main(argv=None):
    p = argparse.ArgumentParser(prog="xphd irreps",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--save", default="../phonons/SAVE")
    p.add_argument("--bse-dir", default="../BSE/output_all")
    p.add_argument("--dmats", default="Dmats.npy")
    p.add_argument("--nv", type=int, required=True)
    p.add_argument("--points", nargs="*", default=["G", "K", "M"],
                   help="any of: " + ", ".join(sorted(HIGH_SYMMETRY)))
    p.add_argument("--nstates", type=int, default=15)
    p.add_argument("--deg-tol", type=float, default=5e-3,
                   help="eV; states closer than this form one multiplet")
    p.add_argument("--closure-tol", type=float, default=1e-2,
                   help="largest |D(R)D(S) - D(RS)| accepted for a multiplet "
                        "(default 1e-2)")
    p.add_argument("-o", "--out", default=None,
                   help="write the table to an .npz (per point: energies, "
                        "dimensions, chi(sh), labels, residuals, closure, "
                        "characters), e.g. for scripts/compare_mos2_refs.py")
    a = p.parse_args(argv)

    ylat = lattice(a.save)
    D = load_dmats(a.dmats)
    kf = full_zone_kpoints(ylat)
    kidx, first = ibz_parents(ylat)
    ops = spatial_ops(a.save, kf, D, a.nv)
    print(f"{len(ops)} spatial operations; {len(kf)} k-points, "
          f"{len(first)} irreducible")
    sh_ops = [o for o in ops if o[0] == "sh"]
    sp = spin_report(ylat, a.save, sh_ops[0] if sh_ops else None)
    saved = {"points": [], "nspinor": -1 if sp["nspinor"] is None else sp["nspinor"],
             "spin_kind": str(sp["kind"]),
             "sign_v": 0 if sp["sign_v"] is None else sp["sign_v"],
             "sign_c": 0 if sp["sign_c"] is None else sp["sign_c"],
             "deg_tol": a.deg_tol, "nstates": a.nstates}

    for pt in a.points:
        if pt not in HIGH_SYMMETRY:
            print(f"\n[skip] unknown point {pt!r}")
            continue
        Qreq = np.array([*HIGH_SYMMETRY[pt], 0.0])
        d = kf[:, :2] - np.mod(Qreq[:2], 1.0)
        d -= np.rint(d)
        i = int(np.argmin(np.linalg.norm(d, axis=1)))
        if np.linalg.norm(d[i]) > 1e-5:
            print(f"\n[skip] {pt} is not on this mesh")
            continue
        j = int(kidx[i])
        Q = kf[first[j]]
        path = f"{a.bse_dir}/ndb.BS_diago_Q{j + 1}"
        print(f"\n{'=' * 64}\n{pt}: requested {np.round(Qreq[:2], 4)}, "
              f"database BS_diago_Q{j + 1} at {np.round(Q[:2], 4)}")
        if not os.path.exists(path):
            print(f"   missing {path}")
            continue

        stab = stabilizer(ops, Q)
        grp = group_name(stab)
        counts = {}
        for o in stab:
            counts[o[0]] = counts.get(o[0], 0) + 1
        print(f"   little group {grp} (order {len(stab)}): "
              + ", ".join(f"{k}x{v}" for k, v in sorted(counts.items())))

        kQ, w = shift_map(kf, Q)
        if kQ is None:
            print(f"   k+Q leaves the grid ({w:.1e}); skipped")
            continue
        _, A, E = load_excitons(path, ylat, a.nstates)
        groups = degeneracy_groups(E, tol=a.deg_tol)

        g0 = groups[0]
        worst = max(np.abs(M @ M.conj().T - np.eye(len(g0))).max()
                    for M in (rep_matrix_Q(A, o[2], o[3], o[4], kQ, g0)
                              for o in stab))
        print(f"   unitarity on the lowest multiplet: {worst:.1e}"
              + ("" if worst < 1e-2 else "   <-- labels below are unsafe"))

        print(f"   {'states':<9} {'E (eV)':>9} {'chi(sh)':>9} "
              f"{'irrep':>12} {'resid':>9} {'closure':>9}")
        bad, rows = [], []
        for g in groups:
            chi = class_characters(A, stab, kQ, g)
            name, res = assign_irreps(chi, grp, verbose=False)
            summed = False
            if res >= 0.2:
                dname, dres, mult = decompose_characters(chi, grp)
                if dres < 0.2 and sum(mult.values()) > 1:
                    name, res, summed = dname, dres, True
            cl = rep_closure(A, stab, kQ, g)
            idx = f"{g[0]+1}" if len(g) == 1 else f"{g[0]+1}-{g[-1]+1}"
            sh = np.real(chi.get("sh", np.nan))
            flag = []
            if res >= 0.2:
                flag.append("no clean match")
            if cl["flips"]:
                flag.append(f"closes only up to -1 on {cl['flips']} pairs")
            elif cl["err"] > a.closure_tol:
                flag.append("does not close")
            if flag:
                bad.append(idx)
            note = list(flag)
            if summed:
                note.append(f"a sum of irreps, {np.ptp(E[g]) * 1e3:.2f} meV wide: "
                            f"lower --deg-tol to split")
            print(f"   {idx:<9} {np.mean(E[g]):9.4f} {sh:+9.3f} "
                  f"{name:>12} {res:9.2e} {cl['err']:9.1e}"
                  + ("" if not note else "   <-- " + "; ".join(note)))
            rows.append((g[0] + 1, len(g), float(np.mean(E[g])), sh, name, res,
                         cl["err"], cl["flips"], "; ".join(note),
                         [chi.get(c, np.nan) for c in CLASSES[grp]]))
        saved["points"].append(pt)
        saved[f"{pt}_group"] = grp
        saved[f"{pt}_Q"] = np.asarray(Q[:2], float)
        saved[f"{pt}_classes"] = np.array(CLASSES[grp])
        for key, col in (("first", 0), ("dim", 1), ("E", 2), ("sh", 3), ("label", 4),
                         ("resid", 5), ("closure", 6), ("flips", 7), ("note", 8)):
            saved[f"{pt}_{key}"] = np.array([r[col] for r in rows])
        saved[f"{pt}_chi"] = np.array([r[9] for r in rows], complex)
        if bad:
            print(f"   {len(bad)} multiplet(s) flagged. 'closes only up to -1': the "
                  f"conduction and valence\n   rotation blocks carry different spin "
                  f"signs for some operation (spinor Dmats from\n   different "
                  f"conventions). 'does not close': a multiplet cut short by --deg-tol, a "
                  f"band\n   window not closed under the group, or Dmats from another "
                  f"SAVE. Labels of flagged\n   multiplets are not reliable.")

    print("\n   A residual near 0 is a clean assignment; 0.1-0.2 reflects the "
          "finite mesh.\n   Larger values mean an incomplete multiplet -- try "
          "another --deg-tol -- or mixed irreps.\n   closure: the largest "
          "|D(R)D(S) - D(RS)| over the little group; near 0 means the multiplet "
          "carries a true representation.")
    if a.out:
        saved["points"] = np.array(saved["points"])
        np.savez(a.out, **saved)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
