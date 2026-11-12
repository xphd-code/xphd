#!/usr/bin/env python3
"""verify_all.py ============= Confirm the full set of archives is present and complete before
the transport runs."""
from __future__ import annotations

import argparse
import os

import numpy as np

NEED = {"G_grid", "g2_grid", "Ge_grid", "Gh_grid", "E_n_grid", "E_m_grid",
        "hw_grid", "q_red", "Q_red", "mesh"}


def main(argv=None):
    p = argparse.ArgumentParser(prog="xphd verify-archives")
    p.add_argument("--dir", default="archives")
    p.add_argument("--n", type=int, default=576)
    a = p.parse_args(argv)

    missing, incomplete, bad_shape = [], [], []
    ref_shape = None
    Qs = []
    for k in range(1, a.n + 1):
        f = os.path.join(a.dir, f"GI_ExcPh_Q{k:04d}.npz")
        if not os.path.exists(f):
            missing.append(k)
            continue
        try:
            d = np.load(f)
        except Exception:
            incomplete.append(k)
            continue
        lack = NEED - set(d.files)
        if lack:
            incomplete.append(k)
            continue
        sh = d["g2_grid"].shape
        if ref_shape is None:
            ref_shape = sh
        elif sh != ref_shape:
            bad_shape.append((k, sh))
        Qs.append(np.round(np.mod(d["Q_red"][:2], 1.0), 5))

    print(f"{a.dir}: expected {a.n} archives")
    print(f"   present and complete : {a.n - len(missing) - len(incomplete)}")
    print(f"   missing              : {len(missing)}"
          + (f"  e.g. {missing[:8]}" if missing else ""))
    print(f"   incomplete/unreadable: {len(incomplete)}"
          + (f"  e.g. {incomplete[:8]}" if incomplete else ""))
    print(f"   g2_grid shape        : {ref_shape}"
          + (f"   {len(bad_shape)} DIFFER" if bad_shape else "  (uniform)"))
    if bad_shape:
        print(f"      Q0001 (the reference)    {ref_shape}")
        for sh in sorted({s for _, s in bad_shape}):
            ks = [k for k, s in bad_shape if s == sh]
            print(f"      {len(ks):4d} archive(s)          {sh}   e.g. Q{ks[0]:04d}")
        print("      axes are (q-points, branches, states, states): a different state count\n"
              "      means --nexc differed between runs; a different q count, --mesh.\n"
              "      run_generate skips archives that exist, so move the wrong ones away\n"
              "      before regenerating them")
    if Qs:
        n_unique = len({tuple(q) for q in Qs})
        print(f"   distinct Q           : {n_unique}"
              + ("" if n_unique == len(Qs) else "   <-- DUPLICATES"))

    ok = not (missing or incomplete or bad_shape)
    if not ok and (missing or incomplete):
        redo = sorted(set(missing) | set(incomplete))
        print(f"\n   to regenerate, run for iQ (0-based) = "
              f"{[k - 1 for k in redo[:20]]}{' ...' if len(redo) > 20 else ''}")
    print(f"\n   {'COMPLETE: the transport can run.' if ok else 'INCOMPLETE: do not run the transport yet.'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
