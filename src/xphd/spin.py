"""Spin-orbit or not, and what it means for the group theory."""
from __future__ import annotations

import numpy as np

__all__ = ["spinor_components", "sigma_h_square", "spin_report"]


def spinor_components(ylat=None, save=None):
    """nspinor of the SAVE: 2 for spin-orbit (spinor wavefunctions), 1 otherwise; None if it
    cannot be read."""
    n = getattr(ylat, "spinor_components", None) if ylat is not None else None
    if n is None and save is not None:
        try:
            from netCDF4 import Dataset
            with Dataset(f"{save}/ns.db1") as d:
                n = np.asarray(d.variables["DIMENSIONS"][:], float).ravel()[11]
        except Exception:
            n = None
    return None if n is None else int(round(float(n)))


def sigma_h_square(D, tol=1e-3):
    """How D(sigma_h)^2 behaves on one block of bands."""
    D = np.asarray(D)
    I = np.eye(D.shape[-1])
    sq = np.einsum("kab,kbc->kac", D, D)
    dp = np.abs(sq - I).max(axis=(1, 2))
    dm = np.abs(sq + I).max(axis=(1, 2))
    plus = int(np.sum(dp < tol))
    minus = int(np.sum(dm < tol))
    sign = -1 if minus > plus else +1
    dev = dm if sign < 0 else dp
    ok = dev < tol
    worst = float(dev[ok].max()) if ok.any() else float(dev.min())
    return sign, float(ok.mean()), worst


def spin_report(ylat=None, save=None, sh_op=None, verbose=True, tol=1e-3):
    """What the SAVE and the rotation matrices say about spin-orbit, and which groups apply.
    sh_op is the sigma_h entry of yambo.spatial_ops (label, R, Dc, Dv, kmap) or None (no
    horizontal mirror)."""
    ns = spinor_components(ylat, save)
    out = dict(nspinor=ns, kind=None if ns is None else ("spinor" if ns == 2 else "spinless"),
               sign_v=None, sign_c=None, frac_v=None, frac_c=None, consistent=None)
    if sh_op is not None:
        Dc, Dv = sh_op[-3], sh_op[-2]
        out["sign_v"], out["frac_v"], wv = sigma_h_square(Dv, tol)
        out["sign_c"], out["frac_c"], wc = sigma_h_square(Dc, tol)
        dm_kind = "spinor" if out["sign_v"] < 0 and out["sign_c"] < 0 else (
            "spinless" if out["sign_v"] > 0 and out["sign_c"] > 0 else "mixed")
        if out["kind"] is None:
            out["kind"] = dm_kind if dm_kind != "mixed" else None
            out["consistent"] = dm_kind != "mixed"
        else:
            out["consistent"] = dm_kind == out["kind"]
    if verbose:
        src = ("ns.db1: nspinor = %d" % ns) if ns is not None else "ns.db1: nspinor unknown"
        print(f"spin: {src}"
              + ("" if ns is None else
                 (" -- spin-orbit, spinor wavefunctions" if ns == 2 else " -- no spin-orbit")))
        if sh_op is not None:
            print(f"   D(sigma_h)^2 = {out['sign_v']:+d} on the valence block "
                  f"({out['frac_v'] * 100:.1f}% of k within {tol:g}), "
                  f"{out['sign_c']:+d} on the conduction block "
                  f"({out['frac_c'] * 100:.1f}%)")
            for name, f in (("valence", out["frac_v"]), ("conduction", out["frac_c"])):
                if f < 0.99:
                    print(f"   note: on {(1 - f) * 100:.1f}% of k the {name} block is not "
                          f"closed under sigma_h (a band whose partner lies outside the "
                          f"window): see scripts/window_closure.py")
        else:
            print("   no horizontal mirror among the operations: D(sigma_h)^2 not checked")
        if out["consistent"] is False:
            print("   WARNING: ns.db1 and the Dmats disagree about spin-orbit -- the Dmats "
                  "were made without\n   spinor rotations, or for another SAVE. Exciton "
                  "labels and parities built from them are unreliable.")
        if out["kind"] == "spinor":
            print("   -> bands carry DOUBLE-group irreps (sigma_h eigenvalues +-i); excitons "
                  "and phonons carry\n      single-group irreps, as used here: the "
                  "electron's and the hole's -1 cancel")
        elif out["kind"] == "spinless":
            print("   -> single groups throughout (bands, excitons, phonons)")
    return out
