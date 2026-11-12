"""Phonon branch characters on the archive mesh -> mode_labels.npy."""
from __future__ import annotations

import argparse

import numpy as np

LABELS = ["ZA", "TA", "LA", "ZO", "TO", "LO"]


def main(argv=None):
    p = argparse.ArgumentParser(prog="xphd mode-labels",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--modes", default="matdyn.modes",
                   help="matdyn eigenvectors on the refined mesh")
    p.add_argument("--fine", type=int, default=360,
                   help="the refined mesh is fine x fine")
    p.add_argument("--mesh", type=int, default=24,
                   help="the source (archive) mesh is mesh x mesh")
    p.add_argument("--masses", type=float, nargs="+", required=True,
                   help="atomic masses in amu, in the order of the cell. No "
                        "default: GaN 69.723 14.007, hBN 10.81 14.007")
    p.add_argument("--archive", default="GI_ExcPh_Q0001.npz",
                   help="any archive; only its q_red ordering is used")
    p.add_argument("--hw-fine", default=None,
                   help="refined frequencies from xphd matdyn, to cross-check "
                        "against the frequencies in --modes")
    p.add_argument("-o", "--out", default="mode_labels.npy")
    a = p.parse_args(argv)

    import xphd
    from .io.matdyn import source_indices
    from .modes import classify, read_modes

    q, freq, ev = read_modes(a.modes, verbose=False)
    if ev.shape[2] != 2:
        raise SystemExit(
            f"{a.modes} has {ev.shape[2]} atoms and {ev.shape[1]} branches. This "
            f"classifier assumes a planar two-atom layer (six branches: ZA TA LA "
            f"ZO TO LO, two out-of-plane modes at every q) and takes parity from "
            f"the out-of-plane weight, which is wrong when sigma_h exchanges "
            f"atoms (1H TMDs such as MoS2). For such a layer take the parity from the "
            f"el-ph database instead: scripts/labels_from_elph.py")
    if len(q) != a.fine * a.fine:
        raise SystemExit(f"{a.modes} has {len(q)} q-points, not "
                         f"{a.fine}x{a.fine}; check --fine")
    idx = source_indices(a.fine, a.mesh)
    lab = classify(ev[idx], q_cart=q[idx], masses=a.masses, freq=freq[idx])

    n = a.mesh
    b1 = q[idx][n] * n
    b2 = q[idx][1] * n
    B = np.stack([b1, b2, [0.0, 0.0, 1.0]])
    q_frac = q[idx] @ np.linalg.inv(B)
    ang = np.degrees(np.arccos(b1 @ b2 / np.linalg.norm(b1)
                               / np.linalg.norm(b2)))
    print(f"  |b1| = {np.linalg.norm(b1):.6f}, |b2| = "
          f"{np.linalg.norm(b2):.6f}, angle {ang:.2f} deg")

    arc = xphd.ExcPhArchive(a.archive)
    Qf = np.mod(arc.q_red, 1.0)
    if len(Qf) != n * n:
        raise SystemExit(f"{a.archive} has {len(Qf)} q-points, not "
                         f"{n}x{n}; check --mesh")
    diff = Qf[:, None, :] - q_frac[None, :, :]
    diff = (diff + 0.5) % 1.0 - 0.5
    dist = np.linalg.norm(diff, axis=-1)
    perm = np.argmin(dist, axis=1)
    dmin = dist[np.arange(len(Qf)), perm]
    print(f"  worst q match {dmin.max():.2e}  (want < 1e-3)")
    if dmin.max() > 1e-3:
        raise SystemExit("q-points do not correspond; check the b1/b2 "
                         "ordering and --mesh")
    if len(np.unique(perm)) != len(Qf):
        raise SystemExit("the q match is not a bijection")

    out = np.asarray(lab)[perm].copy()
    iG = int(np.argmin(np.linalg.norm(((Qf[:, :2] + 0.5) % 1) - 0.5,
                                      axis=1)))
    out[iG, :3] = -1
    np.save(a.out, out)
    counts = np.bincount(out[out >= 0].ravel(), minlength=6)
    print(f"  wrote {a.out}: {out.shape}")
    print("  " + "  ".join(f"{L} {c}" for L, c in zip(LABELS, counts)))

    if a.hw_fine:
        hw = np.load(a.hw_fine).reshape(-1, freq.shape[1])[idx]
        dev = np.abs(freq[idx] * 1.239841984e-4 - hw).max()
        print(f"  freq in {a.modes} vs {a.hw_fine}: {dev * 1e3:.6f} meV")


if __name__ == "__main__":
    main()
