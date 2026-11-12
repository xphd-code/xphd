"""Mirror parity of the electronic bands, along a path and over the zone."""
from __future__ import annotations

import argparse

import numpy as np


def sigma_h_index(R_all):
    """Index of sigma_h among the SPATIAL operations (the first half; the second half repeats
    them with time reversal)."""
    R_all = np.asarray(R_all)
    for iop in range(len(R_all) // 2):
        R = R_all[iop]
        if (round(np.linalg.det(R)) == -1 and round(np.trace(R)) == 1
                and round(R[2, 2]) == -1):
            return iop
    raise SystemExit("no horizontal mirror among the operations; is the "
                     "layer planar?")


def parity_from_D(Dsh, tol=0.1):
    """Band parities and how well each is defined."""
    from .mirror import band_labels
    Dsh = np.asarray(Dsh)
    diag = np.diagonal(Dsh, axis1=1, axis2=2)
    _, kind, _ = band_labels(diag, tol)
    p = np.imag(diag) if kind == "spinor" else np.real(diag)
    row = np.sum(np.abs(Dsh) ** 2, axis=2)
    off = np.sqrt(np.maximum(row - np.abs(diag) ** 2, 0.0))
    defined = (np.abs(np.abs(p) - 1.0) < tol) & (off < tol)
    return p, off, defined


def path_points(kf, rlat, corners, rtol=1e-4):
    """Mesh points on the polyline through `corners`, in order along it."""
    kf = np.asarray(kf, float)[:, :3].copy()
    kf[:, 2] = 0.0
    B = np.asarray(rlat, float)
    imgs = [(m1, m2) for m1 in (-1, 0, 1) for m2 in (-1, 0, 1)]
    idx, xs, ticks, x0 = [], [], [], 0.0
    for s in range(len(corners) - 1):
        (la, a), (lb, b) = corners[s], corners[s + 1]
        va = np.array([a[0], a[1], 0.0]) @ B
        vb = np.array([b[0], b[1], 0.0]) @ B
        d = vb - va
        L = float(np.linalg.norm(d))
        found = {}
        for m1, m2 in imgs:
            vq = (kf + np.array([m1, m2, 0.0])) @ B
            t = (vq - va) @ d / L ** 2
            dist = np.linalg.norm(vq - (va + t[:, None] * d), axis=1)
            ok = (t > -rtol) & (t < 1 + rtol) & (dist < rtol * L)
            for i in np.where(ok)[0]:
                found[int(i)] = float(np.clip(t[i], 0.0, 1.0))
        seg = sorted(found.items(), key=lambda kv: kv[1])
        if s > 0 and seg and idx and seg[0][0] == idx[-1]:
            seg = seg[1:]
        if s == 0:
            ticks.append((x0, la))
        for i, t in seg:
            idx.append(i)
            xs.append(x0 + t * L)
        x0 += L
        ticks.append((x0, lb))
    return np.array(idx, int), np.array(xs), ticks


def flips_along(p_path, defined_path):
    """Sign changes of one band's parity along the path, ignoring points where the parity is
    undetermined."""
    s = np.sign(p_path[defined_path])
    return int(np.sum(s[1:] != s[:-1]))


def main(argv=None):
    p = argparse.ArgumentParser(prog="xphd band-parity",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--save", default="../phonons/SAVE",
                   help="directory holding ns.db1")
    p.add_argument("--bse-dir", default="../BSE/output_all",
                   help="for the band window, read from ndb.BS_diago_Q1")
    p.add_argument("--dmats", default="Dmats.npy")
    p.add_argument("--nv", type=int, required=True,
                   help="valence bands inside the BSE window")
    p.add_argument("--path", nargs="+", default=["G", "M", "K", "G"],
                   help="high-symmetry labels, as in xphd irreps")
    p.add_argument("--tol", type=float, default=0.1,
                   help="how far from +-1 a parity may sit and still count")
    p.add_argument("-o", "--out", default="band_parity.npz")
    a = p.parse_args(argv)

    from yambopy import YamboExcitonDB
    from .arpes import valence_energies
    from .excsym import HIGH_SYMMETRY
    from .symmetry import load_ops
    from .yambo import full_zone_kpoints, lattice, load_dmats

    ylat = lattice(a.save)
    kf = full_zone_kpoints(ylat)
    D = load_dmats(a.dmats)
    nb = D.shape[-1]
    if not 0 < a.nv < nb:
        raise SystemExit(f"--nv {a.nv} is outside 1..{nb - 1}")

    R_all = np.asarray(load_ops(a.save))
    iop = sigma_h_index(R_all)
    dk = kf @ R_all[iop].T - kf
    dk -= np.rint(dk)
    if np.abs(dk).max() > 1e-5:
        raise SystemExit("sigma_h moves in-plane k; the layer is not planar")
    pb, off, ok = parity_from_D(D[iop], a.tol)

    exc = YamboExcitonDB.from_db_file(
        ylat, filename=f"{a.bse_dir}/ndb.BS_diago_Q1", neigs=1)
    b0 = int(min(exc.unique_vbands))
    b1 = int(max(exc.unique_cbands)) + 1
    if b1 - b0 != nb:
        raise SystemExit(f"the BSE window [{b0}, {b1}) has {b1 - b0} bands but "
                         f"Dmats has {nb}; they must describe the same bands")
    E = valence_energies(a.save, ylat, list(range(b0, b1)))

    bad = [lab for lab in a.path if lab not in HIGH_SYMMETRY]
    if bad:
        raise SystemExit(f"unknown labels {bad}; known: "
                         f"{sorted(HIGH_SYMMETRY)}")
    corners = [(lab, HIGH_SYMMETRY[lab]) for lab in a.path]
    idx, x, ticks = path_points(kf, ylat.rlat, corners)
    if not len(idx):
        raise SystemExit("no mesh point lies on the path")

    from .mirror import band_labels
    kind = band_labels(np.diagonal(D[iop], axis1=1, axis2=2), a.tol)[1]
    print(f"sigma_h is operation {iop}; {len(kf)} k-points, bands "
          f"{b0 + 1}-{b1} (1-based), {a.nv} valence; {kind} bands"
          + (" (labels from the +-i eigenvalues; Kramers pairs at Gamma"
             " and M have none)" if kind == "spinor" else ""))
    print(f"  largest ||p| - 1| {np.abs(np.abs(pb) - 1).max():.1e}, "
          f"largest off-diagonal {off.max():.1e}, undetermined at "
          f"{int((~ok).sum())} of {ok.size} (k, band)")
    print(f"  path {'-'.join(a.path)}: {len(idx)} mesh points")

    key = {lab: np.argmin(np.linalg.norm(
        ((kf[:, :2] - np.mod(HIGH_SYMMETRY[lab], 1.0)) + 0.5) % 1.0 - 0.5,
        axis=1)) for lab in dict.fromkeys(a.path)}
    print(f"\n  {'band':>5} {'':>2}" + "".join(f"{lab:>6}" for lab in key)
          + f"{'flips':>7}{'% even':>8}")
    for n in range(nb):
        kind = "v" if n < a.nv else "c"
        row = "".join(
            f"{('+' if pb[k, n] > 0 else '-') if ok[k, n] else '?':>6}"
            for k in key.values())
        fl = flips_along(pb[idx, n], ok[idx, n])
        ev = 100.0 * np.mean(pb[ok[:, n], n] > 0) if ok[:, n].any() else 0.0
        print(f"  {b0 + n + 1:5d} {kind:>2}{row}{fl:7d}{ev:7.1f}%")
    print("\n  + even, - odd, ? undetermined. A flip along the path is a band"
          "\n  crossing between states of opposite parity: the energy-ordered"
          "\n  index changes character there, not any single state.")

    np.savez(a.out, kf=kf, E=E, p=pb, off=off, defined=ok, Q_red=kf, chi=pb,
             bands=np.arange(b0, b1) + 1, nv=a.nv, path_idx=idx, path_x=x,
             tick_x=np.array([t[0] for t in ticks]),
             tick_labels=np.array([t[1] for t in ticks]))
    print(f"\nwrote {a.out}")
    print(f"   dispersion: python scripts/plot_band_parity.py {a.out}")
    print(f"   zone map:   python scripts/plot_parity_bz.py --parity {a.out} "
          f"--only exciton --band <0..{nb - 1}> --title \"band <n>\"")


if __name__ == "__main__":
    main()
