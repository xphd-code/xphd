"""chi(sigma_h) for every exciton state at every Q."""
from __future__ import annotations

import argparse
import os

import numpy as np

from .excsym import rep_matrix_Q, shift_map, sigma_h_trace
from .multiplet import energy_groups, sigma_blocks, time_reversed
from .spin import spin_report
from .yambo import (full_zone_kpoints, ibz_parents, lattice, load_dmats,
                    load_excitons, sigma_h_op, spatial_ops)


def main(argv=None):
    p = argparse.ArgumentParser(prog="xphd parity",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--save", default="../phonons/SAVE")
    p.add_argument("--bse-dir", default="../BSE/output_all")
    p.add_argument("--dmats", default="Dmats.npy")
    p.add_argument("--nv", type=int, required=True,
                   help="valence bands in the BSE window (3 for hBN and GaN)")
    p.add_argument("--nstates", type=int, default=None,
                   help="states per Q; 1 is enough for the lowest-band map "
                        "and is far faster")
    p.add_argument("--deg-tol", type=float, default=0.005,
                   help="eV; states closer than this form one multiplet, whose "
                        "sigma_h matrix is stored so that `xphd selection-rule "
                        "--rotate` can split the coupling by the multiplet's "
                        "parity eigenstates instead of the solver's mixed "
                        "states (default 5 meV; 0 stores none)")
    p.add_argument("-o", "--out", default="parity.npz")
    a = p.parse_args(argv)

    ylat = lattice(a.save)
    D = load_dmats(a.dmats)
    nb = D.shape[-1]
    if not 0 < a.nv < nb:
        raise SystemExit(f"--nv {a.nv} is outside 1..{nb - 1}")
    kf = full_zone_kpoints(ylat)
    ops = spatial_ops(a.save, kf, D, a.nv)
    sh = sigma_h_op(ops)
    print(f"Dmats: {nb} bands, nv={a.nv}, nc={nb - a.nv}; "
          f"sigma_h k-map is the identity: "
          f"{np.array_equal(sh[4], np.arange(len(kf)))}")
    spin_report(ylat, a.save, sh)

    kidx, first = ibz_parents(ylat)
    print(f"{len(kf)} full-zone points, {len(first)} irreducible")

    chi_ibz, E_ibz, Q_ibz, ns = [], [], [], None
    sig_ibz, grp_ibz = [], []
    for j in range(len(first)):
        path = f"{a.bse_dir}/ndb.BS_diago_Q{j + 1}"
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}; need every irreducible "
                             f"database")
        Q = kf[first[j]]
        kQ, w = shift_map(kf, Q)
        if kQ is None:
            raise SystemExit(f"IBZ {j}: k+Q leaves the grid ({w:.1e})")
        _, A, E = load_excitons(path, ylat, a.nstates)
        if ns is None:
            ns = len(E) if a.nstates is None else min(a.nstates, len(E))
        chi = [sigma_h_trace(A, sh, kQ, s) for s in range(ns)]
        chi_ibz.append(chi)
        if a.deg_tol > 0:
            grp = energy_groups(E[:ns], a.deg_tol)
            sig_ibz.append(sigma_blocks(A[:ns], sh, kQ, grp, rep_matrix_Q))
            grp_ibz.append(grp)
        E_ibz.append(E[:ns])
        Q_ibz.append(Q)
        print(f"  IBZ {j:3d}  Q = {np.round(Q[:2], 4)}  E = {E[0]:.4f}  "
              f"chi = {chi[0]:+.3f}")

    chi_ibz, E_ibz = np.array(chi_ibz), np.array(E_ibz)
    chi_full, E_full = chi_ibz[kidx], E_ibz[kidx]
    extra = {}
    if a.deg_tol > 0:
        sig_ibz, grp_ibz = np.array(sig_ibz), np.array(grp_ibz)
        extra = dict(sigma_ibz=sig_ibz, group_ibz=grp_ibz,
                     trev=time_reversed(ylat), deg_tol=a.deg_tol)
    np.savez(a.out, Q_red=kf, iQ_ibz=kidx, chi=chi_full, E=E_full,
             chi_ibz=chi_ibz, Q_ibz=np.array(Q_ibz), **extra)

    low = chi_full[:, 0]
    print(f"\nwrote {a.out}: {chi_full.shape} (full zone x states)")
    print(f"  within 0.2 of +-1: "
          f"{100 * np.mean(np.abs(np.abs(chi_full) - 1) < 0.2):.1f}%")
    if a.deg_tol > 0:
        mu = np.concatenate([np.linalg.eigvalsh(0.5 * (M + M.conj().T)) for M in sig_ibz])
        multi = sum(int((np.bincount(g) > 1).sum()) for g in grp_ibz)
        print(f"  multiplets (states within {a.deg_tol * 1e3:g} meV): "
              f"{multi} with more than one state over {len(grp_ibz)} irreducible Q")
        print(f"  within 0.02 of +-1:  per state {100 * np.mean(np.abs(np.abs(chi_ibz) - 1) < 0.02):.1f}%"
              f",  after rotating each multiplet {100 * np.mean(np.abs(np.abs(mu) - 1) < 0.02):.1f}%")
    print(f"  lowest state: {100 * np.mean(low > 0.5):.1f}% even, "
          f"{100 * np.mean(low < -0.5):.1f}% odd, "
          f"{100 * np.mean(np.abs(low) <= 0.5):.1f}% undetermined")


if __name__ == "__main__":
    main()
