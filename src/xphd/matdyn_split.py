"""Running matdyn.x over a long q-list in chunks, and merging the output."""
from __future__ import annotations

import argparse
import os
import re

import numpy as np


def main_split(argv=None):
    p = argparse.ArgumentParser(prog="xphd matdyn-split")
    p.add_argument("--n", type=int, default=360,
                   help="fine mesh is n x n x 1")
    p.add_argument("--chunks", type=int, default=48,
                   help="number of pieces; match the cores on one node, or "
                        "the array size you intend to submit")
    p.add_argument("--prefix", default="qlist")
    a = p.parse_args(argv)

    i, j = np.meshgrid(np.arange(a.n), np.arange(a.n), indexing="ij")
    q = np.stack([i.ravel() / a.n, j.ravel() / a.n,
                  np.zeros(a.n * a.n)], axis=1)
    parts = np.array_split(q, a.chunks)

    off = 0
    with open("chunk_index.txt", "w") as idx:
        idx.write("# chunk  first_iq  npoints\n")
        for c, part in enumerate(parts):
            fn = f"{a.prefix}_{c:03d}.txt"
            with open(fn, "w") as f:
                f.write(f"{len(part)}\n")
                np.savetxt(f, part, fmt="%18.12f")
            idx.write(f"{c:5d} {off:9d} {len(part):8d}\n")
            off += len(part)

    print(f"{len(q)} q-points -> {a.chunks} chunks of "
          f"{len(parts[0])}-{len(parts[-1])}")
    print(f"wrote {a.prefix}_000.txt .. {a.prefix}_{a.chunks-1:03d}.txt")
    print("wrote chunk_index.txt (used by the concatenation step to verify "
          "ordering)")


def count_q_modes(path):
    """Number of q-blocks in a .modes file."""
    n = 0
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("q ="):
                n += 1
    return n


def concat_modes(chunks, prefix, out, verbose=True):
    """Concatenate the chunk .modes files as written."""
    total = 0
    with open(out, "w") as fo:
        for c in range(chunks):
            fn = f"{prefix}_{c:03d}.modes"
            if not os.path.exists(fn):
                raise SystemExit(f"missing {fn}; chunk {c} did not finish")
            n = 0
            with open(fn) as fi:
                for line in fi:
                    if line.lstrip().startswith("q ="):
                        n += 1
                    fo.write(line)
            if n == 0:
                raise SystemExit(f"{fn} contains no q-blocks")
            total += n
            if verbose:
                print(f"   chunk {c:3d}: {n:6d} q-points")
    return total


def concat_freq(chunks, prefix, out, verbose=True):
    """Write one .freq with the namelist emitted once and nks summed."""
    blocks, nbnd, total = [], None, 0
    for c in range(chunks):
        fn = f"{prefix}_{c:03d}.freq"
        if not os.path.exists(fn):
            raise SystemExit(f"missing {fn}")
        with open(fn) as f:
            txt = f.read()
        m = re.search(r"nbnd\s*=\s*(\d+).*?nks\s*=\s*(\d+)", txt, re.S)
        if m is None:
            raise SystemExit(f"{fn}: no &plot namelist found")
        nb, nk = int(m.group(1)), int(m.group(2))
        if nbnd is None:
            nbnd = nb
        elif nb != nbnd:
            raise SystemExit(f"{fn}: nbnd {nb} differs from {nbnd}")
        body = txt.split("/", 1)[1].lstrip("\n")
        blocks.append(body)
        total += nk
        if verbose:
            print(f"   chunk {c:3d}: {nk:6d} q-points, {nb} bands")
    with open(out, "w") as f:
        f.write(f" &plot nbnd= {nbnd:4d}, nks= {total:7d} /\n")
        for b in blocks:
            f.write(b if b.endswith("\n") else b + "\n")
    return total, nbnd


def concat_gp(chunks, prefix, out, verbose=True):
    """Merge the chunk .gp files, offsetting the cumulative distance."""
    files = [f"{prefix}_{c:03d}.freq.gp" for c in range(chunks)]
    if not all(os.path.exists(f) for f in files):
        if verbose:
            print("   no .gp files from matdyn; skipped")
        return None
    offset, rows = 0.0, 0
    with open(out, "w") as fo:
        for fn in files:
            last = offset
            with open(fn) as fi:
                for line in fi:
                    v = line.split()
                    if not v:
                        continue
                    d = float(v[0]) + offset
                    fo.write(f"{d:14.6f}" + "".join(f"{float(x):12.4f}"
                                                    for x in v[1:]) + "\n")
                    last = d
                    rows += 1
            offset = last
    return rows


def main_merge(argv=None):
    p = argparse.ArgumentParser(prog="xphd matdyn-merge")
    p.add_argument("--chunks", type=int, default=48)
    p.add_argument("--modes-prefix", default="modes")
    p.add_argument("--freq-prefix", default="freq")
    p.add_argument("-o", "--out-modes", default="matdyn.modes")
    p.add_argument("--freq", default="matdyn.freq")
    p.add_argument("--index", default="chunk_index.txt",
                   help="written by xphd matdyn-split; used to check that "
                        "the merged length is what was requested")
    a = p.parse_args(argv)

    print("merging .modes")
    n_modes = concat_modes(a.chunks, a.modes_prefix, a.out_modes)
    print(f"   -> {a.out_modes}: {n_modes} q-points\n")

    print("merging .freq")
    n_freq, nbnd = concat_freq(a.chunks, a.freq_prefix, a.freq)
    print(f"   -> {a.freq}: {n_freq} q-points, {nbnd} bands\n")

    if n_modes != n_freq:
        raise SystemExit(f"MISMATCH: {n_modes} in .modes vs {n_freq} in "
                         f".freq; one set of chunks is incomplete")

    print("merging .gp")
    n_gp = concat_gp(a.chunks, a.freq_prefix, a.freq + ".gp")
    if n_gp is not None:
        print(f"   -> {a.freq}.gp: {n_gp} rows, distance offset per chunk")
        if n_gp != n_modes:
            raise SystemExit(f"MISMATCH: {n_gp} rows in .gp vs {n_modes} "
                             f"q-points")

    if os.path.exists(a.index):
        want = sum(int(l.split()[2]) for l in open(a.index)
                   if not l.startswith("#"))
        ok = "OK" if want == n_modes else "MISMATCH"
        print(f"   chunk_index.txt expects {want}: {ok}")
        if want != n_modes:
            raise SystemExit("refusing to proceed with an incomplete mesh")

    n = int(round(n_modes ** 0.5))
    print(f"\n   {n_modes} = {n}x{n}" if n * n == n_modes else
          f"\n   {n_modes} q-points (not a square mesh)")
    print(f"   size: {os.path.getsize(a.out_modes)/1e9:.2f} GB")


CM1 = 1.239841984e-4


def main_check(argv=None):
    p = argparse.ArgumentParser(prog="xphd modes-check")
    p.add_argument("--modes", default="matdyn.modes")
    p.add_argument("--freq", default="matdyn.freq")
    p.add_argument("--n", type=int, default=360, help="expected mesh")
    a = p.parse_args(argv)

    from xphd.modes import read_modes
    q, freq, ev = read_modes(a.modes, verbose=False)
    nq, nmod = freq.shape
    print(f"{a.modes}: {nq} q-points, {nmod} branches")

    want = a.n * a.n
    print(f"   expected {want} for a {a.n}x{a.n} mesh: "
          f"{'OK' if nq == want else 'MISMATCH -- a chunk is missing'}")

    qf = np.mod(q[:, :2], 1.0)
    on = np.abs(qf * a.n - np.rint(qf * a.n)).max()
    uniq = len(np.unique(np.round(qf, 6), axis=0))
    print(f"   points lie on the grid to {on:.1e}")
    print(f"   distinct q-points: {uniq}"
          + ("" if uniq == nq else "   <-- DUPLICATES, chunks overlap"))

    iG = int(np.argmin(np.linalg.norm(
        (qf + 0.5) % 1.0 - 0.5, axis=1)))
    wG = freq[iG] * CM1 * 1e3
    print(f"\n   Gamma frequencies (meV): {np.round(wG, 4)}")
    res = float(np.abs(wG[:3]).max())
    print(f"   largest acoustic residue: {res:.4f} meV"
          + ("   OK" if res < 0.05 else
             "   <-- set --acoustic-cut above this"))

    neg = (freq < -1.0).sum()
    print(f"   frequencies below -1 cm^-1: {neg}"
          + ("" if neg == 0 else "   <-- dynamical instability, or the "
                                 "force constants are not converged"))

    ev2 = ev.reshape(nq, nmod, -1)
    zw = (np.abs(ev2[:, :, 2::3]) ** 2).sum(-1) / np.maximum(
        (np.abs(ev2) ** 2).sum(-1), 1e-30)
    clean = 100.0 * np.mean((zw < 0.05) | (zw > 0.95))
    nodd = (zw > 0.5).sum(axis=1)
    v, c = np.unique(nodd, return_counts=True)
    print(f"\n   out-of-plane weights within 0.05 of 0 or 1: {clean:.2f}%"
          + ("   OK" if clean > 99 else
             "   <-- the layer is not planar, or the eigenvector "
             "convention\n       is not displacement"))
    print(f"   sigma_h-odd branches per q: "
          + ", ".join(f"{vv}x{100*cc/nq:.1f}%" for vv, cc in zip(v, c)))

    with open(a.freq) as f:
        head = f.readline()
    import re
    m = re.search(r"nbnd\s*=\s*(\d+).*?nks\s*=\s*(\d+)", head)
    if m:
        nb, nk = int(m.group(1)), int(m.group(2))
        print(f"\n   {a.freq}: nbnd={nb}, nks={nk}"
              + ("   OK" if (nb == nmod and nk == nq) else
                 f"   <-- disagrees with {a.modes}"))
