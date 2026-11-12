"""Reader for Quantum ESPRESSO ``matdyn.x`` frequency files."""
from __future__ import annotations

import re
import numpy as np

CM1_TO_EV = 1.239841984e-4

__all__ = ["read_freq", "source_indices", "compare_to_archive", "CM1_TO_EV"]


def read_freq(path, nbnd=None, nks=None, verbose=True):
    """Read a .freq file."""
    with open(path, "rb") as fh:
        text = fh.read().decode("latin-1")
    preview = "\n".join(text.splitlines()[:3])

    head = re.search(r"nbnd\s*=\s*(\d+)\D+nks\s*=\s*(\d+)", text, re.S | re.I)
    if head:
        nbnd = nbnd or int(head.group(1))
        nks = nks or int(head.group(2))

    body = "\n".join(ln for ln in text.splitlines()
                     if "&" not in ln and "/" not in ln)
    try:
        vals = np.array(body.split(), dtype=float)
    except ValueError as exc:
        raise ValueError(f"{path}: non-numeric token ({exc}). "
                         f"First lines:\n{preview}") from None

    if nks and not nbnd:
        if vals.size % nks:
            raise ValueError(f"{vals.size} numbers not divisible by nks={nks}")
        nbnd = vals.size // nks - 3
    if nbnd and not nks:
        nks = vals.size // (3 + nbnd)
    if not (nbnd and nks):
        raise ValueError(f"{path}: cannot determine nbnd/nks; pass them "
                         f"explicitly. File holds {vals.size} numbers.")

    per = 3 + nbnd
    if vals.size < nks * per:
        raise ValueError(
            f"{path}: expected {nks*per} numbers ({nks} points x {per}), "
            f"found {vals.size}. Truncated -- matdyn may have been killed.")
    vals = vals[:nks * per].reshape(nks, per)
    if verbose:
        print(f"  {path}: {nks} q-points x {nbnd} branches")
    return vals[:, :3], vals[:, 3:] * CM1_TO_EV


def source_indices(n_fine: int, n_src: int):
    """Row-major fine-mesh indices coinciding with an n_src x n_src mesh."""
    if n_fine % n_src:
        raise ValueError(f"fine mesh {n_fine} is not a multiple of {n_src}")
    r = n_fine // n_src
    a, b = np.meshgrid(np.arange(n_src), np.arange(n_src), indexing="ij")
    return ((a * r) * n_fine + (b * r)).ravel()


def compare_to_archive(freqs_ev, n_fine, archive, q_order="rowmajor",
                       verbose=True):
    """Compare matdyn output against the archive's source-mesh frequencies."""
    hw = np.asarray(archive._raw("hw"), float)
    n_src = archive.n1

    if q_order == "npz":
        if freqs_ev.shape[0] != hw.shape[0]:
            raise ValueError(f"--q-order npz needs {hw.shape[0]} points, "
                             f"got {freqs_ev.shape[0]}")
        got, ref = freqs_ev, hw
    else:
        q = np.mod(archive.q_red, 1.0)
        key = (np.rint(q[:, 0] * n_src).astype(int) % n_src) * n_src + \
              (np.rint(q[:, 1] * n_src).astype(int) % n_src)
        ref = hw[np.argsort(key)]
        got = freqs_ev[source_indices(n_fine, n_src)]

        if freqs_ev.shape[0] == hw.shape[0]:
            d_row = float(np.median(np.abs(got - ref)))
            d_npz = float(np.median(np.abs(freqs_ev - hw)))
            if d_npz < 0.2 * d_row and verbose:
                print(f"  [!] archive order fits far better (median "
                      f"{d_npz*1e3:.4f} vs {d_row*1e3:.4f} meV) -- your "
                      f"matdyn q-list is in q_red order. Use q_order='npz'.")

    dev = np.abs(got - ref)
    rest = dev.copy()
    rest[0, :3] = 0.0
    res = dict(max=float(dev.max()), gamma_acoustic=float(dev[0, :3].max()),
               elsewhere=float(rest.max()),
               rel=float(rest.max() / max(ref.max(), 1e-300)),
               per_branch=dev.max(axis=0))
    if verbose:
        print(f"  Gamma acoustic {res['gamma_acoustic']*1e3:.4f} meV "
              f"(the ASR correction; matdyn's zeros are correct)")
        print(f"  elsewhere      {res['elsewhere']*1e3:.4f} meV "
              f"({res['rel']:.2e} relative)")
        print("  per-branch max (meV):",
              np.array2string(res["per_branch"] * 1e3, precision=4))
    return res
