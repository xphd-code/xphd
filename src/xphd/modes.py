"""Physical labels for phonon branches, from the eigenvectors."""
from __future__ import annotations

import re

import numpy as np

__all__ = ["read_modes", "classify", "LABELS"]

LABELS = ["ZA", "TA", "LA", "ZO", "TO", "LO"]


def read_modes(path, verbose=True):
    """Read matdyn's `flvec` file (matdyn.modes)."""
    txt = open(path, "rb").read().decode("latin-1")
    blocks = re.split(r"^\s*q\s*=\s*", txt, flags=re.M)[1:]
    if not blocks:
        raise SystemExit(f"{path}: no 'q =' blocks. Is this a matdyn.modes "
                         f"file? flvec must not have been blanked.")
    num = r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?"
    Q, W, V = [], [], []
    for b in blocks:
        head = b.split("\n", 1)
        Q.append([float(x) for x in re.findall(num, head[0])[:3]])
        w, v = [], []
        for m in re.finditer(r"freq\s*\(\s*\d+\s*\)\s*=\s*(" + num + r")"
                             r"\s*\[THz\]\s*=\s*(" + num + r")\s*\[cm-1\]"
                             r"(.*?)(?=freq\s*\(|\Z)", b, re.S):
            w.append(float(m.group(2)))
            vals = np.array([float(x) for x in re.findall(num, m.group(3))])
            v.append(vals.reshape(-1, 3, 2))
        if w:
            W.append(w)
            V.append(np.array(v))
    q = np.array(Q)
    freq = np.array(W)
    ev = np.array(V)
    ev = ev[..., 0] + 1j * ev[..., 1]
    if verbose:
        print(f"  {path}: {len(q)} q-points, {freq.shape[1]} branches, "
              f"{ev.shape[2]} atoms")
    return q, freq, ev


def classify(evec, q_cart=None, masses=None, freq=None, n_acoustic=3,
             z_tol=0.5, verbose=True):
    """Character label index for each branch."""
    ev = np.asarray(evec)
    nq, nmod, nat, _ = ev.shape
    m = np.ones(nat) if masses is None else np.asarray(masses, float)
    norm = np.maximum(np.sqrt(np.sum(np.abs(ev) ** 2, axis=(2, 3))), 1e-300)

    z = np.sum(np.abs(ev[..., 2]) ** 2, axis=2) / norm ** 2
    is_z = z > z_tol

    ac = None
    if freq is None:
        def overlap(w):
            w = w / np.sqrt(np.sum(w ** 2))
            o = np.zeros((nq, nmod))
            for dd in range(3):
                o = np.maximum(o, np.abs(np.einsum("qma,a->qm",
                                                   ev[..., dd], w)))
            return o / norm

        cands = {"mass-weighted": overlap(np.sqrt(m)),
                 "displacement": overlap(np.ones(nat))}
        best, score = None, -np.inf
        for name, o in cands.items():
            so = np.sort(o, axis=1)[:, ::-1]
            gap = float(np.median(so[:, n_acoustic - 1] - so[:, n_acoustic]))
            if verbose:
                print(f"    {name:>14}: median acoustic/optical gap {gap:+.3f}")
            if gap > score:
                best, score = name, gap
        if verbose:
            print(f"    using {best}  [no frequencies given; pass freq= for "
                  f"a more robust split]")
        ac = cands[best]

    lon = np.zeros((nq, nmod))
    if q_cart is not None:
        qc = np.asarray(q_cart, float)[:, :2]
        n = np.linalg.norm(qc, axis=1)
        good = n > 1e-8
        qh = np.zeros_like(qc)
        qh[good] = qc[good] / n[good, None]
        inpl = np.maximum(np.sqrt(np.sum(np.abs(ev[..., :2]) ** 2,
                                         axis=(2, 3))), 1e-300)
        lon = np.abs(np.einsum("qmad,qd->qm", ev[..., :2], qh)) / inpl
        lon[~good] = 0.0

    n_z_ac = 1
    n_p_ac = n_acoustic - n_z_ac
    lab = np.zeros((nq, nmod), int)
    for k in range(nq):
        jz = np.where(is_z[k])[0]
        jp = np.where(~is_z[k])[0]
        key = np.asarray(freq[k], float) if freq is not None else -ac[k]

        o = jz[np.argsort(key[jz])]
        lab[k, o[:n_z_ac]] = 0
        lab[k, o[n_z_ac:]] = 3

        o = jp[np.argsort(key[jp])]
        for grp, base in ((o[:n_p_ac], 0), (o[n_p_ac:], 3)):
            if grp.size == 2:
                a_, b_ = grp[np.argsort(-lon[k, grp])]
                lab[k, a_] = base + 2
                lab[k, b_] = base + 1
            else:
                for r_ in grp:
                    lab[k, r_] = base + (2 if lon[k, r_] > 0.707 else 1)

    if verbose:
        c = np.bincount(lab.ravel(), minlength=6)
        print("    counts: " + "  ".join(f"{LABELS[k]}={c[k]}"
                                         for k in range(6))
              + f"   (expect {nq} each)")
        nz = is_z.sum(axis=1)
        if not np.all(nz == 2):
            print(f"    [warn] {(nz != 2).sum()} q-points do not have exactly "
                  f"two out-of-plane modes;\n    the layer may not be planar, "
                  f"or z_tol needs adjusting")
        two = np.nonzero(nz == 2)[0]
        if freq is not None and len(two):
            g = np.array([np.sort(np.asarray(freq[k], float)[np.where(is_z[k])[0]])
                          for k in two])
            print(f"    ZA/ZO frequency gap: min {(g[:,1]-g[:,0]).min():.2f}, "
                  f"median {np.median(g[:,1]-g[:,0]):.2f} cm-1"
                  + (f" (over the {len(two)} q-points with two)" if len(two) < nq else ""))
    return lab
