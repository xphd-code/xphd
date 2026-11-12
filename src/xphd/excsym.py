"""Symmetry labels for exciton states."""
from __future__ import annotations

import numpy as np

__all__ = ["CHARACTER_TABLES", "CLASSES", "HIGH_SYMMETRY", "LITTLE_GROUP",
           "rep_matrix", "rep_matrix_Q", "shift_map", "characters",
           "assign_irreps", "degeneracy_groups", "sigma_h_trace",
           "little_group_ops", "classify_d3h", "stabilizer", "group_name",
           "class_characters", "rep_closure", "decompose_characters",
           "CLASS_SIZES"]

HIGH_SYMMETRY = {
    "G":  (0.0, 0.0),      "M":  (0.5, 0.0),      "K":  (1/3, 1/3),
    "M'": (0.0, 0.5),      "M''": (0.5, 0.5),     "K'": (2/3, 2/3),
    "Q":  (1/6, 1/6),      "Q'": (5/6, 5/6),
}

LITTLE_GROUP = {
    "G":  ("D3h", ("E", "2C3", "3C2", "sh", "2S3", "3sv")),
    "M":  ("C2v", ("E", "3C2", "sh", "3sv")),
    "M'": ("C2v", ("E", "3C2", "sh", "3sv")),
    "M''":("C2v", ("E", "3C2", "sh", "3sv")),
    "K":  ("C3h", ("E", "2C3", "sh", "2S3")),
    "K'": ("C3h", ("E", "2C3", "sh", "2S3")),
    "Q":  ("Cs",  ("E", "sh")),
    "Q'": ("Cs",  ("E", "sh")),
}

CLASSES = {
    "D3h": ("E", "2C3", "3C2", "sh", "2S3", "3sv"),
    "C3v": ("E", "2C3", "3sv"),
    "C3h": ("E", "2C3", "sh", "2S3"),
    "C3":  ("E", "2C3"),
}

_W = np.exp(2j * np.pi / 3)
_W2 = _W ** 2

CLASSES["C2v"] = ("E", "3C2", "sh", "3sv")
CLASSES["Cs"] = ("E", "sh")
CLASSES["C1"] = ("E",)

CHARACTER_TABLES = {
    "D3h": {"A1'":  (1,  1,  1,  1,  1,  1),
            "A2'":  (1,  1, -1,  1,  1, -1),
            "E'":   (2, -1,  0,  2, -1,  0),
            "A1''": (1,  1,  1, -1, -1, -1),
            "A2''": (1,  1, -1, -1, -1,  1),
            "E''":  (2, -1,  0, -2,  1,  0)},
    "C3v": {"A1": (1,  1,  1),
            "A2": (1,  1, -1),
            "E":  (2, -1,  0)},
    "C3h": {"A'":    (1,  1,  1,  1),
            "E'_1":  (1, _W,  1, _W),
            "E'_2":  (1, _W2, 1, _W2),
            "A''":   (1,  1, -1, -1),
            "E''_1": (1, _W, -1, -_W),
            "E''_2": (1, _W2, -1, -_W2),
            "E'_1+E'_2":   (2, -1, 2, -1),
            "E''_1+E''_2": (2, -1, -2, 1)},
    "C3":  {"A": (1,  1),
            "E": (2, -1)},
    "C2v": {"A1": (1,  1,  1,  1),
            "A2": (1,  1, -1, -1),
            "B1": (1, -1,  1, -1),
            "B2": (1, -1, -1,  1)},
    "Cs":  {"A'":  (1,  1),
            "A''": (1, -1)},
    "C1":  {"A": (1,)},
}


def degeneracy_groups(E, tol=1e-4):
    """Indices grouped into degenerate multiplets."""
    E = np.asarray(E, float)
    groups, cur = [], [0]
    for k in range(1, len(E)):
        if abs(E[k] - E[cur[-1]]) < tol:
            cur.append(k)
        else:
            groups.append(cur)
            cur = [k]
    groups.append(cur)
    return groups


def rep_matrix(A, Dc, Dv, kmap, group):
    """<A^S | R A^T> for the states in `group`."""
    return rep_matrix_Q(A, Dc, Dv, kmap, np.arange(np.asarray(A).shape[1]),
                        group)


def characters(A, ops, group, E=None, tol=1e-4, verbose=True):
    """Character of a degenerate multiplet under each operation."""
    chi = {}
    for lab, Dc, Dv, kmap in ops:
        D = rep_matrix(A, Dc, Dv, kmap, group)
        chi[lab] = complex(np.trace(D))
        if verbose:
            uni = np.abs(D @ np.conj(D).T - np.eye(len(group))).max()
            if uni > 1e-3:
                print(f"    [warn] {lab}: representation is not unitary "
                      f"(dev {uni:.1e}); the multiplet is probably incomplete")
    return chi


def assign_irreps(chi_by_class, table="D3h", tol=0.15, verbose=True):
    """Nearest irrep by character, with the residual reported."""
    if table not in CHARACTER_TABLES:
        raise SystemExit(f"no character table for {table!r}; "
                         f"have {sorted(CHARACTER_TABLES)}")
    cls = CLASSES[table]
    missing = [c for c in cls if c not in chi_by_class]
    if missing:
        raise SystemExit(f"characters missing for classes {missing}")
    v = np.array([chi_by_class[c] for c in cls], complex)
    best, resid = None, np.inf
    for name, row in CHARACTER_TABLES[table].items():
        r = float(np.abs(v - np.array(row, complex)).max())
        if r < resid:
            best, resid = name, r
    if verbose and resid > tol:
        print(f"    [warn] closest irrep {best} but residual {resid:.3f} "
              f"exceeds {tol}; check the degeneracy grouping and the little "
              f"group")
    return best, resid


CLASS_SIZES = {"D3h": (1, 2, 3, 1, 2, 3), "C3v": (1, 2, 3), "C3h": (1, 2, 1, 2),
               "C3": (1, 2), "C2v": (1, 1, 1, 1), "Cs": (1, 1), "C1": (1,)}


def decompose_characters(chi_by_class, table="D3h"):
    """A multiplet's characters as a sum of irreps."""
    cls = CLASSES[table]
    w = np.array(CLASS_SIZES[table], float)
    v = np.array([chi_by_class[c] for c in cls], complex)
    mult, rec = {}, np.zeros(len(cls), complex)
    for name, row in CHARACTER_TABLES[table].items():
        if "+" in name:
            continue
        r = np.array(row, complex)
        n = int(np.rint(np.sum(w * np.real(np.conj(r) * v)) / w.sum()))
        if n > 0:
            mult[name] = n
            rec += n * r
    resid = float(np.abs(v - rec).max())
    label = "+".join((f"{n}{k}" if n > 1 else k) for k, n in mult.items()) or "-"
    return label, resid, mult


def shift_map(kf, Qfrac, tol=1e-5):
    """Index of k_i + Q for every k_i, or (None, worst) if it leaves the grid."""
    d = np.asarray(kf)[:, None, :] - (np.asarray(kf) + np.asarray(Qfrac))[None, :, :]
    d -= np.rint(d)
    dist = np.linalg.norm(d, axis=-1)
    kQ = np.argmin(dist, axis=1)
    worst = float(dist[np.arange(len(kf)), kQ].max())
    if worst > tol or len(np.unique(kQ)) != len(kf):
        return None, worst
    return kQ, worst


def rep_matrix_Q(A, Dc, Dv, kmap, kQ, group):
    """<A^S | R A^T> for an exciton at momentum Q."""
    A = np.asarray(A)
    kmap = np.asarray(kmap, int)
    kQ = np.asarray(kQ, int)
    Dc = np.asarray(Dc)
    Dv = np.asarray(Dv)
    RA = np.einsum("kvw,skwd,kcd->skvc",
                   np.conj(Dv[kQ[kmap]]), A[:, kmap], Dc[kmap])
    g = np.asarray(group, int)
    return np.einsum("skvc,tkvc->st", np.conj(A[g]), RA[g])


def sigma_h_trace(A, sh_op, kQ, state):
    """chi(sigma_h) for a single state."""
    Dc, Dv, kmap = sh_op[-3], sh_op[-2], sh_op[-1]
    return float(np.trace(rep_matrix_Q(A, Dc, Dv, kmap, kQ, [state])).real)


def little_group_ops(ops, point):
    """Operations of `ops` that survive at a high-symmetry point."""
    if point not in LITTLE_GROUP:
        raise SystemExit(f"unknown point {point!r}; have "
                         f"{sorted(LITTLE_GROUP)}")
    table, classes = LITTLE_GROUP[point]
    keep = [o for o in ops if o[0] in classes]
    return table, keep


def classify_d3h(R):
    """Class of a reduced-coordinate D3h operation."""
    tr, det, r33 = (int(round(np.trace(R))), int(round(np.linalg.det(R))),
                    int(round(R[2, 2])))
    if det == 1:
        if tr == 3:
            return "E"
        if tr == 0:
            return "2C3"
        if tr == -1 and r33 == -1:
            return "3C2"
    else:
        if tr == 1 and r33 == -1:
            return "sh"
        if tr == -2:
            return "2S3"
        if tr == 1 and r33 == 1:
            return "3sv"
    return None


def stabilizer(ops, Q, tol=1e-5):
    """Operations with R Q = Q modulo a reciprocal lattice vector."""
    Q = np.asarray(Q, float)
    out = []
    for o in ops:
        dq = Q @ np.asarray(o[1]).T - Q
        if np.abs(dq - np.rint(dq)).max() < tol:
            out.append(o)
    return out


def group_name(stab):
    """Point group of a D3h stabilizer, from its order and classes."""
    n = len(stab)
    labs = {o[0] for o in stab}
    if n == 12:
        return "D3h"
    if n == 6 and "2C3" in labs:
        return "C3h"
    if n == 4:
        return "C2v"
    if n == 2 and "sh" in labs:
        return "Cs"
    if n == 1:
        return "C1"
    raise SystemExit(f"unrecognised little group of order {n}: {sorted(labs)}")


def rep_closure(A, stab, kQ, multiplet, tol=1e-3):
    """Does the multiplet carry a true representation of the little group?"""
    mats = [rep_matrix_Q(A, o[2], o[3], o[4], kQ, multiplet) for o in stab]
    n = len(mats)
    I = np.eye(len(multiplet))
    unitary = max(float(np.abs(M @ M.conj().T - I).max()) for M in mats) if n else 0.0
    Rs = [np.rint(np.asarray(o[1], float)).astype(int) for o in stab]
    index = {tuple(R.ravel()): i for i, R in enumerate(Rs)}
    best = None
    for order in ("RS", "SR"):
        err, flips, missing = 0.0, 0, 0
        for a in range(n):
            for b in range(n):
                P = Rs[a] @ Rs[b] if order == "RS" else Rs[b] @ Rs[a]
                c = index.get(tuple(P.ravel()))
                if c is None:
                    missing += 1
                    continue
                M = mats[a] @ mats[b]
                ep = float(np.abs(M - mats[c]).max())
                em = float(np.abs(M + mats[c]).max())
                err = max(err, ep)
                if em < ep and em < tol:
                    flips += 1
        res = dict(err=err, flips=flips, pairs=n * n, missing=missing, order=order,
                   unitary=unitary)
        if best is None or (res["missing"], res["err"]) < (best["missing"], best["err"]):
            best = res
    return best


def class_characters(A, stab, kQ, multiplet):
    """Characters of a degenerate multiplet -- a list of state indices, not a point group --
    taking one representative per class."""
    reps = {}
    for o in stab:
        reps.setdefault(o[0], o)
    if "2C3" in reps and "2S3" in reps and "sh" in reps:
        want = np.rint(np.asarray(reps["sh"][1], float)
                       @ np.asarray(reps["2C3"][1], float))
        for o in stab:
            if o[0] == "2S3" and np.array_equal(np.rint(np.asarray(o[1], float)), want):
                reps["2S3"] = o
                break
    chi = {}
    for lab, (_, R, Dc, Dv, kmap) in reps.items():
        chi[lab] = complex(np.trace(rep_matrix_Q(A, Dc, Dv, kmap, kQ, multiplet)))
    return chi
