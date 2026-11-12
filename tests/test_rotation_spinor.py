"""Exciton rotation with yambo-like SPINOR D-matrices."""
import numpy as np

from xphd.excsym import (assign_irreps, class_characters, classify_d3h,
                         rep_closure, rep_matrix_Q, shift_map, stabilizer)


def _su2(R, sign):
    """An SU(2) lift of the proper part of R, with an arbitrary sign."""
    P = R / np.linalg.det(R)
    th = np.arccos(np.clip((np.trace(P) - 1) / 2, -1, 1))
    I2 = np.eye(2, dtype=complex)
    if th < 1e-9:
        return sign * I2
    if abs(th - np.pi) < 1e-9:
        w, v = np.linalg.eigh(P)
        n = v[:, np.argmax(w)]
    else:
        n = np.array([P[2, 1] - P[1, 2], P[0, 2] - P[2, 0], P[1, 0] - P[0, 1]]) / (2 * np.sin(th))
    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.array([[1, 0], [0, -1]], complex)
    return sign * (np.cos(th / 2) * I2 - 1j * np.sin(th / 2) * (n[0] * sx + n[1] * sy + n[2] * sz))


def _model(n=12, seed=3):
    rng = np.random.default_rng(seed)
    a = 6.02
    lat = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, 50.0]])
    rlat = np.linalg.inv(lat).T

    def rz(t):
        return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    sh = np.diag([1, 1, -1.0])
    c2 = np.diag([-1, 1, -1.0])
    proper = [rz(2 * np.pi * m / 3) @ x for x in (np.eye(3), c2) for m in range(3)]
    cart = proper + [sh @ p for p in proper]
    red = [np.rint(lat @ s @ np.linalg.inv(lat)) for s in cart]
    U = [_su2(s, rng.choice([-1, 1])) for s in cart]
    kf = np.array([[i / n, j / n, 0] for i in range(n) for j in range(n)])
    nk = len(kf)

    def idx(k):
        d = kf - np.mod(k, 1.0)
        d -= np.rint(d)
        return int(np.argmin(np.linalg.norm(d, axis=1)))
    img = np.array([[idx(R @ k) for k in kf] for R in red])
    S = np.full(nk, -1)
    seen = np.zeros(nk, bool)
    for k in range(nk):
        if seen[k]:
            continue
        for r in range(len(red)):
            j = img[r, k]
            if not seen[j]:
                seen[j] = True
                S[j] = r
    spin = [0, 1, 0, 1]
    nv = 2

    def dmat(r, k):
        kp = img[r, k]
        M = U[S[kp]].conj().T @ U[r] @ U[S[k]]
        return np.diag([M[s, s] for s in spin])
    D = np.array([[dmat(r, k) for k in range(nk)] for r in range(len(red))])

    def kmap_of(R):
        d = kf[:, None, :] - (kf @ R.T)[None, :, :]
        d -= np.rint(d)
        return np.argmin(np.linalg.norm(d, axis=-1), axis=1)
    ops = [(classify_d3h(R), R, D[i, :, nv:, nv:], D[i, :, :nv, :nv], kmap_of(R))
           for i, R in enumerate(red)]

    w = np.zeros(nk)
    for cen in ((1 / 3, 1 / 3), (2 / 3, 2 / 3)):
        d = kf[:, :2] - np.array(cen)
        d -= np.rint(d)
        kc = np.c_[d, np.zeros(nk)] @ rlat
        w += np.exp(-np.sum(kc ** 2, 1) / (2 * (0.05 / (2 * np.pi)) ** 2))
    seed_state = (rng.normal(size=(1, nk, nv, 2)) + 1j * rng.normal(size=(1, nk, nv, 2))) \
        * w[None, :, None, None]
    return kf, cart, ops, seed_state


def _eq49(A, op, kQ):
    _, _, Dc, Dv, kmap = op
    return np.einsum("kvw,skwd,kcd->skvc", np.conj(Dv[kQ[kmap]]), A[:, kmap], Dc[kmap])


def _old(A, Dc, Dv, kmap, kQ, group):
    RA = np.einsum("kcd,ksdb,kba->ksca", np.conj(Dc),
                   A[:, kmap].transpose(1, 0, 3, 2), Dv[kQ])
    RA = RA.transpose(1, 0, 3, 2)
    g = np.asarray(group, int)
    return np.einsum("skvc,tkvc->st", np.conj(A[g]), RA[g])


def test_eq49_doublet_at_gamma_closes():
    kf, cart, ops, seed = _model()
    kQ, _ = shift_map(kf, np.zeros(3))
    rows = []
    for row in (0, 1):
        rows.append(sum(np.conj(cart[i][:2, :2][row, 0] * cart[i][2, 2]) * _eq49(seed, o, kQ)
                        for i, o in enumerate(ops)))
    A = np.concatenate(rows)
    A /= np.linalg.norm(A.reshape(2, -1), axis=1)[:, None, None, None]
    r = rep_closure(A, ops, kQ, [0, 1])
    assert r["missing"] == 0 and r["err"] < 1e-10 and r["unitary"] < 1e-10
    chi = {o[0]: np.trace(rep_matrix_Q(A, o[2], o[3], o[4], kQ, [0, 1])) for o in ops}
    assert abs(chi["2C3"] + 1) < 1e-10 and abs(chi["sh"] + 2) < 1e-10 and abs(chi["2S3"] - 1) < 1e-10
    old = [_old(A, o[2], o[3], o[4], kQ, [0, 1]) for o in ops]
    assert abs(np.trace(old[2]) + 1) < 1e-6
    assert max(np.abs(M @ M.conj().T - np.eye(2)).max() for M in old) > 0.1


def test_eq49_singlet_at_K_unitary():
    kf, cart, ops, seed = _model()
    Q = np.array([1 / 3, 1 / 3, 0])
    kQ, _ = shift_map(kf, Q)
    stab = stabilizer(ops, Q)
    assert sorted({o[0] for o in stab}) == ["2C3", "2S3", "E", "sh"]
    A = sum(_eq49(seed, o, kQ) for o in stab)
    A /= np.linalg.norm(A)
    r = rep_closure(A, stab, kQ, [0])
    assert r["err"] < 1e-10 and r["unitary"] < 1e-10
    for o in stab:
        assert abs(rep_matrix_Q(A, o[2], o[3], o[4], kQ, [0])[0, 0] - 1) < 1e-10
    old = [abs(_old(A, o[2], o[3], o[4], kQ, [0])[0, 0]) for o in stab]
    assert 1 - min(old) > 1e-2


def test_sigma_h_parity_unchanged():
    """sigma_h leaves k in place, so the parity is the same in both forms."""
    kf, cart, ops, seed = _model()
    kQ, _ = shift_map(kf, np.zeros(3))
    sh = [o for o in ops if o[0] == "sh"][0]
    A = seed + _eq49(seed, sh, kQ)
    A /= np.linalg.norm(A)
    new = rep_matrix_Q(A, sh[2], sh[3], sh[4], kQ, [0])[0, 0]
    old = _old(A, sh[2], sh[3], sh[4], kQ, [0])[0, 0]
    assert abs(new.real - 1) < 1e-10 and abs(old.real - 1) < 1e-10


def test_complex_K_singlet_label_whatever_the_operation_order():
    """An E''_1 singlet at K (C3 -> w, sigma_h -> -1, S3 = sigma_h C3 -> -w), with the
    operations ordered so that the first S3 turns the OTHER way from the first C3, as in the
    MoS2 SAVE. class_characters must still pair S3 with its own C3 and give a clean E''_1 or
    E''_2; taking the first S3 gave (1, w, -1, -w^2), "E''_1+E''_2" with residual exactly 1."""
    kf, cart, ops, seed = _model()
    Q = np.array([1 / 3, 1 / 3, 0])
    kQ, _ = shift_map(kf, Q)
    w = np.exp(2j * np.pi / 3)
    st = stabilizer(ops, Q)
    idx = [i for i, o in enumerate(ops) if any(o is x for x in st)]

    def turns(i):
        return int(np.rint(np.arctan2(cart[i][1, 0], cart[i][0, 0]) / (2 * np.pi / 3))) % 3
    chi = {i: (-1.0 if cart[i][2, 2] < 0 else 1.0) * w ** turns(i) for i in idx}
    A = sum(np.conj(chi[i]) * _eq49(seed, ops[i], kQ) for i in idx)
    A /= np.linalg.norm(A)
    c3 = [i for i in idx if ops[i][0] == "2C3"]
    s3 = [i for i in idx if ops[i][0] == "2S3"]
    c3.sort(key=lambda i: turns(i) != 1)
    s3.sort(key=lambda i: turns(i) != 2)
    rest = [i for i in idx if ops[i][0] not in ("2C3", "2S3")]
    stab = [ops[i] for i in rest + c3 + s3]
    name, res = assign_irreps(class_characters(A, stab, kQ, [0]), "C3h", verbose=False)
    assert name in ("E''_1", "E''_2") and res < 1e-8
    first = {}
    for o in stab:
        first.setdefault(o[0], o)
    old = {lab: complex(np.trace(rep_matrix_Q(A, o[2], o[3], o[4], kQ, [0])))
           for lab, o in first.items()}
    name, res = assign_irreps(old, "C3h", verbose=False)
    assert name == "E''_1+E''_2" and abs(res - 1) < 1e-8
