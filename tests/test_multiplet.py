"""Multiplet rotation: parity eigenstates of near-degenerate multiplets."""
import numpy as np
import pytest

from xphd.multiplet import energy_groups, split_by_parity


def _unitary(rng, theta):
    a, b, c = rng.uniform(0, 2 * np.pi, 3)
    return np.array([[np.cos(theta) * np.exp(1j * a), -np.sin(theta) * np.exp(1j * b)],
                     [np.sin(theta) * np.exp(1j * c), np.cos(theta) * np.exp(1j * (b + c - a))]])


def test_energy_groups():
    lab = energy_groups([1.02, 1.0, 1.0201, 1.0001, 1.5], 0.005)
    assert list(lab) == [1, 0, 1, 0, 2]


def test_split_recovers_the_pure_parities():
    rng = np.random.default_rng(0)
    U = _unitary(rng, 0.5)
    assert np.allclose(U.conj().T @ U, np.eye(2))
    Sig = np.diag([1.0, -1.0])
    M = U.conj().T @ Sig @ U
    g_pure = np.array([[0.7 - 0.2j, 0.0]])
    G = (U.conj().T @ g_pure.T).T
    mu, W = split_by_parity(G, M, np.array([0, 0]))
    assert np.allclose(np.sort(mu), [-1, 1])
    assert W[mu > 0].sum() == pytest.approx(abs(0.7 - 0.2j) ** 2)
    assert W[mu < 0].sum() < 1e-28
    assert W.sum() == pytest.approx((np.abs(G) ** 2).sum())


def _case(path, n=6, nmod=3, seed=4):
    """Archive and parity file with mixed opposite-parity pairs (states 1-2) at every q but
    Gamma; points reached by time reversal carry conj(U)."""
    rng = np.random.default_rng(seed)
    qf = np.array([[i / n, j / n] for i in range(n) for j in range(n)])
    nq, ne = len(qf), 4

    def idx(q):
        d = q[None] - qf
        d -= np.rint(d)
        return int(np.argmin(np.linalg.norm(d, axis=1)))

    iqi = -np.ones(nq, int)
    trev = np.zeros(nq, bool)
    U_ibz, reps = [], []
    for f in range(nq):
        if iqi[f] >= 0:
            continue
        j = len(reps)
        reps.append(f)
        U_ibz.append(np.eye(2) if f == 0 else _unitary(rng, rng.uniform(0.3, 0.6)))
        iqi[f] = j
        g = idx(-qf[f])
        if iqi[g] < 0:
            iqi[g], trev[g] = j, True
    Sig = np.diag([1.0, -1.0, 1.0, -1.0])
    sig_ibz = np.zeros((len(reps), ne, ne), complex)
    for j, U in enumerate(U_ibz):
        sig_ibz[j][:2, :2] = U.conj().T @ Sig[:2, :2] @ U
        sig_ibz[j][2, 2], sig_ibz[j][3, 3] = 1.0, -1.0
    grp = np.tile([0, 0, 1, 2], (len(reps), 1))
    lab = np.tile([3, 1, 1], (nq, 1))
    p_nu = np.where(lab == 3, -1, 1)
    G = np.zeros((nq, nmod, ne, ne), complex)
    E = np.zeros((nq, ne))
    chi = np.zeros((nq, ne))
    for f in range(nq):
        Uf = U_ibz[iqi[f]].conj() if trev[f] else U_ibz[iqi[f]]
        Mf = sig_ibz[iqi[f]].conj() if trev[f] else sig_ibz[iqi[f]]
        chi[f] = np.real(np.diag(Mf))
        E[f] = np.array([1.0, 1.0001, 1.05, 1.10]) + 0.01 * np.linalg.norm(qf[f] - np.rint(qf[f]))
        for nu in range(nmod):
            for lam in range(ne):
                gp = np.array([(rng.normal() + 1j * rng.normal()) if p_nu[f, nu] * Sig[p, p] == Sig[lam, lam]
                               else 0.0 for p in range(ne)])
                G[f, nu, lam, :2] = Uf.conj().T @ gp[:2]
                G[f, nu, lam, 2:] = gp[2:]
    G *= 1e-3
    Q3 = np.column_stack([qf, np.zeros(nq)])
    np.savez(path / "a.npz", G_grid=G, g2_grid=np.abs(G) ** 2, Ge_grid=G / 2, Gh_grid=G / 2,
             E_n_grid=E[0], E_m_grid=E, hw_grid=np.tile([0.01, 0.02, 0.03], (nq, 1)),
             q_red=Q3, Q_red=np.zeros(3), mesh=np.array([n, n, 1]))
    np.savez(path / "parity.npz", Q_red=Q3, chi=chi, E=E, iQ_ibz=iqi, sigma_ibz=sig_ibz,
             group_ibz=grp, trev=trev, deg_tol=0.005)
    np.save(path / "labels.npy", lab)
    return trev


def _numbers(out):
    vals = {}
    for key in ("allowed", "forbidden", "undefined"):
        line = [l for l in out.splitlines() if l.strip().startswith(key)][0]
        vals[key] = float(line.split()[1])
    return vals


def _run(tmp_path, capsys, monkeypatch, *extra):
    import matplotlib
    matplotlib.use("Agg")
    from xphd.selection import main as sel
    monkeypatch.chdir(tmp_path)
    sel(["--archive", "a.npz", "--parity", "parity.npz", "--labels", "labels.npy",
         "--state", "1", "--out", "s.pdf", *extra])
    return _numbers(capsys.readouterr().out)


def test_rotate_removes_the_mixing_residue(tmp_path, capsys, monkeypatch):
    trev = _case(tmp_path)
    assert trev.any()
    plain = _run(tmp_path, capsys, monkeypatch)
    rot = _run(tmp_path, capsys, monkeypatch, "--rotate")
    assert plain["forbidden"] / plain["allowed"] > 1e-2
    assert rot["forbidden"] / rot["allowed"] < 1e-12
    assert rot["undefined"] < 1e-12 * rot["allowed"]
    total = lambda v: v["allowed"] + v["forbidden"] + v["undefined"]
    assert total(rot) == pytest.approx(total(plain), rel=1e-3)


def test_rotate_needs_the_time_reversal_conjugate(tmp_path, capsys, monkeypatch):
    _case(tmp_path)
    d = dict(np.load(tmp_path / "parity.npz"))
    d["trev"] = np.zeros_like(d["trev"])
    np.savez(tmp_path / "parity.npz", **d)
    rot = _run(tmp_path, capsys, monkeypatch, "--rotate")
    assert rot["forbidden"] / rot["allowed"] > 1e-3


def test_rotate_refuses_without_matrices(tmp_path, capsys, monkeypatch):
    _case(tmp_path)
    d = dict(np.load(tmp_path / "parity.npz"))
    for k in ("sigma_ibz", "group_ibz", "trev"):
        d.pop(k)
    np.savez(tmp_path / "parity.npz", **d)
    with pytest.raises(SystemExit) as e:
        _run(tmp_path, capsys, monkeypatch, "--rotate")
    assert "--deg-tol" in str(e.value)


def test_sigma_blocks_with_the_real_rep_matrix():
    """Two states mixing an even (c0) and an odd (c1) transition: the block from rep_matrix_Q is
    Hermitian with eigenvalues exactly +-1, and a third, separate state gets its own 1x1
    block."""
    from xphd.excsym import rep_matrix_Q
    from xphd.multiplet import sigma_blocks
    rng = np.random.default_rng(3)
    nk = 4
    U = _unitary(rng, 0.45)
    A = np.zeros((3, nk, 1, 3), complex)
    pure = np.zeros((3, nk, 1, 3), complex)
    pure[0, :, 0, 0] = 0.5
    pure[1, :, 0, 1] = 0.5
    pure[2, :, 0, 2] = 0.5
    A[0] = U[0, 0] * pure[0] + U[1, 0] * pure[1]
    A[1] = U[0, 1] * pure[0] + U[1, 1] * pure[1]
    A[2] = pure[2]
    Dc = np.tile(np.diag([1.0, -1.0, 1.0]).astype(complex), (nk, 1, 1))
    Dv = np.tile(np.eye(1, dtype=complex), (nk, 1, 1))
    op = ("sh", np.eye(3), Dc, Dv, np.arange(nk))
    M = sigma_blocks(A, op, np.arange(nk), np.array([0, 0, 1]), rep_matrix_Q)
    assert np.allclose(M, M.conj().T)
    assert np.allclose(np.sort(np.linalg.eigvalsh(M[:2, :2])), [-1, 1])
    assert M[2, 2] == pytest.approx(1.0) and abs(M[0, 2]) == 0 and abs(M[2, 1]) == 0
    assert np.allclose(np.real(np.diag(M[:2, :2])), [abs(U[0, 0]) ** 2 - abs(U[1, 0]) ** 2,
                                                     abs(U[0, 1]) ** 2 - abs(U[1, 1]) ** 2])


def test_time_reversed_follows_generate():
    from types import SimpleNamespace
    from xphd.multiplet import time_reversed
    lat = SimpleNamespace(symmetry_indexes=[0, 5, 11, 12, 23], sym_car=np.zeros((24, 3, 3)),
                          time_rev=1)
    assert list(time_reversed(lat)) == [False, False, False, True, True]
    no_tr = SimpleNamespace(symmetry_indexes=[0, 5, 11], sym_car=np.zeros((12, 3, 3)),
                            time_rev=0)
    assert not time_reversed(no_tr).any()


def _closure_file(path, cut_pair=True, nibz=8, ns=12, seed=9):
    rng = np.random.default_rng(seed)
    Sig = np.diag([1.0, -1.0] * (ns // 2))
    S = np.zeros((nibz, ns, ns), complex)
    G = np.tile(np.arange(ns), (nibz, 1))
    E = np.tile(1.0 + 0.02 * np.arange(ns), (nibz, 1))
    for j in range(nibz):
        S[j] = Sig.astype(complex)
        pairs = [(3, 4)] + ([(9, 10)] if cut_pair else [])
        for a, b in pairs:
            U = _unitary(rng, rng.uniform(0.4, 0.7))
            blk = np.diag([Sig[a, a], Sig[b, b]])
            S[j][np.ix_([a, b], [a, b])] = U.conj().T @ blk @ U
            G[j, b] = G[j, a]
            E[j, b] = E[j, a] + 1e-4
    chi = np.real(np.einsum("jnn->jn", S))
    np.savez(path / "parity16.npz", sigma_ibz=S, group_ibz=G, chi_ibz=chi,
             E=E, iQ_ibz=np.arange(nibz))


def _dmats(path, leak=False, nk=10, nb=8, op=6):
    rng = np.random.default_rng(5)
    D = np.zeros((12, nk, 1, nb, nb), complex)
    for s in range(12):
        D[s, :, 0] = np.eye(nb)
    lab = rng.choice([1j, -1j], size=(nk, nb))
    for k in range(nk):
        D[op, k, 0] = np.diag(lab[k])
        if leak and k in (2, 7):
            D[op, k, 0, nb - 1, nb - 1] *= np.sqrt(0.8)
    np.save(path / "Dmats.npy", D)
