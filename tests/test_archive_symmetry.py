"""check-archive's symmetry section, on archives built to obey (or break) it."""
import numpy as np
import pytest

from xphd.generate.check import main as check_archive


def _archive(path, n=6, nmod=9, ne=8, seed=3, break_tr=False, mixed=True,
             break_c3=None):
    """|G|^2 that is symmetric under q -> -q and q -> C3 q only in SUM over degenerate partners:
    the time-reversed image swaps two degenerate initial states and two degenerate final
    pairs, as a real rotation may."""
    rng = np.random.default_rng(seed)
    qf = np.array([[i / n, j / n] for i in range(n) for j in range(n)])
    nq = len(qf)

    def idx(q):
        d = q[None, :] - qf
        d -= np.rint(d)
        return int(np.argmin(np.linalg.norm(d, axis=1)))

    M = np.array([[-1, -1], [1, 0]])
    k_tr = np.array([idx(-q) for q in qf])
    k_c3 = np.array([idx(M @ q) for q in qf])
    orbit, flag = -np.ones(nq, int), np.zeros(nq, bool)
    third = np.zeros(nq, int)
    for s in range(nq):
        if orbit[s] >= 0:
            continue
        orbit[s] = s
        todo = [s]
        while todo:
            a = todo.pop()
            for b, f, t in ((k_c3[a], flag[a], (third[a] + 1) % 3),
                            (k_tr[a], not flag[a], third[a])):
                if orbit[b] < 0:
                    orbit[b], flag[b], third[b] = s, f, t
                    todo.append(b)
    En = np.cumsum(rng.uniform(0.02, 0.08, ne)) + 1.0
    En[1] = En[0] + 1e-6
    g2 = np.zeros((nq, nmod, ne, ne))
    Em = np.zeros((nq, ne))
    hw = np.zeros((nq, nmod))
    base = {}
    for s in np.unique(orbit):
        e = np.cumsum(rng.uniform(0.02, 0.08, ne)) + 1.0
        e[2] = e[1] + 1e-6
        e[5] = e[4] + 1e-6
        base[s] = (e, np.sort(rng.uniform(0.005, 0.09, nmod)),
                   rng.uniform(0.1, 1.0, (nmod, ne, ne)))
    for q in range(nq):
        e, w, g = base[orbit[q]]
        Em[q], hw[q], gg = e.copy(), w, g.copy()
        if flag[q] and mixed:
            gg[:, [0, 1], :] = gg[:, [1, 0], :]
            gg[:, :, [1, 2]] = gg[:, :, [2, 1]]
            gg[:, :, [4, 5]] = gg[:, :, [5, 4]]
        if flag[q] and break_tr:
            gg = gg * 2.0
        if break_c3 is not None and third[q] == 1 and orbit[q] != q:
            nu, pair = break_c3
            gg[nu][:, pair] *= 10.0
        g2[q] = gg
    Em[0] = En
    ph = np.exp(1j * rng.uniform(0, 6.28, g2.shape))
    G = np.sqrt(g2) * ph
    np.savez(path, G_grid=G, g2_grid=np.abs(G) ** 2, Ge_grid=G / 2, Gh_grid=G / 2,
             E_n_grid=En, E_m_grid=Em, hw_grid=hw,
             q_red=np.column_stack([qf, np.zeros(nq)]), Q_red=np.zeros(3),
             mesh=np.array([n, n, 1]))


def test_symmetric_archive_passes(tmp_path, capsys):
    _archive(tmp_path / "a.npz")
    check_archive([str(tmp_path / "a.npz")])
    out = capsys.readouterr().out
    assert "OK   time reversal" in out
    assert "OK   C3 rotation" in out and "60 deg basis" in out
    assert "OK below 0.01, FAIL above 0.05" in out
    assert "READY: launch the rest." in out


def test_broken_time_reversal_is_caught(tmp_path, capsys):
    _archive(tmp_path / "a.npz", break_tr=True)
    check_archive([str(tmp_path / "a.npz")])
    out = capsys.readouterr().out
    assert "FAIL time reversal" in out
    assert "OK   C3 rotation" in out
    assert "NOT READY" in out


def test_nonzero_Q_is_skipped_not_failed(tmp_path, capsys):
    _archive(tmp_path / "a.npz")
    d = dict(np.load(tmp_path / "a.npz"))
    d["Q_red"] = np.array([1 / 3, 1 / 3, 0.0])
    np.savez(tmp_path / "b.npz", **d)
    check_archive([str(tmp_path / "b.npz")])
    assert "skipped: Q is not a time-reversal-invariant point" in capsys.readouterr().out


def test_detail_locates_a_c3_fault(tmp_path, capsys):
    _archive(tmp_path / "a.npz", break_c3=(6, [4, 5]))
    check_archive([str(tmp_path / "a.npz"), "--detail"])
    out = capsys.readouterr().out
    assert "FAIL C3 rotation" in out and "OK   time reversal" in out
    assert "where the time reversal mismatch sits" not in out
    lines = out.split("where the C3 rotation mismatch sits")[1].splitlines()
    top = lines[2].split()
    assert top[0] == "7" and top[2] == "5,6"
    assert "worst q-points" in out


def test_split_doublet_fails_narrow_grouping_but_passes_default(tmp_path, capsys):
    """Initial states 3 and 4 are a doublet split by 10 meV whose members couple
    anisotropically."""
    _archive(tmp_path / "a.npz", seed=7)
    d = dict(np.load(tmp_path / "a.npz"))
    En = d["E_n_grid"].copy()
    En[3] = En[2] + 0.010
    En[4:] += 0.2
    d["E_n_grid"] = En
    Em = d["E_m_grid"].copy()
    Em[0] = En
    d["E_m_grid"] = Em
    g2 = d["g2_grid"].copy()
    qf = d["q_red"][:, :2]
    w = np.cos(np.pi * qf[:, 0]) ** 2
    tot = g2[:, :, 2, :] + g2[:, :, 3, :]
    g2[:, :, 2, :] = tot * w[:, None, None]
    g2[:, :, 3, :] = tot * (1 - w)[:, None, None]
    d["g2_grid"], d["G_grid"] = g2, np.sqrt(g2)
    d["Ge_grid"] = d["Gh_grid"] = np.sqrt(g2) / 2
    np.savez(tmp_path / "b.npz", **d)
    check_archive([str(tmp_path / "b.npz"), "--deg-tol", "0.005", "--detail"])
    out = capsys.readouterr().out
    assert "FAIL C3 rotation" in out and "OK   time reversal" in out
    assert "has a partner 10.0 meV away" in out
    check_archive([str(tmp_path / "b.npz")])
    out = capsys.readouterr().out
    assert "OK   C3 rotation" in out and "READY: launch the rest." in out
