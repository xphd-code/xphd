"""Tests for the pieces integrated from standalone scripts."""
import math
import os
import pathlib

import numpy as np
import pytest

from xphd.matdyn_split import concat_gp, main_merge


def _write_chunks(tmp_path, n=9, nch=3):
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    q = np.stack([i.ravel() / n, j.ravel() / n, np.zeros(n * n)], 1)
    off = 0
    idx = ["# chunk first npts\n"]
    for c, part in enumerate(np.array_split(q, nch)):
        m, fr, gp, d = [], [f" &plot nbnd=   6, nks= {len(part):6d} /\n"], [], 0.0
        for k, qq in enumerate(part):
            m += ["     diagonalizing the dynamical matrix ...\n", "\n",
                  f"     q = {qq[0]:12.6f} {qq[1]:12.6f} {qq[2]:12.6f}\n",
                  " " + "*" * 50 + "\n"]
            for mo in range(6):
                m.append(f"     freq ({mo+1:5d}) = 1.0 [THz] = "
                         f"{100.0*(mo+1):15.6f} [cm-1]\n")
                for _ in range(2):
                    m.append("  ( 0.1 0.0 0.1 0.0 0.1 0.0 )\n")
            m.append(" " + "*" * 50 + "\n")
            fr += [f"{qq[0]:12.6f}{qq[1]:12.6f}{qq[2]:12.6f}\n",
                   "  " + "".join(f"{100.0*(mo+1):10.4f}" for mo in range(6))
                   + "\n"]
            if k:
                d += math.dist(qq, part[k - 1])
            gp.append(f"{d:12.6f}" + "".join(f"{100.0*(mo+1):12.4f}"
                                             for mo in range(6)) + "\n")
        (tmp_path / f"modes_{c:03d}.modes").write_text("".join(m))
        (tmp_path / f"freq_{c:03d}.freq").write_text("".join(fr))
        (tmp_path / f"freq_{c:03d}.freq.gp").write_text("".join(gp))
        idx.append(f"{c:5d} {off:9d} {len(part):8d}\n")
        off += len(part)
    (tmp_path / "chunk_index.txt").write_text("".join(idx))
    return n * n


def test_merge_keeps_one_header_per_block(tmp_path):
    nq = _write_chunks(tmp_path)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        main_merge(["--chunks", "3", "-o", "m.modes", "--freq", "x.freq"])
        txt = (tmp_path / "m.modes").read_text()
        assert txt.count("q =") == nq
        assert txt.count("diagonalizing") == nq
        assert (tmp_path / "x.freq").read_text().count("&plot") == 1
    finally:
        os.chdir(cwd)


def test_gp_distance_is_monotonic(tmp_path):
    nq = _write_chunks(tmp_path)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        rows = concat_gp(3, "freq", "out.gp", verbose=False)
        assert rows == nq
        d = np.loadtxt("out.gp")[:, 0]
        assert np.all(np.diff(d) >= -1e-9)
    finally:
        os.chdir(cwd)


def test_gp_absent_returns_none(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        assert concat_gp(2, "nothing", "out.gp", verbose=False) is None
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize("cmd", sorted(__import__("xphd.cli").cli.DELEGATED))
def test_delegated_command_parses_help(cmd, capsys):
    from xphd.cli import _delegate
    with pytest.raises(SystemExit) as e:
        _delegate([cmd, "--help"])
    assert e.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_mode_labels_follow_archive_q_order(tmp_path, monkeypatch):
    """Labels must land on the archive's q-ordering, whatever it is."""
    import xphd.modes
    from xphd.mode_labels import main
    fine, n = 12, 4
    b1 = np.array([1.0, 0.0, 0.0])
    b2 = np.array([0.5, np.sqrt(3) / 2, 0.0])
    i, j = np.meshgrid(np.arange(fine), np.arange(fine), indexing="ij")
    frac = np.stack([i.ravel() / fine, j.ravel() / fine], 1)
    qc = frac[:, :1] * b1 + frac[:, 1:] * b2
    L = []
    for q in qc:
        L += ["     diagonalizing the dynamical matrix ...\n", "\n",
              f"     q = {q[0]:12.6f} {q[1]:12.6f} {q[2]:12.6f}\n",
              " " + "*" * 40 + "\n"]
        for m in range(6):
            L.append(f"     freq ({m+1:5d}) = 1.0 [THz] = "
                     f"{100.0*(m+1):12.6f} [cm-1]\n")
            L += ["  ( 0.1 0.0 0.1 0.0 0.1 0.0 )\n"] * 2
        L.append(" " + "*" * 40 + "\n")
    (tmp_path / "matdyn.modes").write_text("".join(L))
    a, b = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    qr = np.stack([a.ravel() / n, b.ravel() / n, np.zeros(n * n)], 1)
    qr = qr[np.random.default_rng(3).permutation(n * n)]
    G = np.zeros((n * n, 6, 2, 2), complex)
    np.savez(tmp_path / "arc.npz", G_grid=G, g2_grid=np.abs(G) ** 2,
             Ge_grid=G, Gh_grid=G, E_n_grid=np.zeros(2),
             E_m_grid=np.zeros((n * n, 2)), hw_grid=np.full((n * n, 6), 0.1),
             q_red=qr, Q_red=np.zeros(3), mesh=np.array([n, n, 1]))
    monkeypatch.setattr(xphd.modes, "classify",
                        lambda ev, **k: np.repeat(
                            np.arange(len(ev))[:, None], 6, axis=1))
    monkeypatch.chdir(tmp_path)
    main(["--masses", "10.81", "14.007", "--fine", str(fine),
          "--mesh", str(n), "--archive", "arc.npz"])
    out = np.load(tmp_path / "mode_labels.npy")
    want = [int(round(q[0] * n)) * n + int(round(q[1] * n)) for q in qr]
    assert np.array_equal(out[:, 3], want)
    iG = int(np.where((qr[:, 0] == 0) & (qr[:, 1] == 0))[0][0])
    assert np.all(out[iG, :3] == -1)


def test_mode_labels_requires_masses():
    from xphd.mode_labels import main
    with pytest.raises(SystemExit):
        main([])


def test_linewidth_npz_keeps_per_branch_split(archive, tmp_path):
    """`xphd linewidth -o x.npz` once dropped the per-branch arrays it had computed."""
    import xphd
    arc = xphd.ExcPhArchive(archive)
    res = xphd.compute(arc, [10.0, 300.0], refine=2, acoustic_cut=5e-4,
                       verbose=False)
    out = tmp_path / "lw.npz"
    res.save(str(out))
    d = np.load(out)
    assert "mode_em" in d.files and "mode_ab" in d.files
    nT, ne = len(d["T"]), len(d["E_n"])
    assert d["mode_em"].shape[:2] == (nT, ne)
    assert np.allclose(d["mode_em"].sum(-1), d["emission"], rtol=1e-8, atol=1e-14)
    assert np.allclose(d["mode_ab"].sum(-1), d["absorption"], rtol=1e-8, atol=1e-14)


def _hex_mesh(n):
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    kf = np.stack([i.ravel() / n, j.ravel() / n, np.zeros(n * n)], 1)
    rlat = np.array([[1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0],
                     [0.0, 0.0, 1.0]])
    return kf, rlat


def test_band_parity_path_on_24_mesh():
    """Gamma-M-K-Gamma on 24x24: 13 + 5 + 9 points, 25 without repeats, and the M-K points at
    known indices."""
    from xphd.band_parity import path_points
    from xphd.excsym import HIGH_SYMMETRY as H
    kf, rlat = _hex_mesh(24)
    corners = [(lab, H[lab]) for lab in ("G", "M", "K", "G")]
    idx, x, ticks = path_points(kf, rlat, corners)
    assert len(idx) == 25
    assert np.all(np.diff(x) > 0)
    assert [t[1] for t in ticks] == ["G", "M", "K", "G"]
    q = np.rint(kf[idx, :2] * 24).astype(int)
    mk = [tuple(v) for v in q[12:17]]
    assert mk == [(12, 0), (11, 2), (10, 4), (9, 6), (8, 8)]
    assert ticks[-1][0] == pytest.approx(0.5 + 1 / (2 * np.sqrt(3))
                                         + 1 / np.sqrt(3), rel=1e-9)


def test_band_parity_finds_points_stored_as_other_images():
    """A point stored as 23/24 lies on a segment through -1/24."""
    from xphd.band_parity import path_points
    kf, rlat = _hex_mesh(24)
    idx, _, _ = path_points(kf, rlat, [("A", (-0.25, 0.0)), ("B", (0.25, 0.0))])
    assert len(idx) == 13


def test_band_parity_mixed_degenerate_pair_is_undetermined():
    """Even a and odd b returned as (a+b)/sqrt2, (a-b)/sqrt2: sigma_h swaps them, so D =
    [[0,1],[1,0]] -- diagonal 0, off-diagonal 1."""
    from xphd.band_parity import parity_from_D
    D = np.zeros((2, 3, 3))
    D[:, 0, 0] = 1.0
    D[0, 1:, 1:] = np.diag([1.0, -1.0])
    D[1, 1:, 1:] = [[0.0, 1.0], [1.0, 0.0]]
    p, off, ok = parity_from_D(D)
    assert ok[0].all()
    assert np.allclose(p[0], [1, 1, -1])
    assert ok[1, 0] and not ok[1, 1] and not ok[1, 2]
    assert np.allclose(off[1, 1:], 1.0)


def test_band_parity_flips_skip_undetermined_points():
    from xphd.band_parity import flips_along
    pp = np.array([1.0, 1.0, 0.0, -1.0, -1.0, 1.0])
    ok = np.array([True, True, False, True, True, True])
    assert flips_along(pp, ok) == 2


def test_band_parity_locates_sigma_h():
    from xphd.band_parity import sigma_h_index
    E, C3 = np.eye(3), np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]])
    sh = np.diag([1, 1, -1])
    ops = [E, C3, sh, C3 @ sh]
    assert sigma_h_index(ops + ops) == 2


SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"


def test_per_branch_linewidths_sum_to_total(archive):
    import xphd
    r = xphd.compute(xphd.ExcPhArchive(archive), [10.0, 300.0], refine=5,
                     verbose=False)
    assert r.mode_em is not None
    per = (r.mode_em + r.mode_ab).sum(axis=2)
    assert np.allclose(per, r.total, rtol=1e-10, atol=1e-15)
    assert np.allclose(r.mode_em.sum(axis=2), r.emission, rtol=1e-10,
                       atol=1e-15)


def test_linewidth_per_mode_keeps_the_main_table(archive, tmp_path, capsys):
    """--per-mode adds a table and a _modes file; the main .txt must be the same as without it,
    since plotting scripts read its columns by position."""
    from xphd.cli import main
    a, b = tmp_path / "plain.txt", tmp_path / "pm.txt"
    main(["linewidth", str(archive), "--refine", "5", "--T", "10", "300",
          "-o", str(a)])
    main(["linewidth", str(archive), "--refine", "5", "--T", "10", "300",
          "--per-mode", "-o", str(b)])
    assert a.read_text() == b.read_text()
    assert "per-branch linewidth" in capsys.readouterr().out
    rows = [l.split() for l in (tmp_path / "pm_modes.txt").read_text()
            .splitlines() if not l.startswith("#")]
    assert len(rows) == 2 * 4 * 6
    tot = {}
    for T, n, m, e, ab, t in rows:
        assert float(t) == pytest.approx(float(e) + float(ab), abs=2e-6)
        tot[(T, n)] = tot.get((T, n), 0.0) + float(t)
    block = a.read_text().split("\n\n")[0].splitlines()
    main_rows = [l.split() for l in block if l.strip() and not l.startswith("#")]
    for r in main_rows:
        T = f"{float(r[0]):.1f}"
        for n in range(4):
            assert tot[(T, str(n + 1))] == pytest.approx(float(r[1 + n]),
                                                         abs=1e-4)


def _d3h_with_time_reversal():
    C3 = np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]])
    M = np.array([[1, 1, 0], [0, -1, 0], [0, 0, 1]])
    sh = np.diag([1, 1, -1])
    sp = [Y.astype(float) for R in (np.eye(3, dtype=int), C3, C3 @ C3)
          for X in (R, R @ M) for Y in (X, X @ sh)]
    return sp + [-R for R in sp]


@pytest.mark.parametrize("noise", [0.0, 1e-7, 5e-7, 1e-6])
def test_star_map_survives_single_precision_q(noise):
    """hBN's q-points carry ~1e-7."""
    from xphd.symmetry import star_map
    n = 24
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    Q = np.stack([i.ravel() / n, j.ravel() / n, np.zeros(n * n)], 1)
    Q += np.random.default_rng(0).uniform(-noise, noise, Q.shape) * [1, 1, 0]
    assert len(star_map(Q, _d3h_with_time_reversal())[0]) == 61


def test_check_accepts_a_single_precision_split(archive, tmp_path, capsys):
    """Ge + Gh reproduces G only to ~1e-7 in float32; that is correct."""
    import xphd
    d = dict(np.load(archive))
    G = d["G_grid"]
    d["Ge_grid"] = (0.4 * G).astype(np.complex64)
    d["Gh_grid"] = (0.6 * G).astype(np.complex64)
    p = tmp_path / "f32.npz"
    np.savez(p, **d)
    out = xphd.ExcPhArchive(str(p)).check(verbose=True)
    assert 1e-9 < out["Ge_plus_Gh"] < 1e-5
    assert "Ge + Gh vs G" in capsys.readouterr().out


def test_band_parity_flags_a_mixed_degenerate_pair():
    """sigma_h on two degenerate bands of opposite parity, returned rotated by 45 degrees: the
    diagonal is 0, and the point must be undetermined, not a parity of zero."""
    from xphd.band_parity import parity_from_D
    c = np.cos(np.pi / 4)
    U = np.array([[c, -c], [c, c]])
    clean = np.diag([1.0, -1.0])
    mixed = U @ clean @ U.T
    p, off, ok = parity_from_D(np.stack([clean, mixed]), tol=0.1)
    assert np.allclose(p[0], [1, -1]) and ok[0].all()
    assert np.allclose(p[1], [0, 0], atol=1e-12) and not ok[1].any()
    assert np.allclose(off[1], [1, 1])


def _interp_fixture(tmp_path):
    """parity.npz and a yambo-format interpolated file from one dispersion."""
    n = 24
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    Q = np.stack([i.ravel() / n, j.ravel() / n, np.zeros(n * n)], 1)
    r2 = lambda q: ((((q[:, :2] + .5) % 1) - .5) ** 2).sum(1) + \
        (((q[:, 0] + .5) % 1) - .5) * (((q[:, 1] + .5) % 1) - .5)
    E = lambda q: np.array([3.8, 4.3])[None, :] + \
        np.array([-1.2, 1.5])[None, :] * r2(q)[:, None]
    np.savez(tmp_path / "parity.npz", Q_red=Q, E=E(Q),
             chi=np.where(r2(Q)[:, None] < 0.1, 1.0, -1.0) * np.ones((1, 2)))
    M, K = np.array([.5, 0, 0]), np.array([1 / 3, 1 / 3, 0])
    qp = np.array([M * (1 - t) for t in np.linspace(0, 1, 41)]
                  + [K * t for t in np.linspace(0, 1, 41)[1:]])
    rows = [f"{k:.6f} " + " ".join(f"{v:.6f}" for v in E(qp[k:k + 1])[0])
            + " " + " ".join(f"{v:.6f}" for v in qp[k]) for k in range(len(qp))]
    (tmp_path / "o.excitons_interpolated").write_text(
        "# |q| e1 e2 qx qy qz\n" + "\n".join(rows) + "\n")


def _acoustic_data(target):
    """g2, hw, |q| on a 12x12 hexagonal mesh with |G|^2 * 2w = target(|q|)."""
    from xphd.core.mesh import hex_norm
    n = 12
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    q = np.stack([i.ravel() / n, j.ravel() / n, np.zeros(n * n)], 1)
    r = hex_norm(q).reshape(n, n)
    hw = np.zeros((n, n, 3)); g2 = np.zeros((n, n, 3, 1, 1))
    for nu in range(3):
        hw[:, :, nu] = 0.05 * (nu + 1) * r + 1e-6
        g2[:, :, nu, 0, 0] = np.where(r > 0, target(r) / (2 * hw[:, :, nu]), 0)
    return g2, hw, r


def test_q2_recovers_a_pure_q2_coupling():
    from xphd.linewidth import fit_acoustic
    g2, hw, r = _acoustic_data(lambda q: 0.2 * q ** 2)
    for model in ("q2", "auto"):
        A, C, info = fit_acoustic(g2, hw, r, n_shell=3, verbose=False,
                                  model=model)
        assert np.allclose(C[:3], 0.2, rtol=1e-6)
        assert np.allclose(A, 0.0) and not any(d["piezo"] for d in info.values())


def test_q2_uses_the_ratio_mean_not_the_joint_slope():
    """The bug: with a real offset, auto keeps A; q2 used to zero A but keep the slope fitted
    ALONGSIDE it. q2 must be a fit with A = 0."""
    from xphd.linewidth import fit_acoustic
    g2, hw, r = _acoustic_data(lambda q: 0.01 + 0.2 * q ** 2)
    _, Ca, ia = fit_acoustic(g2, hw, r, n_shell=4, verbose=False, model="auto")
    _, Cq, iq = fit_acoustic(g2, hw, r, n_shell=4, verbose=False, model="q2")
    assert ia[0]["piezo"] and np.isclose(Ca[0, 0, 0], 0.2, rtol=1e-6)
    ratio_mean = float(iq[0]["C_shell"].mean())
    assert np.isclose(Cq[0, 0, 0], ratio_mean, rtol=1e-12)
    assert not np.isclose(Cq[0, 0, 0], 0.2, rtol=1e-3)


def test_n_shell_is_honoured_or_refused():
    from xphd.linewidth import fit_acoustic
    g2, hw, r = _acoustic_data(lambda q: 0.2 * q ** 2)
    with pytest.raises(SystemExit):
        fit_acoustic(g2, hw, r, n_shell=2, verbose=False, model="auto")
    for k in (1, 2):
        _, _, info = fit_acoustic(g2, hw, r, n_shell=k, verbose=False,
                                  model="q2")
        assert len(info[0]["C_shell"]) == k


def test_kept_constant_warns_for_excitons(capsys):
    from xphd.linewidth import fit_acoustic
    g2, hw, r = _acoustic_data(lambda q: 0.01 + 0.2 * q ** 2)
    fit_acoustic(g2, hw, r, n_shell=4, verbose=True, model="auto")
    out = capsys.readouterr().out
    assert "WARNING a constant term A was kept" in out
    assert "--acoustic-model q2" in out


def test_falling_ratio_is_reported_as_a_lower_bound(capsys):
    from xphd.linewidth import fit_acoustic
    g2, hw, r = _acoustic_data(lambda q: 0.2 * q ** 2 / (1 + (q / 0.12) ** 2) ** 1.5)
    fit_acoustic(g2, hw, r, n_shell=4, verbose=True, model="q2")
    assert "LOWER bound" in capsys.readouterr().out


def test_sweep_applies_the_same_acoustic_model(archive, tmp_path, monkeypatch):
    """The zone maps must use the same small-q treatment as the single-Q runs, or a minimum
    reads differently in the two figures."""
    import sys, xphd
    from xphd.sweep import sweep
    mod = sys.modules["xphd.sweep"]
    seen, real = [], mod.compute
    def spy(*args, **kw):
        seen.append((kw.get("acoustic_model"), kw.get("n_shell")))
        return real(*args, **kw)
    monkeypatch.setattr(mod, "compute", spy)
    kw = dict(refine=5, acoustic_cut=1e-4)
    fld = sweep([str(archive)], [300.0], verbose=False,
                acoustic_model="q2", n_shell=2, **kw)
    assert seen == [("q2", 2)]
    one = real(xphd.ExcPhArchive(archive), [300.0], verbose=False,
               acoustic_model="q2", n_shell=2, **kw)
    assert np.allclose(fld.LW[0, 0], one.total[0], rtol=1e-12)
    p = tmp_path / "f.npz"
    fld.save(p)
    d = np.load(p)
    assert str(d["acoustic_model"]) == "q2" and int(d["n_shell"]) == 2


def test_linewidth_warns_when_a_minimum_is_left_at_none(archive, capsys,
                                                        monkeypatch):
    """A multiplet with no emission at any temperature sits at a minimum."""
    import types, xphd.linewidth
    from xphd.cli import main
    fake = types.SimpleNamespace(
        groups=[[0], [1]], T=np.array([10.0, 300.0]),
        emission=np.array([[0.0, 1e-3], [0.0, 2e-3]]),
        total=np.array([[1e-4, 2e-3], [2e-3, 4e-3]]))
    monkeypatch.setattr(xphd.linewidth, "compute", lambda *a, **k: fake)
    main(["linewidth", str(archive), "--refine", "5", "--T", "10", "300"])
    out = capsys.readouterr().out
    assert "WARNING {1} emit(s) nothing" in out and "{2}" not in out.split("WARNING")[1]
    main(["linewidth", str(archive), "--refine", "5", "--T", "10", "300",
          "--acoustic-model", "q2"])
    assert "emit(s) nothing" not in capsys.readouterr().out


def _phonon_modes(path, red, split3d=False):
    """matdyn-format eigenvectors for a two-atom planar lattice, q printed to four decimals as
    matdyn does."""
    B = np.array([[1.0, 1 / np.sqrt(3)], [0.0, 2 / np.sqrt(3)]])
    F = (np.cos(2 * np.pi * red[:, 0]) + np.cos(2 * np.pi * red[:, 1])
         + np.cos(2 * np.pi * (red[:, 0] + red[:, 1])))
    s2 = np.clip((3 - F) / 4.5, 0, None)
    s_ = np.sqrt(s2)
    LO = 700 + 80 * s_ + (40 if split3d else 0) * (s_ < 1e-9)
    Fr = np.stack([400 * s2, 450 * s_, 750 * s_, 780 - 160 * s2,
                   700 + 40 * s2, LO], 1)
    odd = [1, 0, 0, 1, 0, 0]
    L = []
    for i, q in enumerate(red @ B):
        L += ["     diagonalizing the dynamical matrix ...\n", "\n",
              f"     q = {q[0]:12.4f} {q[1]:12.4f} {0.0:12.4f}\n",
              " " + "*" * 60 + "\n"]
        for m in np.argsort(Fr[i]):
            L.append(f"     freq ({m + 1:5d}) = 1.0 [THz] = {Fr[i, m]:12.6f} [cm-1]\n")
            e = (0, 0, 1) if odd[m] else (1, 0, 0)
            L += ["  ( " + "  ".join(f"{c / np.sqrt(2):9.6f} 0.000000"
                                     for c in e) + " )\n"] * 2
        L.append(" " + "*" * 60 + "\n")
    pathlib.Path(path).write_text("".join(L))


def _gmkg(npts):
    G, M, K = np.zeros(2), np.array([0.5, 0.0]), np.array([1 / 3, 1 / 3])
    return np.array([a_ + t * (b_ - a_) for a_, b_ in ((G, M), (M, K), (K, G))
                     for t in np.linspace(0, 1, npts)[:-1]] + [G])


def _ticks(out):
    return [l.split(":")[0].strip() for l in out.splitlines()
            if l.strip().startswith(("Gamma:", "M:", "K:"))]


def _valley(tmp_path, tag, b, hc, C, n=24, r=15):
    """An exactly solvable minimum: band E0 + b q^2, LA branch hbar c q and coupling |G|^2 2
    hbar w = C q^2, built from first harmonics only so that Fourier refinement reproduces
    them exactly."""
    from xphd.core.mesh import hex_norm

    def s2(q1, q2):
        c = (np.cos(2 * np.pi * q1) + np.cos(2 * np.pi * q2)
             + np.cos(2 * np.pi * (q1 + q2))) / 3
        return 3 * (1 - c) / (4 * np.pi ** 2)

    def build(m):
        i, j = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
        q1, q2 = i.ravel() / m, j.ravel() / m
        S = s2(q1, q2)
        return q1, q2, S, np.stack([0.8 * S, 0.75 * hc * np.sqrt(S),
                                    hc * np.sqrt(S)], 1)

    q1, q2, S, hw = build(n)
    qn = hex_norm(np.stack([q1, q2, 0 * q1], 1))
    g2 = np.zeros((n * n, 3, 1, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        g2[:, 2, 0, 0] = np.where(hw[:, 2] > 0, C * qn ** 2 / (2 * hw[:, 2]), 0)
    G = np.sqrt(g2).astype(complex)
    np.savez(tmp_path / f"{tag}.npz", G_grid=G, g2_grid=g2, Ge_grid=G * .5,
             Gh_grid=G * .5, E_m_grid=(3.4 + b * S)[:, None], hw_grid=hw,
             E_n_grid=np.array([3.4]), q_red=np.stack([q1, q2, 0 * q1], 1),
             Q_red=np.zeros(3), mesh=np.array([n, n, 1]))
    np.save(tmp_path / f"{tag}_hw.npy", build(n * r)[3])


def _crossing_valley(tmp_path, kink, b=13.0, hc=0.18, C=0.01, n=24, r=15):
    """A minimum whose energy-ordered band has a crossing `kink` eV up, far from q = 0 -- the
    kind of kink a global interpolant rings from."""
    from xphd.core.mesh import hex_norm

    def s2(q1, q2):
        c = (np.cos(2 * np.pi * q1) + np.cos(2 * np.pi * q2)
             + np.cos(2 * np.pi * (q1 + q2))) / 3
        return 3 * (1 - c) / (4 * np.pi ** 2)

    def build(m):
        i, j = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
        q1, q2 = i.ravel() / m, j.ravel() / m
        S = s2(q1, q2)
        band = np.minimum(3.4 + b * S, 3.4 + kink)
        return q1, q2, band, np.stack([0.8 * S, 0.75 * hc * np.sqrt(S),
                                       hc * np.sqrt(S)], 1)

    q1, q2, band, hw = build(n)
    qn = hex_norm(np.stack([q1, q2, 0 * q1], 1))
    g2 = np.zeros((n * n, 3, 1, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        g2[:, 2, 0, 0] = np.where(hw[:, 2] > 0, C * qn ** 2 / (2 * hw[:, 2]), 0)
    G = np.sqrt(g2).astype(complex)
    p = tmp_path / "x.npz"
    np.savez(p, G_grid=G, g2_grid=g2, Ge_grid=G * .5, Gh_grid=G * .5,
             E_m_grid=band[:, None], hw_grid=hw, E_n_grid=np.array([3.4]),
             q_red=np.stack([q1, q2, 0 * q1], 1), Q_red=np.zeros(3),
             mesh=np.array([n, n, 1]))
    kB = 8.617333e-5
    exact = ((4 * np.pi ** 2 / np.sqrt(3)) * C
             / np.expm1(hc ** 2 / b / (kB * 77.0)) / b ** 2)
    return p, build(n * r)[3], exact


def test_local_band_keeps_nodes_and_the_minimum_curvature(tmp_path):
    from xphd.core.interp import fine_band
    n, r = 24, 15
    p, _, _ = _crossing_valley(tmp_path, kink=0.15)
    E = np.load(p)["E_m_grid"][:, 0].reshape(n, n)
    for method in ("local", "fourier"):
        assert np.allclose(fine_band(E, r, method)[::r, ::r], E, atol=1e-12)
    loc, fou = fine_band(E, r, "local"), fine_band(E, r, "fourier")
    q2 = (1 / (n * r)) ** 2
    pts = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    bl = [(loc[i % (n * r), j % (n * r)] - 3.4) / q2 for i, j in pts]
    bf = [(fou[i % (n * r), j % (n * r)] - 3.4) / q2 for i, j in pts]
    assert np.ptp(bl) < 1e-6 * np.mean(bl)
    assert abs(np.mean(bl) / 13.0 - 1) < 0.01
    assert abs(np.mean(bf) / 13.0 - 1) > 0.3


def test_local_band_fixes_the_minimum_linewidth(tmp_path):
    """A crossing 150 meV above the minimum moved its width by 35% under the Fourier
    interpolant; the local one keeps it at the closed form."""
    import xphd
    from xphd.linewidth import compute
    p, hwf, exact = _crossing_valley(tmp_path, kink=0.15)
    kw = dict(hw_fine=hwf, refine=15, acoustic_cut=5e-4, acoustic_model="q2",
              n_shell=1, verbose=False)
    loc = compute(xphd.ExcPhArchive(p), [77.0], band_interp="local", **kw)
    fou = compute(xphd.ExcPhArchive(p), [77.0], band_interp="fourier", **kw)
    assert abs(loc.total[0, 0] / exact - 1) < 0.05
    assert abs(fou.total[0, 0] / exact - 1) > 0.2
    assert loc.meta["band_interp"] == "local"


def test_sweep_applies_the_same_band_interpolation(archive, monkeypatch):
    import sys, xphd
    from xphd.sweep import sweep
    mod = sys.modules["xphd.sweep"]
    seen, real = [], mod.compute

    def spy(*args, **kw):
        seen.append(kw.get("band_interp"))
        return real(*args, **kw)

    monkeypatch.setattr(mod, "compute", spy)
    fld = sweep([str(archive)], [300.0], verbose=False, refine=5,
                band_interp="fourier")
    assert seen == ["fourier"]
    one = real(xphd.ExcPhArchive(archive), [300.0], verbose=False, refine=5,
               band_interp="fourier")
    assert np.allclose(fld.LW[0, 0], one.total[0], rtol=1e-12)
    sweep([str(archive)], [300.0], verbose=False, refine=5)
    assert seen[-1] == "local"


def test_text_headers_record_the_settings(archive, tmp_path):
    """A linewidth file must say which small-q model and band interpolation produced it -- the
    numbers alone cannot."""
    import xphd
    from xphd.linewidth import compute
    res = compute(xphd.ExcPhArchive(archive), [77.0], refine=5, verbose=False,
                  acoustic_model="q2", n_shell=2, band_interp="local")
    res.save(str(tmp_path / "a.txt"))
    res.save_modes(str(tmp_path / "a_modes.txt"))
    for f in ("a.txt", "a_modes.txt"):
        head = (tmp_path / f).read_text()
        assert "# acoustic_model q2 (n_shell 2)   band_interp local" in head, f
    res = compute(xphd.ExcPhArchive(archive), [77.0], refine=5, verbose=False,
                  band_interp="fourier")
    res.save(str(tmp_path / "b.txt"))
    assert "# acoustic_model none   band_interp fourier" in (tmp_path / "b.txt").read_text()


def _qp_file(path, dE63, dE64, skip=None):
    """A yambo-like o-*.qp: 19 k-points, bands 57-68; band 62 peaks at k = 7 (K), where the DFT
    conduction bands are 63 at 1.200 and 64 at 1.236 eV."""
    with open(path, "w") as f:
        f.write("#  K-point  Band  Eo [eV]  E-Eo [eV]  Sc|Eo [eV]\n")
        for k in range(1, 20):
            for b in range(57, 69):
                if skip and (b in skip):
                    continue
                eo = -0.5 + 0.01 * k if b <= 62 else 1.5
                if k == 7:
                    eo = {61: -0.45, 62: 0.0, 63: 1.200, 64: 1.236}.get(b, eo)
                de = {63: dE63, 64: dE64}.get(b, -0.3 if b <= 62 else 0.9)
                f.write(f"  {k:3d}  {b:3d}  {eo:10.4f}  {de:10.4f}  0.0\n")


def _band_parity(path):
    n = 12
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    kf = np.stack([i.ravel() / n, j.ravel() / n, 0 * i.ravel()], 1)
    lab = np.tile([-1, 1, 1, -1, 1, -1, -1, 1], (len(kf), 1)).astype(float)
    np.savez(path, kf=kf, p=lab, bands=np.arange(59, 67))


def _tmd_like_case(path, n=12, ne=6, nmod=9, seed=5, mislabel_branch=None):
    """Nine branches, a 12x12 mesh, couplings that obey the sigma_h rule exactly: the even and
    odd exciton families cross at |Q| ~ 0.25|b|, branches 1, 4 and 8 are odd and branch 4
    turns even beyond 0.30|b|."""
    rng = np.random.default_rng(seed)
    qf = np.array([[i / n, j / n] for i in range(n) for j in range(n)])
    nq = len(qf)
    d = qf - np.rint(qf)
    r = np.linalg.norm(np.stack([d[:, 0] + 0.5 * d[:, 1], np.sqrt(3) / 2 * d[:, 1]], 1), axis=1)
    Ee, Eo = 1.40 + 2.0 * r ** 2, 1.55 - 0.5 * r ** 2
    E, chi = np.zeros((nq, ne)), np.zeros((nq, ne))
    for q in range(nq):
        fam = sorted([(Ee[q], 1), (Eo[q], -1), (Ee[q] + .3, 1), (Eo[q] + .3, -1),
                      (Ee[q] + .5, 1), (Eo[q] + .5, -1)], key=lambda t: t[0])
        E[q] = [t[0] for t in fam][:ne]
        chi[q] = [t[1] for t in fam][:ne]
    p_lam = chi[0].copy()
    lab = np.full((nq, nmod), 1)
    for nu in (0, 3, 7):
        lab[:, nu] = 3
    lab[:, 3] = np.where(r < 0.30, 3, 1)
    p_nu = np.where(lab == 3, -1, 1)
    g2 = np.zeros((nq, nmod, ne, ne))
    for q in range(nq):
        for nu in range(nmod):
            for lam in range(ne):
                for b in range(ne):
                    if p_nu[q, nu] * chi[q, b] == p_lam[lam]:
                        g2[q, nu, lam, b] = rng.uniform(0.2, 1.0) * 1e-6
    lab_used = lab.copy()
    if mislabel_branch is not None:
        lab_used[:, mislabel_branch] = np.where(lab_used[:, mislabel_branch] == 3, 1, 3)
    G = np.sqrt(g2) * np.exp(1j * rng.uniform(0, 6.28, g2.shape))
    Q3 = np.column_stack([qf, np.zeros(nq)])
    np.savez(path / "a.npz", G_grid=G, g2_grid=np.abs(G) ** 2, Ge_grid=G / 2, Gh_grid=G / 2,
             E_n_grid=E[0], E_m_grid=E, hw_grid=np.tile(np.linspace(.003, .04, nmod), (nq, 1)),
             q_red=Q3, Q_red=np.zeros(3), mesh=np.array([n, n, 1]))
    np.savez(path / "parity.npz", Q_red=Q3, chi=chi, E=E)
    np.save(path / "labels.npy", lab_used)


def _shares(stdout):
    import re
    return [float(l.split()[1]) for l in stdout.splitlines()
            if re.match(r"\s+\d+\s+\d\.\d+e[+-]\d+$", l)]


def test_selection_rule_with_nine_branches_and_parity_only_labels(tmp_path, capsys, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from xphd.selection import main as sel
    _tmd_like_case(tmp_path)
    monkeypatch.chdir(tmp_path)
    sel(["--archive", "a.npz", "--parity", "parity.npz", "--labels", "labels.npy",
         "--state", "3", "--by-final", "--nfinal", "2", "--out", "s.pdf"])
    out = capsys.readouterr().out
    assert "forbidden    0.0000e+00    0.00%" in out
    assert "[odd, odd at this q" in out or "[even, even at this q" in out
    assert "FORBIDDEN" not in out.split("COUPLING BY FINAL STATE")[1].split("modes at this q:")[1].split(";")[0]
    assert "zone-wide" in out and "|b|" in out
    with pytest.raises(SystemExit) as e:
        d = dict(np.load("parity.npz"))
        d["chi"], d["E"] = d["chi"][:, :2], d["E"][:, :2]
        np.savez("p2.npz", **d)
        sel(["--archive", "a.npz", "--parity", "p2.npz", "--labels", "labels.npy",
             "--out", "s2.pdf"])
    assert "--nstates 6" in str(e.value)


def _leaky_case(path):
    """The rule-obeying case, plus one final state (index 2) carrying an opposite-parity
    admixture eps = 0.002 that couples at eps times normal."""
    _tmd_like_case(path)
    d = dict(np.load(path / "parity.npz"))
    chi = d["chi"].copy()
    rng0 = np.random.default_rng(2)
    e_q = 10 ** rng0.uniform(-4, -2, chi.shape[0])
    chi[:, 2] *= (1 - 2 * e_q)
    d["chi"] = chi
    np.savez(path / "parity.npz", **d)
    a = dict(np.load(path / "a.npz"))
    g2 = a["g2_grid"].copy()
    lab = np.load(path / "labels.npy")
    p_nu = np.where(lab == 3, -1, 1)
    p_lam = np.sign(chi[0])
    rng = np.random.default_rng(1)
    for q in range(g2.shape[0]):
        for nu in range(g2.shape[1]):
            for lam in range(g2.shape[2]):
                if p_nu[q, nu] * np.sign(chi[q, 2]) != p_lam[lam]:
                    g2[q, nu, lam, 2] = e_q[q] * rng.uniform(0.2, 1.0) * 1e-6
    a["g2_grid"] = g2
    a["G_grid"] = np.sqrt(g2)
    a["Ge_grid"] = a["Gh_grid"] = np.sqrt(g2) / 2
    np.savez(path / "a.npz", **a)


def _snaps(path, tau):
    n = 12
    Q = np.array([[i / n, j / n, 0] for i in range(n) for j in range(n)])
    r = np.linalg.norm(((Q[:, :2] + 0.5) % 1 - 0.5) @ np.array([[1, 0], [0.5, np.sqrt(3) / 2]]), axis=1)
    E = 5.475 + 3.0 * r ** 2
    wB = np.exp(-(E - E.min()) / (8.617333e-5 * 300))
    wB /= wB.sum()
    t = np.linspace(0, 10000, 51)
    F0 = np.exp(-((E - 6.0) / 0.05) ** 2)
    F0 /= F0.sum()
    F = np.array([wB + (F0 - wB) * np.exp(-tt / tau) for tt in t])
    np.savez(path / "snaps.npz", t=t, F=F, E=E, Q_red=Q)


def _qp_rows():
    return [(1, 62, -0.88, -2.18), (1, 63, 1.323, 1.753), (1, 64, 1.527, 1.985),
            (2, 62, 0.00, -1.12), (2, 63, 1.259, 1.320), (2, 64, 1.296, 1.302)]


def _honeycomb_modes(tmp_path):
    """matdyn.modes at K for a planar binary honeycomb (springs to 1st and 2nd neighbours)."""
    import shutil, subprocess, sys
    src = pathlib.Path(__file__).parent / "_honeycomb_model.py"
    shutil.copy(src, tmp_path / "make_model.py")
    subprocess.run([sys.executable, "make_model.py", "lattice"], cwd=tmp_path, check=True,
                   capture_output=True)
