"""Fixes from the external review of 4 Nov 2026 (bundle up to irreps.py)."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from xphd import arpes, bte
from xphd.core.mesh import hex_dist, hex_norm


def _brute(q):
    q = np.asarray(q, float)[:, :2]
    best = np.full(len(q), np.inf)
    for m1 in range(-3, 4):
        for m2 in range(-3, 4):
            a, b = q[:, 0] + m1, q[:, 1] + m2
            best = np.minimum(best, a * a + b * b + a * b)
    return np.sqrt(best)


def test_hex_norm_is_the_minimum_image():
    rng = np.random.default_rng(0)
    q = rng.uniform(-2, 2, size=(2000, 3))
    assert np.allclose(hex_norm(q), _brute(q), atol=1e-12)
    assert hex_norm(np.array([[0.4, 0.4, 0.0]]))[0] == pytest.approx(np.sqrt(0.28))
    assert hex_norm(np.array([[1 / 3, 1 / 3, 0]]))[0] == pytest.approx(1 / np.sqrt(3))
    small = rng.uniform(-0.2, 0.2, size=(200, 2))
    old = np.sqrt(small[:, 0] ** 2 + small[:, 1] ** 2 + small[:, 0] * small[:, 1])
    assert np.allclose(hex_norm(small), old)
    a, b = rng.uniform(0, 1, (5, 1, 2)), rng.uniform(0, 1, (1, 7, 2))
    assert hex_dist(a, b).shape == (5, 7)


def test_valley_masks_hexagonal():
    from xphd.chirality import valley_masks
    k = np.array([[0.86, 0.16, 0.0]])
    nK, nKp = valley_masks(k)
    assert not nK[0] and nKp[0]
    n = 18
    i, j = np.divmod(np.arange(n * n), n)
    kf = np.stack([i / n, j / n, 0 * i], 1)
    nK, nKp = valley_masks(kf)
    assert nK.sum() == nKp.sum()
    neg = ((-i) % n) * n + (-j) % n
    assert np.array_equal(nK, nKp[neg])


def test_path_indices_are_the_nearest_points():
    n = 12
    i, j = np.divmod(np.arange(n * n), n)
    kf = np.stack([i / n, j / n, 0 * i], 1)
    idx, ticks = arpes.path_indices(kf, ["G", "M", "K", "G"], npts=30)
    from xphd.excsym import HIGH_SYMMETRY
    pts = [np.array(HIGH_SYMMETRY[l]) for l in ["G", "M", "K", "G"]]
    seg = [a0 + t * (a1 - a0) for a0, a1 in zip(pts[:-1], pts[1:])
           for t in np.linspace(0, 1, 30, endpoint=False)] + [pts[-1]]
    seg = np.array(seg)[:, :2]
    d = hex_dist(kf[None, :, :2], seg[:, None, :])
    assert np.allclose(d[np.arange(len(seg)), idx], d.min(axis=1))
    assert idx[0] == 0 and idx[ticks[2]] == (n // 3) * n + n // 3


def test_gamma_acoustic():
    hw = np.array([0.0, 0.0, 0.0, 0.034, 0.046])
    ac, res = bte.gamma_acoustic(hw, 1e-4)
    assert list(ac) == [0, 1, 2] and res is None
    hw = np.array([2e-4, 5e-4, 1.2e-3, 0.034, 0.046])
    ac, res = bte.gamma_acoustic(hw, 1e-4)
    assert list(ac) == [0, 1, 2] and res == pytest.approx(1.2e-3)
    hw = np.array([0.0, 0.0, 0.020, 0.034])
    ac, res = bte.gamma_acoustic(hw, 1e-4)
    assert list(ac) == [0, 1] and res is None
    hw = np.array([-3e-5, 0.0, 4e-4, 0.03])
    ac, res = bte.gamma_acoustic(hw, 1e-4)
    assert list(ac) == [0, 1, 2]


def test_outrates_read_back_without_repeating_the_mesh(tmp_path, capsys):
    from test_pipeline import _bte_set
    from xphd.cli import main
    pat = str(_bte_set(tmp_path))
    out = tmp_path / "o.npz"
    main(["bte", pat, "--T", "200", "--refine", "2", "--refine-only",
          "--save-outrates", str(out)])
    main(["bte", pat, "--T", "200", "--outrates", str(out), "--t-end", "100", "--nt", "3"])
    assert "out-rates read from" in capsys.readouterr().out
    main(["bte", pat, "--T", "200", "--refine", "2", "--outrates", str(out),
          "--t-end", "100", "--nt", "3"])
    with pytest.raises(SystemExit, match="refine"):
        main(["bte", pat, "--T", "200", "--refine", "3", "--outrates", str(out),
              "--t-end", "100", "--nt", "3"])
    with pytest.raises(SystemExit, match="'T'"):
        main(["bte", pat, "--T", "250", "--outrates", str(out), "--t-end", "100", "--nt", "3"])


def test_fine_must_be_a_multiple():
    from xphd.cli import _refine_from
    arc = SimpleNamespace(n1=4)
    a = SimpleNamespace(hw_fine=None, refine=1, fine=10)
    with pytest.raises(SystemExit, match="multiple"):
        _refine_from(a, arc)
    a.fine = 2
    with pytest.raises(SystemExit, match="multiple"):
        _refine_from(a, arc)
    a.fine = 12
    assert _refine_from(a, arc)[1] == 3


def test_band_base_respected(tmp_path):
    n = 3
    i, j = np.divmod(np.arange(n * n), n)
    kf = np.stack([i / n, j / n, 0 * i], 1)
    E = np.tile([1.0, 1.1, 1.2], (len(kf), 1))
    F = np.zeros((1, len(kf)))
    F[0, 4] = 1.0
    for base, want in ((0, 1), (None, 0)):
        kw = dict(t=np.array([0.0]), F=F, Q_red=kf, band=np.ones(len(kf), int),
                  T=np.array(77.0))
        if base is not None:
            kw["band_base"] = np.array(base)
        np.savez(tmp_path / "s.npz", **kw)
        N = arpes.populations(kf, np.arange(len(kf)), E, snaps=str(tmp_path / "s.npz"),
                              time=0.0)
        assert N[4, want] == pytest.approx(1.0), base


def test_snapshots_carry_band_base(tmp_path):
    rm = SimpleNamespace(E=np.linspace(0, 0.01, 6), meta={"T": 77.0}, dims=(1, 2, 3),
                         channels=None, P=np.zeros((6, 6)), R=np.zeros((6, 6)))
    sol = SimpleNamespace(t=np.array([0.0, 1.0]), y=np.ones((6, 2)))
    bte.save_snapshots(rm, sol, tmp_path / "s.npz", channels=False, verbose=False)
    d = np.load(tmp_path / "s.npz")
    assert int(d["band_base"]) == 0 and d["band"].min() == 0


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="needs /proc")
def test_archives_hold_no_file_open(tmp_path):
    import xphd
    from test_pipeline import _bte_set
    pat = _bte_set(tmp_path, n=6)
    before = len(os.listdir("/proc/self/fd"))
    arcs = xphd.load_archives(pat, verbose=False)
    for arc in arcs.values():
        arc.grid("g2"), arc.E_n, arc.has("G"), arc.convention
    after = len(os.listdir("/proc/self/fd"))
    assert len(arcs) == 36 and after - before < 5
    rm = bte.build_rates(arcs, T=200.0, verbose=False)
    assert rm.P.shape == (36 * 3, 36 * 3)
    import pickle
    pickle.loads(pickle.dumps(next(iter(arcs.values()))))


def test_windowed_snapshot_number_of_excitons(tmp_path):
    n, f = 3, 2
    i, j = np.divmod(np.arange(n * n), n)
    kf = np.stack([i / n, j / n, 0 * i], 1)
    E = np.tile([1.0, 1.1], (len(kf), 1))
    F = np.full((1, 4), 0.25)
    Q = np.array([[0, 0, 0], [0.5 / n, 0, 0], [0, 0.5 / n, 0], [1 / n, 0, 0]])
    Qc = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1 / n, 0, 0]])
    np.savez(tmp_path / "w.npz", t=np.array([0.0]), F=F, Q_red=Q, Q_coarse_red=Qc,
             band=np.zeros(4, int), band_base=np.array(0), T=np.array(77.0),
             mesh=np.array([n * f, n * f, 1]), coarse_mesh=np.array([n, n, 1]))
    N, tot = arpes.populations(kf, np.arange(len(kf)), E, snaps=str(tmp_path / "w.npz"),
                               time=0.0, return_total=True)
    assert tot == pytest.approx(1.0 / f ** 2) and N.sum() == pytest.approx(1.0)
    assert N[0, 0] == pytest.approx(0.75) and N[3, 0] == pytest.approx(0.25)
