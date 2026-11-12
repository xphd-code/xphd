"""xphd arpes --holes: replica edges with the hole at chosen high-symmetry points, E(Q) +
eps_v(point) - eps_VBM along the path."""
import numpy as np
import pytest

from xphd import arpes
from xphd.excsym import shift_map

from test_arpes_path import band, mesh
from test_arpes_valence import _setup


def test_hole_points():
    kf = mesh(6)
    eps = band(kf)[:, None]
    hp = arpes.hole_points(kf, eps, ["K", "G", "M"])
    names = [h[0] for h in hp]
    assert names == ["K", "G", "M"]
    K, G, M = hp
    assert K[3] == pytest.approx(0.0) and K[4]
    assert G[3] == pytest.approx(band(np.array([[0, 0]]))[0] - eps.max()) and not G[4]
    assert not M[4]
    eps2 = np.column_stack([eps[:, 0] - 0.3, eps[:, 0] - 0.3 + 2e-4, eps[:, 0]])
    hp2 = arpes.hole_points(kf, eps2, ["K"], nbands=2)
    assert hp2[0][2] == [2] and sorted(hp2[1][2]) == [0, 1]
    assert hp2[1][3] == pytest.approx(-0.3 + 2e-4, abs=1e-9)
    with pytest.raises(SystemExit, match="not a point"):
        arpes.hole_points(mesh(4), band(mesh(4))[:, None], ["K"])


def test_selection_by_hole_position():
    kf = mesh(6)
    nk = len(kf)
    A, B = 7, 22
    E_ibz = np.column_stack([np.full(nk, 1.0), np.full(nk, 1.5)])
    shifts, Walls = [], []
    for j in range(nk):
        kq, _ = shift_map(kf, kf[j])
        W = np.zeros((2, nk, 1))
        W[0, np.nonzero(kq == A)[0][0], 0] = 1.0
        W[1, np.nonzero(kq == B)[0][0], 0] = 1.0
        shifts.append(kq)
        Walls.append(W)
    smap = [np.arange(nk)] * nk
    eA = arpes.hole_band_energy(E_ibz, np.arange(nk), smap, shifts, Walls, [0], [A])
    eB = arpes.hole_band_energy(E_ibz, np.arange(nk), smap, shifts, Walls, [0], [B])
    eC = arpes.hole_band_energy(E_ibz, np.arange(nk), smap, shifts, Walls, [0], [3])
    assert np.allclose(eA, 1.0) and np.allclose(eB, 1.5) and np.isnan(eC).all()


def test_curves_on_path_nan_where_unsupported():
    kf = mesh(6)
    E = np.ones(len(kf))
    E[5] = np.nan
    pts = np.array([[0.0, 0.0, 0.0], [5 / 6, 0.0, 0.0], [0.0, 5 / 6, 0.0]])
    c = arpes.curves_on_path([E], kf, pts, [0], [0.1])[0]
    assert c[0] == pytest.approx(1.1)
    assert np.isnan(c).any()


def test_command_holes(monkeypatch, tmp_path):
    kf, A, E, ev, base = _setup(monkeypatch, tmp_path)
    out = tmp_path / "h.npz"
    arpes.main(base + ["--temperature", "300", "--density", "1e13", "--cell-area", "0.0546",
                       "--holes", "K", "G",
                       "--hole-frac", "0", "-o", str(out)])
    d = np.load(out)
    assert list(d["hole_labels"]) == ["K", "G"] and d["hole_curves"].shape[0] == 2
    vbm = ev.max()
    iK = int(np.argmin(np.linalg.norm(kf[:, :2] - [1 / 3, 1 / 3], axis=1)))
    E1 = np.asarray(E).min(axis=1)
    ref = arpes.dispersion_on_path(E1, kf, d["k_fine"], [iK])[0]
    assert np.allclose(d["hole_curves"][0], ref, atol=1e-10)
    assert d["hole_offset"][0] == pytest.approx(ev[iK].max() - vbm)
    assert d["hole_offset"][1] == pytest.approx(ev[0].max() - vbm)
    refG = arpes.dispersion_on_path(E1, kf, d["k_fine"], [0])[0] + d["hole_offset"][1]
    assert np.allclose(d["hole_curves"][1], refG, atol=1e-10)
    assert bool(d["hole_local_max"][0]) and not bool(d["hole_local_max"][1])
