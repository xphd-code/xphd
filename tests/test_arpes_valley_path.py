"""xphd arpes --path M K G K' M: the valley-resolved path through Gamma."""
import numpy as np
import pytest

from xphd.arpes import _path_vertices, _HEX_B, fine_path, interp_matrix, path_indices
from xphd.core.mesh import hex_dist

from test_arpes_path import HEX, band, mesh


def test_short_way_vertices():
    V = _path_vertices(["M", "K", "G", "K'", "M"], _HEX_B)
    assert np.allclose(V, [[0.5, 0.0], [1 / 3, 1 / 3], [0, 0], [-1 / 3, -1 / 3], [-0.5, 0.0]])
    V = _path_vertices(["M", "K", "Q", "G", "Q'", "K'", "M"], _HEX_B)
    assert np.allclose(V[4], [-1 / 6, -1 / 6]) and np.allclose(V[5], [-1 / 3, -1 / 3])
    pts, x, xt = fine_path(["M", "K", "G", "K'", "M"], HEX, 600)
    seg = np.diff(xt)
    assert np.allclose(seg / seg[0], [1, 2, 2, 1], rtol=1e-12)
    assert np.all(np.diff(x) > 0)
    n = len(pts)
    assert np.allclose(pts[::-1, :2][: n // 2], -pts[:, :2][: n // 2], atol=1e-12)


def test_default_path_unchanged():
    V = _path_vertices(["G", "M", "K", "G"], _HEX_B)
    assert np.allclose(V, [[0, 0], [0.5, 0], [1 / 3, 1 / 3], [0, 0]])
    pts, x, xt = fine_path(["G", "M", "K", "G"], HEX, 600)
    seg = np.diff(xt)
    assert np.allclose(seg / seg[0], [1, 1 / np.sqrt(3), 2 / np.sqrt(3)], rtol=1e-12)


def test_path_indices_short_way():
    kf = mesh(12)
    idx, ticks = path_indices(kf, ["M", "K", "G", "K'", "M"], npts=30)
    gk = idx[ticks[2]: ticks[3] + 1]
    K = np.array([1 / 3, 1 / 3])
    Kp = np.array([2 / 3, 2 / 3])
    d_K = hex_dist(kf[gk, :2], K[None, :])
    d_Kp = hex_dist(kf[gk, :2], Kp[None, :])
    assert d_Kp[-1] < 1e-9 and d_K.min() > 0.25
    assert np.allclose(hex_dist(kf[idx[ticks[-1]], :2], np.array([[0.5, 0.0]])), 0)


def test_symmetric_band_and_folding():
    kf = mesh(12)
    pts, x, xt = fine_path(["M", "K", "G", "K'", "M"], HEX, 400)
    assert (pts[:, :2] < 0).any()
    got = interp_matrix(kf, pts) @ band(kf)
    assert np.abs(got - band(pts)).max() < 0.04
    assert np.allclose(got, got[::-1], atol=1e-10)


def test_cli_rejects_unknown_label(monkeypatch, tmp_path):
    from xphd import arpes
    from test_arpes_valence import _setup
    kf, A, E, ev, base = _setup(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="--path"):
        arpes.main(base + ["--temperature", "300", "--path", "M", "X", "G",
                           "-o", str(tmp_path / "x.npz")])
