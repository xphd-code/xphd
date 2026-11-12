import numpy as np
import pytest

from xphd import blockwise, fourier_refine, mesh_indices, ws_vectors
from conftest import hex_vecs


@pytest.mark.parametrize("n,r", [(24, 15), (6, 60), (18, 20), (12, 30)])
def test_refine_exact_at_nodes(n, r):
    i = np.arange(n)
    kx, ky = np.meshgrid(i / n, i / n, indexing="ij")
    f = 6.0 - 0.4 * (np.cos(2 * np.pi * kx) + np.cos(2 * np.pi * ky))
    assert np.abs(fourier_refine(f, r)[::r, ::r] - f).max() < 1e-12


def test_refine_reproduces_band_limited_mode_everywhere():
    n, r = 24, 15
    i = np.arange(n)
    kx, ky = np.meshgrid(i / n, i / n, indexing="ij")
    I = np.arange(n * r)
    KX, KY = np.meshgrid(I / (n * r), I / (n * r), indexing="ij")
    got = fourier_refine(np.cos(2 * np.pi * (3 * kx - 2 * ky)), r)
    assert np.abs(got - np.cos(2 * np.pi * (3 * KX - 2 * KY))).max() < 1e-12


def test_blockwise_uses_nearest_not_floor():
    """Fine points folding to |q| ~ 0 must take the Gamma value, not q=-1/n."""
    n, r = 12, 30
    a = np.arange(n * n, dtype=float).reshape(n, n)
    B = blockwise(a, r)
    I = np.arange(n * r)
    near = np.abs(((I / (n * r) + 0.5) % 1) - 0.5) < 1 / (2 * n)
    assert np.all(B[near, 0] == a[0, 0])
    assert B[n * r - 1, 0] == a[0, 0]


def test_ws_vectors_hexagonal_minimum_image():
    n = 24
    a1, a2 = hex_vecs()
    _, r = ws_vectors(n, n, a1, a2)
    assert r[0, 0] == 0.0
    assert np.isclose(r.max() / np.linalg.norm(a1), n / np.sqrt(3), rtol=1e-6)
    assert r.max() / np.linalg.norm(a1) < n


def test_mesh_indices_rejects_incomplete_mesh():
    q = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    with pytest.raises(ValueError):
        mesh_indices(q, 4, 4)
