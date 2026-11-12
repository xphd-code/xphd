import numpy as np

from xphd import delta_weights_2d
from xphd.tetra import _tri_weights


def test_vertex_weight_integral_is_exact():
    """int dE w_i(E) = A/3 for EACH vertex separately."""
    rng = np.random.default_rng(0)
    ev = np.sort(rng.uniform(-1, 1, size=(500, 3)), axis=1)
    xg, wg = np.polynomial.legendre.leggauss(4)
    acc = np.zeros((len(ev), 3))
    for lo, hi in ((ev[:, 0], ev[:, 1]), (ev[:, 1], ev[:, 2])):
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        for x, w in zip(xg, wg):
            acc += _tri_weights(ev - (mid + half * x)[:, None], 1.0, 1e-14) \
                   * (w * half)[:, None]
    assert np.abs(acc - 1 / 3).max() < 1e-10


def _band(n):
    i = np.arange(n)
    kx, ky = np.meshgrid(i / n, i / n, indexing="ij")
    return -0.5 * (np.cos(2 * np.pi * kx) + np.cos(2 * np.pi * ky)
                   + np.cos(2 * np.pi * (kx + ky))) / 3


def test_energy_sweep_normalisation_and_first_moment():
    n = 24
    rng = np.random.default_rng(0)
    eps = _band(n) + 1e-4 * rng.standard_normal((n, n))
    i = np.arange(n)
    kx, ky = np.meshgrid(i / n, i / n, indexing="ij")
    f = 1 + np.cos(2 * np.pi * kx) + 0.4 * np.sin(2 * np.pi * (kx + 2 * ky))

    Es = np.linspace(eps.min() - 0.02, eps.max() + 0.02, 4001)
    dE = Es[1] - Es[0]
    W = np.array([delta_weights_2d(E - eps, 1e-12) for E in Es])

    assert abs(W.sum() * dE - 1.0) < 3e-3
    assert abs((W * f[None]).sum() * dE - f.mean()) < 3e-3
    assert W.min() >= -1e-14
