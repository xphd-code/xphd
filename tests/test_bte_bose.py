"""Phonon occupations in the BTE rate matrix: bose="node" against "crossing"."""
import numpy as np
import pytest

import xphd
from xphd import bte

KB = 8.617333262e-5


def _set(tmp_path, E_of_q, hw_of_q, n=6, nm=1, seed=0, tag="S"):
    """Archives for bands E_of_q(c) (n, n, ne) and phonons hw_of_q(c) (n, n, nm), with c the
    hexagonal form factor (1 at Gamma)."""
    i = np.arange(n)
    kx, ky = np.meshgrid(i / n, i / n, indexing="ij")
    c = (np.cos(2 * np.pi * kx) + np.cos(2 * np.pi * ky)
         + np.cos(2 * np.pi * (kx + ky))) / 3
    E = E_of_q(c)
    hw = hw_of_q(c)
    ne = E.shape[-1]
    rng = np.random.default_rng(seed)
    qr = np.stack([kx.ravel(), ky.ravel(), np.zeros(n * n)], 1)
    d = tmp_path / tag
    d.mkdir()
    for a in range(n):
        for b in range(n):
            sx, sy = (np.arange(n) + a) % n, (np.arange(n) + b) % n
            Em = E[np.ix_(sx, sy)]
            g2 = np.abs(rng.standard_normal((n, n, nm, ne, ne))) * 4e-6
            np.savez(d / f"Q{a:02d}{b:02d}.npz",
                     g2_grid=g2.reshape(n * n, nm, ne, ne),
                     E_m_grid=Em.reshape(n * n, ne), hw_grid=hw.reshape(n * n, nm),
                     E_n_grid=Em[0, 0], q_red=qr, Q_red=np.array([a / n, b / n, 0.0]),
                     mesh=np.array([n, n, 1]))
    return xphd.load_archives(str(d / "Q*.npz"), verbose=False)


def _generic(tmp_path):
    return _set(tmp_path,
                lambda c: np.array([2.40, 2.45, 2.52])[None, None, :] - 0.03 * (c - 1)[..., None],
                lambda c: np.array([.018, .092])[None, None, :] + 0.006 * (1 - c)[..., None],
                nm=2)


def test_crossing_keeps_detailed_balance_and_equilibrium(tmp_path):
    arcs = _generic(tmp_path)
    for T in (77.0, 300.0):
        rm = bte.build_rates(arcs, T=T, verbose=False, bose="crossing")
        p = np.exp(-(rm.E - rm.E.min()) / (KB * T))
        flux = p[:, None] * rm.P
        assert np.abs(flux - flux.T).max() < 1e-10 * flux.max()
        assert np.abs(rm.P - rm.Q.T).max() == 0.0
        assert np.abs(rm.R + rm.R.T).max() < 1e-12 * max(abs(rm.R).max(), 1e-30)
        r = bte.check_balance(rm, N_tot=1e-3, verbose=False)
        assert r["bose"] < 1e-10


def test_crossing_channels_sum_to_P(tmp_path):
    arcs = _generic(tmp_path)
    rm = bte.build_rates(arcs, T=150.0, verbose=False, bose="crossing", channels=True)
    err = np.abs(rm.channels.astype(float).sum(axis=(0, 1)) - rm.P).max()
    assert err < 1e-6 * rm.P.max()


def test_crossing_equals_node_when_the_crossing_sits_on_the_mesh(tmp_path):
    """Flat bands exactly one (flat) phonon apart: every crossing is at the mesh energies, so
    the two schemes must give the same matrix."""
    w = 0.030
    arcs = _set(tmp_path, lambda c: np.zeros(c.shape + (2,)) + np.array([2.40, 2.40 + w]),
                lambda c: np.full(c.shape + (1,), w))
    for T in (77.0, 300.0):
        a = bte.build_rates(arcs, T=T, verbose=False, flat_width=1e-6, bose="node")
        b = bte.build_rates(arcs, T=T, verbose=False, flat_width=1e-6, bose="crossing")
        assert a.P.max() > 0
        assert np.allclose(a.P, b.P, rtol=1e-6, atol=1e-8 * a.P.max())


def test_crossing_suppresses_absorption_booked_between_mesh_points(tmp_path):
    """Two bands 3 meV apart at Gamma, rising 67 meV to the next mesh point (as the K valley of
    MoS2 at 18x18 rises 43 meV); a flat 34 meV phonon."""
    arcs = _set(tmp_path,
                lambda c: np.array([2.400, 2.403])[None, None, :] + 0.20 * (1 - c)[..., None],
                lambda c: np.full(c.shape + (1,), 0.034), n=6)
    i, j = 1, 0
    T = 77.0
    node = bte.build_rates(arcs, T=T, verbose=False, bose="node").P[i, j]
    cros = bte.build_rates(arcs, T=T, verbose=False, bose="crossing").P[i, j]
    assert node > 0 and cros > 0
    N34, N3 = 1 / np.expm1(0.034 / (KB * T)), 1 / np.expm1(0.003 / (KB * T))
    assert cros / node == pytest.approx(N34 / (1 + N3), rel=0.6)
    assert cros / node < 0.01


def test_unknown_bose_is_refused(tmp_path):
    arcs = _generic(tmp_path)
    with pytest.raises(ValueError):
        bte.build_rates(arcs, T=300.0, verbose=False, bose="bose")
