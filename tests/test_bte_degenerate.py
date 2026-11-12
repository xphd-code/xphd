"""Couplings averaged over degenerate multiplets (bte.degeneracy_average)."""
import numpy as np
import pytest

from xphd import bte

N, NM, NE = 4, 3, 3


def test_multiplets_by_energy_and_parity():
    E = np.array([1.0, 1.00005, 1.2, 1.20003, 1.20008, 1.5])
    assert bte.degenerate_multiplets(E, 1e-4) == [[0, 1], [2, 3, 4], [5]]
    assert bte.degenerate_multiplets(E, 0.0) == [[i] for i in range(6)]
    chi = np.array([1, -1, 1, 1, 0.2, -1])
    assert bte.degenerate_multiplets(E, 1e-4, chi) == [[0], [1], [2, 3], [4], [5]]
    perm = [5, 2, 0, 4, 1, 3]
    g = bte.degenerate_multiplets(E[perm], 1e-4)
    assert sorted(sorted(perm[i] for i in x) for x in g) == [[0, 1], [2, 3, 4], [5]]
    D, info = bte.degeneracy_matrices(E[None, :], 1e-4)
    assert np.allclose(D[0].sum(axis=1), 1) and info["multiplets"] == 2
    assert info["spread"] == pytest.approx(8e-5) and info["gap"] == pytest.approx(0.19995)


def _unitary(rng, d=2):
    q, r = np.linalg.qr(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))
    return q * (np.diag(r) / np.abs(np.diag(r)))


def test_average_is_gauge_invariant_block_mean():
    rng = np.random.default_rng(0)
    nq = 5
    M = rng.normal(size=(nq, NM, 4, 4)) + 1j * rng.normal(size=(nq, NM, 4, 4))
    Ei = np.array([1.0, 1.1, 1.1, 1.3])
    Ef = np.tile(np.array([0.9, 1.2, 1.2, 1.2]), (nq, 1))
    Ef[2] = [0.9, 1.0, 1.1, 1.2]
    Di, _ = bte.degeneracy_matrices(Ei[None], 1e-4)
    Df, _ = bte.degeneracy_matrices(Ef, 1e-4)
    U, V = np.eye(4, dtype=complex), np.eye(4, dtype=complex)
    U[1:3, 1:3] = _unitary(rng)
    V[1:4, 1:4] = _unitary(rng, 3)
    Mr = np.einsum("ab,qmbc,dc->qmad", U, M, V.conj())
    Mr[2] = np.einsum("ab,mbc->mac", U, M[2])
    g, gr = np.abs(M) ** 2, np.abs(Mr) ** 2
    assert not np.allclose(g, gr)
    a, ar = bte.degeneracy_average(g, Di[0], Df), bte.degeneracy_average(gr, Di[0], Df)
    assert np.allclose(a, ar, rtol=1e-12, atol=1e-14)
    blk = g[0, 1][1:3, 1:4]
    assert np.allclose(a[0, 1][1:3, 1:4], blk.mean())
    assert a[0, 1][1:3, 1:4].sum() == pytest.approx(blk.sum())
    assert np.allclose(a[0, 1][0, 0], g[0, 1][0, 0])


def _model(path, gauge_seed=None, seed=0, split=0.004):
    """N x N archives, 3 states; states 1 and 2 degenerate at Gamma, split by `split` (1 - c)
    elsewhere (split 0: degenerate at every Q)."""
    path.mkdir()
    i = np.arange(N)
    kx, ky = np.meshgrid(i / N, i / N, indexing="ij")
    c = (np.cos(2 * np.pi * kx) + np.cos(2 * np.pi * ky) + np.cos(2 * np.pi * (kx + ky))) / 3
    base = -0.03 * (c - 1)
    E = np.stack([2.40 + base, 2.45 + base - split * (1 - c), 2.45 + base + split * (1 - c)], -1)
    rng = np.random.default_rng(seed)
    M = (rng.normal(size=(N, N, N, N, NM, NE, NE))
         + 1j * rng.normal(size=(N, N, N, N, NM, NE, NE))) * 2e-3
    M[:, :, 0, 0, 0] = 0.0
    if gauge_seed is not None:
        U = np.eye(NE, dtype=complex)
        U[1:, 1:] = _unitary(np.random.default_rng(gauge_seed))
        M[0, 0] = np.einsum("ab,ijmbc->ijmac", U, M[0, 0])
        for a in range(N):
            for b in range(N):
                si, sj = (-a) % N, (-b) % N
                M[a, b, si, sj] = np.einsum("mac,dc->mad", M[a, b, si, sj], U.conj())
    qr = np.stack([kx.ravel(), ky.ravel(), np.zeros(N * N)], 1)
    for a in range(N):
        for b in range(N):
            sx, sy = (i + a) % N, (i + b) % N
            hw = np.array([0., .018, .092])[None, None, :] + 0.006 * (1 - c)[..., None]
            np.savez(path / f"Q{a}{b}.npz",
                     g2_grid=(np.abs(M[a, b]) ** 2).reshape(N * N, NM, NE, NE),
                     E_m_grid=E[np.ix_(sx, sy)].reshape(N * N, NE),
                     hw_grid=hw.reshape(N * N, NM), E_n_grid=E[a, b],
                     q_red=qr, Q_red=np.array([a / N, b / N, 0.0]),
                     mesh=np.array([N, N, 1]), osc_Q0=np.array([1.0, 0.5, 0.5]))
    return str(path / "Q*.npz")


def _arcs(tmp_path, name, gauge_seed, split=0.004):
    import xphd
    return xphd.load_archives(_model(tmp_path / name, gauge_seed, split=split), verbose=False)


def test_build_rates_gauge_invariant(tmp_path):
    A = _arcs(tmp_path, "a", None)
    B = _arcs(tmp_path, "b", 7)
    for bose in ("node", "crossing"):
        ra = bte.build_rates(A, T=150.0, bose=bose, verbose=False)
        rb = bte.build_rates(B, T=150.0, bose=bose, verbose=False)
        assert np.allclose(ra.P, rb.P, rtol=1e-10, atol=1e-12 * ra.P.max()), bose
        r0a = bte.build_rates(A, T=150.0, bose=bose, verbose=False, deg_tol=0)
        r0b = bte.build_rates(B, T=150.0, bose=bose, verbose=False, deg_tol=0)
        assert not np.allclose(r0a.P, r0b.P, rtol=1e-3)
        assert "deg" in ra.meta and ra.meta["deg_tol"] == 1e-4
        F0 = np.zeros(len(ra.E))
        F0[[1, 2]] = 1e-3
        ya = bte.propagate(ra, F0, t_end=2000.0, nt=11).y
        yb = bte.propagate(rb, F0, t_end=2000.0, nt=11).y
        assert np.allclose(ya, yb, rtol=1e-6, atol=1e-12)


def test_pair_degenerate_everywhere(tmp_path):
    """Partners whose bands stay degenerate at every Q are one multiplet everywhere: equal rows,
    equal columns, totals unchanged, and equal populations stay equal -- which without the
    averaging they do not."""
    A = _arcs(tmp_path, "a", 3, split=0.0)
    dbl = np.array([1, 2])
    F0 = np.zeros(N * N * NE)
    F0[dbl] = 1e-3
    for bose in ("node", "crossing"):
        rm = bte.build_rates(A, T=150.0, bose=bose, verbose=False)
        r0 = bte.build_rates(A, T=150.0, bose=bose, verbose=False, deg_tol=0)
        odd = np.arange(NE, len(rm.E)) % NE == 0
        rest = np.arange(NE, len(rm.E))[odd]
        assert np.allclose(rm.P[1, rest], rm.P[2, rest]), bose
        assert np.allclose(rm.P[rest, 1], rm.P[rest, 2]), bose
        assert rm.P[dbl].sum() == pytest.approx(r0.P[dbl].sum(), rel=1e-10)
        assert np.allclose(rm.P[rest][:, dbl].sum(axis=1), r0.P[rest][:, dbl].sum(axis=1),
                           rtol=1e-10)
        pairs = np.arange(len(rm.E)) % NE != 0
        assert rm.P[pairs][:, dbl].sum() == pytest.approx(r0.P[pairs][:, dbl].sum(), rel=1e-10)
        for r, same in ((rm, True), (r0, False)):
            y = bte.propagate(r, F0, t_end=2000.0, nt=21).y
            d = np.abs(y[1] - y[2]).max() / y[1].max()
            assert (d < 1e-8) if same else (d > 1e-3), (bose, same, d)


def test_windowed_gauge_invariant(tmp_path):
    A = _arcs(tmp_path, "a", None)
    B = _arcs(tmp_path, "b", 11)
    kw = dict(refine=2, bose="crossing", verbose=False)
    ra = bte.windowed_rates(A, 150.0, 1.0, **kw)
    rb = bte.windowed_rates(B, 150.0, 1.0, **kw)
    assert np.allclose(ra.P, rb.P, rtol=1e-10, atol=1e-12 * ra.P.max())
    r0a = bte.windowed_rates(A, 150.0, 1.0, deg_tol=0, **kw)
    r0b = bte.windowed_rates(B, 150.0, 1.0, deg_tol=0, **kw)
    assert not np.allclose(r0a.P, r0b.P, rtol=1e-3)


def test_rescale_targets_averaged(tmp_path):
    A = _arcs(tmp_path, "a", 5, split=0.0)
    rm = bte.build_rates(A, T=150.0, verbose=False)
    ref = rm.P.sum(axis=1) * (1 + 0.1 * np.arange(len(rm.E)) / len(rm.E))
    ref[1], ref[2] = ref[1] * 0.8, ref[2] * 1.2
    out, _ = bte.rescale_rows(rm, ref, verbose=False)
    assert np.allclose(out.P[1, 3:], out.P[2, 3:], rtol=1e-10)
