"""Tests for chirality and the ARPES driver's pure-numpy core."""
import numpy as np
import pytest

from xphd.arpes import exciton_arpes, star_assignment
from xphd.chirality import (helicity, jz_rotation, valley_masks,
                            valley_polarisation)
from xphd.excsym import shift_map


def mesh(n):
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return np.stack([i.ravel() / n, j.ravel() / n, np.zeros(n * n)], 1)


def hbn_ops(kf):
    """D3h of planar hBN with time reversal, in reduced coordinates."""
    C3 = np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]])
    Mv = np.array([[1, 1, 0], [0, -1, 0], [0, 0, 1]])
    sh = np.diag([1, 1, -1])
    sp = []
    for R in (np.eye(3, dtype=int), C3, C3 @ C3):
        for X in (R, R @ Mv):
            for Y in (X, X @ sh):
                sp.append(Y)

    def kmap(M):
        d = kf[:, None, :] - (kf @ M.T)[None, :, :]
        d -= np.rint(d)
        return np.argmin(np.linalg.norm(d, axis=-1), 1)
    return [(M, kmap(M)) for R in sp for M in (R, -R)]


def test_helicity_of_circular_light():
    D = np.array([1, -1j]) / np.sqrt(2)
    assert helicity(*D) == pytest.approx(1.0)
    assert helicity(*D.conj()) == pytest.approx(-1.0)


def test_helicity_of_linear_light_is_zero():
    assert helicity(1.0 + 0j, 0.5 + 0j) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_one_rotation_recovers_helicity_and_valley(seed):
    """A pure K / K' pair of opposite circular senses, scrambled by an arbitrary unitary as a
    solver would return it: the rotation that restores circular polarisation must also
    restore valley polarisation."""
    kf = mesh(24)
    nK, nKp = valley_masks(kf)
    rng = np.random.default_rng(seed)
    nk, nv, nc = len(kf), 3, 3
    AK = np.zeros((nk, nv, nc), complex)
    AK[nK] = rng.standard_normal((nK.sum(), nv, nc))
    AKp = np.zeros((nk, nv, nc), complex)
    AKp[nKp] = rng.standard_normal((nKp.sum(), nv, nc))
    DK = np.array([1, -1j]) / np.sqrt(2)

    th, ph = rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)
    a, b = np.cos(th), np.sin(th) * np.exp(1j * ph)
    M = np.array([[a, b], [-np.conj(b), np.conj(a)]])
    P = M @ np.stack([DK, DK.conj()])
    A = np.einsum("mn,nkvc->mkvc", M, np.stack([AK, AKp]))

    U = jz_rotation(P)
    Pj, Ar = U @ P, np.einsum("mn,nkvc->mkvc", U, A)
    h = [helicity(*Pj[k]) for k in range(2)]
    pv = [valley_polarisation(Ar[k], nK, nKp)[2] for k in range(2)]
    assert h == pytest.approx([1.0, -1.0], abs=1e-10)
    assert pv == pytest.approx([1.0, -1.0], abs=1e-10)


def test_valley_masks_are_balanced():
    nK, nKp = valley_masks(mesh(24))
    assert nK.sum() == nKp.sum()
    assert not np.any(nK & nKp)


def test_exciton_lands_at_electron_momentum_and_energy():
    """Electron at the table index k0, hole at k0+Q (the yambo storage convention, see
    excsym.shift_map): the feature must sit at p = k0 and at E_S + eps_v(k0+Q)."""
    kf = mesh(12)
    nk, k0 = len(kf), 17
    Q = np.array([2 / 12, 1 / 12, 0.0])
    A = np.zeros((1, nk, 1, 1), complex)
    A[0, k0, 0, 0] = 1.0
    kpQ, _ = shift_map(kf, Q)
    eps = np.full((nk, 1), -1.0)
    eps[kpQ[k0], 0] = -2.0
    om = np.linspace(-1, 3, 801)
    I = exciton_arpes(A, np.array([3.0]), eps, kpQ, np.array([1.0]), om, 0.01)
    p, e = np.unravel_index(np.argmax(I), I.shape)
    assert p == k0
    assert om[e] == pytest.approx(3.0 + (-2.0), abs=0.01)


def test_mirror_parity_is_exact_at_finite_Q():
    """An exact sigma_h eigenstate stored in yambo's convention -- electron at the table index
    k, hole at k+Q -- must give chi = -1 exactly, even where the band parities change across
    its support."""
    from xphd.excsym import rep_matrix_Q, sigma_h_trace
    n = 24
    kf = mesh(n)
    nk = len(kf)
    pv = np.where(np.mod(kf[:, 0] + 0.1, 1) < 0.5, 1.0, -1.0)
    pc = np.where(np.mod(kf[:, 1] + 0.3, 1) < 0.6, 1.0, -1.0)
    Dv = pv[:, None, None].astype(complex)
    Dc = pc[:, None, None].astype(complex)
    Q = np.array([5 / n, 2 / n, 0.0])
    kpQ, _ = shift_map(kf, Q)
    odd = pc * pv[kpQ] < 0
    rng = np.random.default_rng(3)
    A = np.zeros((1, nk, 1, 1), complex)
    A[0, odd, 0, 0] = rng.normal(size=odd.sum()) + 1j * rng.normal(size=odd.sum())
    A /= np.sqrt((np.abs(A) ** 2).sum())
    sh = ("sh", Dc, Dv, np.arange(nk))
    assert sigma_h_trace(A, sh, kpQ, 0) == pytest.approx(-1.0, abs=1e-12)
    ident = np.arange(nk)
    assert np.allclose(rep_matrix_Q(A, Dc, Dv, ident, ident, [0]),
                       np.array([[np.vdot(A[0].ravel(),
                                          (np.conj(Dc) * A[0] * Dv).ravel())]]))


def test_signal_sits_below_conduction_band_by_binding_energy():
    """The measured result."""
    eps_c, eps_v_top, Eb = 3.0, -1.5, 0.5
    ES = (eps_c - eps_v_top) - Eb
    nk = 5
    A = np.zeros((1, nk, 1, 1), complex)
    A[0, 2, 0, 0] = 1.0
    eps = np.full((nk, 1), eps_v_top)
    om = np.linspace(0, 7, 1401)
    I = exciton_arpes(A, np.array([ES]), eps, np.arange(nk), np.array([1.0]),
                      om, 0.01)
    e = om[np.argmax(I[2])]
    assert e == pytest.approx(eps_c - Eb, abs=0.01)
    assert e < eps_c


def test_replica_has_the_valence_band_dispersion():
    """At low temperature the signal is a REPLICA of the valence band -- the same dispersion,
    shifted up by E_S -- not its mirror image."""
    x = np.linspace(-1, 1, 11)
    eps_v = (-0.8 * x ** 2 - 1.5)[:, None]
    A = np.exp(-x ** 2 / 0.5).astype(complex)[None, :, None, None]
    ES = 4.0
    om = np.linspace(0, 8, 3201)
    I = exciton_arpes(A, np.array([ES]), eps_v, np.arange(len(x)),
                      np.array([1.0]), om, 0.01)
    peak = om[np.argmax(I, axis=1)]
    assert np.allclose(peak, ES + eps_v[:, 0], atol=0.01)
    assert peak[5] > peak[0] and peak[5] > peak[-1]


def test_star_count_matches_yambo():
    kf = mesh(24)
    ops = hbn_ops(kf)
    key = lambda q: tuple(np.round(np.mod(q[:2], 1), 6))
    reps = {min(key(M @ q) for M, _ in ops) for q in kf}
    assert len(reps) == 61


def test_star_assembly_is_fully_symmetric():
    """Contributions invariant under their own little group, assembled over every star, must
    give a map invariant under the whole group."""
    kf = mesh(24)
    nk = len(kf)
    ops = hbn_ops(kf)
    key = lambda q: tuple(np.round(np.mod(q[:2], 1), 6))
    par, kidx = {}, np.empty(nk, int)
    for i, q in enumerate(kf):
        rep = min(key(M @ q) for M, _ in ops)
        par.setdefault(rep, len(par))
        kidx[i] = par[rep]
    first = np.array([np.where(kidx == j)[0][0] for j in range(kidx.max() + 1)])
    smap = star_assignment(kf, kidx, first, ops)

    rng = np.random.default_rng(3)
    I = np.zeros((nk, 4))
    for j in range(len(first)):
        Qj = kf[first[j]]
        stab = [km for M, km in ops
                if np.abs((Qj @ M.T - Qj) - np.rint(Qj @ M.T - Qj)).max() < 1e-6]
        raw = rng.random((nk, 4))
        W = np.mean([raw[km] for km in stab], axis=0)
        for i in np.where(kidx == j)[0]:
            I += W[smap[i]]
    for _, km in ops:
        assert np.allclose(I[km], I, atol=1e-12)
