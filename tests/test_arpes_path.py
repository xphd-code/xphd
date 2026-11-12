"""The smooth path of `xphd arpes`: interpolation, path geometry, the mesh/path consistency of
exciton_spectra, QP corrections, and the driver end to end on mocked databases (no yambopy
reads)."""
import numpy as np
import pytest

from xphd import arpes
from xphd.arpes import (exciton_arpes, exciton_spectra, fine_path,
                        grid_lookup, interp_matrix)
from xphd.excsym import shift_map

HEX = np.array([[1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0], [0, 0, 1.0]])


def mesh(n, shuffle=False):
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    k = np.stack([i.ravel() / n, j.ravel() / n, np.zeros(n * n)], 1)
    if shuffle:
        k = k[np.random.default_rng(0).permutation(len(k))]
    return k


def band(q):
    """A smooth hBN-like pi band, maximum at K = (1/3, 1/3), in eV."""
    q = np.atleast_2d(q)
    f = 1 + np.exp(-2j * np.pi * q[:, 0]) + np.exp(2j * np.pi * q[:, 1])
    return -3.8 + 2.3 - np.sqrt(2.3 ** 2 + 2.6 ** 2 * np.abs(f) ** 2)


def envelope(q, width=0.18):
    """Weight around K and K', smooth and periodic."""
    out = 0.0
    for c in ((1 / 3, 1 / 3), (2 / 3, 2 / 3)):
        d = q[:, :2] - np.array(c)
        d -= np.rint(d)
        r = np.linalg.norm(d @ HEX[:2, :2], axis=1)
        out = out + np.exp(-(r / width) ** 2)
    return out


def test_grid_lookup_recovers_a_shuffled_mesh():
    kf = mesh(6, shuffle=True)
    n1, n2, t = grid_lookup(kf)
    assert (n1, n2) == (6, 6)
    for i1 in range(6):
        for i2 in range(6):
            assert np.allclose(kf[t[i1, i2], :2], [i1 / 6, i2 / 6])


def test_interpolation_is_exact_at_nodes_and_rows_sum_to_one():
    kf = mesh(12, shuffle=True)
    M = interp_matrix(kf, kf)
    assert np.allclose(M, np.eye(len(kf)), atol=1e-12)
    pts = np.random.default_rng(1).random((50, 3))
    assert np.allclose(interp_matrix(kf, pts).sum(1), 1.0)


def test_interpolation_reproduces_a_smooth_band():
    kf = mesh(12)
    pts, _, _ = fine_path(["G", "M", "K", "G"], HEX, 400)
    got = interp_matrix(kf, pts) @ band(kf)
    assert np.abs(got - band(pts)).max() < 0.04


def test_path_lengths_follow_the_hexagonal_zone():
    pts, x, xt = fine_path(["G", "M", "K", "G"], HEX, 600)
    seg = np.diff(xt)
    assert np.allclose(seg / seg[0], [1, 1 / np.sqrt(3), 2 / np.sqrt(3)],
                       rtol=1e-12)
    assert np.allclose(pts[0, :2], 0) and np.allclose(pts[-1, :2], 0)
    assert np.all(np.diff(x) > 0)


def _one_exciton(n=12, nv=2, nc=2, E_S=5.475, seed=0):
    kf = mesh(n)
    rng = np.random.default_rng(seed)
    W = envelope(kf)[:, None] * np.array([1.0, 0.05])[None, :nv]
    A = np.sqrt(W[:, :, None] / nc) * np.exp(
        2j * np.pi * rng.random((len(kf), nv, nc)))
    ev = np.column_stack([band(kf), band(kf) - 1.5])[:, :nv]
    return kf, A, ev, E_S


def test_mesh_spectrum_equals_exciton_arpes():
    kf, A, ev, E_S = _one_exciton()
    omega = np.linspace(0, 3, 300)
    kq, _ = shift_map(kf, np.zeros(3))
    ref = exciton_arpes(A[None], np.array([E_S]), ev, kq, np.array([1.0]),
                        omega, 0.05)
    W = np.sum(np.abs(A) ** 2, axis=2)
    Im, Ip = exciton_spectra(W, E_S + ev[kq], [(1.0, np.arange(len(kf)))],
                             omega, 0.05)
    assert Ip is None and np.allclose(Im, ref, atol=1e-12)


def test_path_peaks_sit_on_the_band_and_keep_their_width():
    """Off the mesh the replica follows E_S + eps_v at the true k, and the peak is as narrow as
    sigma -- interpolating the intensity would smear it."""
    kf, A, ev, E_S = _one_exciton(nv=1)
    omega = np.linspace(-3.0, 3.0, 6001)
    sigma = 0.03
    pts, x, xt = fine_path(["G", "M", "K", "G"], HEX, 600)
    M = interp_matrix(kf, pts)
    W = np.sum(np.abs(A) ** 2, axis=2)
    _, Ip = exciton_spectra(W, E_S + ev, [(1.0, np.arange(len(kf)))],
                            omega, sigma, M)
    near = envelope(pts) > 0.3
    peak = omega[np.argmax(Ip[near], axis=1)]
    assert np.abs(peak - (E_S + band(pts[near]))).max() < 0.045
    i = np.where(near)[0][len(np.where(near)[0]) // 3]
    half = Ip[i] >= 0.5 * Ip[i].max()
    fwhm = omega[half][-1] - omega[half][0]
    assert fwhm == pytest.approx(2.3548 * sigma, rel=0.05)


def test_path_agrees_with_mesh_at_mesh_points():
    kf, A, ev, E_S = _one_exciton()
    omega = np.linspace(0, 3, 200)
    W = np.sum(np.abs(A) ** 2, axis=2)
    on = kf[[0, 3, 40]]
    Im, Ip = exciton_spectra(W, E_S + ev, [(0.7, np.arange(len(kf)))],
                             omega, 0.05, interp_matrix(kf, on))
    assert np.allclose(Ip, Im[[0, 3, 40]], atol=1e-10)


class _FakeQP:
    """What yambopy's YamboQPDB exposes."""
    def __init__(self, nk, bmin, bmax, kmin=1, drop=()):
        nb = bmax - bmin + 1
        self.spin = False
        pairs = [(k, b) for k in range(kmin, kmin + nk)
                 for b in range(bmin, bmax + 1) if (k, b) not in drop]
        self.kpoint_index = np.array([k for k, _ in pairs])
        self.band_index = np.array([b for _, b in pairs])
        self.min_kpoint, self.max_kpoint = kmin, kmin + nk - 1
        self.min_band, self.max_band = bmin, bmax
        e0 = np.add.outer(-0.05 * np.arange(nk), 0.5 * (np.arange(bmin, bmax + 1) - 4))
        self.eigenvalues_dft = e0
        self.eigenvalues_qp = e0 + np.add.outer(0.01 * np.arange(nk),
                                                0.1 * np.arange(nb))
        for k, b in drop:
            self.eigenvalues_dft[k - kmin, b - bmin] = 0.0
            self.eigenvalues_qp[k - kmin, b - bmin] = 0.0


def _patch_qp(monkeypatch, **kw):
    import yambopy
    monkeypatch.setattr(yambopy.YamboQPDB, "from_db",
                        classmethod(lambda cls, filename, folder: _FakeQP(**kw)))


def test_qp_corrections_pick_the_right_bands(monkeypatch):
    """Includes the valence maximum, whose stored KS energy is exactly 0.0 on yambo's zero -- it
    must not be taken for a missing entry."""
    _patch_qp(monkeypatch, nk=7, bmin=2, bmax=7)
    dE, e0 = arpes.qp_corrections("out/ndb.QP", 7, np.array([1, 2, 3]))
    want = np.add.outer(0.01 * np.arange(7), 0.1 * np.arange(3))
    assert np.allclose(dE, want)
    assert e0[0, 2] == 0.0


@pytest.mark.parametrize("kw,msg", [(dict(nk=5, bmin=1, bmax=8), "k-points"),
                                    (dict(nk=7, bmin=3, bmax=8), "bands"),
                                    (dict(nk=7, bmin=1, bmax=8,
                                          drop=((4, 3),)), "no QP energy")])
def test_qp_corrections_refuse_incomplete_databases(monkeypatch, kw, msg):
    _patch_qp(monkeypatch, **kw)
    with pytest.raises(SystemExit, match=msg):
        arpes.qp_corrections("ndb.QP", 7, np.array([1, 2]))


def test_qp_alignment_accepts_a_constant_offset_and_refuses_a_shuffle():
    rng = np.random.default_rng(2)
    e0 = rng.normal(size=(61, 3))
    assert arpes.check_qp_alignment(e0 - 3.86, e0) == pytest.approx(-3.86)
    with pytest.raises(SystemExit, match="do not match"):
        arpes.check_qp_alignment(e0[rng.permutation(61)] - 3.86, e0)


def _hbn_ops(kf):
    C3 = np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]])
    Mv = np.array([[1, 1, 0], [0, -1, 0], [0, 0, 1]])
    sp = []
    for R in (np.eye(3, dtype=int), C3, C3 @ C3):
        for X in (R, R @ Mv):
            sp.append(X)

    def kmap(M):
        d = kf[:, None, :] - (kf @ M.T)[None, :, :]
        d -= np.rint(d)
        return np.argmin(np.linalg.norm(d, axis=-1), 1)
    return [(M, kmap(M)) for R in sp for M in (R, -R)]


def test_driver_end_to_end(monkeypatch, tmp_path):
    """`xphd arpes` on mocked databases: thermal occupations, a 6x6 mesh with every point its
    own irreducible representative."""
    import xphd.yambo as yb
    n, nS, nv, nc = 6, 3, 2, 2
    kf = mesh(n)
    rng = np.random.default_rng(3)
    A = rng.normal(size=(len(kf), nS, len(kf), nv, nc)) + 0j
    E = 5.0 + 0.3 * rng.random((len(kf), nS))

    class Lat:
        rlat = HEX
        kpoints_indexes = np.arange(len(kf))
        kmap = np.column_stack([np.arange(len(kf)), np.zeros(len(kf), int)])

    class Exc:
        unique_vbands = np.array([2, 3])

    monkeypatch.setattr(yb, "lattice", lambda save: Lat())
    monkeypatch.setattr(yb, "full_zone_kpoints", lambda ylat: kf)
    monkeypatch.setattr(yb, "load_excitons",
                        lambda path, ylat, ns: (Exc(), A[int(path.split("Q")[-1]) - 1][:ns],
                                                E[int(path.split("Q")[-1]) - 1]))
    monkeypatch.setattr(arpes, "star_kmaps", lambda save, kf_: _hbn_ops(kf_))
    ev = np.column_stack([band(kf), band(kf) - 1.0])
    np.save(tmp_path / "ev.npy", ev)
    out = tmp_path / "a.npz"
    arpes.main(["--temperature", "300", "--nstates", str(nS),
                "--valence-npy", str(tmp_path / "ev.npy"), "--fine", "200",
                "--ne", "150", "--min-occ", "0", "-o", str(out)])
    d = np.load(out)

    N = np.exp(-(E - E.min()) / (8.617333262e-5 * 300))
    N /= N.sum()
    ref = np.zeros((len(kf), 150))
    for j in range(len(kf)):
        kq, _ = shift_map(kf, kf[j])
        ref += exciton_arpes(A[j], E[j], ev, kq, N[j], d["omega"], 0.05)
    assert np.allclose(d["I"] * d["I_norm"], ref, rtol=1e-9, atol=1e-12)
    assert d["I_fine"].shape == (len(d["x_fine"]), 150)
    assert float(d["vbm"]) == pytest.approx(ev.max())
    assert np.isclose(np.diff(d["xt_fine"])[1] / np.diff(d["xt_fine"])[0],
                      1 / np.sqrt(3))


def test_dispersion_is_exact_at_mesh_points():
    kf = mesh(12)
    E1 = np.random.default_rng(4).random(len(kf))
    _, _, table = grid_lookup(kf)
    iK = table[4, 4]
    d = arpes.dispersion_on_path(E1, kf, kf, [iK])[0]
    for p in range(len(kf)):
        q = np.rint(np.mod(kf[p, :2] - kf[iK, :2], 1.0) * 12).astype(int) % 12
        assert d[p] == pytest.approx(E1[table[q[0], q[1]]], abs=1e-12)


def test_vbm_points_are_K_and_Kprime():
    kf = mesh(12)
    holes = arpes.vbm_points(band(kf)[:, None])
    got = {tuple(np.round(kf[h, :2], 6)) for h in holes}
    assert got == {(round(1 / 3, 6), round(1 / 3, 6)), (round(2 / 3, 6), round(2 / 3, 6))}


@pytest.mark.parametrize("q", [(1, 0), (2, 1), (3, 3)])
def test_replica_top_sits_on_the_dispersion(q):
    """One exciton at a finite Q: its replica reaches its top at the k whose hole is at the
    valence maximum, and that top is E1(Q) above the maximum -- the value the overlay draws
    there."""
    n = 12
    kf = mesh(n)
    _, _, table = grid_lookup(kf)
    Q = np.array([q[0] / n, q[1] / n, 0.0])
    E_S = 5.6
    ev = band(kf)[:, None]
    kq, _ = shift_map(kf, Q)
    W = np.ones((len(kf), 1))
    omega = np.linspace(0.0, 3.0, 3001)
    Im, _ = exciton_spectra(W, E_S + ev[kq], [(1.0, np.arange(len(kf)))],
                            omega, 0.01)
    vbm = ev.max()
    top_k = int(np.argmax((E_S + ev[kq])[:, 0]))
    E1 = np.full(len(kf), 9.0)
    E1[table[q[0] % n, q[1] % n]] = E_S
    d = arpes.dispersion_on_path(E1, kf, kf[[top_k]], [kq[top_k]])[0, 0]
    peak = omega[np.argmax(Im[top_k])]
    assert kq[top_k] in arpes.vbm_points(ev)
    assert peak - vbm == pytest.approx(d, abs=2e-3)
