"""xphd arpes --valence: the valence band and the holes the excitons leave."""
import numpy as np
import pytest

from xphd import arpes
from xphd.arpes import carrier_occupations, valence_photoemission
from xphd.excsym import shift_map

from test_arpes_path import HEX, _hbn_ops, band, mesh


def test_hole_of_a_rotated_exciton():
    n = 6
    kf = mesh(n)
    ops = _hbn_ops(kf)
    rng = np.random.default_rng(1)
    Q = kf[7]
    kq, _ = shift_map(kf, Q)
    Wv = rng.random((len(kf), 2))
    Wc = rng.random((len(kf), 3))
    for M, km in ops[:6]:
        fh, fe = carrier_occupations(Wv, Wc, kq, [(0.7, km)])
        Qm = np.mod(Q @ M.T, 1.0)
        kqm, _ = shift_map(kf, Qm)
        ref_h = np.zeros_like(fh)
        np.add.at(ref_h, kqm, 0.7 * Wv[km])
        assert np.allclose(fh, ref_h, atol=1e-14)
        assert np.allclose(fe, 0.7 * Wc[km], atol=1e-14)
        assert fh.sum() == pytest.approx(0.7 * Wv.sum())
        assert fe.sum() == pytest.approx(0.7 * Wc.sum())


def test_identity_member_puts_the_hole_at_k_plus_Q():
    kf = mesh(6)
    Q = kf[9]
    kq, _ = shift_map(kf, Q)
    Wv = np.zeros((len(kf), 1))
    Wv[4, 0] = 1.0
    fh, fe = carrier_occupations(Wv, Wv, kq, [(2.0, np.arange(len(kf)))])
    assert fh[kq[4], 0] == pytest.approx(2.0) and fh.sum() == pytest.approx(2.0)
    assert fe[4, 0] == pytest.approx(2.0)


def test_valence_photoemission_depletion():
    eps = np.array([[-0.5, -1.5], [-0.2, -1.0]])
    fh = np.array([[0.1, 0.0], [0.0, 0.3]])
    om = np.linspace(-2.5, 0.5, 3001)
    I, I0, dI = valence_photoemission(eps, fh, om, 0.02, g_s=2.0)
    assert np.allclose(I, I0 + dI)
    i = np.argmin(np.abs(om - (-0.5)))
    assert I0[0, i] == pytest.approx(2.0, abs=1e-3)
    assert I[0, i] == pytest.approx(1.9, abs=1e-3)
    j = np.argmin(np.abs(om - (-1.0)))
    assert dI[1, j] == pytest.approx(-0.3, abs=1e-3)
    assert np.all(dI <= 0)


def _mock(monkeypatch, kf, A, E):
    import xphd.yambo as yb

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


def _setup(monkeypatch, tmp_path):
    n, nS, nv, nc = 6, 3, 2, 2
    kf = mesh(n)
    rng = np.random.default_rng(3)
    A = rng.normal(size=(len(kf), nS, len(kf), nv, nc)) + 0j
    A /= np.sqrt((np.abs(A) ** 2).sum(axis=(2, 3, 4), keepdims=True))
    E = 5.0 + 0.3 * rng.random((len(kf), nS))
    _mock(monkeypatch, kf, A, E)
    ev = np.column_stack([band(kf), band(kf) - 1.0])
    np.save(tmp_path / "ev.npy", ev)
    base = ["--nstates", str(nS), "--valence-npy", str(tmp_path / "ev.npy"),
            "--fine", "60", "--ne", "200", "--min-occ", "0", "--valence",
            "--vb-ne", "300"]
    return kf, A, E, ev, base


def test_command_thermal_with_density(monkeypatch, tmp_path, capsys):
    kf, A, E, ev, base = _setup(monkeypatch, tmp_path)
    out = tmp_path / "v.npz"
    arpes.main(base + ["--temperature", "300", "--density", "1e13", "--cell-area", "0.0546",
                       "-o", str(out)])
    d = np.load(out)
    Ntot = 1e13 * 1e-14 * 0.0546 * len(kf)
    assert float(d["N_tot"]) == pytest.approx(Ntot)
    assert d["fh"].sum() == pytest.approx(Ntot, rel=1e-10)
    assert d["fe"].sum() == pytest.approx(Ntot, rel=1e-10)
    N = np.exp(-(E - E.min()) / (8.617333262e-5 * 300))
    N = N / N.sum() * Ntot
    ref = np.zeros((len(kf), 2))
    for j in range(len(kf)):
        kq, _ = shift_map(kf, kf[j])
        for S in range(A.shape[1]):
            np.add.at(ref, kq, N[j, S] * (np.abs(A[j, S]) ** 2).sum(axis=2))
    assert np.allclose(d["fh"], ref, rtol=1e-10, atol=1e-14)
    assert np.allclose(d["I_vb_fine"], d["I_vb0_fine"] + d["dI_vb_fine"])
    assert d["I_vb0_fine"].max() == pytest.approx(2.0, abs=0.02)
    assert np.all(d["dI_vb_fine"] <= 0)
    om = d["omega_vb"]
    assert om[0] == pytest.approx(ev.max() - 3.0) and om[-1] == pytest.approx(ev.max() + 0.5)
    Iabs = d["I"] * float(d["I_abs_scale"])
    ref_I = np.zeros_like(Iabs)
    for j in range(len(kf)):
        kq, _ = shift_map(kf, kf[j])
        ref_I += arpes.exciton_arpes(A[j], E[j], ev, kq, N[j], d["omega"], 0.05)
    assert np.allclose(Iabs, ref_I, rtol=1e-9, atol=1e-12)
    assert "emptied by" in capsys.readouterr().out


def test_command_snapshot_and_refusal(monkeypatch, tmp_path):
    kf, A, E, ev, base = _setup(monkeypatch, tmp_path)
    nS = A.shape[1]
    F = np.zeros((2, len(kf) * (nS + 1)))
    Qr = np.concatenate([kf] * (nS + 1))
    band_ = np.repeat(np.arange(1, nS + 2), len(kf))
    F[1, 3] = 0.4
    F[1, len(kf) + 10] = 0.2
    F[1, nS * len(kf) + 5] = 0.05
    snap = tmp_path / "s.npz"
    np.savez(snap, t=np.array([0.0, 10.0]), F=F, Q_red=Qr, band=band_, T=np.array(77.0))
    out = tmp_path / "w.npz"
    arpes.main(base + ["--snapshots", str(snap), "--time", "10", "-o", str(out)])
    d = np.load(out)
    assert float(d["N_tot"]) == pytest.approx(0.6)
    assert d["fh"].sum() == pytest.approx(0.6)
    kq, _ = shift_map(kf, kf[3])
    ref = np.zeros((len(kf), 2))
    np.add.at(ref, kq, 0.4 * (np.abs(A[3, 0]) ** 2).sum(axis=2))
    kq, _ = shift_map(kf, kf[10])
    np.add.at(ref, kq, 0.2 * (np.abs(A[10, 1]) ** 2).sum(axis=2))
    assert np.allclose(d["fh"], ref, atol=1e-14)
    with pytest.raises(SystemExit):
        arpes.main(base + ["--temperature", "300", "-o", str(tmp_path / "x.npz")])
