"""Spin-orbit detection and the closure of exciton representations."""
import numpy as np
import pytest

from xphd.excsym import classify_d3h, rep_closure
from xphd.spin import sigma_h_square, spin_report, spinor_components

from test_arpes_path import mesh


def _d3h_ops(kf, Dc_of=None, Dv_of=None):
    """(label, R, Dc, Dv, kmap) for the twelve D3h operations on a mesh."""
    C3 = np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]])
    Mv = np.array([[1, 1, 0], [0, -1, 0], [0, 0, 1]])
    sh = np.diag([1, 1, -1])
    Rs = []
    for R in (np.eye(3, dtype=int), C3, C3 @ C3):
        for X in (R, R @ Mv):
            Rs += [X, X @ sh]
    nk = len(kf)
    ops = []
    for i, R in enumerate(Rs):
        d = kf[:, None, :] - (kf @ R.T)[None, :, :]
        d -= np.rint(d)
        km = np.argmin(np.linalg.norm(d, axis=-1), 1)
        Dc = np.ones((nk, 1, 1), complex) if Dc_of is None else Dc_of(i, R)
        Dv = np.ones((nk, 1, 1), complex) if Dv_of is None else Dv_of(i, R)
        ops.append((classify_d3h(R), R, Dc, Dv, km))
    return ops


def _delta_states(nk):
    A = np.zeros((nk, nk, 1, 1), complex)
    A[np.arange(nk), np.arange(nk), 0, 0] = 1.0
    return A


def test_sigma_h_square():
    nk = 5
    spinless = np.tile(np.diag([1.0, -1.0, 1.0])[None], (nk, 1, 1)).astype(complex)
    spinor = np.tile(np.diag([1j, -1j, 1j])[None], (nk, 1, 1))
    assert sigma_h_square(spinless)[:2] == (1, 1.0)
    assert sigma_h_square(spinor)[:2] == (-1, 1.0)
    bad = spinor.copy()
    bad[2] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1j]])
    s, f, _ = sigma_h_square(bad)
    assert s == -1 and f == pytest.approx(0.8)


def test_spin_report(tmp_path, capsys):
    class Lat:
        spinor_components = 2
    nk = 4
    I = np.ones((nk, 1, 1), complex)
    sh_spinor = ("sh", np.diag([1, 1, -1]), 1j * I, 1j * I, np.arange(nk))
    sh_plain = ("sh", np.diag([1, 1, -1]), I, -I, np.arange(nk))
    r = spin_report(Lat(), None, sh_spinor)
    assert r["kind"] == "spinor" and r["consistent"] is True
    assert "DOUBLE-group" in capsys.readouterr().out
    r = spin_report(Lat(), None, sh_plain)
    assert r["consistent"] is False and "WARNING" in capsys.readouterr().out
    r = spin_report(None, None, sh_plain, verbose=False)
    assert r["kind"] == "spinless" and r["consistent"] is True
    from netCDF4 import Dataset
    (tmp_path / "SAVE").mkdir()
    with Dataset(tmp_path / "SAVE" / "ns.db1", "w") as d:
        d.createDimension("n", 18)
        v = d.createVariable("DIMENSIONS", "f8", ("n",))
        dims = np.zeros(18)
        dims[11] = 2
        v[:] = dims
    assert spinor_components(None, str(tmp_path / "SAVE")) == 2
    assert spinor_components(None, str(tmp_path / "nowhere")) is None


def test_closure_of_a_permutation_representation():
    kf = mesh(6)
    nk = len(kf)
    A = _delta_states(nk)
    kQ = np.arange(nk)
    allk = list(range(nk))
    ops = _d3h_ops(kf)
    assert sorted({o[0] for o in ops}) == ["2C3", "2S3", "3C2", "3sv", "E", "sh"]
    r = rep_closure(A, ops, kQ, allk)
    assert r["missing"] == 0 and r["err"] < 1e-12 and r["flips"] == 0
    assert r["unitary"] < 1e-12
    rng = np.random.default_rng(0)
    ph = np.exp(2j * np.pi * rng.random(len(ops)))
    ops2 = _d3h_ops(kf, lambda i, R: ph[i] * np.ones((nk, 1, 1)),
                    lambda i, R: ph[i] * np.ones((nk, 1, 1)))
    r = rep_closure(A, ops2, kQ, allk)
    assert r["err"] < 1e-12 and r["flips"] == 0
    C3 = np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]])
    ops3 = _d3h_ops(kf, lambda i, R: (-1.0 if np.array_equal(R, C3) else 1.0)
                    * np.ones((nk, 1, 1)))
    r = rep_closure(A, ops3, kQ, allk)
    assert r["flips"] > 0 and r["err"] > 1.0
    r = rep_closure(A, ops, kQ, allk[:10])
    assert r["err"] > 0.5 and r["unitary"] > 0.5


def test_irreps_command(monkeypatch, tmp_path, capsys):
    from xphd import irreps
    kf = mesh(6)
    nk = len(kf)
    A = np.full((1, nk, 1, 1), 1 / np.sqrt(nk), complex)
    E = np.array([2.0])

    class Lat:
        spinor_components = 2

    (tmp_path / "bse").mkdir()
    (tmp_path / "bse" / "ndb.BS_diago_Q1").write_text("")
    monkeypatch.setattr(irreps, "lattice", lambda s: Lat())
    monkeypatch.setattr(irreps, "load_dmats", lambda p: None)
    monkeypatch.setattr(irreps, "full_zone_kpoints", lambda y: kf)
    monkeypatch.setattr(irreps, "ibz_parents", lambda y: (np.arange(nk), np.arange(nk)))
    monkeypatch.setattr(irreps, "load_excitons", lambda p, y, n: (None, A[:n], E[:n]))
    sh = np.diag([1, 1, -1])

    def spin_ph(i, R):
        return (1j if np.array_equal(R, sh) else 1.0) * np.ones((nk, 1, 1))
    monkeypatch.setattr(irreps, "spatial_ops", lambda s, k, D, nv: _d3h_ops(kf, spin_ph, spin_ph))
    base = ["--bse-dir", str(tmp_path / "bse"), "--nv", "1", "--points", "G",
            "--nstates", "1"]
    irreps.main(base)
    out = capsys.readouterr().out
    assert "nspinor = 2" in out and "DOUBLE-group" in out and "WARNING" not in out
    assert "A1'" in out and "flagged" not in out
    C3 = np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]])
    monkeypatch.setattr(irreps, "spatial_ops", lambda s, k, D, nv: _d3h_ops(
        kf, lambda i, R: (-1.0 if np.array_equal(R, C3) else spin_ph(i, R)) * np.ones((nk, 1, 1)),
        spin_ph))
    irreps.main(base)
    out = capsys.readouterr().out
    assert "closes only up to -1" in out and "flagged" in out


def _a2p_state(kf, ops):
    """An A2' state of the toy: the projection of a delta on a generic k onto A2' (+1 on E, C3,
    sigma_h, S3; -1 on C2' and sigma_v)."""
    sign = {"E": 1, "2C3": 1, "sh": 1, "2S3": 1, "3C2": -1, "3sv": -1}
    for k0 in range(len(kf)):
        orbit = {int(o[4][k0]) for o in ops}
        if len(orbit) == 6:
            break
    f = np.zeros(len(kf), complex)
    for lab, R, Dc, Dv, km in ops:
        f[km[k0]] += sign[lab]
    return f / np.linalg.norm(f)


def test_irreps_sum_of_irreps_and_npz(monkeypatch, tmp_path, capsys):
    """Two states of different irreps closer than --deg-tol: labelled as a sum, A1'+A2', and
    split by a smaller --deg-tol; the table in the -o file."""
    from xphd import irreps
    kf = mesh(6)
    nk = len(kf)
    ops = _d3h_ops(kf)
    A = np.zeros((2, nk, 1, 1), complex)
    A[0, :, 0, 0] = 1 / np.sqrt(nk)
    A[1, :, 0, 0] = _a2p_state(kf, ops)

    class Lat:
        spinor_components = 1

    (tmp_path / "bse").mkdir()
    (tmp_path / "bse" / "ndb.BS_diago_Q1").write_text("")
    monkeypatch.setattr(irreps, "lattice", lambda s: Lat())
    monkeypatch.setattr(irreps, "load_dmats", lambda p: None)
    monkeypatch.setattr(irreps, "full_zone_kpoints", lambda y: kf)
    monkeypatch.setattr(irreps, "ibz_parents", lambda y: (np.arange(nk), np.arange(nk)))
    monkeypatch.setattr(irreps, "spatial_ops", lambda s, k, D, nv: ops)
    base = ["--bse-dir", str(tmp_path / "bse"), "--nv", "1", "--points", "G",
            "--nstates", "2"]

    E = np.array([2.0, 2.0005])
    monkeypatch.setattr(irreps, "load_excitons", lambda p, y, n: (None, A[:n], E[:n]))
    out_npz = tmp_path / "irreps.npz"
    irreps.main(base + ["-o", str(out_npz)])
    out = capsys.readouterr().out
    assert "single groups throughout" in out
    assert "A1'+A2'" in out and "a sum of irreps" in out and "flagged" not in out
    d = np.load(out_npz)
    assert list(d["points"]) == ["G"] and str(d["G_group"]) == "D3h"
    assert list(d["G_label"]) == ["A1'+A2'"] and list(d["G_dim"]) == [2]
    assert d["G_closure"][0] < 1e-8 and int(d["nspinor"]) == 1

    irreps.main(base + ["--deg-tol", "1e-4", "-o", str(out_npz)])
    capsys.readouterr()
    d = np.load(out_npz)
    assert list(d["G_label"]) == ["A1'", "A2'"]
    assert np.allclose(d["G_sh"], [1, 1]) and list(d["G_first"]) == [1, 2]
    assert d["G_chi"].shape == (2, 6)


def test_decompose_characters():
    from xphd.excsym import CHARACTER_TABLES as T, CLASSES as C, decompose_characters

    def chi(tab, *names):
        return dict(zip(C[tab], sum(np.array(T[tab][n], complex) for n in names)))
    assert decompose_characters(chi("D3h", "A1''", "A2''"), "D3h")[:2] == ("A1''+A2''", 0.0)
    assert decompose_characters(chi("D3h", "E'", "E'", "A1'"), "D3h")[0] == "A1'+2E'"
    assert decompose_characters(chi("C3h", "E'_1", "A''"), "C3h")[0] == "E'_1+A''"
    assert decompose_characters(chi("C3h", "E''_2"), "C3h")[0] == "E''_2"
    assert decompose_characters(chi("C2v", "A2", "B2"), "C2v")[0] == "A2+B2"
    bad = dict(zip(C["D3h"], (1, 0.5, 0, 1, 0, 0)))
    assert decompose_characters(bad, "D3h")[1] > 0.2
