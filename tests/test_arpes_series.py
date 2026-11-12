"""xphd arpes --times / --series / --box: the dynamics as Gosetti et al. show it."""
import numpy as np
import pytest

from xphd import arpes
from xphd.arpes import (box_points, exciton_spectra, interp_matrix, occupied_dos,
                        parse_times, star_assignment)
from xphd.excsym import shift_map

from test_arpes_valence import _setup
from test_arpes_path import HEX


def test_parse_times():
    av = np.arange(0.0, 151.0, 1.0)
    assert parse_times(["all"], av) == list(av)
    assert parse_times(["1:5"], av) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert parse_times(["0:150:50"], av) == [0.0, 50.0, 100.0, 150.0]
    assert parse_times(["10.4", "10", "200"], av) == [10.0, 150.0]
    with pytest.raises(ValueError):
        parse_times(["1:2:0"], av)


def test_box_points_and_dos():
    pts = box_points(HEX, "K", 0.3, n_across=21)
    B = 2 * np.pi * HEX[:2, :2] / 0.529177210903
    K = np.array([1 / 3, 1 / 3]) @ B
    r = np.linalg.norm(pts[:, :2] @ B - K, axis=1)
    assert r.max() <= 0.3 + 1e-9 and len(pts) > 250 and np.all(pts[:, 2] == 0)
    E = np.array([[1.0, 1.2], [1.1, 1.5]])
    N = np.array([[0.2, 0.3], [0.4, 0.1]])
    w = np.linspace(0.0, 2.5, 5001)
    d = occupied_dos(E, N, w, 0.01)
    assert np.trapezoid(d, w) == pytest.approx(1.0, rel=1e-6)


def _snap(path, kf, nS):
    F = np.zeros((3, len(kf) * nS))
    F[0, 3] = 0.6
    F[0, len(kf) + 10] = 0.4
    F[1, 5] = 0.5
    F[1, 2 * len(kf) + 7] = 0.5
    F[2, 0] = 1.0
    np.savez(path, t=np.array([0.0, 10.0, 20.0]), F=F, Q_red=np.concatenate([kf] * nS),
             band=np.repeat(np.arange(1, nS + 1), len(kf)), T=np.array(77.0))


def test_series_matches_single_runs(monkeypatch, tmp_path, capsys):
    kf, A, E, ev, base = _setup(monkeypatch, tmp_path)
    nS = A.shape[1]
    snap = tmp_path / "s.npz"
    _snap(snap, kf, nS)
    box = ["--box", "G", "0.8"]
    arpes.main(base + box + ["--snapshots", str(snap), "--times", "0", "10", "20",
                             "-o", str(tmp_path / "multi" / "arpes_{t}fs.npz"),
                             "--series", str(tmp_path / "series.npz")])
    for t in (0, 10, 20):
        arpes.main(base + box + ["--snapshots", str(snap), "--time", str(t),
                                 "-o", str(tmp_path / f"single_{t}.npz")])
    s = np.load(tmp_path / "series.npz")
    assert list(s["times"]) == [0.0, 10.0, 20.0]
    pts = box_points(HEX, "G", 0.8)
    Bm = interp_matrix(kf, pts)
    ops = arpes.star_kmaps(None, kf)
    kidx, first = np.arange(len(kf)), np.arange(len(kf))
    smap = star_assignment(kf, kidx, first, ops)
    for it, t in enumerate((0, 10, 20)):
        m = np.load(tmp_path / "multi" / f"arpes_{t}fs.npz")
        o = np.load(tmp_path / f"single_{t}.npz")
        for key in ("I", "I_fine", "edc_zone", "edc_box", "dos_occ", "N_path", "E_path",
                    "I_vb_fine", "fh"):
            assert np.allclose(m[key], o[key], rtol=1e-12, atol=1e-15), key
        assert float(m["time"]) == t
        assert np.allclose(s["edc_zone_t"][it], m["edc_zone"])
        assert np.allclose(s["edc_box_t"][it], m["edc_box"])
        assert np.allclose(s["dos_occ_t"][it], m["dos_occ"])
        assert np.allclose(m["edc_zone"], (m["I"] * float(m["I_norm"])).mean(axis=0))
        assert np.trapezoid(m["dos_occ"], m["omega_x"]) == pytest.approx(1.0, rel=1e-3)
        N = arpes.populations(kf, kidx, np.array(E), snaps=str(snap), time=t)
        ref = np.zeros((len(pts), len(m["omega"])))
        for j in range(len(kf)):
            kq, _ = shift_map(kf, kf[j])
            W = (np.abs(A[j]) ** 2).sum(axis=3)
            for S in range(nS):
                if N[j, S] <= 0:
                    continue
                mem = [(float(N[i, S]), smap[i]) for i in range(len(kf))
                       if kidx[i] == j and N[i, S] > 0]
                _, Ip = exciton_spectra(W[S], E[j][S] + ev[kq], mem, m["omega"], 0.05, Bm)
                ref += Ip
        assert np.allclose(m["edc_box"][0], ref.mean(axis=0), rtol=1e-10, atol=1e-14)
        fine = interp_matrix(kf, m["k_fine"])
        assert np.allclose(m["E_path"], fine @ np.array(E)[kidx])
        assert np.allclose(m["N_path"], np.clip(fine @ N, 0, None))
    out = capsys.readouterr().out
    assert "--times: 3 delays" in out and "wrote" in out and "edc_zone_t" in out


def test_refusals_and_series_only(monkeypatch, tmp_path):
    kf, A, E, ev, base = _setup(monkeypatch, tmp_path)
    snap = tmp_path / "s.npz"
    _snap(snap, kf, A.shape[1])
    with pytest.raises(SystemExit, match=r"\{t\}"):
        arpes.main(base + ["--snapshots", str(snap), "--times", "0", "10",
                           "-o", str(tmp_path / "a.npz")])
    with pytest.raises(SystemExit, match="--snapshots"):
        arpes.main(base + ["--temperature", "300", "--density", "1e13",
                           "--cell-area", "0.0546", "--times", "0", "10"])
    arpes.main(base + ["--snapshots", str(snap), "--times", "all", "--no-maps",
                       "-o", str(tmp_path / "x" / "arpes_{t}fs.npz")])
    s = np.load(tmp_path / "x" / "arpes_series.npz")
    assert s["edc_zone_t"].shape[0] == 3 and len(s["maps"]) == 0
    assert not (tmp_path / "x" / "arpes_0fs.npz").exists()
    E = np.array(E)
    assert s["E_mean_t"][2] == pytest.approx(E[0, 0])
