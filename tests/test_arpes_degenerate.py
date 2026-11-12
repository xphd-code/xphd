"""xphd arpes with a snapshot that populates a degenerate pair unequally."""
import numpy as np
import pytest

from xphd import arpes

from test_arpes_path import band, mesh
from test_arpes_valence import _mock
from test_arpes_exph import _on_bin


def _unitary(rng, d=2):
    q, r = np.linalg.qr(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))
    return q * (np.diag(r) / np.abs(np.diag(r)))


def _excitons(gauge_seed=None):
    n, nS, nv, nc = 6, 3, 2, 2
    kf = mesh(n)
    rng = np.random.default_rng(3)
    A = rng.normal(size=(len(kf), nS, len(kf), nv, nc)) + 1j * rng.normal(
        size=(len(kf), nS, len(kf), nv, nc))
    A /= np.sqrt((np.abs(A) ** 2).sum(axis=(2, 3, 4), keepdims=True))
    E = 5.0 + 0.3 * rng.random((len(kf), nS))
    E[0, 1:] = 5.1
    if gauge_seed is not None:
        U = _unitary(np.random.default_rng(gauge_seed))
        A[0, 1:] = np.einsum("st,tkvc->skvc", U, A[0, 1:])
    return kf, A, E


def _run(monkeypatch, tmp_path, tag, gauge_seed, extra):
    kf, A, E = _excitons(gauge_seed)
    _mock(monkeypatch, kf, A, E)
    ev = np.column_stack([band(kf), band(kf) - 1.0])
    np.save(tmp_path / "ev.npy", ev)
    nS = A.shape[1]
    nk = len(kf)
    F = np.zeros((2, nk * nS))
    F[1, nk * 1 + 0] = 0.7
    F[1, nk * 2 + 0] = 0.1
    F[1, nk * 0 + 5] = 0.2
    fw = np.full(nk * nS, 2e-3)
    fw[nk * 1], fw[nk * 2] = 4e-3, 1e-3
    snap = tmp_path / f"s_{tag}.npz"
    np.savez(snap, t=np.array([0.0, 10.0]), F=F, Q_red=np.concatenate([kf] * nS),
             band=np.repeat(np.arange(1, nS + 1), nk), T=np.array(77.0),
             fwhm_exph_t=np.stack([fw, fw]).astype(np.float32),
             fwhm_exph0=fw.astype(np.float32))
    out = tmp_path / f"m_{tag}.npz"
    arpes.main(["--nstates", str(nS), "--valence-npy", str(tmp_path / "ev.npy"),
                "--fine", "60", "--ne", "200", "--min-occ", "0", "--valence",
                "--vb-ne", "300", "--box", "G", "0.8", "--time", "10",
                "--snapshots", str(snap), "-o", str(out)] + extra)
    return np.load(out)


@pytest.mark.parametrize("eta", [[], ["--eta-exph"]])
def test_two_gauges_give_one_map(monkeypatch, tmp_path, eta):
    a = _run(monkeypatch, tmp_path, "a", None, eta)
    b = _run(monkeypatch, tmp_path, "b", 5, eta)
    for key in ("I", "I_fine", "fh", "fe", "edc_zone"):
        ref = a[key]
        assert np.allclose(b[key], ref, rtol=1e-9, atol=1e-12 * np.abs(ref).max()), key
    ref = a["edc_box"]
    assert np.allclose(b["edc_box"], ref, rtol=1e-4, atol=1e-4 * np.abs(ref).max())
    a0 = _run(monkeypatch, tmp_path, "a0", None, eta + ["--deg-tol", "0"])
    b0 = _run(monkeypatch, tmp_path, "b0", 5, eta + ["--deg-tol", "0"])
    assert not np.allclose(a0["I"], b0["I"], rtol=1e-3)


def test_shared_occupations(monkeypatch, tmp_path):
    kf, A, E = _excitons()
    _mock(monkeypatch, kf, A, E)
    nk, nS = len(kf), A.shape[1]
    F = np.zeros((1, nk * nS))
    F[0, nk * 1] = 0.7
    F[0, nk * 2] = 0.1
    F[0, 5] = 0.2
    snap = tmp_path / "s.npz"
    np.savez(snap, t=np.array([0.0]), F=F, Q_red=np.concatenate([kf] * nS),
             band=np.repeat(np.arange(1, nS + 1), nk), T=np.array(77.0))
    kidx = np.arange(nk)
    N = arpes.populations(kf, kidx, E, snaps=str(snap), time=0.0)
    assert N[0, 1] == pytest.approx(0.4) and N[0, 2] == pytest.approx(0.4)
    assert N[5, 0] == pytest.approx(0.2) and N.sum() == pytest.approx(1.0)
    N0 = arpes.populations(kf, kidx, E, snaps=str(snap), time=0.0, deg_tol=0.0)
    assert N0[0, 1] == pytest.approx(0.7) and N0[0, 2] == pytest.approx(0.1)
