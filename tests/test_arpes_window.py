"""xphd arpes with a windowed snapshot (xphd bte --window)."""
import numpy as np
import pytest

from xphd import arpes
from xphd.arpes import exph_fwhm, populations

from test_arpes_valence import _setup
from test_arpes_exph import _on_bin, _snapshot


def _window_set(n=6, f=4, nS=3, seed=0):
    """Refined mesh m = n f; parents as bte.windowed_rates sets them (rint of each reduced
    coordinate, half-way points to the even index)."""
    m = n * f
    i, j = np.divmod(np.arange(m * m), m)
    ci, cj = np.rint(i / f).astype(int) % n, np.rint(j / f).astype(int) % n
    a, b = np.divmod(np.arange(n * n), n)
    kf = np.stack([a / n, b / n, np.zeros(n * n)], 1)
    rng = np.random.default_rng(seed)
    keep = rng.random(m * m) < 0.6
    fq = np.repeat(np.nonzero(keep)[0], nS)
    band = np.tile(np.arange(nS), keep.sum())
    Q_red = np.stack([i[fq] / m, j[fq] / m, 0 * fq], 1)
    parent = ci[fq] * n + cj[fq]
    Qc = kf[parent]
    return kf, Q_red, Qc, parent, band


def test_refined_states_go_to_their_parent(tmp_path):
    nS = 3
    kf, Q_red, Qc, parent, band = _window_set(nS=nS)
    nst = len(band)
    rng = np.random.default_rng(1)
    F = rng.random((2, nst)) * (rng.random(nst) < 0.7)
    g = 1e-3 * (1 + rng.random((2, nst)))
    np.savez(tmp_path / "w.npz", t=np.array([0.0, 10.0]), F=F, Q_red=Q_red, Q_coarse_red=Qc,
             band=band, T=np.array(77.0), fwhm_exph_t=g.astype(np.float32),
             fwhm_exph0=g[0].astype(np.float32))
    E_nd = np.tile(0.1 * np.arange(nS), (len(kf), 1))
    N = populations(kf, np.arange(len(kf)), E_nd, snaps=str(tmp_path / "w.npz"), time=10.0)
    ref = np.zeros((len(kf), nS))
    np.add.at(ref, (parent, band), F[1])
    assert np.allclose(N, ref / ref.sum(), rtol=1e-12, atol=1e-15)
    dd = kf[None, :, :2] - np.mod(Q_red, 1.0)[:, None, :2]
    dd -= np.rint(dd)
    nearest = np.argmin(np.linalg.norm(dd, axis=-1), axis=1)
    assert (nearest != parent).any()
    gf = g[1].astype(np.float32).astype(float)
    for mode, gm in (("full", gf), ("dilute", g[0].astype(np.float32).astype(float))):
        G, neg = exph_fwhm(kf, nS, str(tmp_path / "w.npz"), 10.0, mode)
        assert neg == 0
        for (p, s) in {(int(p), int(s)) for p, s in zip(parent, band)}:
            sel = (parent == p) & (band == s)
            w = F[1, sel]
            want = (w @ gm[sel]) / w.sum() if w.sum() > 0 else gm[sel].mean()
            assert G[p, s] == pytest.approx(want, rel=1e-6), (mode, p, s)
    seen = np.zeros((len(kf), nS), bool)
    seen[parent, band] = True
    assert (G[~seen] == 0).all()


def test_windowed_snapshot_gives_the_plain_map(monkeypatch, tmp_path):
    kf, A, E, ev, base = _setup(monkeypatch, tmp_path)
    nS = A.shape[1]
    n = int(round(np.sqrt(len(kf))))
    run = base + ["--box", "G", "0.8", "--time", "10"]
    p = tmp_path / "plain.npz"
    st = ((3, 0.6), (len(kf) + 10, 0.4))
    arpes.main(run + ["--snapshots", str(_plain(p, kf, nS, 0.0, st)), "-o",
                      str(tmp_path / "p0.npz")])
    om = np.load(tmp_path / "p0.npz")["omega"]
    g = float(np.float32(_on_bin(om, 25)))
    _snapshot(p, kf, nS, np.float32(g), states=st)
    arpes.main(run + ["--snapshots", str(p), "-o", str(tmp_path / "p.npz")])
    arpes.main(run + ["--snapshots", str(p), "--eta-exph", "-o", str(tmp_path / "pe.npz")])
    rows = []
    for k, s, tot in ((3, 0, 0.6), (10, 1, 0.4)):
        rows += [(k, s, (0.0, 0.0), tot / 3, 1.5 * g),
                 (k, s, (0.5 / n, 0.0), 2 * tot / 3 * 0.5, 0.75 * g),
                 (k, s, (0.0, -0.25 / n), 2 * tot / 3 * 0.5, 0.75 * g)]
    rows.append((5, 2, (0.25 / n, 0.25 / n), 0.0, 9 * g))
    k_, s_ = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    off = np.array([[*r[2], 0.0] for r in rows])
    F1 = np.array([r[3] for r in rows])
    gw = np.array([r[4] for r in rows])
    w = tmp_path / "win.npz"
    np.savez(w, t=np.array([0.0, 10.0]), F=np.stack([0 * F1, F1]), Q_red=kf[k_] + off,
             Q_coarse_red=kf[k_], band=s_, T=np.array(77.0),
             fwhm_exph_t=np.stack([gw, gw]).astype(np.float32), fwhm_exph0=gw.astype(np.float32))
    arpes.main(run + ["--snapshots", str(w), "-o", str(tmp_path / "w.npz")])
    arpes.main(run + ["--snapshots", str(w), "--eta-exph", "-o", str(tmp_path / "we.npz")])
    for a_, b_ in (("p.npz", "w.npz"), ("pe.npz", "we.npz")):
        P, W = np.load(tmp_path / a_), np.load(tmp_path / b_)
        for key in ("I", "I_fine", "edc_box", "fh"):
            ref = P[key]
            assert np.allclose(W[key], ref, rtol=1e-5, atol=1e-9 * np.abs(ref).max()), (b_, key)


def _plain(path, kf, nS, full, st):
    _snapshot(path, kf, nS, full, states=st)
    return path
