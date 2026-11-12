"""Exciton-phonon line widths in the ARPES map (--eta-exph), as Chen, Sangalli and Bernardi set
them (PRR 4, 043203 (2022): eta = the linewidth of Eq. 1)."""
import numpy as np
import pytest

from xphd import arpes, bte
from xphd.arpes import LorentzBins, lorentz_convolve

from test_arpes_valence import _setup

KB = 8.617333262e-5


def _rates(tmp_path, T=300.0):
    import xphd
    from test_pipeline import _bte_set
    arcs = xphd.load_archives(_bte_set(tmp_path), verbose=False)
    return bte.build_rates(arcs, T=T, verbose=False)


def test_linewidth_is_chen_bernardi_eq1(tmp_path):
    T = 300.0
    rm = _rates(tmp_path, T)
    n = len(rm.E)
    rng = np.random.default_rng(0)
    F = 0.3 * rng.random(n)
    full, dil = bte.exph_linewidths(rm, F)
    ref = np.array([np.sum(rm.P[i] * (1 + F) - rm.P[:, i] * F) for i in range(n)])
    assert np.allclose(full, bte.HBAR_EV_FS * ref, rtol=1e-12, atol=1e-18)
    B = np.abs(rm.R)
    dE = rm.E[:, None] - rm.E[None, :]
    Np = np.zeros_like(dE)
    m = np.abs(dE) > 1e-9
    Np[m] = 1.0 / np.expm1(np.clip(np.abs(dE[m]) / (KB * T), 1e-12, 300.0))
    pair = np.where(dE > 0, Np + 1 + F[None, :], Np - F[None, :]) * B
    assert np.allclose(full, bte.HBAR_EV_FS * pair.sum(axis=1), rtol=1e-10, atol=1e-18)
    assert np.allclose(dil, bte.HBAR_EV_FS * rm.out_rate)
    assert np.allclose(bte.exph_linewidths(rm, 0 * F)[0], dil)
    F2 = np.column_stack([F, 2 * F, 0 * F])
    full2, _ = bte.exph_linewidths(rm, F2)
    assert np.allclose(full2[:, 0], full) and np.allclose(full2[:, 2], dil)


def test_bte_command_saves_the_widths(tmp_path, capsys):
    from test_pipeline import _bte_set
    from xphd.cli import main
    pat = _bte_set(tmp_path)
    snap = tmp_path / "s.npz"
    main(["bte", str(pat), "--T", "200", "--density", "1e-2", "--t-end", "500", "--nt", "3",
          "--snapshots", str(snap)])
    out = capsys.readouterr().out
    d = np.load(snap)
    rm = _rates(tmp_path, 200.0)
    full, dil = bte.exph_linewidths(rm, d["F"].T)
    assert d["fwhm_exph_t"].shape == d["F"].shape
    assert np.allclose(d["fwhm_exph_t"], full.T, rtol=1e-6, atol=1e-9)
    assert np.allclose(d["fwhm_exph0"], dil, rtol=1e-6, atol=1e-9)
    assert "fwhm_exph_t" in out


def test_lorentz_bins():
    om = np.linspace(-1.0, 1.0, 2001)
    d = om[1] - om[0]
    x = np.zeros((2, len(om)))
    x[0, 1000] = 1.0
    x[1, 900] = 2.0
    acc = LorentzBins(om, [x.shape])
    g = 0.5 * d * 1.1 ** 30
    acc.add(g, [x])
    I = acc.finish()[0]
    assert np.allclose(I, lorentz_convolve(x, om, g), rtol=1e-9, atol=1e-14)
    g = 0.5 * d * 1.1 ** 30.5
    acc = LorentzBins(om, [x.shape])
    acc.add(g, [x])
    I = acc.finish()[0]
    ref = lorentz_convolve(x, om, g)
    assert np.abs(I - ref).max() < 5e-3 * ref.max()
    assert I.sum() == pytest.approx(ref.sum(), rel=1e-3)
    acc = LorentzBins(om, [x.shape])
    acc.add(0.4 * d, [x])
    acc.add(0.02, [x])
    with pytest.raises(ValueError, match="ascending"):
        acc.add(0.01, [x])
    I = acc.finish()[0]
    assert np.allclose(I, x + lorentz_convolve(x, om, 0.02), rtol=2e-3, atol=1e-6)


def _snapshot(path, kf, nS, full, dil=None, states=((3, 0.6), None)):
    """States (flat index, occupation); widths given per flat index or one value."""
    nst = len(kf) * nS
    F = np.zeros((2, nst))
    for st in states:
        if st is not None:
            F[1, st[0]] = st[1]
    fw = np.broadcast_to(np.asarray(full, float), (nst,)).copy() if np.ndim(full) == 0 \
        else np.asarray(full, float)
    d0 = fw if dil is None else (np.full(nst, dil) if np.ndim(dil) == 0 else dil)
    np.savez(path, t=np.array([0.0, 10.0]), F=F, Q_red=np.concatenate([kf] * nS),
             band=np.repeat(np.arange(1, nS + 1), len(kf)), T=np.array(77.0),
             fwhm_exph_t=np.stack([fw, fw]).astype(np.float32),
             fwhm_exph0=np.asarray(d0, np.float32))


def _on_bin(omega, b):
    return 0.5 * (omega[1] - omega[0]) * 1.1 ** b


def test_uniform_width_is_a_convolution(monkeypatch, tmp_path, capsys):
    kf, A, E, ev, base = _setup(monkeypatch, tmp_path)
    nS = A.shape[1]
    st = ((3, 0.6), (len(kf) + 10, 0.4))
    _snapshot(tmp_path / "plain.npz", kf, nS, 0.0, states=st)
    run = base + ["--box", "G", "0.8", "--time", "10"]
    arpes.main(run + ["--snapshots", str(tmp_path / "plain.npz"), "-o", str(tmp_path / "p.npz")])
    p = np.load(tmp_path / "p.npz")
    om = p["omega"]
    g = _on_bin(om, 25)
    _snapshot(tmp_path / "w.npz", kf, nS, np.float32(g), dil=np.float32(_on_bin(om, 12)),
              states=st)
    g = float(np.float32(g))
    arpes.main(run + ["--snapshots", str(tmp_path / "w.npz"), "--eta-exph",
                      "-o", str(tmp_path / "w_full.npz")])
    out = capsys.readouterr().out
    assert "exciton-phonon line widths (--eta-exph full)" in out
    w = np.load(tmp_path / "w_full.npz")
    nP, nW = float(p["I_norm"]), float(w["I_norm"])
    for key in ("I", "I_fine"):
        ref = lorentz_convolve(p[key] * nP, om, g)
        assert np.allclose(w[key] * nW, ref, rtol=1e-5, atol=1e-9 * ref.max()), key
    assert np.allclose(w["edc_box"], lorentz_convolve(p["edc_box"], om, g), rtol=1e-5,
                       atol=1e-9 * p["edc_box"].max())
    assert np.allclose(w["I_vb_fine"], p["I_vb_fine"]) and np.allclose(w["fh"], p["fh"])
    assert float(w["fwhm_exph_mean"]) == pytest.approx(g, rel=1e-6)
    assert str(w["eta_exph"]) == "full" and w["fwhm_exph"].shape == (len(kf), nS)
    arpes.main(run + ["--snapshots", str(tmp_path / "w.npz"), "--eta-exph", "dilute",
                      "-o", str(tmp_path / "w_dil.npz")])
    dl = np.load(tmp_path / "w_dil.npz")
    g0 = float(np.float32(_on_bin(om, 12)))
    ref = lorentz_convolve(p["I"] * nP, om, g0)
    assert np.allclose(dl["I"] * float(dl["I_norm"]), ref, rtol=1e-5, atol=1e-9 * ref.max())
    _snapshot(tmp_path / "half.npz", kf, nS, np.float32(g / 2), states=st)
    arpes.main(run + ["--snapshots", str(tmp_path / "half.npz"), "--eta-exph", "--eta-scale",
                      "2", "-o", str(tmp_path / "w2.npz")])
    w2 = np.load(tmp_path / "w2.npz")
    assert np.allclose(w2["I"] * float(w2["I_norm"]), w["I"] * nW, rtol=1e-4,
                       atol=1e-8 * nW)


def test_two_widths_are_two_convolutions(monkeypatch, tmp_path):
    kf, A, E, ev, base = _setup(monkeypatch, tmp_path)
    nS = A.shape[1]
    a_, b_ = 3, len(kf) + 10
    run = base + ["--time", "10", "--fine", "0"]
    _snapshot(tmp_path / "a.npz", kf, nS, 0.0, states=((a_, 1.0), None))
    _snapshot(tmp_path / "b.npz", kf, nS, 0.0, states=((b_, 1.0), None))
    arpes.main(run + ["--snapshots", str(tmp_path / "a.npz"), "-o", str(tmp_path / "pa.npz")])
    arpes.main(run + ["--snapshots", str(tmp_path / "b.npz"), "-o", str(tmp_path / "pb.npz")])
    pa, pb = np.load(tmp_path / "pa.npz"), np.load(tmp_path / "pb.npz")
    om = pa["omega"]
    ga, gb = float(np.float32(_on_bin(om, 40))), float(np.float32(_on_bin(om, 8)))
    fw = np.zeros(len(kf) * nS)
    fw[a_], fw[b_] = ga, gb
    _snapshot(tmp_path / "ab.npz", kf, nS, fw, states=((a_, 0.7), (b_, 0.3)))
    arpes.main(run + ["--snapshots", str(tmp_path / "ab.npz"), "--eta-exph",
                      "-o", str(tmp_path / "ab_out.npz")])
    m = np.load(tmp_path / "ab_out.npz")
    ref = (0.7 * lorentz_convolve(pa["I"] * float(pa["I_norm"]), om, ga)
           + 0.3 * lorentz_convolve(pb["I"] * float(pb["I_norm"]), om, gb))
    got = m["I"] * float(m["I_norm"])
    assert np.allclose(got, ref, rtol=1e-5, atol=1e-9 * ref.max())
    assert not np.allclose(got, 0.7 * pa["I"] * float(pa["I_norm"])
                           + 0.3 * pb["I"] * float(pb["I_norm"]), rtol=1e-2)
    assert float(m["fwhm_exph_mean"]) == pytest.approx(0.7 * ga + 0.3 * gb, rel=1e-6)


def test_series_and_refusals(monkeypatch, tmp_path, capsys):
    kf, A, E, ev, base = _setup(monkeypatch, tmp_path)
    nS = A.shape[1]
    fw = np.full(len(kf) * nS, 0.02)
    fw[3] = -0.01
    _snapshot(tmp_path / "s.npz", kf, nS, fw, states=((3, 0.5), (len(kf) + 10, 0.5)))
    arpes.main(base + ["--snapshots", str(tmp_path / "s.npz"), "--times", "0", "10",
                       "--eta-exph", "--no-maps", "--series", str(tmp_path / "ser.npz"),
                       "-o", str(tmp_path / "x_{t}.npz")])
    out = capsys.readouterr().out
    assert "negative widths (stimulated gain) clipped to 0" in out and "<FWHM>" in out
    s = np.load(tmp_path / "ser.npz")
    assert s["fwhm_exph_mean_t"][1] == pytest.approx(0.5 * 0.0 + 0.5 * 0.02, rel=1e-5)
    assert str(s["eta_exph"]) == "full"
    with pytest.raises(SystemExit, match="--snapshots"):
        arpes.main(base + ["--temperature", "300", "--density", "1e13", "--cell-area",
                           "0.0546", "--eta-exph", "-o", str(tmp_path / "t.npz")])
    old = tmp_path / "old.npz"
    F = np.zeros((1, len(kf) * nS))
    F[0, 3] = 1.0
    np.savez(old, t=np.array([0.0]), F=F, Q_red=np.concatenate([kf] * nS),
             band=np.repeat(np.arange(1, nS + 1), len(kf)), T=np.array(77.0))
    with pytest.raises(SystemExit, match="rerun xphd bte"):
        arpes.main(base + ["--snapshots", str(old), "--eta-exph", "-o", str(tmp_path / "o.npz")])
    with pytest.raises(SystemExit, match="--eta-scale"):
        arpes.main(base + ["--snapshots", str(tmp_path / "s.npz"), "--eta-exph",
                           "--eta-scale", "0", "-o", str(tmp_path / "z.npz")])
