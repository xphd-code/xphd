"""The windowed rate matrix on the refined mesh (bte.windowed_rates)."""
import numpy as np
import pytest

import xphd
from xphd import bte

KB = 8.617333262e-5


def _qnorm(kx, ky):
    from xphd.core.mesh import hex_norm
    return hex_norm(np.stack([kx, ky], -1))


def _archives(tmp_path, E, hw, seed=0, tag="S", gscale=4e-6):
    """Archives for bands E (n, n, ne) and phonons hw (n, n, nm) on the mesh."""
    n, _, ne = E.shape
    nm = hw.shape[-1]
    i = np.arange(n)
    kx, ky = np.meshgrid(i / n, i / n, indexing="ij")
    qr = np.stack([kx.ravel(), ky.ravel(), np.zeros(n * n)], 1)
    rng = np.random.default_rng(seed)
    d = tmp_path / tag
    d.mkdir()
    for a in range(n):
        for b in range(n):
            sx, sy = (i + a) % n, (i + b) % n
            Em = E[np.ix_(sx, sy)]
            g2 = np.abs(rng.standard_normal((n, n, nm, ne, ne))) * gscale
            ac = hw[0, 0] < 1e-4
            g2[:, :, ac] *= (_qnorm(kx, ky) / _qnorm(kx, ky).max())[:, :, None, None, None]
            np.savez(d / f"Q{a:02d}{b:02d}.npz",
                     g2_grid=g2.reshape(n * n, nm, ne, ne),
                     E_m_grid=Em.reshape(n * n, ne), hw_grid=hw.reshape(n * n, nm),
                     E_n_grid=Em[0, 0], q_red=qr, Q_red=np.array([a / n, b / n, 0.0]),
                     mesh=np.array([n, n, 1]))
    return xphd.load_archives(str(d / "Q*.npz"), verbose=False)


def _model(n=6, acoustic=True):
    i = np.arange(n)
    kx, ky = np.meshgrid(i / n, i / n, indexing="ij")
    c = (np.cos(2 * np.pi * kx) + np.cos(2 * np.pi * ky)
         + np.cos(2 * np.pi * (kx + ky))) / 3
    pert = 0.004 * np.sin(2 * np.pi * kx + 0.3) * np.cos(2 * np.pi * ky + 1.1)
    E = np.stack([2.400 + 0.20 * (1 - c) + pert,
                  2.419 + 0.20 * (1 - c) - pert,
                  2.428 + 0.22 * (1 - c) + 0.5 * pert], -1)
    ph = [np.full_like(c, 0.034), np.full_like(c, 0.045)]
    if acoustic:
        ph.insert(0, 0.020 * np.sqrt(np.clip((1 - c) / 1.5, 0, None)))
    return E, np.stack(ph, -1)


def _coarse_index(rm, ne):
    """Flat coarse-state index (a*n2+b)*ne + band of every kept state."""
    n1, n2 = rm.meta["coarse_mesh"]
    qc = np.rint(rm.meta["Q_coarse_red"][:, :2] * [n1, n2]).astype(int)
    return ((qc[:, 0] % n1) * n2 + qc[:, 1] % n2) * ne + rm.meta["band"]


@pytest.mark.parametrize("bose", ["node", "crossing"])
def test_f1_is_build_rates(tmp_path, bose):
    arcs = _archives(tmp_path, *_model())
    a = bte.build_rates(arcs, T=150.0, verbose=False, bose=bose, channels=True)
    b = bte.windowed_rates(arcs, 150.0, 10.0, refine=1, flat_width=0.0,
                           bose=bose, channels=True, verbose=False)
    assert b.P.shape == a.P.shape
    assert np.abs(a.E - b.E).max() == 0.0
    assert np.abs(a.P - b.P).max() < 1e-12 * a.P.max()
    assert np.abs(a.R - b.R).max() < 1e-12 * np.abs(a.R).max()
    assert np.abs(a.channels - b.channels).max() < 1e-6 * a.channels.max()


def test_f1_window_is_the_block_of_build_rates(tmp_path):
    arcs = _archives(tmp_path, *_model())
    full = bte.build_rates(arcs, T=150.0, verbose=False)
    w = bte.windowed_rates(arcs, 150.0, 0.08, refine=1, flat_width=0.0,
                           verbose=False, bose="node")
    keep = _coarse_index(w, 3)
    assert 0 < len(keep) < len(full.E)
    assert np.allclose(full.E[keep], w.E, rtol=0, atol=0)
    assert np.abs(full.P[np.ix_(keep, keep)] - w.P).max() < 1e-12 * w.P.max()


def test_forward_out_rates_match_refined_out_rates(tmp_path):
    arcs = _archives(tmp_path, *_model())
    T, f = 150.0, 3
    ref = bte.refined_out_rates(arcs, T, refine=f, verbose=False)
    rm = bte.windowed_rates(arcs, T, 10.0, refine=f, band_interp="fourier",
                            verbose=False)
    node = rm.meta["on_node"]
    assert node.sum() == 6 * 6 * 3
    r = rm.meta["forward_out"][node] / ref[_coarse_index(rm, 3)[node]]
    assert np.abs(r - 1).max() < 1e-10


def test_forward_out_rates_match_linewidths_with_acoustic_model(tmp_path):
    from xphd.linewidth import compute
    E, hw = _model()
    arcs = _archives(tmp_path, E, hw)
    T, f = 150.0, 3
    rm = bte.windowed_rates(arcs, T, 10.0, refine=f, band_interp="local",
                            acoustic_model="q2", n_shell=2, verbose=False)
    fw = rm.meta["flat_width"]
    node = np.nonzero(rm.meta["on_node"])[0]
    idx = _coarse_index(rm, 3)
    worst = 0.0
    for (a, b) in [(0, 0), (1, 2), (3, 3)]:
        lw = compute(arcs[(a, b)], [T], refine=f, flat_width=fw,
                     acoustic_model="q2", n_shell=2, band_interp="local",
                     verbose=False)
        for n in range(3):
            k = node[idx[node] == (a * 6 + b) * 3 + n]
            assert len(k) == 1
            fwhm = rm.meta["forward_out"][k[0]] * bte.HBAR_EV_FS
            worst = max(worst, abs(fwhm / lw.total[0, n] - 1))
    assert worst < 1e-10
    rm0 = bte.windowed_rates(arcs, T, 10.0, refine=f, verbose=False)
    assert (rm.meta["forward_out"] > rm0.meta["forward_out"] * (1 + 1e-6)).any()


@pytest.mark.parametrize("bose", ["node", "crossing"])
def test_detailed_balance_and_bose_stationarity(tmp_path, bose):
    arcs = _archives(tmp_path, *_model())
    T = 100.0
    rm = bte.windowed_rates(arcs, T, 0.25, refine=3, bose=bose, verbose=False)
    p = np.exp(-(rm.E - rm.E.min()) / (KB * T))
    flux = p[:, None] * rm.P
    assert np.abs(flux - flux.T).max() < 1e-8 * flux.max()
    assert np.abs(rm.R + rm.R.T).max() < 1e-12 * np.abs(rm.R).max()
    assert bte.check_balance(rm, N_tot=1e-3, verbose=False)["bose"] < 1e-8


def test_hw_fine_finer_than_refine_is_subsampled(tmp_path):
    from xphd.core.interp import fourier_refine
    E, hw = _model()
    arcs = _archives(tmp_path, E, hw)
    h6 = np.maximum(fourier_refine(hw, 6), 0).reshape(36 * 36, -1)
    h3 = np.ascontiguousarray(h6.reshape(36, 36, -1)[::2, ::2]).reshape(18 * 18, -1)
    np.save(tmp_path / "h6.npy", h6)
    np.save(tmp_path / "h3.npy", h3)
    a = bte.windowed_rates(arcs, 150.0, 0.2, refine=3, hw_fine=str(tmp_path / "h6.npy"),
                           verbose=False)
    b = bte.windowed_rates(arcs, 150.0, 0.2, hw_fine=str(tmp_path / "h3.npy"),
                           verbose=False)
    assert a.meta["refine"] == b.meta["refine"] == 3
    assert np.array_equal(a.P, b.P)
    with pytest.raises(SystemExit):
        bte.windowed_rates(arcs, 150.0, 0.2, refine=4, hw_fine=str(tmp_path / "h6.npy"),
                           verbose=False)


def test_window_reports_and_labels(tmp_path):
    arcs = _archives(tmp_path, *_model())
    rm = bte.windowed_rates(arcs, 77.0, 0.06, refine=3, verbose=False)
    m1, m2, ne = rm.dims
    assert (m1, m2, ne) == (18, 18, 3)
    assert len(rm.E) < m1 * m2 * ne
    assert (rm.E - rm.E.min()).max() <= 0.06 + 1e-12
    assert 0.0 < rm.meta["outside_weight"] < 1.0
    Q, band = bte.state_labels(rm)
    assert Q.shape == (len(rm.E), 3) and band.shape == (len(rm.E),)
    q0 = bte.q0_states(rm)
    assert q0[0] >= 0 and np.allclose(Q[q0[0], :2], 0.0) and band[q0[0]] == 0
    assert np.abs(rm.meta["Q_coarse_red"] * 6 - np.rint(rm.meta["Q_coarse_red"] * 6)).max() < 1e-9


def test_node_and_crossing_converge_for_optical_phonons(tmp_path):
    """With optical phonons only, both occupations become exact as the cell shrinks below kT:
    their total equilibrium rates move together."""
    arcs = _archives(tmp_path, *_model(acoustic=False))
    T = 300.0
    kT = KB * T

    def total(f, bose):
        rm = bte.windowed_rates(arcs, T, 0.12, refine=f, bose=bose, verbose=False)
        p = np.exp(-(rm.E - rm.E.min()) / kT)
        return (p[:, None] * rm.P).sum() / p.sum()

    r1 = total(1, "node") / total(1, "crossing")
    r5 = total(5, "node") / total(5, "crossing")
    assert abs(np.log(r5)) < 0.5 * abs(np.log(r1))


def test_bad_options_refused(tmp_path):
    arcs = _archives(tmp_path, *_model())
    with pytest.raises(ValueError):
        bte.windowed_rates(arcs, 150.0, 0.1, refine=2, bose="bose", verbose=False)
    with pytest.raises(ValueError):
        bte.windowed_rates(arcs, 150.0, 0.1, refine=2, acoustic_model="auto",
                           verbose=False)


def test_selection_rule_survives_band_reordering(tmp_path):
    """Two bands of opposite sigma_h parity that cross, so the energy-ordered band 1 is even
    near Gamma and odd further out; couplings obey the rule (even phonon: same parity, odd
    phonon: opposite)."""
    n, f = 6, 4
    i = np.arange(n)
    kx, ky = np.meshgrid(i / n, i / n, indexing="ij")
    c = (np.cos(2 * np.pi * kx) + np.cos(2 * np.pi * ky)
         + np.cos(2 * np.pi * (kx + ky))) / 3
    pert = 0.003 * np.sin(2 * np.pi * kx + 0.3) * np.cos(2 * np.pi * ky + 1.1)
    EA = 2.400 + 0.20 * (1 - c) + pert
    EB = 2.430 + 0.08 * (1 - c) - pert
    E = np.sort(np.stack([EA, EB], -1), axis=-1)
    chi = np.where(np.stack([EA, EB], -1).argsort(axis=-1) == 0, 1.0, -1.0)
    assert (chi[..., 0] > 0).any() and (chi[..., 0] < 0).any()
    hw = np.stack([np.full_like(c, 0.025), np.full_like(c, 0.034)], -1)
    qr = np.stack([kx.ravel(), ky.ravel(), np.zeros(n * n)], 1)
    rng = np.random.default_rng(3)
    d = tmp_path / "X"
    d.mkdir()
    for a in range(n):
        for b in range(n):
            sx, sy = (i + a) % n, (i + b) % n
            Em, cm = E[np.ix_(sx, sy)], chi[np.ix_(sx, sy)]
            same = chi[a, b][None, None, :, None] == cm[:, :, None, :]
            g2 = np.abs(rng.standard_normal((n, n, 2, 2, 2))) * 4e-6
            g2[:, :, 0] *= same
            g2[:, :, 1] *= ~same
            np.savez(d / f"Q{a:02d}{b:02d}.npz", g2_grid=g2.reshape(n * n, 2, 2, 2),
                     E_m_grid=Em.reshape(n * n, 2), hw_grid=hw.reshape(n * n, 2),
                     E_n_grid=Em[0, 0], q_red=qr, Q_red=np.array([a / n, b / n, 0.0]),
                     mesh=np.array([n, n, 1]))
    arcs = xphd.load_archives(str(d / "Q*.npz"), verbose=False)
    rm = bte.windowed_rates(arcs, 150.0, 10.0, refine=f, channels=True, verbose=False)
    qc = np.rint(rm.meta["Q_coarse_red"][:, :2] * n).astype(int) % n
    x = chi[qc[:, 0], qc[:, 1], rm.meta["band"]]
    cross = x[:, None] != x[None, :]
    even_ph = rm.channels[:, 0].astype(float).sum(axis=0)
    odd_ph = rm.channels[:, 1].astype(float).sum(axis=0)
    assert odd_ph[cross].sum() > 0
    assert even_ph[cross].sum() == 0.0
    assert odd_ph[~cross].sum() == 0.0


def test_acoustic_scaling(tmp_path):
    """g^2 ~ w carries an acoustic coupling to the fine q: nothing changes at f = 1 (fine q =
    coarse q), and below the first shell, where the fine phonon is softer than the coarse
    one, the acoustic rates can only drop."""
    E, hw = _model()
    arcs = _archives(tmp_path, E, hw)
    a = bte.build_rates(arcs, T=150.0, verbose=False)
    b = bte.windowed_rates(arcs, 150.0, 10.0, refine=1, flat_width=0.0,
                           acoustic_scaling="omega", verbose=False)
    assert np.abs(a.P - b.P).max() < 1e-12 * a.P.max()
    kw = dict(refine=3, channels=True, verbose=False)
    n0 = bte.windowed_rates(arcs, 150.0, 0.2, acoustic_scaling="none", **kw)
    n1 = bte.windowed_rates(arcs, 150.0, 0.2, acoustic_scaling="omega", **kw)
    ac0 = n0.channels[:, 0].astype(float).sum()
    ac1 = n1.channels[:, 0].astype(float).sum()
    assert 0 < ac1 < ac0
    assert np.allclose(n0.channels[:, 1:], n1.channels[:, 1:], rtol=1e-6, atol=0)
    with pytest.raises(ValueError):
        bte.windowed_rates(arcs, 150.0, 0.2, refine=2, acoustic_scaling="q2", verbose=False)


def test_same_parent_pairs_follow_the_scaling(tmp_path):
    """acoustic_model="shell" fills the cell around q = 0 with the first-shell coupling between
    the same states, carried down by the scaling law: nothing changes at f = 1 (no
    same-parent pair except q = 0, where w = 0), the acoustic rates only grow against the
    empty cell, the optical ones are untouched, n_shell does not enter, and for couplings g^2
    = k w on the coarse mesh (omega) every pair couples as k w(q_fine) -- the rate matrix
    cannot depend on where the parent cells sit."""
    E, hw = _model()
    arcs = _archives(tmp_path, E, hw)
    one = np.ones((36, 3))
    a = bte.build_rates(arcs, T=150.0, verbose=False)
    b = bte.windowed_rates(arcs, 150.0, 10.0, refine=1, flat_width=0.0,
                           acoustic_scaling="omega", acoustic_model="shell",
                           parity=one, verbose=False)
    assert np.abs(a.P - b.P).max() < 1e-12 * a.P.max()
    kw = dict(refine=3, channels=True, verbose=False, acoustic_scaling="omega")
    z = bte.windowed_rates(arcs, 150.0, 0.2, acoustic_model="none", **kw)
    q = bte.windowed_rates(arcs, 150.0, 0.2, acoustic_model="shell", parity=one, **kw)
    acz = z.channels[:, 0].astype(float).sum()
    acq = q.channels[:, 0].astype(float).sum()
    assert acq > acz * (1 + 1e-3)
    assert np.allclose(z.channels[:, 1:], q.channels[:, 1:], rtol=1e-6, atol=0)
    q3 = bte.windowed_rates(arcs, 150.0, 0.2, acoustic_model="shell", parity=one,
                            n_shell=3, **kw)
    q1 = bte.windowed_rates(arcs, 150.0, 0.2, acoustic_model="shell", parity=one,
                            n_shell=1, **kw)
    assert np.array_equal(q1.P, q3.P)
    with pytest.raises(ValueError):
        bte.windowed_rates(arcs, 150.0, 0.2, acoustic_model="shell", **kw)

    n, f = 6, 3
    i = np.arange(n)
    kx, ky = np.meshgrid(i / n, i / n, indexing="ij")
    qr = np.stack([kx.ravel(), ky.ravel(), np.zeros(n * n)], 1)
    hw1 = hw[..., :1]

    def band(x, y):
        return 2.4 + 0.05 * (1 - (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)
                                  + np.cos(2 * np.pi * (x + y))) / 3)

    def arcs_for(shift, tag):
        d = tmp_path / tag
        d.mkdir()
        Eb = band(kx + shift, ky)[..., None]
        g2 = (2e-4 * hw1)[:, :, :, None, None]
        for a_ in range(n):
            for b_ in range(n):
                sx, sy = (i + a_) % n, (i + b_) % n
                np.savez(d / f"Q{a_:02d}{b_:02d}.npz", g2_grid=g2.reshape(n * n, 1, 1, 1),
                         E_m_grid=Eb[np.ix_(sx, sy)].reshape(n * n, 1),
                         hw_grid=hw1.reshape(n * n, 1), E_n_grid=Eb[a_, b_], q_red=qr,
                         Q_red=np.array([a_ / n, b_ / n, 0.0]), mesh=np.array([n, n, 1]))
        return xphd.load_archives(str(d / "Q*.npz"), verbose=False)

    from xphd.core.interp import fourier_refine
    np.save(tmp_path / "hf.npy", np.maximum(fourier_refine(hw1, f), 0.0).reshape(-1, 1))
    kk = dict(refine=f, hw_fine=str(tmp_path / "hf.npy"), acoustic_scaling="omega",
              verbose=False, band_interp="fourier", bose="crossing", flat_width=1e-4,
              parity=np.ones((n * n, 1)))
    A0, A1 = arcs_for(0.0, "S0"), arcs_for(1.0 / (n * f), "S1")
    m = n * f
    idx = np.arange(m * m).reshape(m, m)
    perm = np.roll(idx, -1, axis=0).ravel()
    for model, same in (("shell", True), ("none", False)):
        P0 = bte.windowed_rates(A0, 150.0, 10.0, acoustic_model=model, **kk).P
        P1 = bte.windowed_rates(A1, 150.0, 10.0, acoustic_model=model, **kk).P
        d = np.abs(P1 - P0[np.ix_(perm, perm)]).max() / P0.max()
        assert (d < 1e-9) if same else (d > 1e-3), (model, d)


def test_cell_fill_keeps_the_selection_rule_where_bands_reorder(tmp_path):
    """Two bands of opposite parity that cross, so band index 1 is even near Gamma and odd
    further out, and two acoustic branches (zero at q = 0), one even and one odd, whose
    couplings obey the rule."""
    n, f = 6, 4
    i = np.arange(n)
    kx, ky = np.meshgrid(i / n, i / n, indexing="ij")
    c = (np.cos(2 * np.pi * kx) + np.cos(2 * np.pi * ky)
         + np.cos(2 * np.pi * (kx + ky))) / 3
    pert = 0.003 * np.sin(2 * np.pi * kx + 0.3) * np.cos(2 * np.pi * ky + 1.1)
    EA = 2.400 + 0.20 * (1 - c) + pert
    EB = 2.430 + 0.08 * (1 - c) - pert
    E = np.sort(np.stack([EA, EB], -1), axis=-1)
    chi = np.where(np.stack([EA, EB], -1).argsort(axis=-1) == 0, 1.0, -1.0)
    s_ = np.sqrt(np.clip((1 - c) / 1.5, 0, None))
    hw = np.stack([0.020 * s_, 0.008 * s_ ** 2], -1)
    qr = np.stack([kx.ravel(), ky.ravel(), np.zeros(n * n)], 1)
    rng = np.random.default_rng(5)
    d = tmp_path / "R"
    d.mkdir()
    for a in range(n):
        for b in range(n):
            sx, sy = (i + a) % n, (i + b) % n
            Em, cm = E[np.ix_(sx, sy)], chi[np.ix_(sx, sy)]
            same = chi[a, b][None, None, :, None] == cm[:, :, None, :]
            g2 = np.abs(rng.standard_normal((n, n, 2, 2, 2))) * 4e-6
            g2[:, :, 0] *= same
            g2[:, :, 1] *= ~same
            g2 *= (_qnorm(kx, ky) / _qnorm(kx, ky).max())[:, :, None, None, None]
            np.savez(d / f"Q{a:02d}{b:02d}.npz", g2_grid=g2.reshape(n * n, 2, 2, 2),
                     E_m_grid=Em.reshape(n * n, 2), hw_grid=hw.reshape(n * n, 2),
                     E_n_grid=Em[0, 0], q_red=qr, Q_red=np.array([a / n, b / n, 0.0]),
                     mesh=np.array([n, n, 1]))
    arcs = xphd.load_archives(str(d / "Q*.npz"), verbose=False)
    par = chi.reshape(n * n, 2)
    kw = dict(refine=f, channels=True, verbose=False, acoustic_scaling="omega")
    x = None

    def cross_sector(rm):
        nonlocal x
        qc = np.rint(rm.meta["Q_coarse_red"][:, :2] * n).astype(int) % n
        x = chi[qc[:, 0], qc[:, 1], rm.meta["band"]]
        cross = x[:, None] != x[None, :]
        ev = rm.channels[:, 0].astype(float).sum(axis=0)
        od = rm.channels[:, 1].astype(float).sum(axis=0)
        return ev[cross].sum(), od[cross].sum(), ev[~cross].sum(), od[~cross].sum()

    ev_x, od_x, ev_s, od_s = cross_sector(bte.windowed_rates(
        arcs, 150.0, 10.0, acoustic_model="shell", parity=par, **kw))
    assert ev_x == 0.0 and od_s == 0.0
    assert od_x > 0 and ev_s > 0
    e0, o0, _, _ = cross_sector(bte.windowed_rates(arcs, 150.0, 10.0, acoustic_model="none", **kw))
    assert e0 == 0.0 and od_x > o0
    ev_q, _, _, od_q = cross_sector(bte.windowed_rates(
        arcs, 150.0, 10.0, acoustic_model="q2", n_shell=1, **kw))
    assert ev_q > 0 or od_q > 0


@pytest.mark.parametrize("p", [1, 2])
def test_power_law_reduces_to_none_and_omega(tmp_path, p):
    """acoustic_scaling="power" with |M|^2 ~ q^n and g^2 = |M|^2/2w: for an acoustic branch that
    is an exact power w ~ |q|^p on the coarse and the fine mesh it is "none" at n = p and
    "omega" at n = 2p -- for the pairs across a parent boundary and for the filled cell
    alike."""
    E, hw = _model()
    n, f = 6, 3
    i = np.arange(n)
    kx, ky = np.meshgrid(i / n, i / n, indexing="ij")
    hw = hw.copy()
    hw[..., 0] = 0.020 * (n * _qnorm(kx, ky)) ** p
    arcs = _archives(tmp_path, E, hw)
    m = n * f
    j = np.arange(m)
    fx, fy = np.meshgrid(j / m, j / m, indexing="ij")
    hf = np.stack([0.020 * (n * _qnorm(fx, fy)) ** p,
                   np.full_like(fx, 0.034), np.full_like(fx, 0.045)], -1).reshape(m * m, 3)
    np.save(tmp_path / "hf.npy", hf)
    kw = dict(refine=f, hw_fine=str(tmp_path / "hf.npy"), acoustic_model="shell",
              parity=np.ones((n * n, 3)), bose="crossing", verbose=False)
    for ref, npow in (("none", p), ("omega", 2 * p)):
        a = bte.windowed_rates(arcs, 150.0, 0.2, acoustic_scaling=ref, **kw)
        b = bte.windowed_rates(arcs, 150.0, 0.2, acoustic_scaling="power",
                               acoustic_power=npow, **kw)
        assert np.abs(a.P - b.P).max() < 1e-10 * a.P.max(), (ref, npow)
    with pytest.raises(ValueError):
        bte.windowed_rates(arcs, 150.0, 0.2, acoustic_scaling="power", **kw)
