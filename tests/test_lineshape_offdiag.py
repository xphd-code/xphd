"""xphd lineshape for the off-diagonal self-energy against experiment."""
import numpy as np
import pytest

from xphd import lineshape as ls
from xphd.io.excph import ExcPhArchive
from xphd.linewidth import compute


def _grid(arc, n=401, pad=0.05):
    lo = float(arc.grid("E_m").min() - arc.grid("hw").max()) - pad
    hi = float(arc.grid("E_m").max() + arc.grid("hw").max()) + pad
    return np.linspace(lo, hi, n)


def test_temperature_list_equals_single_calls(archive):
    arc = ExcPhArchive(str(archive))
    w = _grid(arc, 161)
    many = ls.sigma_matrix(arc, w, [77.0, 300.0], refine=2, verbose=False)
    for sm in many:
        one = ls.sigma_matrix(arc, w, sm.T, refine=2, verbose=False)
        assert np.allclose(one.sigma, sm.sigma, atol=1e-13 * np.abs(sm.sigma).max())


def test_sweep_equals_whole_mesh_loop(archive):
    """The triangle sweep and the reference loop (delta_weights_2d at every frequency) give the
    same matrices, with and without the acoustic models."""
    arc = ExcPhArchive(str(archive))
    w = _grid(arc, 121)
    par = _parity(arc)
    for kw in (dict(), dict(acoustic_model="shell", acoustic_scaling="omega", parity=par)):
        a = ls.sigma_matrix(arc, w, [77.0, 300.0], refine=3, prune=False, verbose=False, **kw)
        b = ls.sigma_matrix(arc, w, [77.0, 300.0], refine=3, prune=False, method="loop",
                            verbose=False, **kw)
        for x, y in zip(a, b):
            assert np.allclose(x.sigma, y.sigma, rtol=0, atol=1e-12 * np.abs(y.sigma).max())


def test_diagonal_is_the_linewidth(archive):
    arc = ExcPhArchive(str(archive))
    E = np.asarray(arc.E_n, float)
    lo = float(arc.grid("E_m").min() - arc.grid("hw").max()) - 0.05
    step = 5e-4
    k0 = int(np.floor((E.min() - lo) / step)) + 1
    start = E.min() - k0 * step
    w = start + step * np.arange(int((E.max() + 0.25 - start) / step))
    assert all(np.min(np.abs(w - e)) < 1e-12 for e in E)
    for T in (77.0, 300.0):
        lw = compute(arc, [T], refine=3, verbose=False)
        exact = ls.sigma_matrix(arc, w, T, refine=3, prune=False, cells="rhombic",
                                verbose=False)
        assert np.allclose(exact.gamma_diag(), lw.total[0], rtol=1e-9, atol=1e-12)
        fast = ls.sigma_matrix(arc, w, T, refine=3, cells="rhombic", verbose=False)
        assert np.allclose(fast.gamma_diag(), lw.total[0], rtol=0,
                           atol=1e-5 * lw.total[0].max())


def _parity(arc):
    n = arc.n1 * arc.n2
    return np.tile(np.array([1.0, 1.0, -1.0, 1.0])[: arc.nexc], (n, 1))


def test_acoustic_models_add_width_and_stay_covariant(archive):
    arc = ExcPhArchive(str(archive))
    w = _grid(arc, 161)
    par = _parity(arc)
    none = ls.sigma_matrix(arc, w, 300.0, refine=3, verbose=False)
    om = ls.sigma_matrix(arc, w, 300.0, refine=3, acoustic_scaling="omega", verbose=False)
    sh = ls.sigma_matrix(arc, w, 300.0, refine=3, acoustic_model="shell",
                         acoustic_scaling="omega", parity=par, verbose=False)
    d = np.arange(arc.nexc)
    D0 = -none.sigma[:, d, d].imag.sum()
    Ds = -sh.sigma[:, d, d].imag.sum()
    Do = -om.sigma[:, d, d].imag.sum()
    assert Ds > Do and Do != D0
    for sm in (om, sh):
        D = -(sm.sigma - np.conj(np.swapaxes(sm.sigma, 1, 2))) / 2j
        assert np.allclose(D, np.conj(np.swapaxes(D, 1, 2)), atol=1e-15)
        assert (np.diagonal(D, axis1=1, axis2=2).real >= -1e-14).all()
    with pytest.raises(ValueError, match="parit"):
        ls.sigma_matrix(arc, w, 300.0, refine=3, acoustic_model="shell", verbose=False)


def test_fine_grid_and_peak_resolution():
    w = np.linspace(1.0, 3.0, 2001)
    ne = 2
    gam = 1e-4
    S = np.zeros((len(w), ne, ne), complex)
    S[:, 0, 0] = -0.5j * gam
    S[:, 1, 1] = -0.5j * 5e-3
    sm = ls.SigmaMatrix(w, S, np.array([2.0, 2.2]), 10.0)
    d = np.array([1.0, 0.7])
    assert np.allclose(ls.epsilon2(sm, d, 1e-3), ls.epsilon2(sm, d, 1e-3, omega=w))
    m, x, y = ls.peak_metrics(sm, d, (1.9, 2.1), eta=1e-6)
    assert m["fwhm"] == pytest.approx(gam + 2e-6, rel=2e-3)
    assert m["peak"] == pytest.approx(2.0, abs=1e-6)


def test_symmetry_report():
    w = np.linspace(1.0, 3.0, 11)
    S = np.zeros((len(w), 3, 3), complex)
    S[:, 0, 0] = S[:, 1, 1] = -0.01j
    S[:, 2, 2] = -0.02j
    sm = ls.SigmaMatrix(w, S, np.array([2.0, 2.0, 2.1]), 77.0)
    out = ls.symmetry_report(sm, chi0=[1, 1, -1], verbose=False)
    assert out["parity"] == 0.0 and out["multiplet"] == 0.0 and out["groups"] == [[0, 1]]
    S2 = S.copy()
    S2[:, 0, 2] = S2[:, 2, 0] = 0.003
    S2[:, 0, 1] = 0.002j
    S2[:, 1, 0] = -0.002j
    out = ls.symmetry_report(ls.SigmaMatrix(w, S2, sm.E_bare, 77.0), chi0=[1, 1, -1],
                             verbose=False)
    assert out["parity"] == pytest.approx(0.15) and out["multiplet"] == pytest.approx(0.1)


def test_bright_window():
    E = np.array([1.79, 1.795, 1.81, 1.81, 1.94, 1.96, 1.96])
    d = np.array([0, 1e-4, 1.0, 0.9, 1e-3, 0.8, 0.8])
    w = np.linspace(1.5, 2.3, 101)
    lo, hi, mem = ls.bright_window(E, d, w)
    assert sorted(mem) == [2, 3]
    assert lo == pytest.approx(1.56) and hi == pytest.approx(1.885)


def test_command_temperature_table(archive, tmp_path, monkeypatch, capsys):
    from xphd import cli
    arc = ExcPhArchive(str(archive))
    d = np.array([0.0, 1.0, 0.8, 0.1], complex)
    monkeypatch.setattr(ls, "read_dipoles", lambda *a_, **k_: d)
    out = tmp_path / "ls.npz"
    cli.main(["lineshape", str(archive), "--T", "100", "300", "--refine", "2",
              "--n-omega", "301", "--bsdb", "fake", "--n-plot", "501", "-o", str(out)])
    txt = capsys.readouterr().out
    assert "at w = E_A (Chan)" in txt and "lowest bright multiplet" in txt
    z = np.load(out)
    assert list(z["T"]) == [100.0, 300.0]
    assert z["sigma"].shape == (2, 301, arc.nexc, arc.nexc)
    assert z["eps2_full"].shape == (2, 501) and np.isfinite(z["fwhm_full"]).all()
    for k in ("gamma_A", "gamma_A_full", "shift_A_full", "fwhm_diag", "peak_full"):
        assert z[k].shape == (2,) and np.isfinite(z[k]).all(), k
    assert bool(z["recenter"]) and np.allclose(z["recenter_shift"], z["shift_A"])
    assert np.allclose(z["recenter_shift_full"], z["shift_A_full"])
    assert str(z["final_width_mode"]) == "sc"
    assert np.allclose(2 * z["final_width"], z["gamma_A"], rtol=1e-3)
    assert z["gamma_A_0"].shape == (2,) and np.isfinite(z["gamma_A_full_0"]).all()
    assert (np.abs(z["peak_full"] - z["E_A"]) < np.abs(z["shift_A"]) + 1e-4).all()
    assert z["gamma_A_0"][1] == pytest.approx(z["gamma_diag"][1][list(z["bright"])].mean(),
                                              rel=1e-6)


def test_onshell_two_level():
    """A bright level coupled to one other: the on-shell eigen-width is the diagonal one plus
    the second-order correction |v|^2 / (E_A - E_X)."""
    w = np.linspace(1.0, 3.0, 2001)
    S = np.zeros((len(w), 2, 2), complex)
    gA, gX, v = 1e-3, 4e-3, 2e-3 - 1e-3j
    S[:, 0, 0], S[:, 1, 1] = -0.5j * gA, -0.5j * gX
    S[:, 0, 1], S[:, 1, 0] = v, v
    sm = ls.SigmaMatrix(w, S, np.array([2.0, 2.15]), 300.0)
    od = ls.onshell(sm, [0], diagonal=True)
    of = ls.onshell(sm, [0])
    assert od["fwhm"] == pytest.approx(gA) and od["shift"] == pytest.approx(0.0, abs=1e-15)
    lam = 2.0 - 0.5j * gA + v * v / ((2.0 - 0.5j * gA) - (2.15 - 0.5j * gX))
    assert of["fwhm"] == pytest.approx(-2 * lam.imag, rel=2e-3)


def test_interpolation_keeps_widths_positive():
    """At a channel threshold Im Sigma jumps; the interpolation between nodes must not overshoot
    to a negative width (linear: a convex combination)."""
    w = np.linspace(1.0, 2.0, 101)
    S = np.zeros((len(w), 2, 2), complex)
    S[w >= 1.5, 0, 0] = -5e-3j
    S[w >= 1.5, 0, 1] = S[w >= 1.5, 1, 0] = -2e-3j
    S[w >= 1.5, 1, 1] = -1e-3j
    sm = ls.SigmaMatrix(w, S, np.array([1.5, 1.6]), 300.0)
    x = np.linspace(1.4, 1.6, 2001)
    Sx = sm.at(x)
    D = -(Sx - np.conj(np.swapaxes(Sx, 1, 2))) / 2j
    assert np.linalg.eigvalsh(D).min() >= -1e-18
    assert np.allclose(sm.at(w), S)


def test_command_dipoles_from_npy(archive, tmp_path):
    from xphd import cli
    dp = tmp_path / "d.npy"
    np.save(dp, np.array([0.0, 1.0, 0.8, 0.1], complex))
    out = tmp_path / "ls.npz"
    cli.main(["lineshape", str(archive), "--T", "300", "--refine", "2", "--n-omega", "201",
              "--dipoles", str(dp), "--n-plot", "201", "-o", str(out)])
    z = np.load(out)
    assert list(z["bright"]) == [1] and np.isfinite(z["fwhm_full"]).all()


class _SymArc:
    """A doublet (x, y) coupled to one band through a C3v-equivariant vector field, on a
    60-degree hexagonal mesh."""
    def __init__(self, n):
        i = np.arange(n)
        q1, q2 = np.meshgrid(i / n, i / n, indexing="ij")
        c = np.cos(2 * np.pi * q1) + np.cos(2 * np.pi * q2) + np.cos(2 * np.pi * (q1 + q2))
        s = [np.sin(2 * np.pi * q1), np.sin(2 * np.pi * q2), np.sin(-2 * np.pi * (q1 + q2))]
        th = np.deg2rad([0, 120, 240])
        G = np.zeros((n, n, 1, 2, 2), complex)
        G[:, :, 0, 0, 0] = 1e-2 * sum(np.cos(t) * sj for t, sj in zip(th, s))
        G[:, :, 0, 1, 0] = 1e-2 * sum(np.sin(t) * sj for t, sj in zip(th, s))
        self.n1 = self.n2 = n
        self.nexc, self.nmod, self.E_n = 2, 1, np.array([2.0, 2.0])
        self._g = dict(G=G, E_m=np.stack([2.0 + 0.05 * (1 - c / 3), 3.0 + 0.05 * (1 - c / 3)], -1),
                       hw=(0.02 + 0.01 * (1 - c / 3))[..., None])

    def has(self, k):
        return k in self._g

    def grid(self, k):
        return self._g[k]


def test_hex_cells_keep_schur_on_the_fine_mesh():
    w = np.linspace(1.9, 2.25, 241) + 1.2345678e-5
    for r in (2, 3, 4):
        dev = {}
        for cells in ("hex", "rhombic"):
            S = ls.sigma_matrix(_SymArc(12), w, 300.0, refine=r, cells=cells,
                                verbose=False).sigma
            sc = np.abs(S[:, 0, 0]).max()
            dev[cells] = max(np.abs(S[:, 0, 0] - S[:, 1, 1]).max(), np.abs(S[:, 0, 1]).max()) / sc
        assert dev["hex"] < 1e-12, (r, dev)
        assert dev["rhombic"] > 1e-2, (r, dev)
    S = ls.sigma_matrix(_SymArc(12), w, 300.0, refine=1, cells="rhombic", verbose=False).sigma
    assert np.abs(S[:, 0, 1]).max() < 1e-12 * np.abs(S[:, 0, 0]).max()


def test_hex_cells_partition():
    from xphd.core.interp import hex_cells, local_quadratic
    for r in (2, 3, 4, 5):
        idx, w, di, dj = hex_cells(9, 9, r)
        assert np.allclose(w.sum(1), 1.0)
        d = (di / r) ** 2 + (dj / r) ** 2 + (di / r) * (dj / r)
        used = w > 0
        assert np.allclose(d[used], np.min(np.where(used, d, np.inf), axis=1)[np.nonzero(used)[0]])
    n = 9
    i = np.arange(n)
    q1, q2 = np.meshgrid(i / n, i / n, indexing="ij")
    f = (np.cos(2 * np.pi * q1) + np.cos(2 * np.pi * q2))[..., None]
    for r in (2, 3):
        assert np.allclose(local_quadratic(f, r, "hex")[::r, ::r], f)


def test_command_final_width_options(archive, tmp_path, monkeypatch, capsys):
    from xphd import cli
    d = np.array([0.0, 1.0, 0.8, 0.1], complex)
    monkeypatch.setattr(ls, "read_dipoles", lambda *a_, **k_: d)
    base = ["lineshape", str(archive), "--T", "300", "--refine", "2", "--n-omega", "301",
            "--bsdb", "fake", "--n-plot", "201"]
    out = tmp_path / "one.npz"
    cli.main(base + ["--final-width", "0", "-o", str(out)])
    z = np.load(out)
    assert z["final_width"][0] == 0.0
    assert np.allclose(z["gamma_A"], z["gamma_A_0"]) and np.allclose(z["gamma_A_full"],
                                                                      z["gamma_A_full_0"])
    out = tmp_path / "fixed.npz"
    cli.main(base + ["--final-width", "2e-3", "-o", str(out)])
    assert np.load(out)["final_width"][0] == pytest.approx(2e-3)
    with pytest.raises(SystemExit):
        cli.main(base + ["--final-width", "wide"])


def test_fast_kramers_kronig_is_the_reference():
    from xphd.selfenergy import kramers_kronig
    w = np.linspace(1.5, 2.5, 701)
    y = np.random.default_rng(3).random((7, 701)) * np.exp(-((w - 2.0) / 0.15) ** 2)
    a = kramers_kronig(w, y)
    assert np.abs(ls._kk(w, y) - a).max() < 1e-11 * np.abs(a).max()
    assert np.abs(ls._kk(w, y[0]) - a[0]).max() < 1e-11 * np.abs(a).max()


def test_kramers_kronig_is_second_order():
    """Re of a Lorentzian pole from its Im: with the regular integrand's own value at w' = w the
    error falls as dw^2 (without it, as dw)."""
    from xphd.selfenergy import kramers_kronig
    err = []
    for nw in (2001, 4001, 8001):
        w = np.linspace(0.0, 4.0, nw)
        S = 1e-3 / (w - 2.0 + 4e-3j)
        m = (w > 1.8) & (w < 2.2)
        err.append(np.abs(kramers_kronig(w, S.imag) - S.real)[m].max() / np.abs(S[m]).max())
    assert err[-1] < 1e-3
    assert err[0] / err[1] > 3.5 and err[1] / err[2] > 3.5


def test_final_width_is_the_analytic_continuation():
    """Poles with Lorentzian widths g."""
    w = np.linspace(0.0, 4.0, 8001)
    E = np.array([1.9, 2.0, 2.15])
    g0 = np.array([6e-3, 4e-3, 8e-3])
    rng = np.random.default_rng(5)
    V = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))

    def sig(eta):
        out = np.zeros((len(w), 2, 2), complex)
        for p in range(3):
            out += 1e-3 * np.outer(V[p].conj(), V[p])[None] \
                / (w - E[p] + 1j * (g0[p] + eta))[:, None, None]
        return out

    sm = ls.SigmaMatrix(w, sig(0.0), np.array([2.0, 2.0]), 300.0)
    assert sm.with_final_width(0.0) is sm
    got = sm.with_final_width(3e-3).sigma
    ref = sig(3e-3)
    m = (w > 1.7) & (w < 2.3)
    sc_ = np.abs(ref[m]).max()
    assert np.abs(got[m] - ref[m]).max() < 5e-3 * sc_
    D = (np.conj(np.swapaxes(got, 1, 2)) - got) / 2j
    assert np.linalg.eigvalsh(D).min() > -1e-12 * sc_


class _BandBottom:
    """One bright state at the bottom of its own 2D band (mass 1.1 m0 on the MoS2 lattice, |b| =
    2.30 1/A), an acoustic branch (v = 5 km/s, g^2 ~ w) and an optical one (48 meV)."""
    EA = 2.0

    @staticmethod
    def fields(N):
        i = np.arange(N)
        q1, q2 = np.meshgrid(i / N, i / N, indexing="ij")
        c = np.cos(2 * np.pi * q1) + np.cos(2 * np.pi * q2) + np.cos(2 * np.pi * (q1 + q2))
        x = np.clip(3 - c, 0, None) / (2 * np.pi) ** 2
        E = _BandBottom.EA + 0.3 * np.tanh(18.3 * x / 0.3)
        return E, np.stack([0.0757 * np.sqrt(x), 0.048 + 0.002 * (1 - c / 3)], -1)

    def __init__(self, n):
        E, hw = self.fields(n)
        G = np.zeros((n, n, 2, 1, 1), complex)
        G[..., 0, 0, 0] = 0.03 * np.sqrt(hw[..., 0] / 0.01)
        G[..., 1, 0, 0] = 0.15
        self.n1 = self.n2 = n
        self.nexc, self.nmod, self.E_n = 1, 2, np.array([self.EA])
        self._g = dict(G=G, E_m=E[..., None], hw=hw)

    def has(self, k):
        return k in self._g

    def grid(self, k):
        return self._g[k]


def test_band_bottom_line_needs_final_widths():
    n, r, EA = 12, 10, _BandBottom.EA
    arc = _BandBottom(n)
    one, sc = [], []
    for nw in (801, 1001):
        w = np.linspace(EA - 0.06, EA + 0.36, nw)
        sm = ls.sigma_matrix(arc, w, 300.0, hw_fine=_BandBottom.fields(n * r)[1], refine=r,
                             acoustic_cut=2e-5, acoustic_model="shell",
                             acoustic_scaling="omega", parity=np.ones((n * n, 1)),
                             verbose=False)

        def line(s_):
            od = ls.onshell(s_, [0], diagonal=True)
            smr = ls.SigmaMatrix(w, s_.sigma - od["shift"], s_.E_bare, 300.0)
            m, _, _ = ls.peak_metrics(smr, np.array([1.0]), (EA - 0.05, EA + 0.05), eta=1e-6)
            return od["fwhm"], m["fwhm"]

        one.append(line(sm))
        eta, it, ok = ls.final_width_sc(sm, [0])
        assert ok
        g, f = line(sm.with_final_width(eta))
        assert g == pytest.approx(2 * eta, rel=1e-3)
        sc.append((g, f))
    assert all(f < 0.5 * g for g, f in one)
    assert abs(one[0][0] - one[1][0]) > 0.25 * max(one[0][0], one[1][0])
    assert abs(sc[0][0] - sc[1][0]) < 0.05 * sc[0][0]
    assert all(0.6 * g < f < 1.0 * g for g, f in sc)
