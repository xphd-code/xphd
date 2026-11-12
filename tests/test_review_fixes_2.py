"""Fixes from the second part of the external review (4 Nov 2026)."""
import inspect

import numpy as np
import pytest

from xphd import lineshape, modes, symmetry


class _Arc:
    def __init__(self, G, Em, hw, En):
        self.n1 = self.n2 = G.shape[0]
        self._g = dict(G=G, E_m=Em, hw=hw)
        self.nexc, self.nmod, self.E_n = G.shape[-1], G.shape[2], En

    def has(self, k):
        return k in self._g

    def grid(self, k):
        return self._g[k]


def _arc(seed=0, phases=None):
    n, nm, ne = 4, 2, 3
    rng = np.random.default_rng(seed)
    G = (rng.normal(size=(n, n, nm, ne, ne)) + 1j * rng.normal(size=(n, n, nm, ne, ne))) * 3e-3
    i = np.arange(n)
    kx, ky = np.meshgrid(i / n, i / n, indexing="ij")
    c = (np.cos(2 * np.pi * kx) + np.cos(2 * np.pi * ky) + np.cos(2 * np.pi * (kx + ky))) / 3
    Em = np.stack([2.0 + 0.05 * (1 - c), 2.02 + 0.06 * (1 - c), 2.05 + 0.04 * (1 - c)], -1)
    hw = np.stack([0.02 + 0.01 * (1 - c), 0.05 + 0.002 * c], -1)
    if phases is not None:
        G = G * np.exp(1j * np.asarray(phases))[None, None, None, :, None]
    return _Arc(G, Em, hw, Em[0, 0])


def test_sigma_matrix_covariant_and_hermitian():
    om = np.linspace(1.85, 2.35, 401)
    phi = np.array([0.0, np.pi / 2, 1.1])
    a = lineshape.sigma_matrix(_arc(), om, 300.0, verbose=False).sigma
    b = lineshape.sigma_matrix(_arc(phases=phi), om, 300.0, verbose=False).sigma
    U = np.exp(1j * phi)
    want = np.conj(U)[None, :, None] * a * U[None, None, :]
    assert np.allclose(b, want, atol=1e-12 * np.abs(a).max())
    H = (a + np.conj(np.swapaxes(a, 1, 2))) / 2
    D = -(a - np.conj(np.swapaxes(a, 1, 2))) / 2j
    assert np.allclose(np.diagonal(D, axis1=1, axis2=2).imag, 0, atol=1e-15)
    assert (np.diagonal(D, axis1=1, axis2=2).real >= -1e-14).all()
    assert np.abs(D[:, 0, 1].imag).max() > 1e-3 * np.abs(D).max()
    from xphd.selfenergy import kramers_kronig
    dd = np.diagonal(D, axis1=1, axis2=2).real.T
    assert np.allclose(np.diagonal(a, axis1=1, axis2=2).T, kramers_kronig(om, -dd) - 1j * dd)


def test_read_dipoles_axes(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    z = np.array([1 + 2j, -0.5 + 0.1j, 3j])
    for shape in ("2d", "3d"):
        p = tmp_path / f"bs_{shape}.nc"
        with netCDF4.Dataset(p, "w") as f:
            f.createDimension("n", 3)
            f.createDimension("c", 2)
            if shape == "2d":
                v = f.createVariable("BS_Residuals", "f8", ("n", "c"))
                v[:] = np.stack([z.real, z.imag], -1)
            else:
                f.createDimension("p", 2)
                v = f.createVariable("BS_Residuals", "f8", ("p", "n", "c"))
                v[:] = np.stack([np.stack([z.real, z.imag], -1), np.zeros((3, 2))])
        d = lineshape.read_dipoles(str(p))
        assert np.allclose(d, z), shape


def test_symmetry_single_definitions_and_complex_expand():
    src = inspect.getsource(symmetry)
    for name in ("full_mesh", "irreducible", "expand"):
        assert src.count(f"def {name}(") == 1, name
    ops = [np.eye(3)]
    Q = symmetry.full_mesh(2)
    v = np.arange(4) * (1 + 1j)
    out = symmetry.expand(v, Q, Q, ops, verbose=False)
    assert np.iscomplexobj(out) and np.allclose(out, v)
    idx, Qf = symmetry.irreducible(2, ops, verbose=False)
    assert len(idx) == 4 and Qf.shape == (4, 3)


def test_classify_ragged_out_of_plane(capsys):
    nq, nat = 3, 2
    ev = np.zeros((nq, 6, nat, 3), complex)
    for m, d in enumerate([2, 0, 1, 2, 0, 1]):
        ev[:, m, :, d] = 1.0
    ev[1, 3] = 0.0
    ev[1, 3, :, 0] = 1.0
    freq = np.tile(np.arange(6.0), (nq, 1))
    lab = modes.classify(ev, freq=freq, verbose=True)
    assert lab.shape == (nq, 6)
    assert "do not have exactly two" in capsys.readouterr().out


def test_selection_matches_final_states_at_Q_plus_q():
    from xphd import selection
    src = inspect.getsource(selection.main)
    assert "arc.Q_red" in src and "+ Q0" in src


def test_read_dipoles_from_folder(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    job = tmp_path / "bse" / "job"
    job.mkdir(parents=True)
    z = np.array([1 + 2j, -0.5 + 0.1j])
    with netCDF4.Dataset(job / "ndb.BS_diago_Q1", "w") as f:
        f.createDimension("n", 2)
        f.createDimension("c", 2)
        f.createVariable("BS_Residuals", "f8", ("n", "c"))[:] = np.stack([z.real, z.imag], -1)
    assert np.allclose(lineshape.read_dipoles(str(tmp_path / "bse")), z)
    (tmp_path / "bse" / "other").mkdir()
    (tmp_path / "bse" / "other" / "ndb.BS_diago_Q1").write_bytes(b"")
    with pytest.raises(SystemExit, match="2 copies"):
        lineshape.read_dipoles(str(tmp_path / "bse"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no ndb.BS_diago_Q1"):
        lineshape.read_dipoles(str(tmp_path / "empty"))
