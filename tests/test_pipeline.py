import numpy as np

import xphd
from conftest import hex_vecs


def test_archive_loads_and_grids_shuffled_q(archive):
    a = xphd.ExcPhArchive(archive)
    assert (a.n1, a.n2) == (12, 12)
    assert a.grid("g2").shape == (12, 12, 6, 4, 4)


def test_integrity_checks(archive):
    c = xphd.ExcPhArchive(archive).check(verbose=False)
    assert c["g2_vs_G2"] < 1e-12
    assert c["Ge_plus_Gh"] < 1e-12
    assert c["E_n_vs_E_m0"] < 1e-12
    assert c["g2_acoustic_at_Gamma"] == 0.0


def test_linewidth_thermal_structure(archive):
    a = xphd.ExcPhArchive(archive)
    r = xphd.compute(a, [10.0, 300.0], refine=5, verbose=False)
    assert np.all(r.total >= 0)
    assert r.absorption[0].max() < 0.02 * r.emission[0].max()
    assert np.all(r.absorption[1] > r.absorption[0])
    assert np.all(r.total[1] > r.total[0])


def _bte_set(tmp_path, n=4, nm=3, ne=3, seed=0):
    import numpy as np
    i = np.arange(n)
    kx, ky = np.meshgrid(i / n, i / n, indexing="ij")
    c = (np.cos(2 * np.pi * kx) + np.cos(2 * np.pi * ky)
         + np.cos(2 * np.pi * (kx + ky))) / 3
    rng = np.random.default_rng(seed)
    qr = np.stack([kx.ravel(), ky.ravel(), np.zeros(n * n)], 1)
    E0 = np.array([2.40, 2.45, 2.52])
    for a in range(n):
        for b in range(n):
            sx, sy = (np.arange(n) + a) % n, (np.arange(n) + b) % n
            Em = E0[None, None, :] - 0.03 * (c[np.ix_(sx, sy)] - 1)[..., None]
            hw = (np.array([0., .018, .092])[None, None, :]
                  + 0.006 * (1 - c)[..., None])
            g2 = np.abs(rng.standard_normal((n, n, nm, ne, ne))) * 4e-6
            g2[0, 0, :1] = 0.0
            np.savez(tmp_path / f"Q{a}{b}.npz",
                     g2_grid=g2.reshape(n * n, nm, ne, ne),
                     E_m_grid=Em.reshape(n * n, ne),
                     hw_grid=hw.reshape(n * n, nm), E_n_grid=Em[0, 0],
                     q_red=qr, Q_red=np.array([a / n, b / n, 0.0]),
                     mesh=np.array([n, n, 1]),
                     osc_Q0=np.array([1.0, 0.02, 0.5]))
    return tmp_path / "Q*.npz"


def test_triangle_weights_share_the_kernel():
    """Both entry points must give identical weights, flat branch included."""
    import numpy as np
    from xphd.tetra import _tri_weights, triangle_weights
    rng = np.random.default_rng(0)
    v = np.sort(rng.uniform(-0.05, 0.05, size=(3, 400)), axis=0)
    for fw in (1e-14, 1e-3):
        got = triangle_weights(v, area=1 / 288, flat_width=fw)
        ref = _tri_weights(v.T, 1 / 288, fw).T
        assert np.allclose(got, ref)


def test_bte_detailed_balance_and_conservation(tmp_path):
    import numpy as np
    import xphd
    from xphd import bte
    arcs = xphd.load_archives(_bte_set(tmp_path), verbose=False)
    rm = bte.build_rates(arcs, T=300.0, verbose=False)
    assert np.abs(rm.P - rm.Q.T).max() < 1e-12 * max(rm.P.max(), 1e-30)
    assert np.abs(rm.R + rm.R.T).max() < 1e-12 * max(abs(rm.R).max(), 1e-30)
    r = bte.check_balance(rm, N_tot=1e-3, verbose=False)
    assert r["bose"] < 1e-10


def test_bte_conserves_particles_under_propagation(tmp_path):
    import numpy as np
    import xphd
    from xphd import bte
    arcs = xphd.load_archives(_bte_set(tmp_path), verbose=False)
    rm = bte.build_rates(arcs, T=300.0, verbose=False)
    F0 = np.zeros(len(rm.E))
    F0[:3] = bte.inject_optical(arcs[(0, 0)].E_n,
                                np.array([1.0, 0.02, 0.5]), 1e-3,
                                verbose=False)
    sol = bte.propagate(rm, F0, t_end=2000.0, nt=5)
    N = sol.y.sum(axis=0)
    assert np.abs(N / N[0] - 1).max() < 1e-6
    assert sol.y.min() > -1e-12


def test_inject_optical_follows_oscillator_strength():
    import numpy as np
    from xphd import bte
    E = np.array([2.40, 2.45, 2.52])
    F = bte.inject_optical(E, np.array([1.0, 0.0, 0.5]), 1.0, verbose=False)
    assert F[1] == 0.0
    assert np.isclose(F.sum(), 1.0)
    assert np.isclose(F[0] / F[2], 2.0)
    Fp = bte.inject_optical(E, np.array([1.0, 0.0, 0.5]), 1.0,
                            center=2.40, sigma=0.01, verbose=False)
    assert Fp[0] > 0.99


def _hex_ops():
    """C6 rotations in reduced coordinates on a hexagonal lattice."""
    import numpy as np
    C6 = np.array([[1, 1, 0], [-1, 0, 0], [0, 0, 1]], float)
    ops, R = [], np.eye(3)
    for _ in range(6):
        ops.append(R.copy())
        R = C6 @ R
    return ops


def test_sweep_collects_every_q(tmp_path):
    import numpy as np
    import xphd
    pat = _bte_set(tmp_path, n=4, ne=3)
    fld = xphd.sweep(str(pat), [10.0, 300.0], verbose=False)
    assert fld.LW.shape == (2, 16, 3)
    assert fld.Q_red.shape == (16, 3)
    assert np.all(fld.LW >= 0)
    assert len({tuple(np.round(q, 6)) for q in fld.Q_red}) == 16


def test_star_map_partitions_the_zone():
    import numpy as np
    from xphd.symmetry import star_map
    n = 6
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    Q = np.stack([i.ravel() / n, j.ravel() / n, np.zeros(n * n)], 1)
    reps, owner = star_map(Q, _hex_ops())
    assert (owner >= 0).all()
    assert set(owner) == set(reps)
    assert len(reps) < n * n
    assert owner[0] == 0


def test_unfold_removes_within_star_noise():
    import numpy as np
    from xphd.symmetry import star_map, unfold, violation
    n = 6
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    Q = np.stack([i.ravel() / n, j.ravel() / n, np.zeros(n * n)], 1)
    ops = _hex_ops()
    reps, owner = star_map(Q, ops)

    clean = np.linspace(1.0, 2.0, len(reps))[
        np.searchsorted(reps, owner)]
    assert violation(clean, Q, ops) < 1e-12

    rng = np.random.default_rng(0)
    noisy = clean * (1 + 0.05 * rng.standard_normal(len(clean)))
    assert violation(noisy, Q, ops) > 0.01
    assert violation(unfold(noisy, Q, ops, verbose=False), Q, ops) < 1e-12
