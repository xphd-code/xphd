"""The dipole check: agreement means a constant ratio, not only the same ranking."""
from xphd.dipoles import residual_verdict


def test_constant_ratio_agrees():
    v, spread = residual_verdict([3.0, 2.0, 1.0], [3e-7, 2e-7, 1e-7])
    assert v == "AGREE" and abs(spread - 1) < 1e-9


def test_same_ranking_but_varying_ratio_disagrees():
    mine = [3.1404e-06, 1.8491e-06, 8.0320e-07]
    ref = [5.0026e-08, 1.3298e-08, 3.2489e-09]
    v, spread = residual_verdict(mine, ref)
    assert v.startswith("DISAGREE") and "NOT proportional" in v
    assert 3.9 < spread < 4.0


def test_different_ranking_disagrees():
    v, _ = residual_verdict([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], k=1)
    assert v.startswith("DISAGREE -- different")


import numpy as np

from xphd.dipoles import multiplet_checks


def _pair(rng, d=1.0):
    """An E' pair in a random basis: helicity states (1, +-i) d / sqrt2 mixed by a random
    unitary, as a degenerate solver returns them."""
    hel = d / np.sqrt(2) * np.array([[1, 1], [1j, -1j], [0, 0]])
    u, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    return hel @ u


def test_degenerate_pair_any_basis_passes():
    """Isotropic and equal partners in whatever basis the solver chose."""
    rng = np.random.default_rng(1)
    D = np.concatenate([_pair(rng, 1.0), _pair(rng, 0.7)], axis=1)
    rows, acc, former = multiplet_checks(D, [[0, 1], [2, 3]], np.arange(4.0))
    assert len(rows) == 2 and not acc
    assert max(r["aniso"] for r in rows) < 1e-12
    assert max(r["imbalance"] for r in rows) < 1e-12
    assert former[0] < 1e-12


def test_dark_state_merged_by_the_tolerance():
    """A bright pair and a dark state closer than --deg-tol: the per-state spread of the former
    check is 1, the tensor test is still 0 -- the MoS2 printout, spread 1.0 while its
    brightest pairs were equal to 0.3 %."""
    rng = np.random.default_rng(2)
    D = np.concatenate([_pair(rng), np.zeros((3, 1))], axis=1)
    rows, acc, former = multiplet_checks(D, [[0, 1, 2]], np.zeros(3))
    assert abs(former[0] - 1.0) < 1e-12
    assert rows[0]["aniso"] < 1e-12 and rows[0]["imbalance"] is None
    assert acc == [[0, 1, 2]]


def test_noise_on_a_dark_pair_is_not_judged():
    """An in-plane-dark pair with numerical noise above the old absolute threshold 1e-6 gave
    spread ~1; relative to the brightest it is ignored."""
    rng = np.random.default_rng(3)
    noise = np.zeros((3, 2), complex)
    noise[:2, 0] = 2e-3
    noise[:2, 1] = 1e-5
    D = np.concatenate([_pair(rng), noise], axis=1)
    rows, _, former = multiplet_checks(D, [[0, 1], [2, 3]], np.arange(4.0))
    assert former[0] > 0.99 and former[1] == [2, 3]
    assert [r["states"] for r in rows] == [[0, 1]]


def test_broken_symmetry_and_split_pair_fail():
    """One helicity alone (a pair split by more than --deg-tol) and an anisotropic pair are
    caught."""
    rng = np.random.default_rng(4)
    P = _pair(rng)
    hel = np.array([[1], [1j], [0]]) / np.sqrt(2)
    rows, _, _ = multiplet_checks(hel, [[0]], np.zeros(1))
    assert abs(rows[0]["aniso"] - 1.0) < 1e-12
    Q = P.copy()
    Q[0] *= 1.1
    rows, _, _ = multiplet_checks(Q, [[0, 1]], np.zeros(2))
    assert rows[0]["aniso"] > 0.05


def _mixed(theta, rng):
    """Two E' pairs 0.2 meV apart whose first states the solver mixed by an angle theta: each
    pair alone is anisotropic, the four together are not."""
    P, Q = _pair(rng, 1.0), _pair(rng, 0.8)
    c, s = np.cos(theta), np.sin(theta)
    p1, q1 = c * P[:, 0] + s * Q[:, 0], -s * P[:, 0] + c * Q[:, 0]
    D = np.stack([p1, P[:, 1], q1, Q[:, 1]], axis=1)
    E = np.array([3.0451, 3.0451, 3.0453, 3.0453])
    return D, E


def test_solver_mixing_closes_with_the_neighbours():
    """The MoS2 680-681 case: a pair anisotropic on its own (mixed with the next pair) becomes
    isotropic when that pair joins."""
    from xphd.dipoles import isotropic_cluster
    D, E = _mixed(0.3, np.random.default_rng(5))
    rows, _, _ = multiplet_checks(D, [[0, 1], [2, 3]], E)
    assert max(r["aniso"] for r in rows) > 0.05
    c = isotropic_cluster(D, E, [0, 1])
    assert c["closed"] and c["states"] == [0, 1, 2, 3]
    assert c["aniso"] < 1e-10 and abs(c["span"] - 2e-4) < 1e-9


def test_symmetry_error_does_not_close():
    """Dipoles with x and y unequal everywhere: no cluster becomes isotropic, and the most
    isotropic one met is reported, not the last."""
    from xphd.dipoles import isotropic_cluster
    D, E = _mixed(0.3, np.random.default_rng(6))
    D = D.copy()
    D[0] *= 1.2
    c = isotropic_cluster(D, E, [0, 1])
    assert not c["closed"] and c["stop"] == "edge" and c["aniso"] > 0.05
    assert c["aniso"] <= min(_aniso_of(D, s) for s in ([0, 1], [0, 1, 2], [0, 1, 2, 3])) + 1e-12


def _aniso_of(D, idx):
    lo, hi = np.linalg.eigvalsh(D[:2, idx] @ D[:2, idx].conj().T)
    return (hi - lo) / (hi + lo)


def test_weak_multiplet_borrowing_from_a_distant_bright_state():
    """Why "open" is not "error": a weak pair with a 2 % admixture of a pair 100 times brighter,
    3 meV away, is strongly anisotropic and does not close within 2 meV; the strongest
    neighbour is reported."""
    from xphd.dipoles import isotropic_cluster, strongest_neighbour
    rng = np.random.default_rng(7)
    W, B = _pair(rng, 0.1), _pair(rng, 1.0)
    th = 0.02
    w1, b1 = np.cos(th) * W[:, 0] + np.sin(th) * B[:, 0], -np.sin(th) * W[:, 0] + np.cos(th) * B[:, 0]
    D = np.stack([w1, W[:, 1], b1, B[:, 1]], axis=1)
    E = np.array([3.0, 3.0, 3.003, 3.003])
    rows, _, _ = multiplet_checks(D, [[0, 1], [2, 3]], E)
    weak = [r for r in rows if r["states"] == [0, 1]][0]
    assert weak["aniso"] > 0.1
    c = isotropic_cluster(D, E, [0, 1])
    assert not c["closed"] and c["stop"] == "span"
    n, ratio, de = strongest_neighbour(D, E, [0, 1])
    assert n in (2, 3) and ratio > 50 and abs(de - 3e-3) < 1e-9
    assert isotropic_cluster(D, E, [0, 1], max_span=4e-3)["closed"]
