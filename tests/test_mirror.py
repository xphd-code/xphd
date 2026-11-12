"""sigma_h permutation, non-planar phonon parity and spinor band labels."""
import numpy as np
import pytest

from xphd.mirror import band_labels, phonon_parity, sigma_h_permutation

A = 3.3
CELL = np.array([[A, 0, 0], [A / 2, A * np.sqrt(3) / 2, 0], [0, 0, 20.0]])


def tmd(wrap=False):
    """1H MoS2."""
    se = np.array([A / 2, A / (2 * np.sqrt(3)), 0.0])
    pos = np.array([[0, 0, 10.0], se + [0, 0, 11.67], se + [0, 0, 8.33]])
    if wrap:
        pos[2] += CELL[0]
    return pos, np.array([42, 16, 16])


@pytest.mark.parametrize("wrap", [False, True])
def test_tmd_permutation_swaps_the_chalcogen_pair(wrap):
    pos, sp = tmd(wrap)
    assert list(sigma_h_permutation(pos, sp, CELL)) == [0, 2, 1]


@pytest.mark.parametrize("name, vec, want", [
    ("A1' (S breathing)",       [[0, 0, 0], [0, 0, 1], [0, 0, -1]], +1),
    ("A2'' (Mo against S, z)",   [[0, 0, 1], [0, 0, -.5], [0, 0, -.5]], -1),
    ("E'' (S antiphase, x)",    [[0, 0, 0], [1, 0, 0], [-1, 0, 0]], -1),
    ("E' (Mo against S, x)",     [[1, 0, 0], [-.5, 0, 0], [-.5, 0, 0]], +1),
    ("ZA (all along z)",         [[0, 0, 1], [0, 0, 1], [0, 0, 1]], -1),
    ("LA (all along x)",         [[1, 0, 0], [1, 0, 0], [1, 0, 0]], +1),
])
def test_tmd_textbook_parities(name, vec, want):
    pos, sp = tmd()
    perm = sigma_h_permutation(pos, sp, CELL)
    e = np.array(vec, complex) * np.exp(0.7j)
    assert np.isclose(phonon_parity(e, perm), want), name


def test_planar_layer_reduces_to_out_of_plane_weight():
    pos = np.array([[0, 0, 10.0], [A / 2, A / (2 * np.sqrt(3)), 10.0]])
    perm = sigma_h_permutation(pos, [31, 7], CELL)
    assert list(perm) == [0, 1]
    rng = np.random.default_rng(1)
    e = rng.normal(size=(50, 6, 2, 3)) + 1j * rng.normal(size=(50, 6, 2, 3))
    wz = np.sum(np.abs(e[..., 2]) ** 2, -1) / np.sum(np.abs(e) ** 2, (-2, -1))
    assert np.allclose(phonon_parity(e, perm), 1 - 2 * wz)


def test_buckled_layer_has_no_mirror():
    pos = np.array([[0, 0, 10.0], [A / 2, A / (2 * np.sqrt(3)), 10.5]])
    with pytest.raises(ValueError):
        sigma_h_permutation(pos, [31, 7], CELL)


def test_band_labels_spinless_and_spinor():
    lab, kind, ok = band_labels(np.array([1, -1, 1, 1]) + 0j)
    assert kind == "spinless" and list(lab) == [1, -1, 1, 1] and ok.all()
    lam = np.array([1j, -1j, 1j])
    lab, kind, ok = band_labels(lam)
    assert kind == "spinor" and list(lab) == [1, -1, 1] and ok.all()
    for c in range(3):
        for v in range(3):
            assert np.isclose(lam[c] * np.conj(lam[v]), lab[c] * lab[v])


def test_parity_from_D_spinless_unchanged_and_spinor_labels():
    from xphd.band_parity import parity_from_D
    D = np.zeros((3, 2, 2), complex)
    D[:, 0, 0], D[:, 1, 1] = 1, -1
    p, off, ok = parity_from_D(D)
    assert np.allclose(p, [[1, -1]] * 3) and ok.all()
    D = np.zeros((2, 2, 2), complex)
    D[0] = np.diag([1j, -1j])
    D[1] = np.array([[0.36j, 0.93], [-0.93, -0.36j]])
    p, off, ok = parity_from_D(D)
    assert np.allclose(p[0], [1, -1]) and ok[0].all()
    assert not ok[1].any()
