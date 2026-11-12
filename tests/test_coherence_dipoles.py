"""xphd coherence --dipoles, rebuilt on xphd.dipoles (review of 4 Nov 2026)."""
import numpy as np
import pytest

pytest.importorskip("yambopy")
pytest.importorskip("netCDF4")

from test_dipoles_reader import NK, lattice, write

from xphd import dipoles


def _lat():
    lat = lattice()
    R = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    lat.sym_car = np.stack([np.eye(3), R, np.eye(3), R])
    lat.time_rev = 1
    lat.kmap = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [1, 1], [2, 2], [3, 3]])
    return lat


def test_full_zone_dipoles(tmp_path):
    path = tmp_path / "ndb.dipoles"
    write(path, 7, 14, False)
    lat = _lat()
    ibz = dipoles.read_cv_dipoles(lat, str(path), [7, 12])
    full, dtype = dipoles.full_zone_dipoles(lat, str(path), [7, 12])
    assert full.shape == (7, 3, 3, 3) and dtype == ibz.dtype
    for k, (j, s) in enumerate(lat.kmap):
        ref = np.einsum("ij,jcv->icv", lat.sym_car[s], ibz[j])
        assert np.allclose(full[k], ref.conj() if s >= 2 else ref), k
