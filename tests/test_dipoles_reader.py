"""read_cv_dipoles against planted ndb.dipoles files (needs yambopy)."""
import numpy as np
import pytest

yambopy = pytest.importorskip("yambopy")
netCDF4 = pytest.importorskip("netCDF4")
from yambopy.dbs.latticedb import YamboLatticeDB

from xphd.dipoles import read_cv_dipoles

NOCC, NK = 9, 4


def planted(v_band, c_band, i, k):
    return 1000 * v_band + 10 * c_band + i + 1j * (k + 1)


def lattice(spinor=False):
    """A lattice stub with NOCC occupied bands, spinless or spinor."""
    lat = YamboLatticeDB.__new__(YamboLatticeDB)
    lat.spinor_components = 2 if spinor else 1
    lat.nelectrons = NOCC if spinor else 2 * NOCC
    return lat


def write(path, bmin, bmax, ordered, lim_v=NOCC, lim_c=NOCC + 1):
    """ndb.dipoles as yambo writes it: without DipBandsAll the valence axis runs bmin..lim_v and
    the conduction axis lim_c..bmax (PARS[2], PARS[3]); with it both axes run bmin..bmax."""
    n1, n2 = (lim_v - bmin + 1, bmax - lim_c + 1) if ordered else (bmax - bmin + 1,) * 2
    with netCDF4.Dataset(path, "w") as db:
        for n, s in (("four", 4), ("one", 1), ("pars", 9), ("sp", 1),
                     ("nk", NK), ("n1", n1), ("n2", n2), ("x", 3), ("c", 2)):
            db.createDimension(n, s)
        db.createVariable("HEAD_R_LATT", "f8", ("four",))[:] = [1, 1, NK, NK]
        db.createVariable("SPIN_VARS", "f8", ("one",))[:] = [1]
        db.createVariable("PARS", "f8", ("pars",))[:] = [
            bmin, bmax, lim_v, lim_c, 0, 0, 0, 0, int(ordered)]
        a = np.zeros((1, NK, n1, n2, 3, 2))
        for k in range(NK):
            for b1 in range(n1):
                for b2 in range(n2):
                    vb = bmin + b1
                    cb = (lim_c + b2) if ordered else (bmin + b2)
                    for i in range(3):
                        z = planted(vb, cb, i, k)
                        a[0, k, b1, b2, i] = [z.real, z.imag]
        db.createVariable("DIP_iR", "f8",
                          ("sp", "nk", "n1", "n2", "x", "c"))[:] = a


def want():
    return np.array([[[[planted(7 + v, NOCC + 1 + c, i, k) for v in range(3)]
                       for c in range(3)] for i in range(3)] for k in range(NK)])


@pytest.mark.parametrize("bmin, ordered", [(7, False), (1, False), (7, True)])
def test_reader_returns_the_bse_window(tmp_path, bmin, ordered):
    """DipBandsAll from band 7 is the case yambopy's own reader rejects as 'open-shell'; every
    returned element must be the one planted there."""
    path = tmp_path / "ndb.dipoles"
    write(path, bmin, 14, ordered)
    d = read_cv_dipoles(lattice(), str(path), [7, 12])
    assert d.shape == (NK, 3, 3, 3)
    assert np.array_equal(d, want())


@pytest.mark.parametrize("lim_v, lim_c", [(NOCC + 1, NOCC), (NOCC, NOCC),
                                          (NOCC + 1, NOCC + 1), (NOCC, 1)])
@pytest.mark.parametrize("spinor", [False, True])
def test_limits_off_the_gap(tmp_path, lim_v, lim_c, spinor):
    """yambo's limits off the gap: the conduction block of the file starts at lim_c, which may
    lie below the top valence band -- at band 1 in the MoS2 spin-orbit database (v 1-26, c
    1-200), where reading it as starting above the gap took bands 1-4 for the conduction
    bands 27-30."""
    path = tmp_path / "ndb.dipoles"
    write(path, 1, 14, True, lim_v, lim_c)
    d = read_cv_dipoles(lattice(spinor), str(path), [7, 12])
    assert np.array_equal(d, want())


def test_window_outside_the_database_stops(tmp_path):
    """A conduction block starting above the window's first conduction band cannot hold it:
    stop, do not return a shifted block."""
    path = tmp_path / "ndb.dipoles"
    write(path, 1, 14, True, NOCC + 1, NOCC + 2)
    with pytest.raises(SystemExit):
        read_cv_dipoles(lattice(), str(path), [7, 12])
