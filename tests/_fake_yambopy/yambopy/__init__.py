"""A stand-in for yambopy, just large enough to drive `xphd generate`."""
import os
import numpy as np

N = 3
NB0, NV, NC = 2, 2, 2
NM = 3


def _kred():
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return np.stack([i.ravel() / N, j.ravel() / N, np.zeros(N * N)], 1)


class YamboLatticeDB:
    @classmethod
    def from_db_file(cls, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        self = cls()
        k = _kred()
        self.car_kpoints = k.copy()
        self.red_kpoints = k.copy()
        self.ibz_kpoints = k.copy()
        self.rlat = np.eye(3)
        self.lat = np.eye(3)
        self.kpoints_indexes = np.arange(N * N)
        self.symmetry_indexes = np.zeros(N * N, int)
        self.sym_car = np.array([np.eye(3)])
        self.time_rev = 0
        return self


class LetzElphElectronPhononDB:
    def __init__(self, path, read_all=True):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        self.kpoints = _kred()
        self.qpoints = _kred()
        self.nq = N * N
        q = self.qpoints
        c = (np.cos(2 * np.pi * q[:, 0]) + np.cos(2 * np.pi * q[:, 1])) / 2
        self.ph_energies = (np.array([0.0, 0.0, 0.06])[None, :]
                            + 0.01 * (1 - c)[:, None] + 0.001)

    def read_iq(self, iq, bands_range, convention="standard"):
        nb = bands_range[1] - bands_range[0]
        rng = np.random.default_rng(1000 + iq)
        em = (rng.standard_normal((N * N, NM, 1, nb, nb))
              + 1j * rng.standard_normal((N * N, NM, 1, nb, nb))) * 1e-3
        return None, em.astype(np.complex64)


def build_ktree(k):
    return None
