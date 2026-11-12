import os
import re
import numpy as np
from yambopy import N, NB0, NV, NC


class YamboExcitonDB:
    @classmethod
    def from_db_file(cls, lat, filename, Load_WF=True, neigs=-1):
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        iq = int(re.search(r"Q(\d+)$", filename).group(1)) - 1
        self = cls()
        n = neigs if neigs > 0 else 6
        self.unique_vbands = np.arange(NB0, NB0 + NV)
        self.unique_cbands = np.arange(NB0 + NV, NB0 + NV + NC)
        self.eigenvalues = (3.0 + 0.01 * iq + 0.02 * np.arange(n)).astype(complex)
        rng = np.random.default_rng(iq)
        self._A = (rng.standard_normal((n, N * N, NV, NC))
                   + 1j * rng.standard_normal((n, N * N, NV, NC)))
        return self

    def get_Akcv(self):
        return self._A
