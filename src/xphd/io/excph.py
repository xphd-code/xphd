"""Reader for exciton-phonon archives produced by ``xphd-generate``."""
from __future__ import annotations

import numpy as np

from ..core.mesh import mesh_indices, to_grid

__all__ = ["ExcPhArchive", "load_archives"]

_ALIASES = {
    "G": ("G_grid", "G", "Gexcph"),
    "g2": ("g2_grid", "g2"),
    "Ge": ("Ge_grid", "Ge"),
    "Gh": ("Gh_grid", "Gh"),
    "E_n": ("E_n_grid", "E_n"),
    "E_m": ("E_m_grid", "E_m"),
    "hw": ("hw_grid", "hw"),
    "q_red": ("q_red", "qpts", "qpoints"),
    "Q_red": ("Q_red", "Q"),
}


_SMALL = 1 << 20


class ExcPhArchive:
    """Lazily-gridded view of one exciton-phonon archive at fixed Q."""

    def __init__(self, path, mesh=None):
        self.path = str(path)
        self._cache: dict[str, np.ndarray] = {}
        self._small: dict[str, np.ndarray] = {}
        with np.load(self.path, allow_pickle=True) as d:
            self._files = tuple(d.files)
            m = np.asarray(d["mesh"]).ravel() if "mesh" in d.files else None

        if mesh is not None:
            self.n1, self.n2 = int(mesh[0]), int(mesh[1])
        elif m is not None:
            self.n1, self.n2 = int(m[0]), int(m[1])
        else:
            raise ValueError(f"{self.path} has no 'mesh'; pass mesh=(n1, n2)")

        self.q_red = np.asarray(self._raw("q_red"), float)
        self._i, self._j = mesh_indices(self.q_red, self.n1, self.n2)

    def _read(self, key):
        if key in self._small:
            return self._small[key]
        with np.load(self.path, allow_pickle=True) as d:
            a = d[key]
        if getattr(a, "nbytes", 0) <= _SMALL:
            self._small[key] = a
        return a

    def _key(self, name):
        for key in _ALIASES.get(name, (name,)):
            if key in self._files:
                return key
        return None

    def _raw(self, name):
        key = self._key(name)
        if key is None:
            raise KeyError(f"{self.path} has no '{name}' "
                           f"(tried {_ALIASES.get(name, (name,))})")
        return self._read(key)

    def has(self, name) -> bool:
        return self._key(name) is not None

    def grid(self, name):
        """Array scattered onto the (n1, n2, ...) mesh, cached."""
        if name not in self._cache:
            a = np.asarray(self._raw(name))
            self._cache[name] = (a if a.shape[:2] == (self.n1, self.n2)
                                 else to_grid(a, self._i, self._j,
                                              self.n1, self.n2))
        return self._cache[name]

    @property
    def E_n(self):
        return np.asarray(self._raw("E_n"), float)

    @property
    def Q_red(self):
        return np.asarray(self._raw("Q_red"), float).ravel()

    @property
    def nexc(self) -> int:
        return int(self.grid("g2").shape[-1])

    @property
    def nmod(self) -> int:
        return int(self.grid("g2").shape[2])

    @property
    def convention(self):
        k = "elph_convention"
        return str(self._read(k)) if k in self._files else None

    def __repr__(self):
        return (f"<ExcPhArchive {self.n1}x{self.n2}, {self.nexc} excitons, "
                f"{self.nmod} branches, Q={np.round(self.Q_red, 4)}>")

    def check(self, verbose: bool = True):
        """Internal-consistency checks."""
        out = {}

        if self.has("G"):
            ref = np.abs(np.asarray(self._raw("G"))) ** 2
            g2 = np.asarray(self._raw("g2"), float)
            out["g2_vs_G2"] = (float(np.abs(ref - g2).max()
                                     / max(g2.max(), 1e-300))
                               if ref.shape == g2.shape else np.inf)

        if self.has("Ge") and self.has("Gh") and self.has("G"):
            G = np.asarray(self._raw("G"))
            s = np.asarray(self._raw("Ge")) + np.asarray(self._raw("Gh"))
            out["Ge_plus_Gh"] = float(np.abs(s - G).max()
                                      / max(np.abs(G).max(), 1e-300))

        out["E_n_vs_E_m0"] = float(np.abs(self.grid("E_m")[0, 0]
                                          - self.E_n).max())

        g2g = self.grid("g2")
        out["g2_acoustic_at_Gamma"] = float(g2g[0, 0, :3].max())

        nz = np.sqrt(g2g[g2g > 1e-18])
        out["G_median_meV"] = float(np.median(nz) * 1e3) if nz.size else 0.0
        out["G_max_meV"] = float(np.sqrt(g2g.max()) * 1e3)

        if verbose:
            self._print_check(out)
        return out

    @staticmethod
    def _print_check(out):
        def verdict(ok):
            return "OK" if ok else "*** FAIL ***"

        print("integrity checks")
        if "g2_vs_G2" in out:
            v = out["g2_vs_G2"]
            print(f"  |G|^2 vs g2            {v:10.2e}   {verdict(v < 1e-10)}")
        if "Ge_plus_Gh" in out:
            v = out["Ge_plus_Gh"]
            print(f"  Ge + Gh vs G           {v:10.2e}   {verdict(v < 1e-5)}")
        v = out["E_n_vs_E_m0"]
        print(f"  E_n vs E_m(q=0)   {v*1e3:11.4f} meV   {verdict(v < 1e-6)}")
        v = out["g2_acoustic_at_Gamma"]
        print(f"  acoustic |G|^2 at Gamma {v:9.2e}   {verdict(v < 1e-18)}")
        print(f"  |G| median {out['G_median_meV']:.3f} meV, "
              f"max {out['G_max_meV']:.1f} meV")
        if out["G_median_meV"] > 100:
            print("    median above 100 meV is not physical -- check the "
                  "Ry/Ha conversion in the generator")

    def q_index(self):
        """Integer mesh index (i, j) of this archive's Q."""
        Q = np.mod(self.Q_red, 1.0)
        return (int(np.rint(Q[0] * self.n1)) % self.n1,
                int(np.rint(Q[1] * self.n2)) % self.n2)


def load_archives(pattern, mesh=None, verbose=True):
    """Load every GI_ExcPh_Q*.npz matching `pattern`, keyed by Q mesh index."""
    import glob
    files = sorted(glob.glob(str(pattern)))
    if not files:
        raise SystemExit(f"no files matching {pattern}")
    out = {}
    for f in files:
        a = ExcPhArchive(f, mesh=mesh)
        out[a.q_index()] = a
    if verbose:
        first = next(iter(out.values()))
        print(f"loaded {len(out)} Q-points, mesh {first.n1}x{first.n2}, "
              f"{first.nexc} excitons")
    return out
