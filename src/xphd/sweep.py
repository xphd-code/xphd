"""Linewidths at every Q in the zone."""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np

from . import parallel
from .io.excph import ExcPhArchive
from .linewidth import compute

__all__ = ["LinewidthField", "sweep"]


@dataclass
class LinewidthField:
    """Gamma(Q) over the zone."""
    Q_red: np.ndarray
    T: np.ndarray
    LW: np.ndarray
    E_n: np.ndarray
    meta: dict = field(default_factory=dict)
    mode_em: np.ndarray | None = None
    mode_ab: np.ndarray | None = None

    def dominant_mode(self, it=0):
        """(mode, is_emission, fraction) per (Q, state), for colouring bands."""
        if self.mode_em is None:
            return None
        tot = self.mode_em[it] + self.mode_ab[it]
        best = np.argmax(tot, axis=2)
        q, n = np.ogrid[:tot.shape[0], :tot.shape[1]]
        frac = tot[q, n, best] / np.maximum(tot.sum(axis=2), 1e-300)
        return best, self.mode_em[it][q, n, best] > self.mode_ab[it][q, n, best], frac

    @property
    def nq(self):
        return self.LW.shape[1]

    @property
    def nexc(self):
        return self.LW.shape[2]

    def trace(self, it=0):
        """sum over exciton states: basis-independent, so the right thing to test for symmetry."""
        return self.LW[it].sum(axis=1)

    def save(self, path, legacy=False, it=None):
        """Write the field."""
        extra = {}
        if legacy:
            k = 0 if it is None else int(it)
            if len(self.T) > 1 and it is None:
                print(f"  [legacy] LW written for T = {self.T[0]:.1f} K "
                      f"(pass it= to choose another of {len(self.T)})")
            extra = dict(Qx=self.Q_red[:, 0], Qy=self.Q_red[:, 1],
                         LW=self.LW[k].T * 1e3, T=float(self.T[k]))
        native = dict(Q_red=self.Q_red, E_n=self.E_n)
        if self.mode_em is not None:
            native["mode_em"] = self.mode_em
            native["mode_ab"] = self.mode_ab
        if legacy:
            native["LW_eV"] = self.LW
            native["T_all"] = self.T
        else:
            native["LW"] = self.LW
            native["T"] = self.T
        np.savez(path, **native,
                 **{k: np.asarray(v) for k, v in self.meta.items()}, **extra)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        lw = d["LW_eV"] if "LW_eV" in d.files else d["LW"]
        T = d["T_all"] if "T_all" in d.files else np.atleast_1d(d["T"])
        skip = {"Q_red", "T", "LW", "E_n", "LW_eV", "T_all", "Qx", "Qy",
                "mode_em", "mode_ab"}
        return cls(d["Q_red"], T, lw, d["E_n"],
                   {k: d[k] for k in d.files if k not in skip},
                   d["mode_em"] if "mode_em" in d.files else None,
                   d["mode_ab"] if "mode_ab" in d.files else None)

    def report(self, it=0, top=10):
        g = self.LW[it]
        print(f"T = {self.T[it]:.1f} K, {self.nq} Q-points, "
              f"{self.nexc} states  [FWHM, meV]")
        print(f"   {'state':>6} {'min':>10} {'median':>10} {'max':>10}")
        for n in range(min(self.nexc, top)):
            v = g[:, n] * 1e3
            print(f"   {n+1:6d} {v.min():10.3f} {np.median(v):10.3f} "
                  f"{v.max():10.3f}")


def _worker(args):
    path, cfg = args
    arc = ExcPhArchive(path, mesh=cfg.get("mesh"))
    hw = np.load(cfg["hw_fine"]) if cfg.get("hw_fine") else None
    res = compute(arc, cfg["T"], hw_fine=hw, refine=cfg["refine"],
                  scheme=cfg["scheme"], flat_width=cfg.get("flat_width"),
                  acoustic_cut=cfg["acoustic_cut"],
                  acoustic_model=cfg.get("acoustic_model", "none"),
                  n_shell=cfg.get("n_shell", 3),
                  band_interp=cfg.get("band_interp", "local"), verbose=False)
    return (np.asarray(arc.Q_red, float), res.total, arc.E_n,
            res.mode_em, res.mode_ab)


def sweep(paths, temperatures, hw_fine=None, refine=None, fine=None,
          scheme="const", flat_width=None, acoustic_cut=1e-4,
          mesh=None, workers=1, expand_to=None, verbose=True,
          acoustic_model="none", n_shell=3, band_interp="local",
          checkpoint=None):
    """Linewidths for every archive matching `paths`."""
    if isinstance(paths, str):
        files = sorted(glob.glob(paths))
    else:
        files = [str(p) for p in paths]
    if not files:
        raise SystemExit(f"no files matching {paths}")

    probe = ExcPhArchive(files[0], mesh=mesh)
    r = refine or 1
    if hw_fine is not None:
        hw = np.load(hw_fine, mmap_mode="r")
        N = int(round(np.sqrt(hw.shape[0]))) if hw.ndim == 2 else hw.shape[0]
        if N % probe.n1:
            raise SystemExit(f"fine mesh {N} is not a multiple of the source "
                             f"mesh {probe.n1}")
        r = N // probe.n1
    elif fine:
        r = fine // probe.n1

    T = np.atleast_1d(np.asarray(temperatures, float))
    cfg = dict(T=T, refine=r, scheme=scheme, flat_width=flat_width,
               acoustic_cut=acoustic_cut, hw_fine=hw_fine, mesh=mesh,
               acoustic_model=acoustic_model, n_shell=n_shell,
               band_interp=band_interp)

    if verbose and parallel.is_root():
        print(f"{len(files)} Q-points, source {probe.n1}x{probe.n2} -> fine "
              f"{probe.n1*r}x{probe.n2*r}, scheme={scheme}, acoustic model "
              f"{acoustic_model}" + (f" ({n_shell} shells)"
                                     if acoustic_model != "none" else ""))
        if acoustic_model == "none":
            print("  [note] acoustic model 'none': a state at a band minimum "
                  "loses its only\n         channel, small-q acoustic "
                  "absorption. Use --acoustic-model q2.")
        if scheme == "fft":
            print("  [warn] scheme='fft' interpolates |G|^2, which rings at "
                  "the symmetry nodes.\n         'const' is the defensible "
                  "choice; see the README.")

    keys = None
    if checkpoint is not None:
        tag = parallel.config_key("sweep", {k: (v.tolist() if
                                                isinstance(v, np.ndarray)
                                                else v)
                                            for k, v in cfg.items()})
        keys = [f"lw-{os.path.basename(f)}-{tag}" for f in files]

    out = parallel.pmap(_worker, [(f, cfg) for f in files], workers=workers,
                        checkpoint=checkpoint, keys=keys, desc="linewidths",
                        verbose=verbose)
    if out is None:
        return None

    Q = np.array([o[0] for o in out])
    LW = np.stack([o[1] for o in out], axis=1)
    E_n = np.array([o[2] for o in out])
    mode_em = mode_ab = None
    if all(o[3] is not None for o in out):
        mode_em = np.stack([o[3] for o in out], axis=1)
        mode_ab = np.stack([o[4] for o in out], axis=1)

    if expand_to is not None and len(files) < probe.n1 * probe.n2:
        from .symmetry import expand, full_mesh, load_ops
        ops = load_ops(expand_to) if isinstance(expand_to, str) else expand_to
        Qf = full_mesh(probe.n1, probe.n2)
        LW = np.stack([expand(LW[it], Q, Qf, ops, verbose=(it == 0))
                       for it in range(len(T))])
        if mode_em is not None:
            mode_em = np.stack([expand(mode_em[it], Q, Qf, ops, verbose=False)
                                for it in range(len(T))])
            mode_ab = np.stack([expand(mode_ab[it], Q, Qf, ops, verbose=False)
                                for it in range(len(T))])
        E_n = expand(E_n, Q, Qf, ops, verbose=False)
        Q = Qf

    return LinewidthField(Q, T, LW, E_n,
                          dict(mesh=np.array([probe.n1, probe.n2]), refine=r,
                               scheme=scheme, acoustic_cut=acoustic_cut,
                               acoustic_model=acoustic_model,
                               n_shell=n_shell, band_interp=band_interp),
                          mode_em, mode_ab)
