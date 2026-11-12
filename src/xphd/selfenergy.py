"""Frequency-dependent exciton-phonon self-energy."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core.interp import fine_band, fourier_refine
from .core.stats import HBAR_EV_PS
from .linewidth import FLAT_WIDTH_SCALE, linewidth_one

__all__ = ["SelfEnergy", "kramers_kronig", "self_energy"]


def kramers_kronig(omega, im_sigma):
    """Re Sigma from Im Sigma on a uniform grid."""
    w = np.asarray(omega, float)
    y = np.asarray(im_sigma, float)
    if w.ndim != 1 or y.shape[-1] != w.size:
        raise ValueError("im_sigma's last axis must match omega")
    dw = w[1] - w[0]
    if not np.allclose(np.diff(w), dw, rtol=1e-6):
        raise ValueError("kramers_kronig needs a uniform frequency grid")

    y2 = np.atleast_2d(y)
    out = np.zeros_like(y2)
    for i in range(w.size):
        d = w - w[i]
        m = d != 0.0
        reg = np.zeros_like(y2)
        reg[:, m] = (y2[:, m] - y2[:, i][:, None]) / d[m]
        out[:, i] = reg.sum(axis=1) * dw / np.pi
        if 0 < i < w.size - 1:
            out[:, i] += (y2[:, i + 1] - y2[:, i - 1]) / (2.0 * np.pi)
        lo, hi = w[i] - w[0], w[-1] - w[i]
        if lo > 0 and hi > 0:
            out[:, i] += y2[:, i] * np.log(hi / lo) / np.pi
    return out.reshape(y.shape)


@dataclass
class SelfEnergy:
    """Sigma_lambda(omega) for one state, in eV."""
    omega: np.ndarray
    im: np.ndarray
    re: np.ndarray
    E_bare: float
    T: float
    meta: dict = field(default_factory=dict)

    def spectral(self, eta=1e-3):
        """A(omega) = -(1/pi) Im [omega - E - Sigma]^{-1}, normalised to 1."""
        num = np.abs(self.im) + eta
        den = (self.omega - self.E_bare - self.re) ** 2 + num ** 2
        A = num / (np.pi * den)
        trap = getattr(np, "trapezoid", None) or np.trapz
        return A / max(trap(A, self.omega), 1e-300)

    def quasiparticle(self):
        """Solve E = E_bare + Re Sigma(E)."""
        f = self.omega - self.E_bare - self.re
        s = np.where(np.diff(np.sign(f)) != 0)[0]
        if not len(s):
            return np.nan, np.nan, np.nan
        k = s[np.argmin(np.abs(self.omega[s] - self.E_bare))]
        x0, x1 = self.omega[k], self.omega[k + 1]
        E_qp = x0 - f[k] * (x1 - x0) / (f[k + 1] - f[k])
        dre = np.gradient(self.re, self.omega)
        Z = 1.0 / (1.0 - np.interp(E_qp, self.omega, dre))
        G = -2.0 * np.interp(E_qp, self.omega, self.im)
        return float(E_qp), float(Z), float(G)

    def report(self):
        E_qp, Z, G = self.quasiparticle()
        G_on = -2.0 * float(np.interp(self.E_bare, self.omega, self.im))
        A = self.spectral()
        peak = float(self.omega[np.argmax(A)])
        rs = float(np.interp(self.E_bare, self.omega, self.re))
        print("  -- the two standard quantities ---------------------------")
        print(f"  bare energy        {self.E_bare:12.4f} eV")
        print(f"  LINEWIDTH  Gamma   {G_on*1e3:12.3f} meV   "
              f"tau = {HBAR_EV_PS/max(G_on,1e-30):.4f} ps")
        print(f"  SHIFT  Re Sigma    {rs*1e3:12.3f} meV   "
              f"renormalised E = {self.E_bare + rs:.4f} eV")
        print("     Im Sigma at the BARE energy is the linewidth; Re Sigma is")
        print("     the energy renormalisation, reported separately. This is")
        print("     the standard convention (Chan 2023 Eq. 5; Chen 2020).")
        print("  -- beyond the standard treatment -------------------------")
        print(f"  quasiparticle E    {E_qp:12.4f} eV   "
              f"shift {(E_qp-self.E_bare)*1e3:+.1f} meV")
        print(f"  Z                  {Z:12.4f}")
        print(f"  Gamma at E_qp      {G*1e3:12.3f} meV")
        print(f"  spectral peak      {peak:12.4f} eV")
        print("     Gamma at E_qp is NOT the linewidth. It shifts the initial")
        print("     state while leaving the final states at their bare")
        print("     energies, so for an intraband channel -- where the shift")
        print("     should cancel in E_n - E_m -- it is spurious. Compare")
        print("     RENORMALISED GAPS between states instead: that comparison")
        print("     is consistent and is where a channel opening or closing")
        print("     actually shows up.")
        if not np.isfinite(Z) or Z < 0.5:
            print("\n  Z < 0.5: most of the spectral weight is NOT in the")
            print("  quasiparticle peak. A single linewidth is not a")
            print("  meaningful description of this state; quote the spectral")
            print("  function instead.")
        if abs(G) > 1e-4 and abs(peak - E_qp) > 0.5 * abs(G):
            print("\n  the spectral maximum is displaced from the")
            print("  quasiparticle solution by more than half a width: the")
            print("  lineshape is not Lorentzian here.")


def self_energy(archive, state, T, hw_fine=None, refine=1, window=None,
                n_omega=241, flat_width=None, acoustic_cut=1e-4, pad=0.15,
                band_interp="local", verbose=True):
    """Sigma_lambda(omega) for one initial state."""
    n1, n2 = archive.n1, archive.n2
    r = int(refine)
    fw = flat_width if flat_width is not None else FLAT_WIDTH_SCALE / (n1 * r)

    g2 = archive.grid("g2")[:, :, :, state, :]
    E_m = fine_band(archive.grid("E_m"), r, band_interp)
    hw_c = archive.grid("hw")
    if hw_fine is not None:
        hw = np.asarray(hw_fine, float)
        if hw.ndim == 2:
            hw = hw.reshape(n1 * r, n2 * r, archive.nmod)
    else:
        hw = fourier_refine(hw_c, r)

    E0 = float(archive.E_n[state])
    if window is None:
        wmax = float(hw_c.max())
        lo = float(archive.grid("E_m").min()) - wmax - pad
        hi = float(archive.grid("E_m").max()) + wmax + pad
    else:
        lo, hi = E0 - window, E0 + window
    omega = np.linspace(lo, hi, n_omega)
    dw = omega[1] - omega[0]
    omega = omega + (E0 - omega[np.argmin(np.abs(omega - E0))])
    if verbose:
        print(f"state {state+1}: E = {E0:.4f} eV, scanning "
              f"{lo:.3f} .. {hi:.3f} eV in {n_omega} steps (refine {r})"
              + ("   [auto window]" if window is None else ""))

    im = np.empty(n_omega)
    for k, w in enumerate(omega):
        im[k] = -0.5 * linewidth_one(w, E_m, hw, g2, r, T, fw,
                                     acoustic_cut=acoustic_cut)[0]
        if verbose and n_omega >= 20 and (k + 1) % (n_omega // 10) == 0:
            print(f"  {k+1}/{n_omega}", end="\r")

    edge = max(abs(im[0]), abs(im[-1])) / max(abs(im).max(), 1e-300)
    if verbose:
        print(f"  grid spacing {dw*1e3:.2f} meV, E_bare on a node")
        print(f"  |Im Sigma| at the window edges: {edge:.2%} of its maximum")
        g_on = -2.0 * im[np.argmin(np.abs(omega - E0))]
        if 0 < abs(g_on) < 3 * dw:
            print(f"    on-shell Gamma ({abs(g_on)*1e3:.3f} meV) is below "
                  f"3x the grid spacing;\n    raise --n-omega before reading "
                  f"the quasiparticle solution.")
        if edge > 0.05:
            print("    truncated: Kramers-Kronig will pick up the cut. Use "
                  "--window 0\n    for the automatic range, or widen it.")

    re = kramers_kronig(omega, im)
    se = SelfEnergy(omega, im, re, E0, T,
                    dict(refine=r, flat_width=fw, edge=edge,
                         acoustic_cut=acoustic_cut))
    if verbose:
        print()
        se.report()
    return se
