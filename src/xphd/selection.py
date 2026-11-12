#!/usr/bin/env python3
"""selection_rule_test.py ====================== The exciton-phonon selection rule, tested
directly."""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

from .io.excph import ExcPhArchive

_have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({
    "font.size": 12,
    "font.family": "Montserrat" if "Montserrat" in _have else "DejaVu Sans",
    "mathtext.fontset": "stix", "axes.linewidth": 1.6,
})

ROT = np.radians(-30.0)
CR, SR = np.cos(ROT), np.sin(ROT)


def to_rot(qf):
    x = qf[:, 0] + 0.5 * qf[:, 1]
    y = (np.sqrt(3.0) / 2.0) * qf[:, 1]
    return x * CR - y * SR, x * SR + y * CR


def grid(Q, data, res=350, limit=0.8, method="cubic"):
    w = Q[:, :2] - np.rint(Q[:, :2])
    qx, qy, dd = [], [], []
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            qx.append(w[:, 0] + dx); qy.append(w[:, 1] + dy); dd.append(data)
    kx, ky = to_rot(np.column_stack([np.concatenate(qx), np.concatenate(qy)]))
    dd = np.concatenate(dd)
    xi = np.linspace(-limit, limit, res)
    X, Y = np.meshgrid(xi, xi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = griddata((kx, ky), dd, (X, Y), method=method)
        if np.isnan(Z).any():
            Z = np.where(np.isnan(Z),
                         griddata((kx, ky), dd, (X, Y), "nearest"), Z)
    return X, Y, Z


def zone(ax, color="white", lw=1.5):
    v = np.array([[1/3, 1/3], [-1/3, 2/3], [-2/3, 1/3],
                  [-1/3, -1/3], [1/3, -2/3], [2/3, -1/3]])
    vx, vy = to_rot(v)
    ax.add_patch(patches.Polygon(np.column_stack([vx, vy]), closed=True,
                                 fill=False, edgecolor=color, lw=lw,
                                 zorder=10))


def panel(ax, Q, dat, vmax, title, cmap="afmhot", limit=0.8):
    X, Y, Z = grid(Q, dat, limit=limit)
    im = ax.pcolormesh(X, Y, np.clip(Z, 0, vmax), cmap=cmap, vmin=0,
                       vmax=vmax, shading="auto", zorder=1, rasterized=True)
    zone(ax)
    ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=11, pad=5)
    return im


def _rotated_split(arc, d, idx, S, nexc, p_nu, ok_nu, p_lam, tol):
    """Allowed / forbidden / undefined |G|^2 per q, split by the parity eigenstates of each
    near-degenerate multiplet of final states."""
    from .multiplet import split_by_parity
    need = ("sigma_ibz", "group_ibz", "trev", "iQ_ibz")
    if not all(k in d.files for k in need):
        raise SystemExit("--rotate needs the multiplet matrices: rerun "
                         "xphd parity ... --deg-tol 0.005")
    sig, grp, trev, iqi = (d["sigma_ibz"], d["group_ibz"], d["trev"], d["iQ_ibz"])
    if sig.shape[1] < nexc:
        raise SystemExit(f"parity file has {sig.shape[1]} states per Q, the archive {nexc}")
    G = np.asarray(arc.grid("G"))[arc._i, arc._j][:, :, S, :nexc]
    nq = G.shape[0]
    G_all, G_for, G_und = np.zeros(nq), np.zeros(nq), np.zeros(nq)
    mus = []
    for iq in range(nq):
        f = idx[iq]
        M = sig[iqi[f]][:nexc, :nexc]
        if trev[f]:
            M = M.conj()
        mu, W = split_by_parity(G[iq], M, grp[iqi[f]][:nexc], tol)
        mus.append(mu)
        ok_mu = np.abs(mu) > tol
        prod = np.sign(mu)[:, None] * p_nu[iq][None, :]
        good = ok_mu[:, None] & ok_nu[iq][None, :]
        G_all[iq] = W[good & (np.abs(prod - p_lam) < 0.5)].sum()
        G_for[iq] = W[good & (np.abs(prod + p_lam) < 0.5)].sum()
        G_und[iq] = W[~good].sum()
    mus = np.concatenate(mus)
    print(f"   multiplet eigen-parities defined (|mu| > {tol:g}) for "
          f"{100 * np.mean(np.abs(mus) > tol):.1f}% of (q, state); "
          f"within 0.02 of +-1: {100 * np.mean(np.abs(np.abs(mus) - 1) < 0.02):.1f}%")
    return G_all, G_for, G_und

def main(argv=None):
    p = argparse.ArgumentParser(prog="xphd selection-rule")
    p.add_argument("--archive", default="GI_ExcPh_Q0001.npz")
    p.add_argument("--parity", default="parity.npz")
    p.add_argument("--labels", default="mode_labels.npy")
    p.add_argument("--state", type=int, default=1, help="initial exciton")
    p.add_argument("--tol", type=float, default=0.5,
                   help="|chi| below this is treated as undefined and the "
                        "point is excluded rather than forced to a parity")
    p.add_argument("--rotate", action="store_true",
                   help="split the coupling by the parity EIGENSTATES of each "
                        "near-degenerate multiplet of final states (needs a "
                        "parity.npz written with --deg-tol), instead of by "
                        "the per-state chi of the solver's possibly mixed "
                        "states. Nothing is excluded for being mixed; only "
                        "eigenvectors with |mu| < --tol remain undefined")
    p.add_argument("--by-final", action="store_true",
                   help="also map the coupling resolved by final state")
    p.add_argument("--nfinal", type=int, default=4)
    p.add_argument("--modes-for", type=int, default=None,
                   help="final state index; adds one panel per "
                        "phonon branch for that channel")
    p.add_argument("--clip", type=float, default=98.0)
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--out", default="selection_rule.pdf")
    a = p.parse_args(argv)

    arc = ExcPhArchive(a.archive)
    g2 = arc.grid("g2")
    print(f"{a.archive}: g2 {g2.shape}")
    n1, n2, nmod, nexc, _ = g2.shape
    S = a.state - 1

    d = np.load(a.parity)
    Qp, chi = d["Q_red"], d["chi"]
    from .core.mesh import hex_dist
    qgrid = np.mod(arc.q_red, 1.0)
    Q0 = np.mod(arc.Q_red[:2], 1.0)
    dist = hex_dist(np.mod(qgrid[:, None, :2] + Q0, 1.0), np.mod(Qp[None, :, :2], 1.0))
    idx = np.argmin(dist, axis=1)
    worst = float(dist[np.arange(len(qgrid)), idx].max())
    if worst > 1e-5:
        raise SystemExit(f"meshes differ: worst match {worst:.2e}")
    print(f"   matched all {len(qgrid)} q-points to {a.parity} "
          f"(worst {worst:.1e})")

    if chi.shape[1] < nexc:
        raise SystemExit(
            f"{a.parity} holds {chi.shape[1]} states per Q but the archive has "
            f"{nexc}: the final-state parity is needed for every final state. "
            f"Rerun  xphd parity ... --nstates {nexc}")
    p_beta = np.sign(chi[idx][:, :nexc])
    ok_beta = np.abs(chi[idx][:, :nexc]) > a.tol
    print(f"   final-state parity defined at "
          f"{100*ok_beta.mean():.1f}% of (q, state) pairs")

    lab = np.load(a.labels)
    if lab.shape != (len(qgrid), nmod):
        raise SystemExit(
            f"{a.labels} has shape {lab.shape}, the archive needs "
            f"{(len(qgrid), nmod)} (q, branch). For a layer with {nmod} "
            f"branches use scripts/labels_from_elph.py, not xphd mode-labels")
    p_nu = np.where(np.isin(lab, [0, 3]), -1.0, 1.0)
    p_nu[lab < 0] = 0.0
    ok_nu = lab >= 0
    print(f"   phonon parity defined at {100*ok_nu.mean():.1f}% of "
          f"(q, mode) pairs")

    iG = int(np.argmin(hex_dist(qgrid[:, :2], np.zeros(2))))
    p_lam = float(np.sign(chi[idx[iG], S]))
    print(f"   initial state {a.state} at Q = {np.round(Q0, 4)}: chi = "
          f"{chi[idx[iG], S]:+.3f}  -> p = {p_lam:+.0f}")

    g2f = np.zeros((len(qgrid), nmod, nexc))
    g2f[:] = g2[arc._i, arc._j][:, :, S, :]
    prod = p_nu[:, :, None] * p_beta[:, None, :]
    good = ok_nu[:, :, None] & ok_beta[:, None, :]
    allowed = good & (np.abs(prod - p_lam) < 0.5)
    forbid = good & (np.abs(prod + p_lam) < 0.5)

    G_all = np.where(allowed, g2f, 0.0).sum(axis=(1, 2))
    G_for = np.where(forbid, g2f, 0.0).sum(axis=(1, 2))
    G_und = np.where(~good, g2f, 0.0).sum(axis=(1, 2))
    if a.rotate:
        G_all, G_for, G_und = _rotated_split(arc, d, idx, S, nexc, p_nu, ok_nu,
                                             p_lam, a.tol)
    tot = G_all.sum() + G_for.sum() + G_und.sum()
    print(f"\n   SELECTION RULE, initial state {a.state}"
          + ("  (multiplets rotated to sigma_h eigenstates)" if a.rotate else ""))
    print(f"     allowed    {G_all.sum():12.4e}  {100*G_all.sum()/tot:6.2f}%")
    print(f"     forbidden  {G_for.sum():12.4e}  {100*G_for.sum()/tot:6.2f}%")
    print(f"     undefined  {G_und.sum():12.4e}  {100*G_und.sum()/tot:6.2f}%"
          f"   (parity not assignable)")
    r = G_for.sum() / max(G_all.sum(), 1e-30)
    print(f"     forbidden / allowed = {r:.2e}")
    print("     a working rule makes this small; a value near 1 means the"
          "\n     rule is not operating, or a parity assignment is wrong")

    vmax = np.percentile(np.concatenate([G_all, G_for]), a.clip)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1))
    panel(axes[0], arc.q_red, G_all, vmax, "symmetry-allowed")
    im = panel(axes[1], arc.q_red, G_for, vmax, "symmetry-forbidden")
    cax = make_axes_locatable(axes[1]).append_axes("right", size="5%",
                                                  pad=0.08)
    cb = plt.colorbar(im, cax=cax, ticks=[0, vmax])
    cb.set_ticklabels(["0", "max"])
    for ax, t in zip(axes, "ab"):
        ax.text(0.02, 0.98, f"({t})", transform=ax.transAxes, ha="left",
                va="top", fontsize=13, fontweight="bold", color="white")
    fig.subplots_adjust(wspace=0.06)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    print(f"\nsaved {a.out}")

    if not a.by_final:
        return
    print(f"\n   COUPLING BY FINAL STATE (where does state {a.state} go?)")
    LAB = ["ZA", "TA", "LA", "ZO", "TO", "LO"]
    parity_only = set(np.unique(lab)) <= {-1, 1, 3}

    def branch_name(v):
        return {1: "even", 3: "odd"}[int(v)] if parity_only else LAB[int(v)]
    nf = min(a.nfinal, nexc)
    per = [g2f[:, :, m].sum(axis=1) for m in range(nf)]

    def folded_length(q2):
        """|q| in units of |b| for the shortest image of q (60-degree basis)."""
        best = np.inf
        for g1 in (-1, 0, 1):
            for g2_ in (-1, 0, 1):
                v = (q2 - np.rint(q2)) + np.array([g1, g2_])
                best = min(best, float(np.hypot(v[0] + 0.5 * v[1], np.sqrt(3) / 2 * v[1])))
        return best

    for m in range(nf):
        w = per[m]
        j = int(np.argmax(w))
        loc = g2f[j, :, m]
        zone = g2f[:, :, m].sum(axis=0)
        order = np.argsort(loc)[::-1]
        par = ("odd" if p_beta[j, m] < 0 else "even") if ok_beta[j, m] \
            else "undefined"
        print(f"     -> state {m+1}: {100*w.sum()/g2f.sum():5.1f}% of the "
              f"coupling")
        print(f"        peaks at q = {np.round(qgrid[j, :2], 4)}, "
              f"|q| = {folded_length(qgrid[j, :2]):.3f} |b|; the final state "
              f"is {par} there")
        parts = []
        for nu in order[:3]:
            ch = branch_name(lab[j, nu]) if lab[j, nu] >= 0 else "--"
            pp = "odd" if p_nu[j, nu] < 0 else "even"
            if ok_beta[j, m] and ok_nu[j, nu]:
                rule = "allowed" if abs(p_nu[j, nu] * p_beta[j, m] - p_lam) < 0.5 else "FORBIDDEN"
            else:
                rule = "undefined"
            parts.append(f"nu{nu+1} [{ch}, {pp} at this q, {rule}] "
                         f"{100*loc[nu]/max(loc.sum(), 1e-30):.0f}% here, "
                         f"{100*zone[nu]/max(zone.sum(), 1e-30):.0f}% zone-wide")
        print(f"        modes at this q: " + ";\n                         ".join(parts))

    vmax2 = np.percentile(np.concatenate(per), a.clip)
    fig2, ax2 = plt.subplots(1, nf, figsize=(3.2 * nf, 3.1))
    ax2 = np.atleast_1d(ax2)
    for m in range(nf):
        panel(ax2[m], arc.q_red, per[m], vmax2,
              rf"$\lambda={a.state} \to \beta={m+1}$")
    fig2.subplots_adjust(wspace=0.06)
    out2 = a.out.replace(".pdf", "_by_final.pdf")
    fig2.savefig(out2, dpi=a.dpi, bbox_inches="tight")
    print(f"saved {out2}")
    print("   these are intervalley maps: the value at q is the coupling for")
    print("   the exciton at Gamma to scatter INTO that state at momentum q.")
    print("   They show where coupling EXISTS, not where scattering happens:")
    print("   energy conservation still has to be satisfied, and it is what")
    print("   selects the active channels.")

    if a.modes_for:
        m = a.modes_for - 1
        per_mode = [g2f[:, nu, m] for nu in range(nmod)]
        share = np.array([x.sum() for x in per_mode])
        print(f"\n   BY PHONON BRANCH, for {a.state} -> {a.modes_for}")
        for nu in range(nmod):
            frac_odd = (np.where(p_nu[:, nu] < 0, per_mode[nu], 0).sum()
                        / max(share[nu], 1e-30))
            print(f"     nu{nu+1}: {100*share[nu]/max(share.sum(),1e-30):5.1f}%"
                  f" of this channel, {100*frac_odd:5.1f}% of it where the "
                  f"branch is sigma_h-odd")
        vm = np.percentile(np.concatenate(per_mode), a.clip)
        fig3, ax3 = plt.subplots(1, nmod, figsize=(2.6 * nmod, 2.9))
        ax3 = np.atleast_1d(ax3)
        for nu in range(nmod):
            panel(ax3[nu], arc.q_red, per_mode[nu], vm, rf"$\nu={nu+1}$")
        fig3.subplots_adjust(wspace=0.05)
        out3 = a.out.replace(".pdf", f"_modes_b{a.modes_for}.pdf")
        fig3.savefig(out3, dpi=a.dpi, bbox_inches="tight")
        print(f"saved {out3}")


if __name__ == "__main__":
    main()
