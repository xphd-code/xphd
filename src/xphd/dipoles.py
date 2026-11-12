"""Exciton dipoles for emission, written where exc_ph_get_inputs finds them."""
from __future__ import annotations

import argparse

import numpy as np

from .yambo import lattice, load_excitons


def dipole_layout(ydip, nocc):
    """Fortran band numbers on the two axes of a yambopy dipole array."""
    b0, b1 = int(ydip.min_band), int(ydip.max_band)
    if ydip.dip_bands_ordered:
        return b0, int(ydip.indexv) + 1, int(ydip.indexc) + 1, b1
    return b0, nocc, nocc + 1, b1


def read_cv_dipoles(ylat, path, bs_bands):
    """<c k|r|v k> on the irreducible k-points, (nk_ibz, 3, nc, nv), for the BSE window bs_bands
    = [first valence band, last conduction band] (Fortran numbering)."""
    from yambopy.dbs.dipolesdb import YamboDipolesDB

    class _Fixed(YamboDipolesDB):
        def __init__(self, lattice, nq_ibz, nq_bz, nk_ibz, nk_bz, spin,
                     min_band, max_band, indexv, indexc, bands_range, nbands,
                     nbandsv, nbandsc, dip_bands_ordered, dipoles, **kw):
            if not dip_bands_ordered:
                nbandsc = nbands - nbandsv
            super().__init__(lattice, nq_ibz, nq_bz, nk_ibz, nk_bz, spin,
                             min_band, max_band, indexv, indexc, bands_range,
                             nbands, nbandsv, nbandsc, dip_bands_ordered,
                             dipoles, **kw)

    ydip = _Fixed.from_db_file(ylat, filename=path, project=False,
                               expand=False)
    raw = np.asarray(ydip.dipoles)
    nocc = int(ylat.nbandsv)
    v_first, v_last, c_first, c_last = dipole_layout(ydip, nocc)
    bv, bc = int(bs_bands[0]), int(bs_bands[1])
    ordered = bool(ydip.dip_bands_ordered)
    print(f"dipoles: database bands {v_first}-{c_last}, {nocc} occupied; "
          f"{'valence/conduction blocks' if ordered else 'DipBandsAll'}: "
          f"v {v_first}-{v_last}, c {c_first}-{c_last}")
    if raw.shape[3] != v_last - v_first + 1 or raw.shape[2] != c_last - c_first + 1:
        raise SystemExit(f"the dipole array is {raw.shape[2]} x {raw.shape[3]} "
                         f"(c x v), the database limits imply "
                         f"{c_last - c_first + 1} x {v_last - v_first + 1}: "
                         f"a layout this reader does not know")
    if ordered and (v_last != nocc or c_first != nocc + 1):
        print(f"   the blocks do not meet at the gap (top valence {nocc}); yambopy's "
              f"open-shell warning\n   above refers to this. The bands are "
              f"placed by the database's limits")
    if not (v_first <= bv <= nocc <= v_last and c_first <= nocc + 1 and bc <= c_last):
        raise SystemExit(f"BSE window {[bv, bc]} is not inside the dipole "
                         f"database (v {v_first}-{v_last}, c {c_first}-"
                         f"{c_last}, {nocc} occupied)")
    v = slice(bv - v_first, nocc - v_first + 1)
    c = slice(nocc + 1 - c_first, bc - c_first + 1)
    print(f"   BSE window {[bv, bc]} -> valence {bv}-{nocc}, conduction "
          f"{nocc + 1}-{bc} ({v.stop - v.start} x {c.stop - c.start})")
    return raw[:, :, c, v]


def full_zone_dipoles(ylat, path, bs_bands):
    """<c k|r|v k> on the FULL zone, (nk, 3, nc, nv), in the k order of the BSE eigenvectors:
    read_cv_dipoles on the irreducible points, rotated by the operation that carries each
    point's parent onto it, and conjugated where that operation includes time reversal."""
    dip = read_cv_dipoles(ylat, path, bs_bands)
    nk_ibz = int(ylat.kmap[:, 0].max()) + 1
    if dip.shape[0] != nk_ibz:
        raise SystemExit(f"the dipoles hold {dip.shape[0]} irreducible k-points, "
                         f"the SAVE {nk_ibz}: not the same k grid")
    rot = ylat.sym_car[ylat.kmap[:, 1], ...]
    dexp = np.einsum("kij,kjcv->kicv", rot, dip[ylat.kmap[:, 0], ...], optimize=True)
    tr = ylat.kmap[:, 1] >= ylat.sym_car.shape[0] / (int(ylat.time_rev) + 1)
    dexp[tr] = dexp[tr].conj()
    return dexp, dip.dtype


def residual_verdict(mine, ref, k=3, max_spread=1.25):
    """Compare per-multiplet |D|^2 with yambo's residuals."""
    mine, ref = np.asarray(mine, float), np.asarray(ref, float)
    k = min(k, len(mine))
    tm = set(np.argsort(mine)[::-1][:k])
    tr = set(np.argsort(ref)[::-1][:k])
    use = sorted(i for i in (tm | tr) if ref[i] > 0 and mine[i] > 0)
    r = mine[use] / ref[use] if use else np.array([np.nan])
    spread = float(np.nanmax(r) / np.nanmin(r)) if use else np.inf
    if tm == tr and spread <= max_spread:
        return "AGREE", spread
    if tm == tr:
        return (f"DISAGREE -- same brightest multiplets, but the ratio varies x{spread:.1f}: "
                f"the dipoles are NOT proportional to yambo's"), spread
    return "DISAGREE -- different brightest multiplets; inspect before use", spread


def multiplet_checks(dexc, groups, E, rel=1e-3):
    """Symmetry checks on the in-plane dipoles of each multiplet."""
    D = np.asarray(dexc)[:2]
    fin = (np.abs(D) ** 2).sum(axis=0)
    trace = np.array([fin[g].sum() for g in groups])
    top = trace.max() if trace.size else 0.0
    rows, accidental = [], []
    for g, t in zip(groups, trace):
        if len(g) >= 3 and t > 0:
            accidental.append(list(g))
        if top <= 0 or t < rel * top:
            continue
        lo, hi = np.linalg.eigvalsh(D[:, g] @ D[:, g].conj().T)
        rows.append(dict(states=list(g), E=float(E[g[0]]), strength=float(t),
                         aniso=float((hi - lo) / (hi + lo)),
                         imbalance=(float((fin[g].max() - fin[g].min()) / fin[g].max())
                                    if len(g) == 2 else None)))
    former = max((((fin[g].max() - fin[g].min()) / fin[g].max(), list(g))
                  for g in groups if len(g) > 1 and fin[g].max() > 1e-6),
                 default=(0.0, None), key=lambda x: x[0])
    return rows, accidental, former


def _aniso(D, idx):
    """(l_max - l_min) / (l_max + l_min) of the in-plane tensor of states idx."""
    lo, hi = np.linalg.eigvalsh(D[:2, idx] @ D[:2, idx].conj().T)
    return float((hi - lo) / (hi + lo)) if hi + lo > 0 else 0.0


def isotropic_cluster(dexc, E, g, tol=2e-2, max_span=2e-3, max_add=12):
    """Grow multiplet g by its energy neighbours until its in-plane tensor is isotropic."""
    D = np.asarray(dexc)
    order = np.argsort(E, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    lo, hi = int(rank[list(g)].min()), int(rank[list(g)].max())
    cur = list(order[lo:hi + 1])
    an = _aniso(D, cur)
    best_an, best = an, list(cur)
    stop, added = "closed", 0
    while an >= tol:
        if added >= max_add:
            stop = "size"
            break
        mid = E[cur].mean()
        cand = [r for r in (lo - 1, hi + 1) if 0 <= r < len(order)]
        if not cand:
            stop = "edge"
            break
        r = min(cand, key=lambda r: abs(E[order[r]] - mid))
        if max(E[cur].max(), E[order[r]]) - min(E[cur].min(), E[order[r]]) > max_span:
            stop = "span"
            break
        lo, hi = min(lo, r), max(hi, r)
        cur = list(order[lo:hi + 1])
        added += 1
        an = _aniso(D, cur)
        if an < best_an:
            best_an, best = an, list(cur)
    closed = an < tol
    st = cur if closed else best
    return dict(closed=closed, states=[int(n) for n in sorted(st, key=lambda n: E[n])],
                span=float(E[st].max() - E[st].min()),
                aniso=float(an if closed else best_an),
                stop="closed" if closed else stop)


def strongest_neighbour(dexc, E, g, window=5e-3):
    """The brightest state (in-plane) within window eV of multiplet g, outside it."""
    D = np.asarray(dexc)
    fin = (np.abs(D[:2]) ** 2).sum(axis=0)
    e0, inside = float(np.mean(E[list(g)])), set(int(n) for n in g)
    near = [int(n) for n in np.where(np.abs(E - e0) <= window)[0] if int(n) not in inside]
    if not near:
        return None
    n = max(near, key=lambda n: fin[n])
    return n, float(fin[n] / max(fin[list(g)].mean(), 1e-300)), float(E[n] - e0)


def _states(g):
    return f"{g[0]+1}" if len(g) == 1 else f"{g[0]+1}-{g[-1]+1}"


def main(argv=None):
    p = argparse.ArgumentParser(prog="xphd dipoles",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--save", default="../phonons/SAVE")
    p.add_argument("--bse-dir", default="../BSE/output_all")
    p.add_argument("--dipoles-dir", default=None,
                   help="directory holding ndb.dipoles; defaults to --bse-dir")
    p.add_argument("-o", "--out", default="exc_dipoles.npy")
    p.add_argument("--window", type=float, default=1.5,
                   help="eV above the lowest exciton for the residual check")
    p.add_argument("--deg-tol", type=float, default=2e-3)
    p.add_argument("--compare", default=None,
                   help="a reference exc_dipoles.npy, e.g. GaN's from the "
                        "working yambopy path, to compare |D|^2 against")
    a = p.parse_args(argv)

    ylat = lattice(a.save)
    exc, A, E = load_excitons(f"{a.bse_dir}/ndb.BS_diago_Q1", ylat)
    A = A.transpose(0, 1, 3, 2)
    nc, nv = A.shape[2], A.shape[3]
    print(f"A {A.shape} (n, k, c, v)")

    dexp, dtype = full_zone_dipoles(ylat, f"{a.dipoles_dir or a.bse_dir}/ndb.dipoles",
                                    exc.bs_bands)
    if dexp.shape[2:] != (nc, nv):
        raise SystemExit(f"dipole window {dexp.shape[2:]} != BSE (nc, nv) "
                         f"{(nc, nv)}")
    if A.shape[1:] != (dexp.shape[0], dexp.shape[2], dexp.shape[3]):
        raise SystemExit(f"shape mismatch A {A.shape[1:]} vs dipoles "
                         f"{(dexp.shape[0], dexp.shape[2], dexp.shape[3])}")

    dexc = (np.einsum("nkcv,kicv->in", A, dexp, optimize=True)
            .astype(dtype) / np.sqrt(exc.nkpoints))
    np.save(a.out, dexc)
    print(f"\nwrote {a.out}: {dexc.shape} (xyz, n)")

    fin = (np.abs(dexc[:2]) ** 2).sum(axis=0)
    fz = np.abs(dexc[2]) ** 2
    print("\n   brightest states:")
    for n in np.argsort(fin + fz)[::-1][:6]:
        print(f"     state {n+1:5d}  E = {E[n]:8.4f}  in-plane {fin[n]:.3e}  "
              f"z {fz[n]:.3e}")

    rr = np.asarray(getattr(exc, "r_residual", []))
    order = np.argsort(E)
    low = order[E[order] < E[order[0]] + a.window]
    groups, cur = [], [low[0]]
    for n in low[1:]:
        if abs(E[n] - E[cur[-1]]) < a.deg_tol:
            cur.append(n)
        else:
            groups.append(cur); cur = [n]
    groups.append(cur)

    if rr.size:
        rr = np.abs(rr) ** 2
        yam = rr.reshape(-1, rr.shape[-1]).sum(0) if rr.ndim > 1 else rr
        mine = np.array([fin[g].sum() for g in groups])
        ref = np.array([yam[g].sum() for g in groups])
        k = min(3, len(groups))
        tm, tr_ = set(np.argsort(mine)[::-1][:k]), set(np.argsort(ref)[::-1][:k])
        print(f"\n   against yambo's residuals, by degenerate multiplet:")
        for gi in sorted(tm | tr_, key=lambda g: E[groups[g][0]]):
            g = groups[gi]
            lab = f"{g[0]+1}" if len(g) == 1 else f"{g[0]+1}-{g[-1]+1}"
            ratio = mine[gi] / ref[gi] if ref[gi] > 0 else np.nan
            print(f"     {lab:<10} {E[g[0]]:.4f}  this {mine[gi]:.4e}  "
                  f"yambo {ref[gi]:.4e}  ratio {ratio:.4e}")
        verdict, spread = residual_verdict(mine, ref, k)
        print(f"   brightest {k} multiplets: {verdict}")
        print("   Agreement means a CONSTANT ratio across multiplets (the units differ).")
        if verdict.startswith("DISAGREE") and tm == tr_:
            print("   Common causes: a BSE solved beyond Tamm-Dancoff (BSEmod = coupling), whose\n"
                  "   residuals mix resonant and antiresonant parts; or yambo rescaling the dipoles\n"
                  "   by quasiparticle energies. Relative weights WITHIN one multiplet are unaffected.")
        elif verdict.startswith("DISAGREE"):
            print("   Common causes: the dipoles are from another SAVE than the BSE (another nscf\n"
                  "   run: the wavefunction phases differ and do not cancel), or the band layout\n"
                  "   printed at the top is not the file's. Use the ndb.dipoles the BSE read.")
    rows, accidental, former = multiplet_checks(dexc, groups, E)
    TOL = 2e-2
    print(f"\n   symmetry of the bright multiplets ({len(rows)} within 1e-3 of the "
          f"brightest; C3 requires 0, {TOL:g} allowed):")
    if rows:
        wa = max(rows, key=lambda r: r["aniso"])
        print(f"     in-plane anisotropy      {wa['aniso']:.1e}  states "
              f"{_states(wa['states'])}  {wa['E']:.4f} eV")
        pairs = [r for r in rows if r["imbalance"] is not None]
        if pairs:
            wi = max(pairs, key=lambda r: r["imbalance"])
            print(f"     imbalance within a pair  {wi['imbalance']:.1e}  states "
                  f"{_states(wi['states'])}  {wi['E']:.4f} eV")
        bad = sorted((r for r in rows if r["aniso"] > TOL or
                      (r["imbalance"] or 0.0) > TOL), key=lambda r: r["E"])
        if not bad:
            print("     OK")
        else:
            top = max(r["strength"] for r in rows)
            below = [r["aniso"] for r in rows if r["E"] < bad[0]["E"] - 1e-9]
            res = [(r, isotropic_cluster(dexc, E, r["states"], TOL)) for r in bad]
            shut = [x for x in res if x[1]["closed"]]
            left = [x for x in res if not x[1]["closed"]]
            print(f"     above {TOL:g}: {len(bad)} multiplets, the lowest at {bad[0]['E']:.4f} eV"
                  + (f"; below it the largest anisotropy is {max(below):.1e}" if below else ""))
            print("     each grown by its nearest neighbours in energy until isotropic (within 2 meV):")
            print("       states     E (eV)  strength  aniso    result")
            for r, c in (shut[:4] + left):
                tail = (f"-> {c['aniso']:.1e} with {_states(c['states'])} "
                        f"({1e3 * c['span']:.2f} meV)" if c["closed"] else
                        f"-> open: best {c['aniso']:.1e} with {_states(c['states'])} "
                        f"({1e3 * c['span']:.2f} meV; stopped: {c['stop']})")
                print(f"       {_states(r['states']):<10} {r['E']:.4f}  "
                      f"{r['strength'] / top:.1e}   {r['aniso']:.1e}  {tail}")
                if not c["closed"]:
                    nb = strongest_neighbour(dexc, E, r["states"])
                    if nb is not None:
                        print(f"{'':39}brightest state within 5 meV: {nb[0] + 1}, "
                              f"{nb[1]:.0f}x per state, {1e3 * nb[2]:+.2f} meV")
            if len(shut) > 4:
                print(f"       ... {len(shut) - 4} more that close")
            if shut:
                print(f"     {len(shut)} of {len(bad)} close with their neighbours (spans "
                      f"{1e3 * min(c['span'] for _, c in shut):.2f}-"
                      f"{1e3 * max(c['span'] for _, c in shut):.2f} meV): the solver "
                      f"mixed near-degenerate states;\n     per-state dipoles there "
                      f"depend on that mixing, the sums over each cluster do not")
            if left:
                print(f"     {len(left)} do not close within 2 meV: not decided by this "
                      f"test (a weak multiplet can borrow\n     from a bright state "
                      f"further away). The expansion of the dipoles is the same for "
                      f"every\n     state, so an error in it would show below "
                      f"{bad[0]['E']:.4f} eV as well")
            print(f"     verdict: OK up to {bad[0]['E']:.4f} eV" +
                  ("" if not left else f"; above it {len(left)} multiplet(s) undecided"))
    if accidental:
        print(f"   {len(accidental)} group(s) of 3+ states closer than --deg-tol "
              f"{a.deg_tol:g} eV: accidental, not compared state by state, e.g. "
              + ", ".join(f"{_states(g)} ({E[g[0]]:.4f})" for g in accidental[:3]))
    if former[1] is not None:
        g = former[1]
        vals = ", ".join(f"{fin[n]:.2e}" for n in g[:6]) + (", ..." if len(g) > 6 else "")
        print(f"   per-state spread over every group (the check before 25 Oct): "
              f"{former[0]:.1e} at states {_states(g)} ({E[g[0]]:.4f} eV, "
              f"in-plane {vals})")

    if a.compare:
        ref = np.load(a.compare)
        if ref.shape == dexc.shape:
            r1, r2 = (np.abs(ref) ** 2).sum(0), (np.abs(dexc) ** 2).sum(0)
            dev = np.abs(r1 - r2).max() / max(r1.max(), 1e-300)
            print(f"\n   --compare: max relative deviation in |D|^2 "
                  f"{dev:.2e}  {'AGREES' if dev < 1e-6 else 'DIFFERS'}")
        else:
            print(f"\n   --compare: shapes differ {ref.shape} vs {dexc.shape}")


if __name__ == "__main__":
    main()
