"""Snapshots of `xphd bte --snapshots`."""
from types import SimpleNamespace

import numpy as np
import pytest

from xphd import bte


def _reference(rm, Y):
    """The per-snapshot loop of xphd <= 2.0.0.dev0 (Nov 2026)."""
    nch, nmq = rm.channels.shape[:2]
    flat = rm.channels.reshape(nch * nmq, *rm.channels.shape[2:])
    nst, nt = Y.shape
    dm = np.zeros((nt, nst), np.int16)
    ie = np.zeros((nt, nst), bool)
    df = np.zeros((nt, nst), np.float32)
    fl = np.zeros((nt, nch, nmq), np.float32)
    for k in range(nt):
        F = Y[:, k]
        infl = np.stack([c.T @ F for c in flat])
        tot = infl.sum(axis=0)
        best = np.argmax(infl, axis=0)
        dm[k] = (best % nmq).astype(np.int16)
        ie[k] = (best // nmq) == 1
        with np.errstate(invalid="ignore", divide="ignore"):
            df[k] = np.where(tot > 0, infl.max(axis=0) / np.maximum(tot, 1e-300), 0.0)
        fl[k] = np.array([F @ c.sum(axis=1) for c in flat]).reshape(nch, nmq)
    return dict(dom_mode_t=dm, is_emission_t=ie, dom_frac_t=df, flux_t=fl)


def _save(rm, Y, path, block=None, monkeypatch=None):
    if block is not None:
        monkeypatch.setattr(bte, "_SNAPSHOT_BLOCK", block)
    sol = SimpleNamespace(t=np.linspace(0.0, 100.0, Y.shape[1]), y=Y)
    return bte.save_snapshots(rm, sol, path, channels=True, verbose=False)


def _check(got, ref):
    for k in ("dom_mode_t", "is_emission_t"):
        assert np.array_equal(got[k], ref[k]), k
    assert np.allclose(got["dom_frac_t"], ref["dom_frac_t"], rtol=1e-6, atol=1e-7)
    assert np.allclose(got["flux_t"], ref["flux_t"], rtol=1e-6, atol=1e-30)


def _rates_with_channels(tmp_path, T=200.0):
    import xphd
    from test_pipeline import _bte_set
    arcs = xphd.load_archives(_bte_set(tmp_path), verbose=False)
    rm = bte.build_rates(arcs, T=T, channels=True, verbose=False)
    assert rm.channels is not None and rm.channels.shape[0] == 2
    return rm


@pytest.mark.parametrize("block", [None, 3.0, 1.0])
def test_batched_decomposition_equals_the_snapshot_loop(tmp_path, monkeypatch, block):
    rm = _rates_with_channels(tmp_path)
    nst = len(rm.E)
    rng = np.random.default_rng(1)
    Y = rng.random((nst, 7)) * np.geomspace(1, 1e-6, 7)[None, :]
    Y[:, 0] = 0.0
    Y[0, 0] = 1.0
    got = _save(rm, Y, tmp_path / "s.npz",
                None if block is None else block * nst, monkeypatch)
    _check(got, _reference(rm, Y))
    d = np.load(tmp_path / "s.npz")
    _check({k: d[k] for k in ("dom_mode_t", "is_emission_t", "dom_frac_t", "flux_t")},
           _reference(rm, Y))


def test_ties_go_to_the_first_channel_as_argmax(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    nst, nch, nmq = 6, 2, 3
    ch = rng.integers(0, 3, size=(nch, nmq, nst, nst)).astype(float)
    ch[:, :, :, 2] = 0.0
    rm = SimpleNamespace(E=np.linspace(0, 0.05, nst), meta={}, dims=(1, nst, 1),
                         channels=ch, P=ch.sum(axis=(0, 1)),
                         R=ch.sum(axis=(0, 1)) - ch.sum(axis=(0, 1)).T)
    Y = rng.integers(0, 2, size=(nst, 5)).astype(float)
    Y[:, 0] = 1.0
    for block in (None, 2.0 * nst):
        got = _save(rm, Y, tmp_path / "t.npz", block, monkeypatch)
        _check(got, _reference(rm, Y))
    assert (got["dom_frac_t"][:, 2] == 0).all()


def test_nt_1_is_refused_before_loading(tmp_path):
    from xphd.cli import main
    with pytest.raises(SystemExit, match="--nt 1 keeps only t = 0"):
        main(["bte", str(tmp_path / "missing_Q*.npz"), "--nt", "1"])
