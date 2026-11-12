"""Parallel execution must not change a single number."""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

import xphd
from xphd import bte, parallel
from test_pipeline import _bte_set

HERE = os.path.dirname(os.path.abspath(__file__))


def _square(x):
    return x * x


def _fail_on_three(x):
    if x == 3:
        raise ValueError("task three is broken")
    return x


def _count_and_square(args):
    x, d = args
    open(os.path.join(d, f"ran{x}"), "w").close()
    return x * x


def test_pmap_keeps_task_order_with_processes():
    tasks = list(range(23))
    assert parallel.pmap(_square, tasks, verbose=False) == [x * x for x in tasks]
    assert parallel.pmap(_square, tasks, workers=3,
                         verbose=False) == [x * x for x in tasks]


def test_pmap_checkpoint_skips_finished_tasks(tmp_path):
    ck, log = tmp_path / "ck", tmp_path / "log"
    log.mkdir()
    keys = [f"t{i}" for i in range(6)]
    first = parallel.pmap(_count_and_square, [(i, str(log)) for i in range(6)],
                          checkpoint=str(ck), keys=keys, verbose=False)
    for f in log.iterdir():
        f.unlink()
    again = parallel.pmap(_count_and_square, [(i, str(log)) for i in range(6)],
                          checkpoint=str(ck), keys=keys, verbose=False)
    assert first == again == [i * i for i in range(6)]
    assert not list(log.iterdir())


def test_pmap_failure_is_reported_not_swallowed(capsys):
    with pytest.raises(SystemExit):
        parallel.pmap(_fail_on_three, list(range(6)), verbose=False)
    assert "task three is broken" in capsys.readouterr().err


def test_config_key_changes_with_settings():
    a = parallel.config_key("sweep", {"T": [10, 300], "refine": 5})
    b = parallel.config_key("sweep", {"T": [10, 300], "refine": 15})
    assert a != b and a == parallel.config_key("sweep", {"refine": 5,
                                                         "T": [10, 300]})


def test_all_temperatures_in_one_pass_is_bit_identical(archive):
    a = xphd.ExcPhArchive(archive)
    T = [10.0, 77.0, 300.0]
    together = xphd.compute(a, T, refine=3, acoustic_model="q2",
                            verbose=False)
    for k, t in enumerate(T):
        alone = xphd.compute(a, [t], refine=3, acoustic_model="q2",
                             verbose=False)
        assert np.array_equal(together.total[k], alone.total[0])
        assert np.array_equal(together.mode_em[k], alone.mode_em[0])
        assert np.array_equal(together.mode_ab[k], alone.mode_ab[0])


def test_sweep_with_processes_equals_serial_and_keeps_modes(tmp_path):
    pat = _bte_set(tmp_path, n=4, ne=3)
    s = xphd.sweep(str(pat), [10.0, 300.0], refine=2, verbose=False)
    p = xphd.sweep(str(pat), [10.0, 300.0], refine=2, workers=2,
                   verbose=False)
    assert np.array_equal(s.LW, p.LW) and np.array_equal(s.Q_red, p.Q_red)
    assert s.mode_em is not None and s.mode_em.shape == (2, 16, 3, 3)
    assert np.allclose(s.mode_em.sum(-1) + s.mode_ab.sum(-1), s.LW)


def _rates_add_at(archives, T):
    """build_rates' geometric accumulation as it was written before 1.1, with np.add.at, kept
    here as the reference for the bincount version."""
    from xphd.bte import HBAR_EV_FS, _triangulate
    from xphd.tetra import triangle_weights
    first = next(iter(archives.values()))
    n1, n2, ne = first.n1, first.n2, first.nexc
    nst = n1 * n2 * ne
    q = np.mod(first.q_red, 1.0)
    qi = np.rint(q[:, 0] * n1).astype(int) % n1
    qj = np.rint(q[:, 1] * n2).astype(int) % n2
    E = np.zeros(nst)
    for (a, b), arc in archives.items():
        E[(a * n2 + b) * ne:(a * n2 + b) * ne + ne] = arc.E_n
    tri = _triangulate(n1, n2)
    area = 1.0 / len(tri)
    pref = 2.0 * np.pi / HBAR_EV_FS
    A_em, A_ab = np.zeros((nst, nst)), np.zeros((nst, nst))
    for (a, b), arc in archives.items():
        target = ((a + qi) % n1) * n2 + (b + qj) % n2
        g2 = np.maximum(np.asarray(arc._raw("g2"), float), 0.0)
        hw = np.asarray(arc._raw("hw"), float)
        g2 = np.where((hw > 1e-4)[:, :, None, None], g2, 0.0)
        Et = np.transpose(E[target[:, None] * ne + np.arange(ne)][tri], (1, 0, 2))
        Wt = np.transpose(hw[tri], (1, 0, 2))
        gt = np.transpose(g2[tri], (1, 0, 2, 3, 4))
        nodes = np.transpose(target[tri][:, :, None] * ne + np.arange(ne),
                             (1, 0, 2))
        dE = arc.E_n[None, None, :, None] - Et[:, :, None, :]
        Wb = Wt[:, :, :, None, None]
        Wa = triangle_weights(dE[:, :, None] + Wb, area, 0.0)
        We = triangle_weights(dE[:, :, None] - Wb, area, 0.0)
        base = (a * n2 + b) * ne + np.arange(ne)
        for v in range(3):
            wt = pref * gt[v]
            em, ab = (wt * We[v]).sum(axis=1), (wt * Wa[v]).sum(axis=1)
            nl = em.shape[0]
            II = np.broadcast_to(base[None, :, None], (nl, ne, ne)).ravel()
            JJ = np.broadcast_to(nodes[v][:, None, :], (nl, ne, ne)).ravel()
            np.add.at(A_em, (II, JJ), em.ravel())
            np.add.at(A_ab, (II, JJ), ab.ravel())
    return A_em, A_ab


def test_bincount_assembly_is_bit_identical_to_add_at(tmp_path):
    arcs = xphd.load_archives(_bte_set(tmp_path), verbose=False)
    rm = bte.build_rates(arcs, T=300.0, verbose=False)
    A_em, A_ab = _rates_add_at(arcs, 300.0)
    B = 0.5 * (A_em + A_ab.T)
    assert np.array_equal(rm.R, (B + B.T) * np.sign(rm.E[:, None] - rm.E[None, :]))


def test_channels_still_sum_to_the_total(tmp_path):
    arcs = xphd.load_archives(_bte_set(tmp_path), verbose=False)
    rm = bte.build_rates(arcs, T=300.0, channels=True, verbose=False)
    err = np.abs(rm.channels.sum(axis=(0, 1)) - rm.P).max() / rm.P.max()
    assert err < 1e-6


def test_analytic_jacobian_equals_finite_differences(tmp_path):
    arcs = xphd.load_archives(_bte_set(tmp_path), verbose=False)
    rm = bte.build_rates(arcs, T=300.0, verbose=False)
    rhs, jac = bte.rhs_factory(rm), bte.jac_factory(rm)
    F = np.random.default_rng(0).uniform(0, 1e-3, len(rm.E))
    J = jac(0.0, F)
    h = 1e-7
    Jfd = np.stack([(rhs(0.0, F + h * e) - rhs(0.0, F - h * e)) / (2 * h)
                    for e in np.eye(len(F))], axis=1)
    assert np.abs(J - Jfd).max() < 1e-6 * np.abs(J).max()


@pytest.mark.parametrize("opts", [dict(jac="analytic"),
                                  dict(jac="analytic", matrix="sparse")])
def test_propagation_agrees_with_the_old_solver(tmp_path, opts):
    arcs = xphd.load_archives(_bte_set(tmp_path), verbose=False)
    rm = bte.build_rates(arcs, T=300.0, verbose=False)
    F0 = np.zeros(len(rm.E))
    F0[:3] = bte.inject_optical(arcs[(0, 0)].E_n, np.array([1.0, 0.02, 0.5]),
                                1e-3, verbose=False)
    ref = bte.propagate(rm, F0, t_end=2000.0, nt=5, jac="fd")
    new = bte.propagate(rm, F0, t_end=2000.0, nt=5, **opts)
    assert np.abs(new.y - ref.y).max() < 1e-5 * F0.max()
    assert np.abs(new.y.sum(0) / new.y[:, 0].sum() - 1).max() < 1e-6


def test_out_rate_chunking_only_moves_round_off(tmp_path):
    arcs = xphd.load_archives(_bte_set(tmp_path), verbose=False)
    whole = bte.refined_out_rates(arcs, 300.0, refine=3, verbose=False)
    tiny = bte.refined_out_rates(arcs, 300.0, refine=3, verbose=False,
                                 mem_gb=1e-9)
    assert np.allclose(whole, tiny, rtol=1e-12, atol=0)
    pooled = bte.refined_out_rates(arcs, 300.0, refine=3, verbose=False,
                                   workers=2)
    assert np.array_equal(whole, pooled)


def _fake_generate(tmp_path, *args, mpi=0):
    pytest.importorskip("tqdm")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [os.path.join(HERE, "_fake_yambopy"), os.environ.get("PYTHONPATH", "")]))
    inp = tmp_path / "in"
    if not inp.exists():
        subprocess.run([sys.executable, os.path.join(HERE, "_fake_generate_inputs.py"),
                        str(inp)], check=True)
    cmd = ["xphd", "generate", *args, "--nexc", "3",
           "--savepath", str(inp / "SAVE"), "--ndb-elph", str(inp / "ndb.elph"),
           "--bse-dir", str(inp / "BSE"), "--dmats", str(inp / "Dmats.npy"),
           "--mesh", "3", "3"]
    if mpi:
        cmd = _mpirun(mpi) + cmd
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout


def _same_archives(d1, d2):
    names = sorted(os.listdir(d1))
    assert names == sorted(os.listdir(d2)) and len(names) == 9
    for n in names:
        x, y = np.load(os.path.join(d1, n)), np.load(os.path.join(d2, n))
        assert set(x.files) == set(y.files)
        for k in x.files:
            assert np.array_equal(x[k], y[k]), (n, k)


def test_generate_many_q_equals_one_at_a_time(tmp_path):
    for q in range(9):
        _fake_generate(tmp_path, str(q), str(tmp_path / "one"))
    _fake_generate(tmp_path, "all", str(tmp_path / "many"))
    _same_archives(tmp_path / "one", tmp_path / "many")
    out = _fake_generate(tmp_path, "0-8", str(tmp_path / "many"))
    assert "9 already complete" in out


def _mpirun(n):
    exe = shutil.which("mpirun") or shutil.which("mpiexec")
    if exe is None:
        pytest.skip("no mpirun")
    try:
        import mpi4py
    except ImportError:
        pytest.skip("no mpi4py")
    cmd = [exe, "-np", str(n)]
    if "Open MPI" in subprocess.run([exe, "--version"], capture_output=True,
                                    text=True).stdout:
        cmd += ["--oversubscribe"]
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            cmd += ["--allow-run-as-root"]
    return cmd


def test_mpi_generate_equals_serial(tmp_path):
    for q in range(9):
        _fake_generate(tmp_path, str(q), str(tmp_path / "one"))
    _fake_generate(tmp_path, "all", str(tmp_path / "mpi"), mpi=3)
    _same_archives(tmp_path / "one", tmp_path / "mpi")


def test_mpi_sweep_and_bte_equal_serial(tmp_path):
    pat = str(_bte_set(tmp_path, n=4, ne=3))
    run = lambda *a, mpi=0: subprocess.run(
        (_mpirun(mpi) if mpi else []) + ["xphd", *a], capture_output=True,
        text=True, check=True)
    run("sweep", pat, "--T", "10", "300", "--refine", "2", "-o",
        str(tmp_path / "s.npz"))
    run("sweep", pat, "--T", "10", "300", "--refine", "2", "-o",
        str(tmp_path / "m.npz"), mpi=3)
    s, m = np.load(tmp_path / "s.npz"), np.load(tmp_path / "m.npz")
    assert np.array_equal(s["LW"], m["LW"])
    assert np.array_equal(s["mode_em"], m["mode_em"])

    common = ["bte", pat, "--refine", "2", "--t-end", "500", "--nt", "4",
              "--jac", "fd"]
    run(*common, "--snapshots", str(tmp_path / "bs.npz"))
    run(*common, "--snapshots", str(tmp_path / "bm.npz"), mpi=3)
    assert np.array_equal(np.load(tmp_path / "bs.npz")["F"],
                          np.load(tmp_path / "bm.npz")["F"])
