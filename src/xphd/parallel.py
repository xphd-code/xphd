"""Parallel execution: serial, process pool, MPI, and BLAS/OpenMP threads."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
import traceback

__all__ = ["comm", "rank", "size", "is_root", "barrier", "bcast", "abort",
           "cpu_count", "set_blas_threads", "blas_info", "pmap",
           "config_key", "silence_nonroot", "finish", "describe"]

_LAUNCH_VARS = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_RANK",
                "MV2_COMM_WORLD_SIZE", "MPI_LOCALNRANKS", "PMI_RANK")

_state = {"init": False, "comm": None, "blas": None}


def _launched_ntasks():
    """Ranks the launcher claims to have started, 1 if it does not say."""
    for k in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE",
              "SLURM_STEP_NUM_TASKS"):
        v = os.environ.get(k)
        if v and v.isdigit():
            return int(v)
    return 1


def comm():
    """The MPI communicator, or None when not running under MPI."""
    if _state["init"]:
        return _state["comm"]
    _state["init"] = True
    want = os.environ.get("XPHD_MPI", "auto").strip().lower()
    if want in ("0", "no", "off", "false"):
        return None
    launched = any(k in os.environ for k in _LAUNCH_VARS)
    if not (launched or want in ("1", "yes", "on", "true")):
        ntask = os.environ.get("SLURM_STEP_NUM_TASKS", "1")
        procid = os.environ.get("SLURM_PROCID", "0")
        if ntask.isdigit() and int(ntask) > 1:
            msg = (f"xphd: srun started {ntask} tasks without an MPI "
                   f"environment (use 'srun --mpi=pmix' or pmi2 with mpi4py "
                   f"built against this MPI).")
            if procid not in ("", "0"):
                sys.stderr.write(msg + f" Task {procid} leaves the work to "
                                 f"task 0.\n")
                raise SystemExit(0)
            sys.stderr.write(msg + " Running serially on task 0.\n")
        return None
    try:
        from mpi4py import MPI
    except Exception as e:
        if _launched_ntasks() > 1:
            raise SystemExit(
                f"xphd: launched with {_launched_ntasks()} MPI ranks but "
                f"mpi4py cannot be imported ({e}).\nEvery rank would repeat "
                f"the whole job and overwrite the same files. Install mpi4py "
                f"against the cluster's MPI (pip install mpi4py with the MPI "
                f"module loaded), or run without mpirun.")
        return None
    c = MPI.COMM_WORLD
    _state["comm"] = c if c.Get_size() > 1 else None
    return _state["comm"]


def rank():
    c = comm()
    return 0 if c is None else c.Get_rank()


def size():
    c = comm()
    return 1 if c is None else c.Get_size()


def is_root():
    return rank() == 0


def barrier():
    c = comm()
    if c is not None:
        c.Barrier()


def bcast(obj):
    c = comm()
    return obj if c is None else c.bcast(obj, root=0)


def silence_nonroot():
    """Send stdout of ranks > 0 to /dev/null. stderr is left alone, so a failure on any rank is
    still reported."""
    if rank() > 0 and os.environ.get("XPHD_ALL_RANKS", "0") != "1":
        sys.stdout = open(os.devnull, "w")


def abort(code=1):
    """Stop every rank."""
    c = comm()
    if c is not None:
        sys.stderr.flush()
        c.Abort(code)


def finish(poll=0.2):
    """End of an MPI run."""
    c = comm()
    if c is None:
        return
    req = c.Ibarrier()
    if c.Get_rank() == 0:
        req.Wait()
        return
    while not req.Test():
        time.sleep(poll)


def cpu_count():
    """Cores this process may use: the affinity mask (respects SLURM/cgroups), not the node
    total."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def set_blas_threads(n):
    """Limit BLAS/OpenMP threads in THIS process to n."""
    if n is None:
        return
    n = max(1, int(n))
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = str(n)
    try:
        from threadpoolctl import threadpool_limits
        _state["blas"] = threadpool_limits(limits=n)
    except ImportError:
        pass


def blas_info():
    """[(library, threads, version)] for every thread pool numpy has loaded."""
    try:
        from threadpoolctl import threadpool_info
        return [(d.get("internal_api"), d.get("num_threads"),
                 d.get("version")) for d in threadpool_info()]
    except ImportError:
        return [("threadpoolctl not installed", None, None)]


def config_key(*parts):
    """Short stable hash of a configuration, for checkpoint file names."""
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


def _ck_path(directory, key):
    return os.path.join(directory, f"{key}.pkl")


def _ck_load(directory, key):
    p = _ck_path(directory, key)
    if not os.path.isfile(p):
        return False, None
    try:
        with open(p, "rb") as fh:
            return True, pickle.load(fh)
    except Exception:
        return False, None


def _ck_save(directory, key, value):
    """Atomic: written to a temporary name, then renamed."""
    p = _ck_path(directory, key)
    tmp = f"{p}.tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, p)


def _pool_init(threads):
    set_blas_threads(threads)


def _call(job):
    fn, i, task = job
    try:
        return i, True, fn(task)
    except BaseException:
        return i, False, traceback.format_exc()


def _assign(n, nranks, cost=None):
    """Task -> rank."""
    if cost is None:
        return [i % nranks for i in range(n)]
    load = [0.0] * nranks
    owner = [0] * n
    for i in sorted(range(n), key=lambda k: -float(cost[k])):
        r = min(range(nranks), key=load.__getitem__)
        owner[i] = r
        load[r] += float(cost[i])
    return owner


def pmap(fn, tasks, *, workers=1, threads_per_worker=1, checkpoint=None,
         keys=None, cost=None, desc="tasks", verbose=True, to_all=False):
    """[fn(t) for t in tasks], spread over MPI ranks and/or local processes."""
    tasks = list(tasks)
    n = len(tasks)
    c = comm()
    me, nr = rank(), size()
    if checkpoint is not None:
        if keys is None or len(keys) != n:
            raise ValueError("checkpoint needs one key per task")
        if me == 0:
            os.makedirs(checkpoint, exist_ok=True)
        barrier()

    owner = _assign(n, nr, cost)
    mine = [i for i in range(n) if owner[i] == me]

    done, todo = {}, []
    for i in mine:
        if checkpoint is not None:
            ok, val = _ck_load(checkpoint, keys[i])
            if ok:
                done[i] = val
                continue
        todo.append(i)

    restored = len(done) if c is None else c.allreduce(len(done))
    if verbose and me == 0:
        lay = f"{nr} rank{'s' if nr > 1 else ''}"
        if workers > 1:
            lay += f" x {workers} processes"
        print(f"  {desc}: {n} tasks on {lay}"
              + (f", {restored} restored from {checkpoint}" if restored else ""))

    errors = []
    t0 = time.perf_counter()
    step = max(len(todo) // 20, 1)

    def take(k, res):
        i, ok, val = res
        if ok:
            done[i] = val
            if checkpoint is not None:
                _ck_save(checkpoint, keys[i], val)
        else:
            errors.append((i, val))
        if verbose and me == 0 and (k % step == 0 or k == len(todo)):
            el = time.perf_counter() - t0
            eta = el / k * (len(todo) - k)
            print(f"    rank 0: {k}/{len(todo)}  {el:7.1f} s elapsed, "
                  f"~{eta:7.1f} s left", flush=True)

    jobs = [(fn, i, tasks[i]) for i in todo]
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        ctx = mp.get_context(os.environ.get("XPHD_MP_START") or None)
        with ctx.Pool(min(workers, len(jobs)), initializer=_pool_init,
                      initargs=(threads_per_worker,)) as pool:
            for k, res in enumerate(pool.imap_unordered(_call, jobs), 1):
                take(k, res)
    else:
        for k, job in enumerate(jobs, 1):
            take(k, _call(job))

    if c is None:
        parts = [(done, errors)]
    else:
        parts = c.gather((done, errors), root=0)

    failed = None
    if me == 0:
        allres, allerr = {}, []
        for d, e in parts:
            allres.update(d)
            allerr.extend(e)
        if allerr:
            i, tb = allerr[0]
            failed = (f"{len(allerr)} of {n} {desc} failed. First, task "
                      f"{i} ({tasks[i]!r:.200}):\n{tb}")
        elif len(allres) != n:
            failed = f"{n - len(allres)} of {n} {desc} produced no result"
    if c is not None:
        failed = c.bcast(failed, root=0)
    if failed:
        if me == 0:
            sys.stderr.write(failed + "\n")
        raise SystemExit(1)

    out = [allres[i] for i in range(n)] if me == 0 else None
    if to_all and c is not None:
        out = c.bcast(out, root=0)
    return out


def describe():
    """Lines describing the parallel environment, for `xphd hpc-info`."""
    import platform
    import numpy as np
    c = comm()
    lines = [f"python      {platform.python_version()} ({sys.executable})",
             f"numpy       {np.__version__}"]
    try:
        import scipy
        lines.append(f"scipy       {scipy.__version__}")
    except ImportError:
        lines.append("scipy       not installed (needed by the BTE)")
    if c is None:
        lines.append("MPI         not active (single process)")
    else:
        from mpi4py import MPI
        lines.append(f"MPI         {c.Get_size()} ranks; "
                     f"{MPI.Get_library_version().splitlines()[0][:70]}")
    host = platform.node()
    info = (host, cpu_count(), os.environ.get("OMP_NUM_THREADS", "unset"))
    allinfo = [info] if c is None else c.gather(info, root=0)
    if rank() == 0:
        hosts = {}
        for h, ncpu, omp in allinfo:
            hosts.setdefault(h, []).append((ncpu, omp))
        for h, v in hosts.items():
            lines.append(f"node        {h}: {len(v)} rank(s), cores per rank "
                         f"{sorted({x[0] for x in v})}, OMP_NUM_THREADS "
                         f"{sorted({x[1] for x in v})}")
        for lib, nt, ver in blas_info():
            lines.append(f"threads     {lib}: {nt} ({ver})")
        try:
            with open("/proc/meminfo") as fh:
                mem = {l.split(":")[0]: l.split()[1] for l in fh}
            lines.append(f"memory      {int(mem['MemTotal'])/1048576:.0f} GB "
                         f"total, {int(mem['MemAvailable'])/1048576:.0f} GB "
                         f"available on this node")
        except Exception:
            pass
    return lines
