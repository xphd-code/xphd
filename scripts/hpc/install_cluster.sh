#!/bin/bash
# ---------------------------------------------------------------------------
# install_cluster.sh -- one-time install of xphd with MPI on a cluster.
#
# Run on a LOGIN node, interactively, once per cluster. Afterwards every job
# script only needs the two `source` lines at the end of this file.
#
# The one thing that matters: mpi4py must be COMPILED against the MPI that
# srun/mpirun will launch. The binary wheel from PyPI bundles its own MPI
# and, on a cluster, either fails to start or silently runs N independent
# copies of the job. Load the MPI module first, then build from source.
# ---------------------------------------------------------------------------
set -euo pipefail

XPHD_SRC=${XPHD_SRC:-$HOME/xphd}          # this repository
ENV=${ENV:-$HOME/xphd-env}                # where the Python environment goes

# ---- 1. compilers and MPI: the same ones QE/yambo were built with ---------
# Spack (the toolchain used for the GaN runs):
. /home/apps/spack/share/spack/setup-env.sh
spack load /6asbh6t
spack load /r7xt55l
spack load /pmjv32t
# or environment modules, e.g.
#   module load gcc/12.2 openmpi/4.1.5 python/3.11
which mpicc mpirun python3

# ---- 2. a Python environment ---------------------------------------------
python3 -m venv "$ENV"
source "$ENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# ---- 3. mpi4py from source, against THIS MPI ------------------------------
# OpenMPI / MPICH:       MPICC=mpicc
# Intel MPI (classic):   MPICC=mpiicc      Intel MPI (oneAPI LLVM): MPICC=mpiicx
MPICC=${MPICC:-$(which mpicc)} \
    python -m pip install --no-cache-dir --no-binary=mpi4py mpi4py

# ---- 4. xphd with the HPC extras (mpi4py is already satisfied) -----------
# plot: matplotlib. On Linux each run leaves a text report, <output>_report.txt;
# with matplotlib, XPHD_REPORT=both also writes the PDF with the plots
python -m pip install -e "$XPHD_SRC[hpc,generate,plot]"

# ---- 5. yambopy, for `xphd generate`, `dmats`, `parity`, ... -------------
# from source, as on the workstation:
#   git clone https://github.com/yambo-code/yambopy.git && pip install -e yambopy

# ---- 6. check: one line per rank must agree with what you asked for ------
mpirun -np 4 xphd hpc-info
python -c "import xphd, mpi4py; print('xphd', xphd.__version__, '| mpi4py', mpi4py.__version__)"

echo
echo "Done. In every job script:"
echo "  . /home/apps/spack/share/spack/setup-env.sh && spack load ... (as above)"
echo "  source $ENV/bin/activate"
