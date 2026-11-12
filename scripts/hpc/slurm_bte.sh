#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=16         # refinement ranks per node (see MEMORY)
#SBATCH --cpus-per-task=3
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --job-name=xphd-bte
#SBATCH --output=bte_%j.out
#SBATCH --error=bte_%j.err
##SBATCH --partition=standard

# ---------------------------------------------------------------------------
# slurm_bte.sh -- the real-time BTE in the two layouts its two halves need.
#
# Step 1, MPI over Q: the out-rate refinement (one task per Q, on the fine
#   mesh) is by far the most expensive part and is embarrassingly parallel.
#   Every rank takes a share; the result goes to outrates.npz.
# Step 2, threads on one node: assembling the rate matrix and integrating
#   the stiff ODE are dense linear algebra on ONE process -- it runs with
#   all the cores of one node as BLAS threads. The settings of step 1 are
#   stored in outrates.npz and checked here, so a mismatch stops the run.
#
# MEMORY. One refinement task holds the fine-mesh arrays of one archive. It
#   is processed in chunks of triangles sized to XPHD_TASK_MEM_GB per task
#   (set below to 80% of node memory / ranks per node).
#
# RESUMABLE: --checkpoint keeps each refined Q; resubmit after a time limit.
# ---------------------------------------------------------------------------
set -u
. /home/apps/spack/share/spack/setup-env.sh
spack load /6asbh6t
spack load /r7xt55l
spack load /pmjv32t
source ~/xphd-env/bin/activate

ARCH=${ARCH:-archives}
HWF=${HWF:-hw_fine.npy}
T=${T:-300}
LAUNCH=${LAUNCH:-"srun --mpi=pmix --cpu-bind=cores"}

NODE_GB=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)
export XPHD_TASK_MEM_GB=$(( NODE_GB * 8 / 10 / SLURM_NTASKS_PER_NODE ))
echo "node memory ${NODE_GB} GB -> ${XPHD_TASK_MEM_GB} GB per refinement task"

# ---- step 1: refined out-rates, all ranks, one thread each ---------------
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$LAUNCH xphd bte "$ARCH/GI_ExcPh_Q*.npz" --T "$T" --hw-fine "$HWF" \
    --refine-only --save-outrates "outrates_${T}K.npz" --checkpoint "ck_bte_${T}K" \
    || exit 1

# ---- step 2: assembly + propagation, one process, a whole node of threads
NT=$(( SLURM_CPUS_PER_TASK * SLURM_NTASKS_PER_NODE ))
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT
srun -N 1 -n 1 -c "$NT" --cpu-bind=none \
    xphd bte "$ARCH/GI_ExcPh_Q*.npz" --T "$T" --hw-fine "$HWF" \
    --outrates "outrates_${T}K.npz" --threads "$NT" \
    --inject optical --t-end 10000 --nt 200 \
    --channels --snapshots "snaps_${T}K.npz"
