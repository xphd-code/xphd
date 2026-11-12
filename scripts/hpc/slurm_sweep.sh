#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=48         # one rank per core
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --job-name=xphd-sweep
#SBATCH --output=sweep_%j.out
#SBATCH --error=sweep_%j.err
##SBATCH --partition=standard

# ---------------------------------------------------------------------------
# slurm_sweep.sh -- linewidths at every Q, one MPI rank per core.
#
# Each Q is an independent task and costs the same, so the work divides
# evenly: 576 Q over 96 ranks is 6 each. Every temperature in --T is done in
# the same pass (the delta-function weights are computed once and reused),
# so list all the temperatures you need here rather than running the sweep
# once per temperature.
#
# Memory per rank is one archive plus the fine mesh -- small; use every core.
#
# RESUMABLE: with --checkpoint each finished Q is saved, so a job stopped by
# the time limit is resubmitted unchanged and continues.
# ---------------------------------------------------------------------------
set -u
. /home/apps/spack/share/spack/setup-env.sh
spack load /6asbh6t
spack load /r7xt55l
spack load /pmjv32t
source ~/xphd-env/bin/activate

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ARCH=${ARCH:-archives}
HWF=${HWF:-hw_fine.npy}
SAVE=${SAVE:-../phonons/SAVE}
LAUNCH=${LAUNCH:-"srun --mpi=pmix --cpu-bind=cores"}

$LAUNCH xphd sweep "$ARCH/GI_ExcPh_Q*.npz" --hw-fine "$HWF" \
    --T 5 10 20 40 77 100 150 200 250 300 \
    --acoustic-cut 5e-4 --acoustic-model q2 --n-shell 1 \
    --unfold "$SAVE" --checkpoint ck_sweep -o lw_field.npz
