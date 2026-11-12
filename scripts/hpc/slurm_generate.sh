#!/bin/bash
#SBATCH -N 4
#SBATCH --ntasks-per-node=8          # ranks per node: as many as fit in MEMORY
#SBATCH --cpus-per-task=6            # cores per rank, used as BLAS threads
#SBATCH --mem=0                      # all of each node's memory
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --job-name=xphd-gen
#SBATCH --output=gen_%j.out
#SBATCH --error=gen_%j.err
##SBATCH --partition=standard

# ---------------------------------------------------------------------------
# slurm_generate.sh -- every exciton-phonon archive of the full zone, over
# several nodes with MPI.
#
# What changed from run_generate.sh (one node, one process per Q):
#   * the Q-list is shared between MPI ranks on any number of nodes;
#   * each rank loads the electron-phonon and BSE databases ONCE and reuses
#     them for all of its Q, instead of reloading them for every Q;
#   * the remaining cores of each rank work as BLAS threads inside yambopy's
#     contractions (--blas-threads).
# The archives are byte-for-byte the ones `xphd generate <iQ>` writes.
#
# SIZING. Memory, not cores, sets the ranks per node: each rank holds one
# copy of the databases. Measure one Q first, on a compute node:
#     /usr/bin/time -v xphd generate 0 test --nexc $NEXC ...   # "Maximum resident set size"
# then   ranks per node = floor(0.85 x node memory / that size)
# and    cpus-per-task  = cores per node / ranks per node.
#
# RESUMABLE. Complete archives are skipped, so after a time limit simply
# resubmit this script unchanged.
#
# Before submitting: generate the Gamma archive by hand and make sure
#     xphd check-archive <OUT>/GI_ExcPh_Q0001.npz
# prints READY. This script refuses to start otherwise.
#
# Paths and sizes for the material (examples):
#   GaN : N=24 NEXC=15 SAVE=../ELPH/SAVE    ELPH=../ELPH/ndb.elph   BSE=../BSE_EXCPH/output
#   hBN : N=24 NEXC=15 SAVE=../phonons/SAVE ELPH=../phonons/ndb.elph BSE=../BSE/output_all
# ---------------------------------------------------------------------------
set -u
. /home/apps/spack/share/spack/setup-env.sh
spack load /6asbh6t
spack load /r7xt55l
spack load /pmjv32t
source ~/xphd-env/bin/activate

# one BLAS thread pool per rank, sized to the rank's cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores OMP_PROC_BIND=close

N=${N:-24}
NEXC=${NEXC:-15}
OUT=${OUT:-archives}
SAVE=${SAVE:-../ELPH/SAVE}
ELPH=${ELPH:-../ELPH/ndb.elph}
BSE=${BSE:-../BSE_EXCPH/output}
DMATS=${DMATS:-Dmats.npy}
# srun with PMIx (OpenMPI, most MPICH); Intel MPI: LAUNCH="mpirun -np $SLURM_NTASKS"
LAUNCH=${LAUNCH:-"srun --mpi=pmix --cpu-bind=cores"}
mkdir -p "$OUT"

G0="$OUT/GI_ExcPh_Q0001.npz"
if [ ! -s "$G0" ] || ! xphd check-archive "$G0" | grep -q "READY: launch the rest"; then
    echo "CRITICAL: $G0 missing or not READY. Generate it by hand first:"
    echo "  xphd generate 0 $OUT --nexc $NEXC --savepath $SAVE --ndb-elph $ELPH \\"
    echo "        --bse-dir $BSE --dmats $DMATS --mesh $N $N"
    echo "  xphd check-archive $G0"
    exit 1
fi

echo "$(date)  $SLURM_NTASKS ranks x $SLURM_CPUS_PER_TASK threads on $SLURM_JOB_NUM_NODES nodes"
$LAUNCH xphd generate all "$OUT" --resume \
    --nexc "$NEXC" --savepath "$SAVE" --ndb-elph "$ELPH" --bse-dir "$BSE" \
    --dmats "$DMATS" --mesh "$N" "$N" --blas-threads "$SLURM_CPUS_PER_TASK"
rc=$?
echo "$(date)  generate exited with $rc"

# the transport needs EVERY Q -- confirm before anything downstream
xphd verify-archives --dir "$OUT" --n $((N * N))
exit $rc
