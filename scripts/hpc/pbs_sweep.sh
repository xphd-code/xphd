#!/bin/bash
#PBS -N xphd-sweep
#PBS -l select=2:ncpus=48:mpiprocs=48
#PBS -l walltime=04:00:00
#PBS -j oe

# ---------------------------------------------------------------------------
# pbs_sweep.sh -- slurm_sweep.sh for PBS Pro / OpenPBS clusters. The other
# stages translate the same way: ranks = select x mpiprocs, threads =
# ncpus / mpiprocs, and mpirun in place of srun.
# ---------------------------------------------------------------------------
cd "$PBS_O_WORKDIR"
# module load gcc openmpi          # the MPI mpi4py was built against
source ~/xphd-env/bin/activate

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
NP=$(wc -l < "$PBS_NODEFILE")

mpirun -np "$NP" -hostfile "$PBS_NODEFILE" \
    xphd sweep "archives/GI_ExcPh_Q*.npz" --hw-fine hw_fine.npy \
    --T 5 10 20 40 77 100 150 200 250 300 \
    --acoustic-cut 5e-4 --acoustic-model q2 --n-shell 1 \
    --checkpoint ck_sweep -o lw_field.npz
