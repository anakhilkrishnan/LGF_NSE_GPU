#!/bin/bash
#SBATCH --job-name=IFHERK_move_test
#SBATCH --partition=small
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

source ~/.bashrc
set -euo pipefail

# Load Pravega environment for Akhil
source "$HOME/env-pravega.sh" || exit 1

# OpenMP parameters, set OMP_NUM_THREADS=1 and comment the rest when not using
export OMP_NUM_THREADS=8
export OMP_PLACES=threads
export OMP_PROC_BIND=close

# AMReX Arguments
AMREX_ARGS=""

ulimit -s unlimited
cd $SLURM_SUBMIT_DIR
mkdir -p Logs Results

EXEC="./nserun"
INPUTS_FILE="inputs"

if ldd $EXEC | grep -q "not found"; then
    echo "UNRESOLVED:"; ldd $EXEC | grep "not found"; exit 1
fi

# Ranks mapped as per OpenMP rules
mpiexec -n $((SLURM_JOB_NUM_NODES * 12)) --map-by ppr:6:socket:PE=4 --bind-to core $EXEC $INPUTS_FILE $AMREX_ARGS | tee ./Logs/log_ifherk_move_test.txt

# One rank per core
# mpiexec -n $SLURM_NTASKS --map-by core --bind-to core $EXEC $INPUTS_FILE $AMREX_ARGS | tee ./Logs/log.txt
