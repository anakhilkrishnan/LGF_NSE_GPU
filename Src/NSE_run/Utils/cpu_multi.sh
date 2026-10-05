#!/bin/bash
#SBATCH --job-name=3D_LGF_NSE_CPU_run
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

source ~/.bashrc
set -o pipefail

# Load Pravega environment for Akhil
source "$HOME/env-pravega.sh" || exit 1

export OMP_NUM_THREADS=1

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

# One rank per core, pinned to its core
mpiexec -n $SLURM_NTASKS --map-by core --bind-to core $EXEC $INPUTS_FILE $AMREX_ARGS | tee ./Logs/log_vortexring_cpu.txt