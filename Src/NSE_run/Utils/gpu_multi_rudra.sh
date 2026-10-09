#!/bin/bash

#SBATCH --job-name=LGF_NSE_GPU_test
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

set -o pipefail

# Environment
module purge
module load spack
. /home/apps/spack/share/spack/setup-env.sh
module load nvhpc-hpcx-cuda13/25.11
spack load gcc@13.4.0/azyvhui
spack load cmake@3.31.12/q57c2r6
export CC=gcc CXX=g++ OMPI_CC=gcc OMPI_CXX=g++

which cmake nvcc mpicxx g++

# MPI / UCX
export OMP_NUM_THREADS=1

# AMReX Arguments
AMREX_ARGS="amrex.abort_on_out_of_gpu_memory=1 amrex.use_gpu_aware_mpi=1"

ulimit -s unlimited
cd $SLURM_SUBMIT_DIR
mkdir -p Logs Results

EXEC="./nserun"
INPUTS_FILE="inputs"

if ldd $EXEC | grep -q "not found"; then
    echo "UNRESOLVED:"; ldd $EXEC | grep "not found"; exit 1
fi

mpiexec -n $SLURM_NTASKS --map-by socket --bind-to socket --report-bindings $EXEC $INPUTS_FILE $AMREX_ARGS | tee ./Logs/log_run.txt


