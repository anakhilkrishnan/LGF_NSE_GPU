#!/bin/sh

#SBATCH --job-name=LGF_NSE_GPU_run
#SBATCH --partition=gpusinglenode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:15:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

# Ensure the correctly configured CUDAROOT path is loaded
source ~/.bashrc

# Environment Initialization 
# (PARAM Pravega often utilizes 'spack load' alongside standard modules, adjust if required)
module load openmpi/openmpi_4.0.5_ucx_cuda_11.2_with_gcc

# MPI and Networking Parameters
export OMPI_MCA_btl_openib_allow_ib=1
export OMPI_MCA_btl_openib_if_include="mlx5_0:1"
export OMP_NUM_THREADS=1

# export AMREX_USE_GPU_AWARE_MPI=1

ulimit -s unlimited

cd $SLURM_SUBMIT_DIR

EXEC="./nserun"
INPUTS_FILE="inputs"

mpiexec -n $SLURM_NTASKS $EXEC $INPUTS_FILE