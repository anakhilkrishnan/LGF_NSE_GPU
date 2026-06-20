#!/bin/sh

#SBATCH -N 2
#SBATCH --ntasks-per-node=48
#SBATCH --time=00:10:00
#SBATCH --job-name=LGF_solver_test
#SBATCH --error=job.%J.err_node_48
#SBATCH --output=job.%J.out_node_48
#SBATCH --partition=debug

module load gnu11/11.4.0
module load openmpi/4.1.2

export OMPI_MCA_btl_openib_allow_ib=1
export OMPI_MCA_btl_openib_if_include="mlx5_0:1"

ulimit -s unlimited

cd $SLURM_SUBMIT_DIR
export OMP_NUM_THREADS=1

EXEC="./lgfpoisson"
INPUTS_FILE="inputs"

mpiexec -n $SLURM_NTASKS $EXEC $INPUTS_FILE
