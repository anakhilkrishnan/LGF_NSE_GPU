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

# Environment
module purge
module load cmake/3.27.7
module load spack/0.17
. /home-ext/apps/spack/share/spack/setup-env.sh
module load nvhpc/23.9-gcc-13.1.0-go44
module load gcc/11.2.0-gcc-4.8.5-yqde

NVROOT=$(dirname $(dirname $(which nvc++)))
MPIROOT=$NVROOT/../comm_libs/12.2/openmpi4/openmpi-4.1.5
CUDALIBS=$NVROOT/../math_libs/12.2/targets/x86_64-linux/lib
CUDART=$NVROOT/../cuda/12.2/targets/x86_64-linux/lib
export PATH=$MPIROOT/bin:$PATH
export LD_LIBRARY_PATH=$MPIROOT/lib:$CUDALIBS:$CUDART:$LD_LIBRARY_PATH

export CC=gcc CXX=g++ CUDAHOSTCXX=g++ OMPI_CC=gcc OMPI_CXX=g++

# Local FFTW build
FFTWROOT=$HOME/softwares/fftw/3.3.10-gcc11.2
export LD_LIBRARY_PATH=$FFTWROOT/lib:$LD_LIBRARY_PATH

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