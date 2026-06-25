#!/bin/sh
#SBATCH --job-name=LGF_NSE_GPU_run
#SBATCH --partition=gpumultinode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
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

# OpenMPI parameters
export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx
export UCX_TLS=rc,sm,cuda_copy,cuda_ipc,gdr_copy
export UCX_MEMTYPE_CACHE=n
export UCX_RNDV_THRESH=8192
export AMREX_USE_GPU_AWARE_MPI=1

ulimit -s unlimited
cd $SLURM_SUBMIT_DIR

EXEC="./nserun"
INPUTS_FILE="inputs"

cat > select_gpu.sh <<'EOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
exec "$@"
EOF
chmod +x select_gpu.sh

mpiexec -n $SLURM_NTASKS ./select_gpu.sh $EXEC $INPUTS_FILE