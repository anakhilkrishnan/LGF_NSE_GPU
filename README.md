# 3D Unbounded-Domain Incompressible Viscous Flow Solver

Akhil Krishnan\
November 11, 2026\
Indian Institute of Science

[GitHub Repository](https://github.com/anakhilkrishnan/LGF_NSE_GPU.git)

## Introduction

The solver is written using [AMReX](https://amrex-codes.github.io). Consequently, the code is performant and portable across architectures. The solver is written to preserve AMReX's own performance benefits without introducing any  bottlenecks. 

The methods implemented in the solver are extensions to Liska and Colonius' work on [Unbounded Flow Solvers using LGFs](https://www.sciencedirect.com/science/article/pii/S002199911630626X), at Caltech.

The solver is part of a larger developmental process of applying LGF-based methods to simulate multiphase flows, flows past solid objects, bubbles and bubble clusters, and so on,  with a sharp interface treatment. A secondary goal is to keep the solver massively parallelizable on hundreds of GPUs. 

Development of the code began in March 2026, as a CFD course project. The progress thus far is the culmination of countless inputs from my PhD advisor, lab mates, friends, and popular LLM's, paired with long hours of writing, debugging, rewriting, and refining code (all while having 5 different code explanation websites open), and having a supercomputer to test and run those codes on. Ofcourse, none of this would be possible without the support of my loved ones, and my sister.

## Using the code

Building and running the code requires the following:
1. CUDA Toolkits/NVHPC kit
2. OpenMPI, OpenMP if desired
3. CMake (build system)
4. Latest GCC
5. FFTW, for CPU builds (cuFFT used for GPU builds)

To build the executable:
```bash
## start in the parent directory containing CMakeLists.txt and Src directory
mkdir Exec && cd Exec

#(a) for OpenMP CPU executable
cmake -B build_omp -S .. -DAMReX_SPACEDIM=3 -DAMReX_GPU_BACKEND=NONE

#(b) for pure MPI (no OpenMP) CPU executable
cmake -B build_mpi -S .. -DAMReX_SPACEDIM=3 -DAMReX_GPU_BACKEND=NONE -DAMReX_OMP=OFF

#(c) for GPU executable 
cmake -B build_gpu -S .. -DAMReX_SPACEDIM=3 -DAMReX_GPU_BACKEND=CUDA

## this creates the build dir and NSE_test inside Exec. Complete the build process
cmake --build <build_dir_name> -j <n_procs>
```

AMReX is built along with the solver by default, pre-existing AMReX files can be provided using `-DFETCHCONTENT_SOURCE_DIR_AMREX_CODE=<path-to-'amrex'>` flag in build dir creation step.

To run the executable:
```bash
## good practice: do not alter in NSE_test directory, make a copy of NSE_test with an apt name
cp -r NSE_test NSE_threeringrun && cd NSE_threeringrun

## edit inputs file
vim inputs

## good practice: update the description.txt file with a brief description of the particular run, with context on what this run achieves
vim description.txt

#(a) run directly
mpiexec -n <nprocs> nserun inputs | tee ./Logs/log.txt

#(b) submit a jobscript on SLURM (many options available: cpu_multi.sh, gpu_multi.sh, gpu_multi_rudra.sh)
sbatch <jobscript>.sh

## all plot/log/checkpoint files are written to folders specified in inputs file, typical values are:

#(a)plotsfiles are written in Results/
ls Results/ # shows all files written so far

#(b) logfiles are written in Logs/
vim Logs/log.txt # opens log file in vim

#(c) checkpoint files (main and alt) are written to Checkpoints/
vim Checkpoints/main/status.dat # shows if checkpoint was written safely without interrupts and saved time step
```

## Contributing to the code

Any suggestions and bug fixes are welcome. You may reach out to the author via email (kakhil@iisc.ac.in). Alternatively, you may create a new branch and raise a pull request.

Please do not push/force push changes to main.
