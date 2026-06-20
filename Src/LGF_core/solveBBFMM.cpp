#include <vector>

#include <LGFCore.H>
// PENDING: requires more kernels from BBFMM2d=
#include <BBFMM2D.hpp>
#include <LGFKernel.hpp>

void solveFMM(const amrex::MultiFab& source, amrex::MultiFab& target, const amrex::Geometry& geom, int n_chebyshev, int n_lookup) 
{
    // Profiling block for AMReX's TinyProfiler
    BL_PROFILE("<Compute> solveFMM()");

    // packing step: take data from AMReX MultiFabs for cell locations
    // and convert them to struct Point and store as std::vector<Point>

    // BBFMM2d considers source locations = target locations
    // PENDING: box tagging has not been worked out for this yet

    const amrex::Box& domain = geom.Domain();

    int nx = domain.length(0);
    int ny = domain.length(1);

    // computing values to include ghost cells in FMM computations
    int n_ghost = target.nGrow();
    int nx_ghost = nx + (2 * n_ghost);
    int ny_ghost = ny + (2 * n_ghost);

    unsigned long N_total = nx_ghost * ny_ghost;

    std::vector<Point> all_locations(N_total);
    std::vector<double> local_charges(N_total, 0.0);

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    amrex::Real dvol = dx[0] * dx[1];

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo_ghost;
    prob_lo_ghost[0] = prob_lo[0] - (n_ghost * dx[0]);
    prob_lo_ghost[1] = prob_lo[1] - (n_ghost * dx[1]);

    // fill all_locations, including ghost cells
    for (int j = 0; j < ny_ghost; ++j)
    {
        amrex::Real y_src = prob_lo_ghost[1] + (j + 0.5) * dx[1];

        for (int i = 0; i < nx_ghost; ++i)
        {
            amrex::Real x_src = prob_lo_ghost[0] + (i + 0.5) * dx[0];

            int flat_idx = j * nx_ghost + i;
            all_locations[flat_idx] = {x_src, y_src};
        }
    }

    // fill local_charges, only in validbox()
    for (amrex::MFIter mfi(source); mfi.isValid(); ++mfi) 
    {
        const amrex::Box& valid_box = mfi.validbox();
        auto const& phi_arr = source.const_array(mfi);

        for (int j = valid_box.smallEnd()[1]; j <= valid_box.bigEnd()[1]; ++j) 
        {
            for (int i = valid_box.smallEnd()[0]; i <= valid_box.bigEnd()[0]; ++i) 
            {
                // Shift AMReX index to match the expanded flat array index
                int shifted_i = i + n_ghost;
                int shifted_j = j + n_ghost;
                int flat_idx = shifted_j * nx_ghost + shifted_i;

                local_charges[flat_idx] = phi_arr(i, j, 0) * dvol; 
            }
        }
    }
    
    std::vector<double> global_charges(N_total, 0.0);
    // because box-tagging is ignored for this implementation, no packing is needed
    // before casting
    // IMPORTANT: This adds another layer of issues. Since the target = source
    // location, each MPI rank essentially performs the full tree computation
    // independent of each other. The original goal of splitting atleast the 
    // target cells and using their results is only possible if the solver
    // can take separate source and target locations

#ifdef AMREX_USE_MPI
    // Because the arrays are identical size and rigidly ordered (dense), 
    // we can just use MPI_Allreduce to sum the arrays across all ranks
    MPI_Allreduce(local_charges.data(), global_charges.data(), N_total, 
                  MPI_DOUBLE, MPI_SUM, amrex::ParallelDescriptor::Communicator());
#else
    global_charges = local_charges;
#endif

    // setup the BBFMM solver 
    unsigned short nChebNodes = n_chebyshev; 
    unsigned m = 1;                 

    // Instantiate the tree without any padded targets!
    H2_2D_Tree myTree(nChebNodes, global_charges.data(), all_locations, N_total, m);

    // instantiate LGF Kernel that inherits pre-built Log kernel
    // explicitly casted into double in case AMReX is compiled as float
    kernel_LGF myKernel((double)dx[0], (double)dx[1], n_lookup); 

    // Evaluate
    std::vector<double> calculated_potentials(N_total, 0.0);
    myKernel.calculate_Potential(myTree, calculated_potentials.data());

    // unpacking step: take the long vector of computed potentials and reconstruct
    // the target MultiFab 

    for (amrex::MFIter mfi(target, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) 
    {
        const amrex::Box& grown_box = mfi.growntilebox(n_ghost);
        auto const& tar_arr = target.array(mfi);

        for (int j = grown_box.smallEnd()[1]; j <= grown_box.bigEnd()[1]; ++j) 
        {
            for (int i = grown_box.smallEnd()[0]; i <= grown_box.bigEnd()[0]; ++i) 
            {
                // Shift AMReX index to match the expanded flat index
                int shifted_i = i + n_ghost;
                int shifted_j = j + n_ghost;
                int flat_idx = shifted_j * nx_ghost + shifted_i;

                // update corrected target values
                tar_arr(i, j, 0) = calculated_potentials[flat_idx];
            }
        }
    }
}