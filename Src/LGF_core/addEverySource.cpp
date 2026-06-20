#include <LGFCore.H>

using namespace amrex;

void addEverySourceBox(const amrex::MultiFab& source, amrex::MultiFab& target, const amrex::Geometry& geom, amrex::Vector<int> box_tag_arr, int n_lookup) 
{
    // adding profiling blocks for Tiny/Base profilers
    BL_PROFILE("<Compute> addEverySourceBox()");

    //extract cell-sizes and physical dom_lo for x,y,z computations
    GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    // Read data from the source MultiFab and make it available to all processes
    ConsolidatedData consolSource = consolidateMultiFab(source, geom, box_tag_arr);
    
    // allocating space in VRAM for the source data and metadata
    amrex::Gpu::DeviceVector<Real> d_data(consolSource.data.size());
    amrex::Gpu::DeviceVector<FabMetaData> d_meta(consolSource.metadata.size());

    // copying the consolidated data from CPU to GPU
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, 
                     consolSource.data.begin(), consolSource.data.end(), 
                     d_data.begin());
                     
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, 
                     consolSource.metadata.begin(), consolSource.metadata.end(), 
                     d_meta.begin());

    // the above lines collapse into CPU use when compiled without USE_CUDA=TRUE

    // export the consolidated data as pointers to the target MFIter
    int num_blocks = consolSource.metadata.size();
    const Real* data_ptr = d_data.data();
    const FabMetaData* meta_ptr = d_meta.data();

    amrex::Box dom = geom.Domain();
    
    // Loop over target boxes in a separate MFIter
#ifdef AMREX_USE_OMP
    #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(target, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& targetbox = mfi.growntilebox(target.nGrow());
        const amrex::Box& valid_box = mfi.tilebox();
        const Array4<Real>& phi = target.array(mfi);

        amrex::ParallelFor(targetbox, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {   
            amrex::IntVect cell(AMREX_D_DECL(i,j,k));
            
            if (!valid_box.contains(cell))
            {
                if (dom.contains(cell))
                {
                    return;
                }
            }
            // extract physical coordinates of target cell
            amrex::Real AMREX_D_DECL(x_tar = prob_lo[0] + (i + 0.5) * dx[0],
                                      y_tar = prob_lo[1] + (j + 0.5) * dx[1],
                                      z_tar = prob_lo[2] + (k + 0.5) * dx[2]);

            amrex::Real total_contribution = 0.0;

            // Iterate over every source block gathered from all MPI processes
            for (int b = 0; b < num_blocks; ++b) 
            {
                const auto& block = meta_ptr[b];
                int idx = block.offset;

                amrex::Real dvol = AMREX_D_TERM(block.dx[0], * block.dx[1], * block.dx[2]);
                
                // Unpack and sum every cell in the source block
                for (int sk = AMREX_D_PICK(0, 0, block.lo[2]); sk <= AMREX_D_PICK(0, 0, block.hi[2]); ++sk) 
                {
                    // conditionally extract physical coordinates of source cell
                    #if AMREX_SPACEDIM == 3
                        amrex::Real z_src = prob_lo[2] + ((sk + 0.5) * block.dx[2]);
                    #endif

                    for (int sj = AMREX_D_PICK(0, block.lo[1], block.lo[1]); sj <= AMREX_D_PICK(0, block.hi[1], block.hi[1]); ++sj) 
                    {
                        #if AMREX_SPACEDIM >= 2
                            amrex::Real y_src = prob_lo[1] + ((sj + 0.5) * block.dx[1]);
                        #endif

                        for (int si = block.lo[0]; si <= block.hi[0]; ++si) 
                        {
                            amrex::Real x_src = prob_lo[0] + ((si + 0.5) * block.dx[0]);
                            
                            // compute the LGF kernel for the current source-target cell pair
                            amrex::Real lgf = computeLGF(n_lookup, 
                                                        AMREX_D_DECL(x_tar, y_tar, z_tar),
                                                        AMREX_D_DECL(x_src, y_src, z_src),
                                                        AMREX_D_DECL(block.dx[0], block.dx[1], block.dx[2]));
                                                            
                            
                            
                            // add the contribution of source cell based on lgf
                            total_contribution += (data_ptr[idx++] * lgf * dvol);
                        }
                    }
                }
            }
            phi(i, j, k) = total_contribution;
        });
    }

    // wait for the GPU to synchnorize
    amrex::Gpu::streamSynchronize();
}