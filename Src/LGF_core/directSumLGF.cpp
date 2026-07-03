#include <directSumLGF.H>

directSumLGF::directSumLGF(const amrex::Geometry& geom_in, const int n_look_in) 
    : geom(geom_in), n_lookup(n_look_in)
{

}

// Note: This function does not use tiling because all indexing and box tag reading requires use of boxes as the smallest
// divisible unit.
// Note: This function isn't given any openmp support because half of it requires serial looping and the other half isn't
// a bottleneck at all. If it does turn out to be, one can add over the MFIter calling ParallelFor() the desired
// pragma openmp for cpu builds
void directSumLGF::consolidateMultiFab(const amrex::MultiFab& phi, const amrex::Gpu::DeviceVector<int>& source_box_tag_arr)
{
    BL_PROFILE("<Communicate> consolidateMultiFab()");

    const int nprocs = amrex::ParallelDescriptor::NProcs();
    const int num_local_boxes = phi.local_size();

    // copy the box tagging array to host to setup h_local_meta to handle device side copy
    h_source_box_tag_arr.resize(num_local_boxes);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, source_box_tag_arr.begin(), source_box_tag_arr.end(), h_source_box_tag_arr.begin());

    int my_data_size = 0;
    int my_meta_size = 0; // Index counter for active boxes

    // ensure host buffer capacity (only allocates on first run if capacity is low)
    h_box_data_offsets.resize(num_local_boxes);
    h_local_meta.resize(num_local_boxes);

    for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi)
    {
        const int local_idx = mfi.LocalIndex();
        if (h_source_box_tag_arr[local_idx] == 0) 
        {
            continue;
        }

        const amrex::Box& bx = mfi.validbox();
        
        // track the starting index for this box's floating point data
        h_box_data_offsets[local_idx] = my_data_size; 

        // direct assignment using index counter (overwrites old data, no push_back)
        h_local_meta[my_meta_size] = {my_data_size, bx.smallEnd(), bx.bigEnd(), geom.CellSizeArray()};

        my_data_size += bx.numPts();
        my_meta_size++;
    }

    // resize device buffers to requested capacity (zero-cost if capacity already exists)
    d_local_data.resize(my_data_size);
    d_local_meta.resize(my_meta_size);

    amrex::Real* d_data_ptr = d_local_data.dataPtr();

    // push only the active segment of the metadata array to the device
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, 
                     h_local_meta.begin(), 
                     h_local_meta.begin() + my_meta_size, 
                     d_local_meta.begin());

    // native device packing kernel
    for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi)
    {
        const int local_idx = mfi.LocalIndex();
        if (h_source_box_tag_arr[local_idx] == 0) 
        {
            continue;
        }

        const amrex::Box& bx = mfi.validbox();
        auto const& phi_arr = phi.const_array(mfi);
        const int offset = h_box_data_offsets[local_idx];
        
        const auto lo = bx.smallEnd();
        const auto len = bx.length();

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // flatten 3D coordinates to 1D continuous memory array
            int ii = i - lo[0];
            int jj = AMREX_SPACEDIM >= 2 ? j - lo[1] : 0;
            int kk = AMREX_SPACEDIM == 3 ? k - lo[2] : 0;

            int len_x = len[0];
            int len_y = AMREX_SPACEDIM >= 2 ? len[1] : 1;

            int flat_idx = ii + (jj * len_x) + (kk * len_x * len_y);

            d_data_ptr[offset + flat_idx] = phi_arr(i, j, k);
        });
    }

    // sync gpus before starting mpi sharing
    amrex::Gpu::streamSynchronize();

    // size the MPI bookkeeping vectors once; subsequent calls are no-ops
    data_counts.resize(nprocs);
    meta_counts.resize(nprocs);
    data_displs.resize(nprocs + 1);
    meta_displs.resize(nprocs + 1);
    d_byte_counts.resize(nprocs);
    m_byte_counts.resize(nprocs);
    d_byte_displs.resize(nprocs + 1);
    m_byte_displs.resize(nprocs + 1);

#ifdef BL_USE_MPI
    MPI_Allgather(&my_data_size, 1, MPI_INT, data_counts.data(), 1, MPI_INT, amrex::ParallelDescriptor::Communicator());
    MPI_Allgather(&my_meta_size, 1, MPI_INT, meta_counts.data(), 1, MPI_INT, amrex::ParallelDescriptor::Communicator());
#else
    data_counts[0] = my_data_size;
    meta_counts[0] = my_meta_size;
#endif

    data_displs[0] = 0;
    meta_displs[0] = 0;
    d_byte_displs[0] = 0;
    m_byte_displs[0] = 0;

    // calculate memory displacements for the global arrays
    for (int i = 0; i < nprocs; ++i)
    {
        data_displs[i+1] = data_displs[i] + data_counts[i];
        meta_displs[i+1] = meta_displs[i] + meta_counts[i];

        d_byte_counts[i]   = data_counts[i] * sizeof(amrex::Real);
        m_byte_counts[i]   = meta_counts[i] * sizeof(FabMetaData);
        d_byte_displs[i+1] = d_byte_displs[i] + d_byte_counts[i];
        m_byte_displs[i+1] = m_byte_displs[i] + m_byte_counts[i];
    }

    const int total_data_size = data_displs[nprocs];
    const int total_meta_size = meta_displs[nprocs];

    // Resize global target buffers (zero-cost if capacity already exists)
    consolData.resize(total_data_size);
    consolMetadata.resize(total_meta_size);

#ifdef BL_USE_MPI
    // transmitting natively from VRAM to VRAM bypassing the CPU entirely
    MPI_Allgatherv(d_local_data.dataPtr(),
                   d_byte_counts[amrex::ParallelDescriptor::MyProc()], MPI_BYTE,
                   consolData.dataPtr(),
                   d_byte_counts.data(), d_byte_displs.data(), MPI_BYTE,
                   amrex::ParallelDescriptor::Communicator());

    MPI_Allgatherv(d_local_meta.dataPtr(),
                   m_byte_counts[amrex::ParallelDescriptor::MyProc()], MPI_BYTE,
                   consolMetadata.dataPtr(),
                   m_byte_counts.data(), m_byte_displs.data(), MPI_BYTE,
                   amrex::ParallelDescriptor::Communicator());
#else
    amrex::Gpu::copy(amrex::Gpu::deviceToDevice,
                     d_local_data.begin(),
                     d_local_data.begin() + my_data_size,
                     consolData.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToDevice,
                     d_local_meta.begin(),
                     d_local_meta.begin() + my_meta_size,
                     consolMetadata.begin());
#endif

    // syncs GPUs before offset correction
    amrex::Gpu::streamSynchronize();

    // device side offset updating for correct unpacking
    d_data_displs.resize(nprocs + 1);
    d_meta_displs.resize(nprocs + 1);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                     data_displs.begin(), data_displs.end(),
                     d_data_displs.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                     meta_displs.begin(), meta_displs.end(),
                     d_meta_displs.begin());

    FabMetaData* global_meta_ptr = consolMetadata.dataPtr();
    int* meta_displs_ptr = d_meta_displs.dataPtr();
    int* data_displs_ptr = d_data_displs.dataPtr();

    if (total_meta_size > 0)
    {
        amrex::ParallelFor(total_meta_size, [=] AMREX_GPU_DEVICE (int b)
        {
            int proc = 0;

            // loop to find which proc the metadata from total_meta_size belongs to and appending appropriately
            for (int p = 0; p < nprocs; ++p) 
            {
                if (b >= meta_displs_ptr[p] && b < meta_displs_ptr[p+1]) 
                {
                    proc = p;
                    break;
                }
            }
            // align block offsets natively on the GPU
            global_meta_ptr[b].offset += data_displs_ptr[proc];
        });
    }
}

void directSumLGF::solvePoisson(const amrex::MultiFab& source, amrex::MultiFab& target, const amrex::Gpu::DeviceVector<int>& source_box_tag_arr)
{
    // adding profiling blocks for Tiny/Base profilers
    BL_PROFILE("<Compute> solvePoisson()");

    //extract cell-sizes and physical dom_lo for x,y,z computations
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    // Read data from the source MultiFab and make it available to all processes
    consolidateMultiFab(source, source_box_tag_arr);

    // export the consolidated data as pointers to the target MFIter
    int num_blocks = consolMetadata.size();
    const amrex::Real* data_ptr = consolData.dataPtr();
    const FabMetaData* meta_ptr = consolMetadata.dataPtr();

    amrex::Box dom = geom.Domain();
    
    // Loop over target boxes in a separate MFIter
#ifdef AMREX_USE_OMP
    #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(target, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& targetbox = mfi.growntilebox(target.nGrow());
        const amrex::Box& valid_box = mfi.tilebox();
        const amrex::Array4<amrex::Real>& phi = target.array(mfi);

        const int n_lookup_local = n_lookup;

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
                            amrex::Real lgf = computeLGF(n_lookup_local, 
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
}