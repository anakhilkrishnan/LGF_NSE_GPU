#include <LGFCore.H>

using namespace amrex;

void consolidateMultiFab(ConsolidatedData& consolPhi, const amrex::MultiFab& phifab, const amrex::Geometry& geom, amrex::Gpu::DeviceVector<int>& box_tag_arr, LGFCommBuffers& buff)
{
    BL_PROFILE("<Communicate> consolidateMultiFab()");

    const int nprocs = ParallelDescriptor::NProcs();
    const int num_local_boxes = phifab.local_size();

    // copy the box tagging array to host to setup h_local_meta to handle device side copy
    amrex::Vector<int> h_box_tag_arr(num_local_boxes);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, box_tag_arr.begin(), box_tag_arr.end(), h_box_tag_arr.begin());

    int my_data_size = 0;
    int my_meta_size = 0; // Index counter for active boxes
    amrex::Vector<int> box_data_offsets(num_local_boxes, 0);

    // ensure host buffer capacity (only allocates on first run if capacity is low)
    buff.h_local_meta.resize(num_local_boxes);

    for (MFIter mfi(phifab); mfi.isValid(); ++mfi)
    {
        int local_idx = mfi.LocalIndex();
        if (h_box_tag_arr[local_idx] == 0) 
        {
            continue;
        }

        const Box& bx = mfi.validbox();
        
        // track the starting index for this box's floating point data
        box_data_offsets[local_idx] = my_data_size; 

        // direct assignment using index counter (overwrites old data, no push_back)
        buff.h_local_meta[my_meta_size] = {my_data_size, bx.smallEnd(), bx.bigEnd(), geom.CellSizeArray()};

        my_data_size += bx.numPts();
        my_meta_size++;
    }

    // resize device buffers to requested capacity (zero-cost if capacity already exists)
    buff.d_local_data.resize(my_data_size);
    buff.d_local_meta.resize(my_meta_size);

    Real* d_data_ptr = buff.d_local_data.dataPtr();

    // push only the active segment of the metadata array to the device
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, 
                     buff.h_local_meta.begin(), 
                     buff.h_local_meta.begin() + my_meta_size, 
                     buff.d_local_meta.begin());

    // native device packing kernel
    for (MFIter mfi(phifab, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        int local_idx = mfi.LocalIndex();
        if (h_box_tag_arr[local_idx] == 0) 
        {
            continue;
        }

        const Box& bx = mfi.validbox();
        auto const& phi_arr = phifab.const_array(mfi);
        int offset = box_data_offsets[local_idx];
        
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

    Vector<int> data_counts(nprocs), data_displs(nprocs + 1, 0);
    Vector<int> meta_counts(nprocs), meta_displs(nprocs + 1, 0);

#ifdef BL_USE_MPI
    MPI_Allgather(&my_data_size, 1, MPI_INT, data_counts.data(), 1, MPI_INT, ParallelDescriptor::Communicator());
    MPI_Allgather(&my_meta_size, 1, MPI_INT, meta_counts.data(), 1, MPI_INT, ParallelDescriptor::Communicator());
#else
    data_counts[0] = my_data_size;
    meta_counts[0] = my_meta_size;
#endif

    // calculate memory displacements for the global arrays
    for (int i = 0; i < nprocs; ++i)
    {
        data_displs[i+1] = data_displs[i] + data_counts[i];
        meta_displs[i+1] = meta_displs[i] + meta_counts[i];
    }

    int total_data_size = data_displs[nprocs];
    int total_meta_size = meta_displs[nprocs];

    // Resize global target buffers (zero-cost if capacity already exists)
    consolPhi.data.resize(total_data_size);
    consolPhi.metadata.resize(total_meta_size);

    Vector<int> d_byte_counts(nprocs), d_byte_displs(nprocs + 1, 0);
    Vector<int> m_byte_counts(nprocs), m_byte_displs(nprocs + 1, 0);

    for (int i = 0; i < nprocs; ++i)
    {
        d_byte_counts[i]   = data_counts[i] * sizeof(amrex::Real);
        d_byte_displs[i+1] = d_byte_displs[i] + d_byte_counts[i];

        m_byte_counts[i]   = meta_counts[i] * sizeof(FabMetaData);
        m_byte_displs[i+1] = m_byte_displs[i] + m_byte_counts[i];
    }

#ifdef BL_USE_MPI
    // transmitting natively from VRAM to VRAM bypassing the CPU entirely
    MPI_Allgatherv(buff.d_local_data.dataPtr(),
                   d_byte_counts[ParallelDescriptor::MyProc()], MPI_BYTE,
                   consolPhi.data.dataPtr(),
                   d_byte_counts.data(), d_byte_displs.data(), MPI_BYTE,
                   ParallelDescriptor::Communicator());

    MPI_Allgatherv(buff.d_local_meta.dataPtr(),
                   m_byte_counts[ParallelDescriptor::MyProc()], MPI_BYTE,
                   consolPhi.metadata.dataPtr(),
                   m_byte_counts.data(), m_byte_displs.data(), MPI_BYTE,
                   ParallelDescriptor::Communicator());
#else
    amrex::Gpu::copy(amrex::Gpu::deviceToDevice, buff.d_local_data.begin(), buff.d_local_data.begin() + my_data_size, consolPhi.data.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToDevice, buff.d_local_meta.begin(), buff.d_local_meta.begin() + my_meta_size, consolPhi.metadata.begin());
#endif

    // syncs GPUs before offset correction
    amrex::Gpu::streamSynchronize();

    // device side offset updating for correct unpacking
    buff.d_data_displs.resize(nprocs + 1);
    buff.d_meta_displs.resize(nprocs + 1);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, data_displs.begin(), data_displs.end(), buff.d_data_displs.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, meta_displs.begin(), meta_displs.end(), buff.d_meta_displs.begin());

    FabMetaData* global_meta_ptr = consolPhi.metadata.dataPtr();
    int* meta_displs_ptr = buff.d_meta_displs.dataPtr();
    int* data_displs_ptr = buff.d_data_displs.dataPtr();

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