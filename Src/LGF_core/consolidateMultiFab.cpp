#include <LGFCore.H>

using namespace amrex;

ConsolidatedData consolidateMultiFab(const amrex::MultiFab& phifab, const amrex::Geometry& geom, amrex::Vector<int> box_tag_arr)
{
    BL_PROFILE("<Communicate> consolidateMultiFab()");

    // Each MPI rank packs its tagged boxes into a flat data vector and metadata vector
    Vector<Real> local_data;
    Vector<FabMetaData> local_meta;

    // Bulk copy from device to a pinned host MultiFab once
    amrex::MultiFab host_mf(phifab.boxArray(), phifab.DistributionMap(), phifab.nComp(), phifab.nGrow(),
                            amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::dtoh_memcpy(host_mf, phifab);

    for (MFIter mfi(host_mf); mfi.isValid(); ++mfi)
    {
        if (box_tag_arr[mfi.LocalIndex()] == 0) continue;

        const Box& bx = mfi.validbox();
        auto const& phi_arr = host_mf.const_array(mfi);

        local_meta.push_back({static_cast<int>(local_data.size()), bx.smallEnd(), bx.bigEnd(), geom.CellSizeArray()});

        // Pack in k-outer, j-middle, i-inner order — must match decode in addEverySourceBox
        for (int k = AMREX_D_PICK(0, 0, bx.smallEnd()[2]); k <= AMREX_D_PICK(0, 0, bx.bigEnd()[2]); ++k)
            for (int j = AMREX_D_PICK(0, bx.smallEnd()[1], bx.smallEnd()[1]); j <= AMREX_D_PICK(0, bx.bigEnd()[1], bx.bigEnd()[1]); ++j)
                for (int i = bx.smallEnd()[0]; i <= bx.bigEnd()[0]; ++i)
                    local_data.push_back(phi_arr(i, j, k));
    }

    const int nprocs = ParallelDescriptor::NProcs();

    // Element counts and element-wise displacements per rank
    // data_displs is nprocs+1 so data_displs[nprocs] gives the total element count
    Vector<int> data_counts(nprocs), data_displs(nprocs + 1, 0);
    Vector<int> meta_counts(nprocs), meta_displs(nprocs + 1, 0);

    const int my_data_size = static_cast<int>(local_data.size());
    const int my_meta_size = static_cast<int>(local_meta.size());

#ifdef BL_USE_MPI
    // --- REPLACED: was Gather-to-root + Bcast (2 round trips each) ---
    // MPI_Allgather: every rank sends its size, every rank receives all sizes in one step
    MPI_Allgather(&my_data_size, 1, MPI_INT,
                  data_counts.data(), 1, MPI_INT,
                  ParallelDescriptor::Communicator());

    MPI_Allgather(&my_meta_size, 1, MPI_INT,
                  meta_counts.data(), 1, MPI_INT,
                  ParallelDescriptor::Communicator());
#else
    // Single-process path: trivially assign
    data_counts[0] = my_data_size;
    meta_counts[0] = my_meta_size;
#endif

    // Build element-wise displacement arrays (used for offset adjustment below)
    for (int i = 0; i < nprocs; ++i)
    {
        data_displs[i+1] = data_displs[i] + data_counts[i];
        meta_displs[i+1] = meta_displs[i] + meta_counts[i];
    }

    // Allocate global receive buffers
    Vector<Real>        global_data(data_displs[nprocs]);
    Vector<FabMetaData> global_meta(meta_displs[nprocs]);

    // Build byte-wise count and displacement arrays for MPI_BYTE transfers.
    // Using MPI_BYTE for both Real data and the FabMetaData struct avoids
    // MPI type-map registration for the struct and handles float/double builds uniformly.
    Vector<int> d_byte_counts(nprocs), d_byte_displs(nprocs + 1, 0);
    Vector<int> m_byte_counts(nprocs), m_byte_displs(nprocs + 1, 0);

    for (int i = 0; i < nprocs; ++i)
    {
        d_byte_counts[i]     = data_counts[i] * static_cast<int>(sizeof(amrex::Real));
        d_byte_displs[i+1]   = d_byte_displs[i] + d_byte_counts[i];

        m_byte_counts[i]     = meta_counts[i] * static_cast<int>(sizeof(FabMetaData));
        m_byte_displs[i+1]   = m_byte_displs[i] + m_byte_counts[i];
    }

#ifdef BL_USE_MPI
    // --- REPLACED: was Gatherv-to-root + Bcast (2 round trips each) ---
    // MPI_Allgatherv: every rank contributes its slice, every rank receives the full buffer

    MPI_Allgatherv(local_data.data(),
                   my_data_size * static_cast<int>(sizeof(amrex::Real)), MPI_BYTE,
                   global_data.data(),
                   d_byte_counts.data(), d_byte_displs.data(), MPI_BYTE,
                   ParallelDescriptor::Communicator());

    MPI_Allgatherv(reinterpret_cast<char*>(local_meta.data()),
                   my_meta_size * static_cast<int>(sizeof(FabMetaData)), MPI_BYTE,
                   reinterpret_cast<char*>(global_meta.data()),
                   m_byte_counts.data(), m_byte_displs.data(), MPI_BYTE,
                   ParallelDescriptor::Communicator());
#else
    std::copy(local_data.begin(),  local_data.end(),  global_data.begin());
    std::copy(local_meta.begin(),  local_meta.end(),  global_meta.begin());
#endif

    // Adjust each box's offset from rank-local to unified global array indexing.
    // Uses element displacements (data_displs), not byte displacements.
    for (int p = 0; p < nprocs; ++p)
        for (int b = meta_displs[p]; b < meta_displs[p+1]; ++b)
            global_meta[b].offset += data_displs[p];

    ConsolidatedData consolPhi;
    consolPhi.data     = std::move(global_data);
    consolPhi.metadata = std::move(global_meta);

    return consolPhi;
}