#ifndef DIRECTSUMLGF_H_
#define DIRECTSUMLGF_H_

#include <directSumKernel.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_BCRec.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_BLProfiler.H>

struct FabMetaData
{
    // ensure everything in here is Plain Old Data
    int offset;
    amrex::IntVect lo;
    amrex::IntVect hi;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx;
};

class DirectSumLGF
{
    private:

    // parameters required for the direct summation
    amrex::Geometry geom;
    int n_lookup;

    // data from what was once LGFCommBuffers
    // host buffers
    amrex::Vector<FabMetaData> h_local_meta;
    amrex::Vector<int>         h_box_data_offsets;

    // MPI bookkeeping (host; sized to nprocs once)
    amrex::Vector<int> data_counts, data_displs;
    amrex::Vector<int> meta_counts, meta_displs;
    amrex::Vector<int> d_byte_counts, d_byte_displs;
    amrex::Vector<int> m_byte_counts, m_byte_displs;

    // device buffers 
    amrex::Gpu::DeviceVector<amrex::Real>     d_local_data;
    amrex::Gpu::DeviceVector<FabMetaData>     d_local_meta;
    amrex::Gpu::DeviceVector<int>             d_data_displs;
    amrex::Gpu::DeviceVector<int>             d_meta_displs;
    
    public:
    // keeping this public so that multifab creating loop can work
    amrex::Vector<int>         h_source_box_tag_arr;
    
    // data from what was once ConsolidatedData
    amrex::Gpu::DeviceVector<amrex::Real> consolData;
    amrex::Gpu::DeviceVector<FabMetaData> consolMetadata;

    // supporting functions
    void consolidateMultiFab(const amrex::MultiFab& phi, const amrex::Gpu::DeviceVector<int>& source_box_tag_arr);
    void regridOnto(const amrex::Geometry& new_geom, const amrex::BoxArray& new_ba, const amrex::DistributionMapping& new_dm);

    DirectSumLGF(const amrex::Geometry& geom_in, const int n_look_in);
    void solvePoisson(const amrex::MultiFab& source, amrex::MultiFab& target, const amrex::Gpu::DeviceVector<int>& source_box_tag_arr);
    void solveNodalPoisson(const amrex::MultiFab& source, amrex::MultiFab& target, const amrex::Gpu::DeviceVector<int>& source_box_tag_arr);
};

#endif
