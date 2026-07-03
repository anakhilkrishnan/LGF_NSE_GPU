#include <directSumLGF.H>

using namespace amrex;

void tagSource(amrex::Gpu::DeviceVector<int>& box_tag_arr, const amrex::MultiFab& phi, const amrex::Real tag_thresh)
{
    // perform grid tagging by assigning an int to each box
    // 0 = to be excluded during packing
    // 1 = to be packed and shipped

    // adding profiling blocks for Tiny/Base profilers
    BL_PROFILE("<Communicate> tagSource()");

    const int num_local_boxes = phi.local_size();

    // ensure capacity matches without forcing a reallocation if it's already sized
    if (box_tag_arr.size() != num_local_boxes) 
    {
        box_tag_arr.resize(num_local_boxes);
    }

    // obtain raw pointers for GPU
    int* d_flags_ptr = box_tag_arr.dataPtr();
    
    // utilize the AMReX compute stream to zero the array natively on the GPU
    amrex::ParallelFor(num_local_boxes, [=] AMREX_GPU_DEVICE (int i) 
    {
        d_flags_ptr[i] = 0;
    });

    auto const& ma = phi.const_arrays();   // MultiArray4: all local boxes

    amrex::ParallelFor(phi, [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k)
    {
        if (d_flags_ptr[box_no] != 0) return;   // early-out, now indexed by box_no
        if (amrex::Math::abs(ma[box_no](i,j,k)) > tag_thresh)
        {
            amrex::Gpu::Atomic::Max(&d_flags_ptr[box_no], 1);
        }
    });
}