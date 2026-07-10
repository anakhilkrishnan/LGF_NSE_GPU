#include <DomainManager.H>

DomainManager::DomainManager(const SolverConfig& config)
{   
    // at constructor call, a coarse mesh with 'search' params is created
    // this facilictates the direct natural follow up of initializeSnugDomain();
    // alternatively, by excluding the call to any of the functions, the solver
    // can be run on the prespecified search grid.

    // creating domain data objects
    amrex::IntVect dom_lo_iv(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect dom_hi_iv(AMREX_D_DECL(config.n_cell_search-1, config.n_cell_search-1, config.n_cell_search-1));
    amrex::Box domain(dom_lo_iv, dom_hi_iv);

    ba.define(domain);
    ba.maxSize(config.max_grid_size_search);
    
    dm.define(ba);

    amrex::RealBox real_box(config.dom_lo, config.dom_hi);
    amrex::Vector<int> is_periodic(AMREX_SPACEDIM, 0); // infinite domain using zero-grad BC
    geom.define(domain, &real_box, amrex::CoordSys::cartesian, is_periodic.data());

    // initializing domain handling parameters
    supp_tag_eps = config.supp_tag_eps;
    regrid_int = config.regrid_int; // TEMP tuning parameter for now
    n_buffer = config.n_buffer;
}

void DomainManager::initializeSnugDomain()
{
    // create coarse multifab to store velocity and vorticity pass them for
    // tagging use tag information to update Geom, BoxArr, DistMap; IMPORTANT:
    // refine Geom, BoxArr and DistMap to match desired resolution

}

int DomainManager::computeRegridInterval(const FlowField& state) const 
{
    // TODO: q_max = floor(beta * Nb * nb / consumption_rate(state)), asserted >= 1
    AMREX_ALWAYS_ASSERT(regrid_int >= 1);
    return regrid_int;
}

void DomainManager::tagSupportRegion(const FlowField& state, const MultiFab& divU_fine) 
{
    // computes vorticity and divergence of lamb vector tags accordingly and
    // stores in supp_tag_arr
    // IMPORTANT: ensure that the function always fills the tag_arr based on 
    // the MultiFab on which the source field is computed. Ideally it should
    // be Dxsoln (all MultiFabs in the domain need to adhere to this)

    // TEMP checker for upd4-3-1: divU_fine included
    const int num_local_boxes = divU_fine.local_size();

    // ensure capacity matches without forcing a reallocation if it's already sized
    if (supp_tag_arr.size() != num_local_boxes) 
    {
        supp_tag_arr.resize(num_local_boxes);
    }

    // obtain raw pointers for GPU
    int* d_flags_ptr = supp_tag_arr.dataPtr();
    
    // utilize the AMReX compute stream to zero the array natively on the GPU
    amrex::ParallelFor(num_local_boxes, [=] AMREX_GPU_DEVICE (int i) 
    {
        d_flags_ptr[i] = 1;
    });
}

void DomainManager::updateSnugDomain(const FlowField& state)
{
    // reads current flowfield state, grows boxarr outward a bit more to create
    // new search space; tags on updated search space; uses tag informatino to
    // update Geom, BoxArr, DistMap; 
}