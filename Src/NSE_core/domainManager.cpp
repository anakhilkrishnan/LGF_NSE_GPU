#include <DomainManager.H>

DomainManager::DomainManager(const SolverConfig& config)
{   
    // at constructor call, a coarse mesh with 'search' params is created
    // this facilictates the direct natural follow up of initializeSnugDomain();
    // alternatively, by excluding the call to any of the functions, the solver
    // can be run on the prespecified search grid.

    // initializing domain handling parameters
    regrid_int = config.regrid_int; // TEMP tuning parameter for now

    // initializing domain parameters needed from config
    n_cell = config.n_cell;
    max_grid_size = config.max_grid_size;
    dom_lo = config.dom_lo;
    dom_hi = config.dom_hi;
    n_buffer_box = config.n_buffer_box;
    search_to_fine_ref_ratio = config.search_to_fine_ref_ratio;
    supp_tag_eps = config.supp_tag_eps;


    // creating domain data objects
    amrex::IntVect dom_lo_iv(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect dom_hi_iv(AMREX_D_DECL(config.n_cell_search-1, config.n_cell_search-1, config.n_cell_search-1));
    amrex::Box domain(dom_lo_iv, dom_hi_iv);

    ba.define(domain);
    ba.maxSize(config.max_grid_size_search);
    
    dm.define(ba);

    amrex::RealBox real_box(dom_lo, dom_hi);
    amrex::Vector<int> is_periodic(AMREX_SPACEDIM, 0); // infinite domain using zero-grad BC
    geom.define(domain, &real_box, amrex::CoordSys::cartesian, is_periodic.data());

    DSupp.define(ba, dm, 1, 0);
    DSupp.setVal(0.0);
}

amrex::BoxArray DomainManager::gatherBoxArr(const FlowField& state)
{
    // ensuring the h_tag_arr is updated with respect to state
    AMREX_ALWAYS_ASSERT(h_tag_arr.size() == state.getPres().local_size());

    // create a box vector containing only the tagged boxes
    amrex::Vector<amrex::Box> local_tagged_boxes;
    {
        int local_i = 0;
        for (amrex::MFIter mfi(state.getPres()); mfi.isValid(); ++mfi, ++local_i)
        {
            if (h_tag_arr[local_i] != 0) {
                local_tagged_boxes.push_back(mfi.validbox());   // the coarse Box for this tagged cell-region
            }
        }
    }

    // gather across all ranks for boxes, updates in place and create box list
    amrex::AllGatherBoxes(local_tagged_boxes);
    amrex::BoxList bl(std::move(local_tagged_boxes));
    amrex::BoxArray tag_ba(std::move(bl));

    return tag_ba;
}

void DomainManager::growBoxArr(amrex::BoxArray& tag_ba, int nBuff, int max_grid_size)
{
    tag_ba.grow(nBuff);          // grow each box (now overlapping)
    tag_ba.removeOverlap(true);     // BoxArray method: removes overlap AND simplifies
    tag_ba.maxSize(max_grid_size);  // re-chunk to compute box size
}

void DomainManager::updateGeomBaDm(amrex::BoxArray& tag_ba)
{    
    // update members
    ba = tag_ba;
    dm.define(ba);

    // geom: same physical RealBox, but FINE resolution domain box
    amrex::Box fine_domain(amrex::IntVect(0), amrex::IntVect(AMREX_D_DECL(n_cell-1, n_cell-1, n_cell-1)));
    amrex::RealBox real_box({AMREX_D_DECL(dom_lo[0], dom_lo[1], dom_lo[2])}, {AMREX_D_DECL(dom_hi[0], dom_hi[1], dom_hi[2])});

    amrex::Vector<int> is_periodic(AMREX_SPACEDIM, 0);
    geom.define(fine_domain, &real_box, amrex::CoordSys::cartesian, is_periodic.data());
}

const amrex::MultiFab& DomainManager::refreshAndGetDSuppFab()
{
    // redefine DSupp to match current grid
    DSupp.define(ba, dm, 1, 0);

    // export tagging data into plotting multifab
    for (MFIter mfi(DSupp); mfi.isValid(); ++mfi) 
    {
        const amrex::Real v = (h_tag_arr[mfi.LocalIndex()] == 1) ? 1.0 : 0.0;
        DSupp[mfi].setVal<RunOn::Device>(v);
    }

    return DSupp;   // by REFERENCE
}

void DomainManager::initializeSnugDomain(const SolverConfig& config)
{
    BL_PROFILE("<Compute>initializeSnugDomain()")
    // create coarse multifab to store velocity and vorticity pass them for
    // tagging use tag information to update Geom, BoxArr, DistMap; IMPORTANT:
    // refine Geom, BoxArr and DistMap to match desired resolution

    // create FlowField data using search params
    FlowField search_state(geom, ba, dm, config);

    // initializing coarse search domain with velocity
    initializeVelField(search_state);
    search_state.setBoundary();

    tagSupportRegion(search_state);

    amrex::BoxArray tag_ba = gatherBoxArr(search_state);
    
    tag_ba.refine(search_to_fine_ref_ratio);   // now at fine resolution

    growBoxArr(tag_ba, n_buffer_box * max_grid_size, max_grid_size);
    updateGeomBaDm(tag_ba);
}

int DomainManager::computeRegridInterval(const FlowField& state) const 
{
    // TODO: q_max = floor(beta * Nb * nb / consumption_rate(state)), asserted >= 1
    AMREX_ALWAYS_ASSERT(regrid_int >= 1);
    return regrid_int;
}

void DomainManager::tagSupportRegion(const FlowField& state) 
{
    BL_PROFILE("<Compute>tagSupportRegion()")
    // computes vorticity and divergence of lamb vector tags accordingly and
    // stores in supp_tag_arr
    // IMPORTANT: ensure that the function always fills the tag_arr based on 
    // the MultiFab on which the source field is computed. Ideally it should
    // be Dxsoln (all MultiFabs in the domain need to adhere to this)

    // this call is feasible because we ensure that at any given time, vort, divN and state.getPres()
    // all live on the same ba and dm
    vort = computePlotVorticity(state);
    divN = computeDivNonLinearTerm(state);

    // normalize vorticity by its global max (DECIDE: 3D magnitude, not comp 0)
    amrex::Real vort_max_norm = vort.norm0(0, 0, false);
    if (vort_max_norm > 0.0)
    {
        vort.mult(1.0 / vort_max_norm, 0, 1, vort.nGrow());
    }

    // normalize lamb-divergence by its global max
    amrex::Real divN_max_norm = divN.norm0(0, 0, false);
    if (divN_max_norm > 0.0)
    {
        divN.mult(1.0 / divN_max_norm, 0, 1, divN.nGrow());
    }

    // create a tagging criterion for "where the action is"
    const int num_local_boxes = divN.local_size();

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
        d_flags_ptr[i] = 0;
    });

    // local copy for GPU lambda
    auto const& vort_arrs = vort.const_arrays();   // MultiArray4: all local boxes
    auto const& divN_arrs = divN.const_arrays();
    amrex::Real tag_thresh = supp_tag_eps;

    amrex::ParallelFor(divN, [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k)
    {
        if (d_flags_ptr[box_no] != 0) return;   // early-out, now indexed by box_no
        if (amrex::max(amrex::Math::abs(vort_arrs[box_no](i,j,k)), amrex::Math::abs(divN_arrs[box_no](i,j,k))) > tag_thresh)
        {
            amrex::Gpu::Atomic::Max(&d_flags_ptr[box_no], 1);
        }
    });
    
    // update h_tag_arr for the box aggregation and plotting
    h_tag_arr.resize(supp_tag_arr.size());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, supp_tag_arr.begin(), supp_tag_arr.end(), h_tag_arr.begin());
}

void DomainManager::updateSnugDomain(const FlowField& state)
{
    BL_PROFILE("<Compute>updateSnugDomain()")
    // reads current flowfield state, grows boxarr outward a bit more to create
    // new search space; tags on updated search space; uses tag informatino to
    // update Geom, BoxArr, DistMap; 

    tagSupportRegion(state);

    amrex::BoxArray tag_ba = gatherBoxArr(state);

    growBoxArr(tag_ba, n_buffer_box * max_grid_size, max_grid_size);

    updateGeomBaDm(tag_ba);
}

// Computes cell-centered vorticity for plotting, via the staggered discrete
// curl (discreteCurlF2E) followed by native averaging to cell centers.
// The edge/node intermediate is the SAME vorticity vort2vel consumes.
amrex::MultiFab computePlotVorticity(const FlowField& state)
{
    BL_PROFILE("computePlotVorticity()");

    const amrex::Geometry& geom = state.getGeom();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx = geom.InvCellSizeArray();

    const amrex::BoxArray& ba = state.getPres().boxArray();
    const amrex::DistributionMapping& dm = state.getPres().DistributionMap();

    const int ncomp = (AMREX_SPACEDIM == 2) ? 1 : 3;
    amrex::MultiFab vort_cc(ba, dm, ncomp, 0);

    // package velocity Array4s once (per-box handles fetched inside MFIter)
    // ---- build the staggered curl, then average to cell centers ----

#if AMREX_SPACEDIM == 2
    // 2D: omega_z lives at NODES. One nodal MultiFab.
    amrex::BoxArray ba_nd = amrex::convert(ba, amrex::IntVect::TheNodeVector());
    amrex::MultiFab vort_nd(ba_nd, dm, 1, 0);

    for (amrex::MFIter mfi(vort_nd, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& out = vort_nd.array(mfi);
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel{
            state.getVel(0).const_array(mfi),
            state.getVel(1).const_array(mfi)
        };
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            out(i,j,k) = discreteCurlF2E<2>(i, j, k, invdx, vel);  // omega_z at node
        });
    }
    amrex::average_node_to_cellcenter(vort_cc, 0, vort_nd, 0, ncomp, 0);

#elif AMREX_SPACEDIM == 3
    // 3D: each omega component lives on a DIFFERENT edge type.
    // omega_x on x-edges (nodal in y,z), omega_y on y-edges, omega_z on z-edges.
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> vort_edge;
    for (int n = 0; n < AMREX_SPACEDIM; ++n)
    {
        // edge type for component n: nodal in the two directions != n
        amrex::IntVect etype = amrex::IntVect::TheNodeVector();
        etype[n] = 0;  // cell-centered along axis n -> that axis's edge
        amrex::BoxArray ba_e = amrex::convert(ba, etype);
        vort_edge[n].define(ba_e, dm, 1, 0);
    }

    for (amrex::MFIter mfi(vort_cc, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel{
            state.getVel(0).const_array(mfi),
            state.getVel(1).const_array(mfi),
            state.getVel(2).const_array(mfi)
        };
        auto const& ox = vort_edge[0].array(mfi);
        auto const& oy = vort_edge[1].array(mfi);
        auto const& oz = vort_edge[2].array(mfi);

        // each component on its own edge box
        const amrex::Box bx0 = mfi.tilebox(vort_edge[0].ixType().toIntVect());
        const amrex::Box bx1 = mfi.tilebox(vort_edge[1].ixType().toIntVect());
        const amrex::Box bx2 = mfi.tilebox(vort_edge[2].ixType().toIntVect());

        amrex::ParallelFor(bx0, [=] AMREX_GPU_DEVICE (int i,int j,int k){
            ox(i,j,k) = discreteCurlF2E<0>(i,j,k,invdx,vel);
        });
        amrex::ParallelFor(bx1, [=] AMREX_GPU_DEVICE (int i,int j,int k){
            oy(i,j,k) = discreteCurlF2E<1>(i,j,k,invdx,vel);
        });
        amrex::ParallelFor(bx2, [=] AMREX_GPU_DEVICE (int i,int j,int k){
            oz(i,j,k) = discreteCurlF2E<2>(i,j,k,invdx,vel);
        });
    }

    amrex::average_edge_to_cellcenter(
        vort_cc, 0,
        amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>{
            &vort_edge[0], &vort_edge[1], &vort_edge[2]}, 0);
#endif

    return vort_cc;
}

amrex::MultiFab computeDivNonLinearTerm(const FlowField& state)
{
    BL_PROFILE("computeDivNonLinearTerm()");

    const amrex::Geometry& geom = state.getGeom();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx = geom.InvCellSizeArray();

    const amrex::BoxArray& ba = state.getPres().boxArray();
    const amrex::DistributionMapping& dm = state.getPres().DistributionMap();

    // create base multifab to work on

    amrex::MultiFab divN(ba, dm, state.getPres().nComp(), 0);

    for (amrex::MFIter mfi(divN, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& divN_arr = divN.array(mfi);
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel{AMREX_D_DECL(state.getVel(0).const_array(mfi), state.getVel(1).const_array(mfi), state.getVel(2).const_array(mfi))};
        
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            divN_arr(i,j,k) = divMorinishiConvective(i,j,k, invdx, vel);
        });
    }
    
    return divN;
}

void computeDivU(amrex::MultiFab& output_divU, const FlowField& input_state)
{
    BL_PROFILE("<Compute> computeDivU()");

    // set divU to 0.0 and store fresh data
    output_divU.setVal(0.0);
    
    // extracting physical dx for computations
    const amrex::Geometry& geom = input_state.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx = geom.InvCellSizeArray();

    // compute divU and store in input_state
    for(amrex::MFIter mfi(output_divU, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& divU_arr = output_divU.array(mfi);
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel_arr;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) 
        {
            vel_arr[d] = input_state.getVel(d).const_array(mfi);
        }
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            divU_arr(i,j,k) = discreteDivergenceF2C(i, j, k, invdx, vel_arr);
        });
    }
}

// function to compute and RETURN divU for plotting divU_at_end
amrex::MultiFab computePlotDivU(const FlowField& state)
{
    // thin wrapper for plot convenience
    MultiFab out(state.getPres().boxArray(), state.getPres().DistributionMap(), 1, 0);
    computeDivU(out, state);

    return out;
}