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
    n_comp = config.n_comp;
    n_ghost = config.n_ghost;
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

amrex::BoxArray DomainManager::gatherTaggedBoxArr(const FlowField& state)
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
    DSupp.define(ba, dm, 1, 1);
    DSupp.setVal(0.0);

    // export tagging data into plotting multifab
    for (MFIter mfi(DSupp); mfi.isValid(); ++mfi) 
    {
        const amrex::Real v = (h_tag_arr[mfi.LocalIndex()] == 1) ? 1.0 : 0.0;
        DSupp[mfi].setVal<RunOn::Device>(v);
    }

    DSupp.FillBoundary(geom.periodicity());

    return DSupp;   // by REFERENCE
}

void DomainManager::initializeSnugDomain()
{
    BL_PROFILE("<Compute>initializeSnugDomain()")
    // create coarse multifab to store velocity and vorticity pass them for
    // tagging use tag information to update Geom, BoxArr, DistMap; IMPORTANT:
    // refine Geom, BoxArr and DistMap to match desired resolution

    // create FlowField data using search params
    FlowField search_state(geom, ba, dm, n_comp, n_ghost);

    // initializing coarse search domain with velocity
    initializeVelField(search_state);
    search_state.setBoundary();

    tagSupportRegion(search_state);

    amrex::BoxArray tag_ba = gatherTaggedBoxArr(search_state);
    
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
    // reads current flowfield state (as new search space), tags on search space;
    // uses tag information to update Geom, BoxArr, DistMap; 

    tagSupportRegion(state);

    amrex::BoxArray tag_ba = gatherTaggedBoxArr(state);

    growBoxArr(tag_ba, n_buffer_box * max_grid_size, max_grid_size);

    updateGeomBaDm(tag_ba);
}

void DomainManager::vor2vel(FlowField& state, DirectSumLGF& lgf_nodal_poisson_solver)
{
    BL_PROFILE("<Compute> DomainManager::vor2vel()");

    // extract nodal vorticity
    amrex::MultiFab vort_nd = computeNodalVorticity(state);

    // allocate target multifab
    amrex::MultiFab psi_nd(vort_nd.boxArray(), vort_nd.DistributionMap(), 1, vort_nd.nGrow());
    psi_nd.setVal(0.0);

    // 3. Generate the cell-centered mask to skip interior math
    const amrex::MultiFab& mask = refreshAndGetDSuppFab();

    // 4. Compute streamfunction ONLY on buffer nodes
    lgf_nodal_poisson_solver.solveNodalPoisson(vort_nd, psi_nd, supp_tag_arr, &mask);

    // update velocities in Dbuff
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx = geom.InvCellSizeArray();

    // X-Velocity (u = d(psi)/dy)
    for (amrex::MFIter mfi(state.getVel(0), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& u_arr = state.getVel(0).array(mfi);
        auto const& psi   = psi_nd.const_array(mfi);
        auto const& m_arr = mask.const_array(mfi); // Cell-centered mask

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // An x-face borders cell(i-1,j,k) and cell(i,j,k). 
            // If EITHER bounding cell is buffer (0.0), overwrite the velocity.
            if (m_arr(i-1, j, k) < 0.5 || m_arr(i, j, k) < 0.5)
            {
                u_arr(i, j, k) = (psi(i, j+1, k) - psi(i, j, k)) * invdx[1];
            }
        });
    }

    // Y-Velocity (v = -d(psi)/dx)
    for (amrex::MFIter mfi(state.getVel(1), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& v_arr = state.getVel(1).array(mfi);
        auto const& psi   = psi_nd.const_array(mfi);
        auto const& m_arr = mask.const_array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // A y-face borders cell(i,j-1,k) and cell(i,j,k).
            if (m_arr(i, j-1, k) < 0.5 || m_arr(i, j, k) < 0.5)
            {
                v_arr(i, j, k) = -(psi(i+1, j, k) - psi(i, j, k)) * invdx[0];
            }
        });
    }

    // Refresh ghosts since buffer data was manually overwritten
    state.setBoundary();
}

void DomainManager::regridFlowFieldOntoNewSnugDomain(FlowField& state, DirectSumLGF& lgf_nodal_poisson_solver)
{
    // recognizes geom, ba, dm members have been updated, do not match current input state
    
    // create new state with new geom, ba, dm
    FlowField new_state(geom, ba, dm, n_comp, n_ghost);

    // copy Dxsoln from old state into new state
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        new_state.getVel(idim).ParallelCopy(state.getVel(idim), 0, 0, state.getVel(idim).nComp(), 0, 0);
        new_state.getKEComp(idim).ParallelCopy(state.getKEComp(idim), 0, 0, state.getKEComp(idim).nComp(), 0, 0);
    }
    new_state.getPres().ParallelCopy(state.getPres(), 0, 0, state.getPres().nComp(), 0, 0);
    new_state.setBoundary();

    // mask transplant to new grid
    amrex::MultiFab new_DSupp(ba, dm, 1, 1);
    new_DSupp.setVal(0.0); 
    new_DSupp.ParallelCopy(DSupp, 0, 0, 1, 0, 0);
    DSupp = std::move(new_DSupp);
    DSupp.FillBoundary(geom.periodicity());

    // update supp_tag_arr and htag)arr
    const int num_local_boxes = DSupp.local_size();
    if (supp_tag_arr.size() != num_local_boxes) 
    {
        supp_tag_arr.resize(num_local_boxes);
    }

    int* d_flags_ptr = supp_tag_arr.dataPtr();
    amrex::ParallelFor(num_local_boxes, [=] AMREX_GPU_DEVICE (int i)
    {
        d_flags_ptr[i] = 0;
    });

    auto const& dsupp_arrs = DSupp.const_arrays();
    amrex::ParallelFor(DSupp, [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k)
    {
        if (d_flags_ptr[box_no] != 0) return;
        if (dsupp_arrs[box_no](i,j,k) == 1.0)
        {
            amrex::Gpu::Atomic::Max(&d_flags_ptr[box_no], 1);
        }
    });

    h_tag_arr.resize(supp_tag_arr.size());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, supp_tag_arr.begin(), supp_tag_arr.end(), h_tag_arr.begin());

    // use supp_tag_arr (now updated with Dsupp found when creating new geom, ba, dm)
    // to avoid vel refresh on Dsupp. Only Dbuff needs the update
    vor2vel(new_state, lgf_nodal_poisson_solver);

    // move new_state back into state to continue
    state = std::move(new_state);
}

// computes nodal vorticity for vor2vel(), via the discreteCurlF2E.
amrex::MultiFab computeNodalVorticity(const FlowField& state)
{
    // IMPORTANT: Limited to 2D at present
    BL_PROFILE("<Compute> computeNodalVorticity()")

    const amrex::Geometry& geom = state.getGeom();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx = geom.InvCellSizeArray();

    const amrex::BoxArray& ba = state.getPres().boxArray();
    const amrex::DistributionMapping& dm = state.getPres().DistributionMap();

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

    return vort_nd;
}

// computes cell-centered vorticity for plotting
amrex::MultiFab computePlotVorticity(const FlowField& state)
{
    BL_PROFILE("<Compute> computePlotVorticity()");

    amrex::MultiFab vort_nd = computeNodalVorticity(state);

    const amrex::BoxArray& ba = state.getPres().boxArray();
    const amrex::DistributionMapping& dm = state.getPres().DistributionMap();

    amrex::MultiFab vort_cc(ba, dm, 1, 0);
    amrex::average_node_to_cellcenter(vort_cc, 0, vort_nd, 0, 1, 0);

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