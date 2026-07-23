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
    n_shed_box = config.n_shed_box;
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
}

void DomainManager::growBoxArr(amrex::BoxArray& xsoln_ba, int nBuff, int max_grid_size_req)
{
    xsoln_ba.grow(nBuff);          // grow each box (now overlapping)
    xsoln_ba.removeOverlap(true);     // BoxArray method: removes overlap AND simplifies
    xsoln_ba.maxSize(max_grid_size_req);  // re-chunk to compute box size
}

void DomainManager::updateGeomBaDm(amrex::BoxArray& new_ba)
{    
    // update members
    ba = new_ba;
    dm.define(ba);

    // geom: same physical RealBox, but FINE resolution domain box
    amrex::Box fine_domain(amrex::IntVect(0), amrex::IntVect(AMREX_D_DECL(n_cell-1, n_cell-1, n_cell-1)));
    amrex::RealBox real_box({AMREX_D_DECL(dom_lo[0], dom_lo[1], dom_lo[2])}, {AMREX_D_DECL(dom_hi[0], dom_hi[1], dom_hi[2])});

    amrex::Vector<int> is_periodic(AMREX_SPACEDIM, 0);
    geom.define(fine_domain, &real_box, amrex::CoordSys::cartesian, is_periodic.data());
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

    computeSuppBoxArr(search_state);

    psi.define(vort.boxArray(), vort.DistributionMap(), vort.nComp(), vort.nGrow());
    psi.setVal(0.0);

    amrex::BoxArray xsoln_ba = supp_ba;
    
    // refine to desired resolution
    xsoln_ba.refine(search_to_fine_ref_ratio);

    // pad with buffer and update
    growBoxArr(xsoln_ba, n_buffer_box * max_grid_size, max_grid_size);
    updateGeomBaDm(xsoln_ba);

    // initialize error multifab to zero
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) 
    {
        vel_refresh_err[idim].define(ba, dm, 1, 0);
        vel_refresh_err[idim].setVal(0.0);
    }
}

int DomainManager::computeRegridInterval(const FlowField& state) const 
{
    // TODO: q_max = floor(beta * Nb * nb / consumption_rate(state)), asserted >= 1
    AMREX_ALWAYS_ASSERT(regrid_int >= 1);
    return regrid_int;
}

void DomainManager::computeSuppBoxArr(const FlowField& state) 
{
    BL_PROFILE("<Compute>computeSuppBoxArr()")
    // computes vorticity and divergence of lamb vector tags accordingly and
    // stores in supp_tag_arr AND NOW AS supp_ba
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
    const int num_local_boxes = state.getPres().local_size();

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
    supp_ba = amrex::BoxArray(std::move(bl));
}

void DomainManager::updateSnugDomain(const FlowField& state)
{
    BL_PROFILE("<Compute>updateSnugDomain()")
    // reads current flowfield state (as new search space), tags on search space;
    // uses tag information to update Geom, BoxArr, DistMap; 

    computeSuppBoxArr(state);

    amrex::BoxArray xsoln_ba = supp_ba;

    growBoxArr(xsoln_ba, n_buffer_box * max_grid_size, max_grid_size);

    updateGeomBaDm(xsoln_ba);
}

void DomainManager::vor2vel(FlowField& state, DirectSumLGF& lgf_nodal_poisson_solver)
{
    BL_PROFILE("<Compute> DomainManager::vor2vel()");

    // extract nodal vorticity
    amrex::MultiFab vort_nd = computeNodalVorticity(state);

    // allocate target multifab
    psi.define(vort_nd.boxArray(), vort_nd.DistributionMap(), 1, vort_nd.nGrow());
    psi.setVal(0.0);// compute streamfunction ONLY on buffer nodes
    vort_nd.mult(-1.0); // source term is -omega_z
    lgf_nodal_poisson_solver.solveNodalPoisson(vort_nd, psi, supp_ba);
    psi.FillBoundary(geom.periodicity());

    // initialize error multifab to zero
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) 
    {
        vel_refresh_err[idim].define(state.getVel(idim).boxArray(),
                                        state.getVel(idim).DistributionMap(), 1, 0);
        vel_refresh_err[idim].setVal(0.0);
    }

    // update velocities in Dbuff
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx = geom.InvCellSizeArray();

    // X-Velocity (u = d(psi)/dy)
    for (amrex::MFIter mfi(state.getVel(0), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        // early exit if box is fully within support
        if (supp_ba.contains(amrex::enclosedCells(bx))) { continue; }

        auto const& u_arr = state.getVel(0).array(mfi);
        auto const& psi_arr   = psi.const_array(mfi);
        auto const& err_arr = vel_refresh_err[0].array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::Real u_refresh = (psi_arr(i, j+1, k) - psi_arr(i, j, k)) * invdx[1];
                err_arr(i, j, k) = u_refresh - u_arr(i, j, k);
                u_arr(i, j, k) = u_refresh;
        });
    }

    // Y-Velocity (v = -d(psi)/dx)
    for (amrex::MFIter mfi(state.getVel(1), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        // early exit if box is fully within support
        if (supp_ba.contains(amrex::enclosedCells(bx))) { continue; }

        auto const& v_arr = state.getVel(1).array(mfi);
        auto const& psi_arr   = psi.const_array(mfi);
        auto const& err_arr = vel_refresh_err[1].array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::Real v_refresh = -(psi_arr(i+1, j, k) - psi_arr(i, j, k)) * invdx[0];
            err_arr(i, j, k) = v_refresh - v_arr(i, j, k);
            v_arr(i, j, k) = v_refresh;
        });
    }

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        refresh_err_max_norm[idim] = vel_refresh_err[idim].norm0(0, 0, false);
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

    // local test to see if grow/rechunk actually affects the solver due to the choice between intersect tag and contain tag
    int n_contain = 0, n_intersect = 0;
    for (MFIter mfi(new_state.getPres()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
        if (supp_ba.contains(bx))   n_contain++;
        if (supp_ba.intersects(bx)) n_intersect++;
    }
    amrex::Print() << "contain-tagged: " << n_contain
                << " | intersect-tagged: " << n_intersect << "\n";

    // use supp_tag_arr (now updated with Dsupp found when creating new geom, ba, dm)
    // to avoid vel refresh on Dsupp. Only Dbuff needs the update
    vor2vel(new_state, lgf_nodal_poisson_solver);

    // move new_state back into state to continue
    state = std::move(new_state);
}