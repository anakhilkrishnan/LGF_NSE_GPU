#include <DomainManager.H>

DomainManager::DomainManager(const SolverConfig& config)
{   
    // at constructor call, a coarse mesh with 'search' params is created this
    // facilictates the direct natural follow up of initializeSnugDomain();
    // alternatively, by excluding the call to any of the functions, the solver
    // can be run on the prespecified search grid.

    // initializing domain handling parameters
    regrid_int = config.regrid_int; // TEMP tuning parameter for now

    // initializing domain parameters needed from config
    n_cell = config.n_cell;
    max_grid_size = config.max_grid_size;
    n_lookup = config.n_lookup;
    n_comp = config.n_comp;
    n_ghost = config.n_ghost;
    n_ghost_max = config.n_ghost_max;
    dom_lo = config.dom_lo;
    dom_hi = config.dom_hi;
    n_buffer_box = config.n_buffer_box;
    n_shed_box = config.n_shed_box;
    search_to_fine_ref_ratio = config.search_to_fine_ref_ratio;
    supp_tag_eps = config.supp_tag_eps;

    // rules on what can be initialized and what can't be
    AMREX_ALWAYS_ASSERT(n_ghost >= 2);
    AMREX_ALWAYS_ASSERT(regrid_int >= 1);
    AMREX_ALWAYS_ASSERT(n_shed_box >= 1 && n_shed_box < n_buffer_box);
        
    // creating domain data objects
    amrex::IntVect dom_lo_iv(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect dom_hi_iv(AMREX_D_DECL(config.n_cell_search-1, config.n_cell_search-1, config.n_cell_search-1));
    amrex::Box domain(dom_lo_iv, dom_hi_iv);

    ba.define(domain);
    ba.maxSize(config.max_grid_size_search);
    
    dm = amrex::DistributionMapping(ba);

    amrex::RealBox real_box(dom_lo, dom_hi);
    amrex::Vector<int> is_periodic(AMREX_SPACEDIM, 0); // infinite domain using zero-grad BC
    geom.define(domain, &real_box, amrex::CoordSys::cartesian, is_periodic.data());

    // initialization of other members
    defineScratch();

    did_snug_domain_change = false;
    refresh_err_max_norm.fill(0.0);

    // no snug behaviour by default: whole domain is the support
    supp_ba     = ba;
    old_supp_ba = ba;

    // set q-max and check if domain parameters will work
}

void DomainManager::defineScratch()
{
    tag_vort.define(ba, dm, 1, n_ghost);
    tag_divN.define(ba, dm, 1, n_ghost);
    tag_vort.setVal(0.0); 
    tag_divN.setVal(0.0);

    for (int icomp = 0; icomp < N_VORT; ++icomp)
    {
        vort[icomp].define(amrex::convert(ba, TheVortEdgeVector(icomp)), dm, 1, n_ghost);
        psi[icomp].define(amrex::convert(ba, TheVortEdgeVector(icomp)), dm, 1, n_ghost);

        vort[icomp].setVal(0.0);
        psi[icomp].setVal(0.0);
    }

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        amrex::BoxArray ba_face = amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim));
        vel_refresh_err[idim].define(ba_face, dm, 1, 0);
        vel_refresh_err[idim].setVal(0.0);
    }
}

void DomainManager::growBoxArr(amrex::BoxArray& xsoln_ba, int nBuff, int max_grid_size_req)
{
    xsoln_ba.grow(nBuff);          // grow each box (now overlapping)
    xsoln_ba.removeOverlap(true);     // BoxArray method: removes overlap AND simplifies
    xsoln_ba.maxSize(max_grid_size_req);  // re-chunk to compute box size
}

void DomainManager::updateGeomBaDm(const amrex::BoxArray& new_ba)
{   
    // box array check
#ifdef ENABLE_DEBUG_CHECKS 
#ifdef AMREX_USE_MPI
    long h = new_ba.size();
    for (int i = 0; i < new_ba.size(); ++i) {
        const amrex::Box& b = new_ba[i];
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            h = h*1000003L + b.smallEnd(d);
            h = h*1000003L + b.bigEnd(d);
        }
    }
    long hmin=h, hmax=h;
    amrex::ParallelDescriptor::ReduceLongMin(hmin);
    amrex::ParallelDescriptor::ReduceLongMax(hmax);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(hmin==hmax, "new_ba differs across ranks after growBoxArr");
#endif
#endif

    // update members
    ba = new_ba;
    dm = amrex::DistributionMapping(ba);

    // geom: same physical RealBox, but FINE resolution domain box
    amrex::Box fine_domain(amrex::IntVect(0), amrex::IntVect(AMREX_D_DECL(n_cell-1, n_cell-1, n_cell-1)));
    amrex::RealBox real_box({AMREX_D_DECL(dom_lo[0], dom_lo[1], dom_lo[2])}, {AMREX_D_DECL(dom_hi[0], dom_hi[1], dom_hi[2])});

    amrex::Vector<int> is_periodic(AMREX_SPACEDIM, 0);
    geom.define(fine_domain, &real_box, amrex::CoordSys::cartesian, is_periodic.data());
}

void DomainManager::initializeSnugDomain(amrex::Real time)
{
    BL_PROFILE("<Compute>initializeSnugDomain()")
    // create coarse multifab to store velocity and vorticity pass them for
    // tagging use tag information to update Geom, BoxArr, DistMap; IMPORTANT:
    // refine Geom, BoxArr and DistMap to match desired resolution

    // compute vorticity field in search domain
    amrex::Array<amrex::MultiFab, N_VORT> search_vort;
    for (int icomp = 0; icomp < N_VORT; ++icomp)
    {
        search_vort[icomp].define(amrex::convert(ba, TheVortEdgeVector(icomp)), dm, n_comp, n_ghost);
    }

    // compute discrete divergence free vorticity field
    fluxAverageVorticity(search_vort, geom);

    // create FlowField data using search params
    FlowField search_state(geom, ba, dm, n_comp, n_ghost);

    // setting supp_ba to ba so that everything in the domain is computed
    supp_ba = ba;

    // initializing coarse search domain with velocity using vor2vel
    LGFOpenBC search_poisson_solver(geom, ba, n_lookup, n_ghost_max, 1);
    vor2vel(search_state.getVelArr(), search_vort, search_poisson_solver);
    // setBoundary() call not needed as vor2vel handles ghost cells as well

    computeSuppBoxArr(search_state, time, false); // can't shed outer layer as it has never been tagged before

    // refine to desired resolution
    supp_ba.refine(search_to_fine_ref_ratio);
    old_supp_ba = supp_ba;

    // pad with buffer and update
    amrex::BoxArray xsoln_ba = supp_ba;
    growBoxArr(xsoln_ba, n_buffer_box * max_grid_size, max_grid_size);
    updateGeomBaDm(xsoln_ba);

    // redefine scratch variables onto new grid
    defineScratch();
}

void DomainManager::initializeVelField(FlowField& init_state, LGFOpenBC& lgf_poisson_solver)
{
    // setting supp_ba as ba to initialize everywhere in the domain
    supp_ba = ba;

    // compute vorticity field in initializing domain
    amrex::Array<amrex::MultiFab, N_VORT> init_vort;
    for (int icomp = 0; icomp < N_VORT; ++icomp)
    {
        init_vort[icomp].define(amrex::convert(ba, TheVortEdgeVector(icomp)), dm, n_comp, n_ghost);
    }

    // compute discrete divergence free vorticity field
    fluxAverageVorticity(init_vort, geom);

    // computing refresh in full ba
    vor2vel(init_state.getVelArr(), init_vort, lgf_poisson_solver);

    // restoring supp_ba to the one that ba was built on
    supp_ba = old_supp_ba;

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        vel_refresh_err[idim].setVal(0.0);
    }
}

void DomainManager::restartSnugDomain(const amrex::BoxArray& chk_ba, const amrex::BoxArray& chk_supp_ba)
{
    BL_PROFILE("<IO> restartSnugDomain()")
    // the checkpoint's BoxArray already encodes the fine-resolution Dxsoln
    // (support + buffer) from when the checkpoint was written. Adopt it directly
    // and rebuild geom/ba/dm around it, bypassing the search-grid path in
    // initializeSnugDomain()
    updateGeomBaDm(chk_ba);
    supp_ba = chk_supp_ba;

    // redefine scratch variables onto new grid
    defineScratch();
}

int DomainManager::computeRegridInterval(const FlowField& state) const 
{
    // TODO: q_max = floor(beta * Nb * nb / consumption_rate(state)), asserted
    // >= 1
    return regrid_int;
}

void DomainManager::computeSuppBoxArr(const FlowField& state, amrex::Real time, bool shedOuterLayer) 
{
    BL_PROFILE("<Compute>computeSuppBoxArr()")
    // computes vorticity and divergence of lamb vector tags accordingly and
    // stores in supp_ba
    // IMPORTANT: shedOuterLayer is implicit in its nature. If true, it cannot be
    // run if the old and new supp_ba are on different refinement levels or if 
    // there is no old supp_ba, like in the first initialization call

    // this call is feasible because we ensure that at any given time, tag_vort,
    // tag_divN and state.getPres() all live on the same ba and dm
    tag_vort = computeTagVorticity(state);
    tag_divN = computeDivNonLinearTerm(state, time);

    // add an if(not_initialization) branch to zero all of the outermost buffer layer's cells by a 
    // specified number of boxes
    if (shedOuterLayer)
    {
        amrex::BoxArray xsoln_ba_without_outer = supp_ba;
        growBoxArr(xsoln_ba_without_outer, (n_buffer_box - n_shed_box) * max_grid_size, max_grid_size);

        // loop over tag_vort and tag_divN, zeroing all cells that are outside this layer
        for (amrex::MFIter mfi(tag_divN, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox();
            if(xsoln_ba_without_outer.contains(mfi.validbox())) { continue; }

            auto const& tag_divN_arr = tag_divN.array(mfi);
            auto const& tag_vort_arr = tag_vort.array(mfi);
            
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                tag_divN_arr(i, j, k) = 0.0;
                tag_vort_arr(i, j, k) = 0.0;
            });
        }
    }

    // normalize vorticity by its global max (DECIDE: 3D magnitude, not comp 0)
    amrex::Real vort_max_norm = tag_vort.norm0(0, 0, false);
    if (vort_max_norm > 0.0)
    {
        tag_vort.mult(1.0 / vort_max_norm, 0, 1, tag_vort.nGrow());
    }

    // normalize lamb-divergence by its global max
    amrex::Real divN_max_norm = tag_divN.norm0(0, 0, false);
    if (divN_max_norm > 0.0)
    {
        tag_divN.mult(1.0 / divN_max_norm, 0, 1, tag_divN.nGrow());
    }

    // create a tagging criterion for "where the action is"
    const int num_local_boxes = state.getPres().local_size();

    // ensure capacity matches without forcing a reallocation if it's already
    // sized
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
    auto const& tag_vort_arrs = tag_vort.const_arrays();   // MultiArray4: all local boxes
    auto const& tag_divN_arrs = tag_divN.const_arrays();
    amrex::Real tag_thresh = supp_tag_eps;

    amrex::ParallelFor(tag_divN, [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k)
    {
        if (d_flags_ptr[box_no] != 0) return;   // early-out, now indexed by box_no
        if (amrex::max(amrex::Math::abs(tag_vort_arrs[box_no](i,j,k)), amrex::Math::abs(tag_divN_arrs[box_no](i,j,k))) > tag_thresh)
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

    // check for multi-rank solver to ensure supp_ba that is generated is consistent
#ifdef ENABLE_DEBUG_CHECKS
#ifdef AMREX_USE_MPI
    {
        // build a cheap order-sensitive hash of the box list
        long h = supp_ba.size();
        for (int i = 0; i < supp_ba.size(); ++i) {
            const amrex::Box& b = supp_ba[i];
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                h = h * 1000003L + b.smallEnd(d);
                h = h * 1000003L + b.bigEnd(d);
            }
        }

        long hmin = h, hmax = h;
        amrex::ParallelDescriptor::ReduceLongMin(hmin);
        amrex::ParallelDescriptor::ReduceLongMax(hmax);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(hmin == hmax, "supp_ba differs across ranks");
    }
#endif
#endif
}

void DomainManager::checkAndUpdateSnugDomain(const FlowField& state, amrex::Real time)
{
    BL_PROFILE("<Compute> checkAndUpdateSnugDomain");

    // save previous for checks and refresh
    old_supp_ba = supp_ba;

    // tag on current flow field
    computeSuppBoxArr(state, time);

    // checks if region is still same, weaker check than == because ordering and
    // indices are irrelevent to the solver; guarded against case if either is 
    // zero

    const bool same_region =
    (old_supp_ba.size() == 0 || supp_ba.size() == 0)
    ? (old_supp_ba.size() == supp_ba.size())
    : (old_supp_ba.contains(supp_ba) && supp_ba.contains(old_supp_ba));

    amrex::Print() << "  [diag] nbox " << old_supp_ba.size() << " -> " << supp_ba.size()
                << " | same region? " << same_region << "\n";

    if (same_region)
    {
        // set refresh flag to false and exit
        did_snug_domain_change = false;
    }
    else
    {   
        // set refresh flag to true
        did_snug_domain_change = true;

        // update ba, geom, dm
        amrex::BoxArray xsoln_ba = supp_ba;
        growBoxArr(xsoln_ba, n_buffer_box * max_grid_size, max_grid_size); // grown to full buffer size
        updateGeomBaDm(xsoln_ba);
    }
    amrex::Print() << "Support changed? " << did_snug_domain_change << "\n";

#ifdef AMREX_USE_MPI
    int flag_int = did_snug_domain_change ? 1 : 0;
    int flag_min = flag_int, flag_max = flag_int;
    amrex::ParallelDescriptor::ReduceIntMin(flag_min);
    amrex::ParallelDescriptor::ReduceIntMax(flag_max);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(flag_min == flag_max,
        "did_snug_domain_change diverged across ranks!");
#endif
}

template <int dim>
void DomainManager::fillVelocityComponent(amrex::MultiFab& vel_comp,
                                        amrex::MultiFab& vel_refresh_err_comp,
                                        const amrex::Array<amrex::MultiFab, N_VORT>& psi,
                                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& invdx)
{
    // velocity can be updated to all of psi's ghost cells
    const int ng_u = amrex::min(vel_comp.nGrow(), psi[0].nGrow());

    for (amrex::MFIter mfi(vel_comp, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box bx  = mfi.growntilebox(ng_u);   // valid + ghosts
        const amrex::Box vbx = mfi.tilebox();            // valid only

        auto const& u = vel_comp.array(mfi);
        auto const& e = vel_refresh_err_comp.array(mfi);
        amrex::GpuArray<amrex::Array4<amrex::Real const>, N_VORT> p;
        for (int c = 0; c < N_VORT; ++c)
        {
            p[c] = psi[c].const_array(mfi);
        }

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const amrex::Real u_new = -discreteCurlE2F<dim>(i, j, k, invdx, p);
            if (vbx.contains(amrex::IntVect(AMREX_D_DECL(i, j, k))))
            {
                e(i,j,k) = u_new - u(i,j,k);
            }

            u(i,j,k) = u_new;
        });
    }
}

template <int... dim>
void DomainManager::fillVelocity(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& vel,
                    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& vel_refresh_err,
                    const amrex::Array<amrex::MultiFab, N_VORT>& psi,
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& invdx,
                    std::integer_sequence<int, dim...>)
{
    (fillVelocityComponent<dim>(vel[dim], vel_refresh_err[dim],psi, invdx), ...);
}

void DomainManager::vor2vel(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& vel_arr,
                            amrex::Array<amrex::MultiFab, N_VORT>& vort,
                            LGFOpenBC& lgf_poisson_solver)
{
    BL_PROFILE("<Compute> DomainManager::vor2vel()");

    // create intersection of supp_ba and old_supp_ba, for the vorticity that
    // must be considered for the streamfunction compute
    amrex::BoxArray tag_ba = supp_ba;
    if (did_snug_domain_change)
    {
        tag_ba = amrex::intersect(old_supp_ba, supp_ba);
    }

    // initialize error multifab to zero
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) 
    {
        vel_refresh_err[idim].define(vel_arr[idim].boxArray(),vel_arr[idim].DistributionMap(), 1, 0);
        vel_refresh_err[idim].setVal(0.0);
    }

    // find dx, dy, dz for stencil operations
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx = geom.InvCellSizeArray();

    for (int icomp = 0; icomp < N_VORT; ++icomp)
    {
        // allocating psi multifabs
        psi[icomp].define(vort[icomp].boxArray(), vort[icomp].DistributionMap(), 1, vort[icomp].nGrow());
        // setVal(0.0) not needed as every cell, including ghost cells, is rewritten

        // CONVENTION: The actual source term is -omega; instead of a full pass
        // on vort multifabs, it is flipped when writing each element of
        // velocity components
        lgf_poisson_solver.solvePoisson(vort[icomp], psi[icomp], tag_ba);
        // fillBoundary can be skipped here, under the guarantee that
        // solvePoisson fills ghost cells
    }

    fillVelocity(vel_arr, vel_refresh_err, psi, invdx, std::make_integer_sequence<int, AMREX_SPACEDIM>{});

    // setBoundary call not needed here, since refresh now updates ghost cells
}

void DomainManager::checkAndRefreshVelocity(FlowField& state, LGFOpenBC& lgf_poisson_solver, int step)
{
#ifdef ENABLE_DEBUG_CHECKS
#ifdef AMREX_USE_MPI
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(state.getPres().DistributionMap().size() == state.getPres().boxArray().size(),
                "DistributionMapping was mutated underneath this MultiFab");
                
        // 1. Hash the BoxArray (Order-sensitive geometric hash)
        const amrex::BoxArray& check_ba = state.getPres().boxArray();
        long local_ba_hash = check_ba.size();
        for (int i = 0; i < check_ba.size(); ++i) 
        {
            const amrex::Box& b = check_ba[i];
            for (int d = 0; d < AMREX_SPACEDIM; ++d) 
            {
                local_ba_hash = local_ba_hash * 1000003L + b.smallEnd(d);
                local_ba_hash = local_ba_hash * 1000003L + b.bigEnd(d);
            }
        }

        // 2. Hash the DistributionMapping (Processor assignment hash)
        const amrex::DistributionMapping& check_dm = state.getPres().DistributionMap();
        auto pmap = check_dm.ProcessorMap();
        long local_dm_hash = pmap.size();
        for (int v : pmap) 
        {
            local_dm_hash = local_dm_hash * 1000003L + v;
        }

        // 3. Cross-rank reduction
        long ba_hmin = local_ba_hash, ba_hmax = local_ba_hash;
        long dm_hmin = local_dm_hash, dm_hmax = local_dm_hash;

        amrex::ParallelDescriptor::ReduceLongMin(ba_hmin);
        amrex::ParallelDescriptor::ReduceLongMax(ba_hmax);
        amrex::ParallelDescriptor::ReduceLongMin(dm_hmin);
        amrex::ParallelDescriptor::ReduceLongMax(dm_hmax);

        // 4. Global synchronization verification
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ba_hmin == ba_hmax, 
            "CRITICAL: BoxArray desynchronization detected across ranks before ParallelCopy!");
        
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dm_hmin == dm_hmax, 
            "CRITICAL: DistributionMapping desynchronization detected across ranks before ParallelCopy!");
    }
#endif
#endif

    // update flow field onto the new geom, ba, dm
    if (did_snug_domain_change)
    {
        // create new state with new geom, ba, dm
        FlowField new_state(geom, ba, dm, n_comp, n_ghost);

        // copy Dxsoln from old state into new state
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            new_state.getVel(idim).ParallelCopy(state.getVel(idim), 0, 0, state.getVel(idim).nComp(), 0, 0);
        }
        new_state.getPres().setVal(0.0);
        new_state.setBoundary();

        // move new_state back into state to continue
        state = std::move(new_state);
    }

    // check and perform if a refresh is needed
    if (step % computeRegridInterval(state) == 0 || did_snug_domain_change)
    {
        vort = computeVorticity(state);
        vor2vel(state.getVelArr(), vort, lgf_poisson_solver);

        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            refresh_err_max_norm[idim] = vel_refresh_err[idim].norm0(0, 0, false);
        }

        amrex::Print() << "Velocity refresh performed!"
                            AMREX_D_TERM(<< " | u-refresh err: " << refresh_err_max_norm[0],
                                        << " | v-refresh err: " << refresh_err_max_norm[1],
                                        << " | w-refresh err: " << refresh_err_max_norm[2]) << "\n";
    }
}