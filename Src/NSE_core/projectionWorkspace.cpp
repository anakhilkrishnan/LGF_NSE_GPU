#include <ProjectionWorkspace.H>

ProjectionWorkspace::ProjectionWorkspace(const amrex::Geometry& geom_in, const amrex::BoxArray& ba_in, const amrex::DistributionMapping& dm_in, const int n_comp, const int n_ghost)
    : geom(geom_in), ba(ba_in), dm(dm_in)
{
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // convert the box array to face centered
        amrex::BoxArray ba_face = amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim));

        // declare the specific velocity component
        rhs_vel[idim].define(ba_face, dm, n_comp, n_ghost);
        rhs_vel_corr[idim].define(ba_face, dm, n_comp, n_ghost);

        rhs_kecomp[idim].define(ba_face, dm, n_comp, n_ghost);

        // initialize velocities upon creation
        rhs_vel[idim].setVal(0.0);
        rhs_vel_corr[idim].setVal(0.0);

        rhs_kecomp[idim].setVal(0.0);

        // initialize global ke component storage variables
        global_kecomp[idim] = 0.0;
        global_kecomp_dir[idim] = 0.0;
        global_kecomp_err[idim] = 0.0;
    }

    // initialize corr_pres upon creation
    corr_pres.define(ba, dm, n_comp, n_ghost);
    corr_pres.setVal(0.0);

    divU_max_norm = 0.0;
}

amrex::Real ProjectionWorkspace::computeDt(const FlowField& state, amrex::Real cfl, amrex::Real Re)
{
    const amrex::Geometry& geom = state.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // cfl constraint: advective limit on dt
    amrex::Real u_max = state.getVel(0).norm0(0, 0, false);
    amrex::Real adv_metric = u_max / dx[0];

    #if AMREX_SPACEDIM >= 2
        amrex::Real v_max = state.getVel(1).norm0(0, 0, false);
        adv_metric += v_max / dx[1];
    #endif

    #if AMREX_SPACEDIM == 3
        amrex::Real w_max = state.getVel(2).norm0(0, 0, false);
        adv_metric += w_max / dx[2];
    #endif

    // dt_adv = CFL / ( |u|/dx + |v|/dy + |w|/dz )
    amrex::Real dt_adv = cfl / (adv_metric + 1.0e-12); // epsilon to prevent div-by-zero

    // diffusive limit on dt
    amrex::Real diff_metric = 1.0 / (dx[0] * dx[0]);
    
    #if AMREX_SPACEDIM >= 2
        diff_metric += 1.0 / (dx[1] * dx[1]);
    #endif

    #if AMREX_SPACEDIM == 3
        diff_metric += 1.0 / (dx[2] * dx[2]);
    #endif

    // for explicit schemes, Fourier number <= 0.5 dt_diff <= 0.5 * Re / (
    // 1/dx^2 + 1/dy^2 + 1/dz^2 )
    amrex::Real dt_diff = 0.5 * Re / diff_metric;

    return amrex::min(dt_adv, dt_diff);
}

void ProjectionWorkspace::initializePresField(FlowField& init_state, amrex::Real Re, amrex::Real source_tag_thresh, int nChebyshev, int nLookup)
{
    BL_PROFILE("<Setup> InitializePresField()");
    
    init_state.getPres().setVal(0.0);

    computeMomentumFluxes(init_state, Re);

    // compute divergence of rhs_vel and store in divU
    const amrex::Geometry& geom = init_state.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    for(amrex::MFIter mfi(init_state.getDivU(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& div_arr = init_state.getDivU().array(mfi);
        
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> rhs_arr;
        for(int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            rhs_arr[d] = rhs_vel[d].const_array(mfi);
        }

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) 
        {
            // discreteDivergence works here because rhs_vel is face-centered
            div_arr(i,j,k) = discreteDivergence(i, j, k, dx, rhs_arr);
        });
    }

    // solving the poisson equation to get correct pressure initial conditions
    amrex::Vector<int> init_tag_region = tagSource(init_state.getDivU(), source_tag_thresh);
    solveFMM(init_state.getDivU(), init_state.getPres(), geom, nChebyshev, nLookup);

    // export tagged cells used for pressure computation
    for (MFIter mfi(init_state.getTagRegion()); mfi.isValid(); ++mfi) 
    {
        if (init_tag_region[mfi.LocalIndex()] == 1) 
        {
            // If active, fill the entire box with 1.0 (on the GPU)
            init_state.getTagRegion()[mfi].setVal<RunOn::Device>(1.0); 
        } 
        else 
        {
            // If inactive, fill the entire box with 0.0 (on the GPU)
            init_state.getTagRegion()[mfi].setVal<RunOn::Device>(0.0); 
        }
    }

    // return divU to undisturbed state
    init_state.getDivU().setVal(0.0);

    // compute divU_at_end
    for(amrex::MFIter mfi(init_state.getDivUAtEnd(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& divU_at_end_arr = init_state.getDivUAtEnd().array(mfi);

        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel_arr;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) 
        {
            vel_arr[d] = init_state.getVel(d).const_array(mfi);
        }

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            divU_at_end_arr(i,j,k) = discreteDivergence(i, j, k, dx, vel_arr);
        });
    }
}

void ProjectionWorkspace::computeKECompFluxes(const FlowField& stage, amrex::Real Re)
{
    BL_PROFILE("<Compute> advanceTimeStep(): computeKEFluxes()");
    // compute the right hand side of the KE evolution equations along x,y,z
    // at the given stage discretized using a second order finite difference
    // KEP scheme as outlined in Morinish et. al.

    // extracting physical dx for computations
    const amrex::Geometry& geom = stage.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // for each velocity direction, rhs is computed accordingly
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        for (amrex::MFIter mfi(rhs_kecomp[idim], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox();

            // .........................KEFlux directly from MomentumFlux....................................
            // auto const& rhs_ke_arr  = rhs_kecomp[idim].array(mfi);
            // auto const& rhs_vel_arr = rhs_vel[idim].const_array(mfi);
            // auto const& vel_arr     = stage.getVel(idim).const_array(mfi);

            // amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            // {
            //     // directly compute the kinetic energy fluxes from the momentum equation 
            //     rhs_ke_arr(i,j,k) = vel_arr(i,j,k) * rhs_vel_arr(i,j,k);
            // });
            // .....................................END.......................................................
            // .........................Separate KEFlux Kernel Compute .......................................
            amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel_arr;
            amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> kecomp_arr;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) 
            {
                vel_arr[d] = stage.getVel(d).const_array(mfi);
                kecomp_arr[d] = stage.getKEComp(d).const_array(mfi);
            }
            auto const& pres_arr = stage.getPres().const_array(mfi);
            auto const& rhs_ke_arr  = rhs_kecomp[idim].array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // evaluating one at a time for template variable idim, because
                // these happen at compile time
                if (idim == 0) 
                {
                    rhs_ke_arr(i,j,k) = morinishiKECompFlux<0>(i, j, k, vel_arr, pres_arr, kecomp_arr, dx, Re);
                }
            #if AMREX_SPACEDIM >= 2
                else if (idim == 1) 
                {
                    rhs_ke_arr(i,j,k) = morinishiKECompFlux<1>(i, j, k, vel_arr, pres_arr, kecomp_arr, dx, Re);
                }
            #endif
            #if AMREX_SPACEDIM == 3
                else if (idim == 2) 
                {
                    rhs_ke_arr(i,j,k) = morinishiKECompFlux<2>(i, j, k, vel_arr, pres_arr, kecomp_arr, dx, Re);
                }
            #endif
            });
            // ............................................END.................................................
        }
    }
}

void ProjectionWorkspace::evolveKE(const FlowField& state_n, FlowField& stage, amrex::Real dt, amrex::Real alpha, amrex::Real beta, amrex::Real gamma)
{
    BL_PROFILE("<Compute> advanceTimeStep(): evolveKE()");
    // use the right hand side to compute the next stage kinetic energy

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // using amrex's linalg functions for this step; not updating any ghost
        // cell data here, those are updated by BCs
        amrex::MultiFab::LinComb(stage.getKEComp(idim), alpha, state_n.getKEComp(idim), 0, beta, stage.getKEComp(idim), 0, 0, stage.getKEComp(idim).nComp(), 0);
        amrex::Real dt_by_gam = dt / gamma;
        amrex::MultiFab::Saxpy(stage.getKEComp(idim), dt_by_gam, rhs_kecomp[idim], 0, 0, stage.getKEComp(idim).nComp(), 0);
    }
}

void ProjectionWorkspace::compareKE(const FlowField& state_n)
{
    BL_PROFILE("<Compute> advanceTimeStep(): compareKE()");
    // compute KE components directly from velocity fields; global reduce both
    // KE and KEdir into component-wise sums and compare/writeout/print

    auto kecomp_dir = computeKEFromState(state_n);

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // summing up across all MPI ranks
        global_kecomp[idim] = state_n.getKEComp(idim).sum(0, false);
        global_kecomp_dir[idim] = kecomp_dir[idim].sum(0, false);

        // computing and storing error
        global_kecomp_err[idim] = global_kecomp_dir[idim] - global_kecomp[idim];
    }
}

void ProjectionWorkspace::computeMomentumFluxes(const FlowField& stage, amrex::Real Re)
{
    BL_PROFILE("<Compute> advanceTimeStep(): computeMomentumFluxes()");
    // compute the right hand side which is of the form 1/Re(laplacian(u)) -
    // grad(P) - u.divergence(u) all taken at the given stage discretized
    // using a second order finite difference KEP scheme as outlined in
    // Morinishi et. al.

    // extracting physical dx for computations
    const amrex::Geometry& geom = stage.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // for each velocity direction, rhs is computed accordingly
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        for (amrex::MFIter mfi(stage.getVel(idim), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox();
            amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel_arr;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) 
            {
                vel_arr[d] = stage.getVel(d).const_array(mfi);
            }
            auto const& pres_arr = stage.getPres().const_array(mfi);
            auto const& rhs = rhs_vel[idim].array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // evaluating one at a time for template variable idim, because
                // these happen at compile time
                if (idim == 0) 
                {
                    rhs(i,j,k) = morinishiFlux<0>(i, j, k, vel_arr, pres_arr, dx, Re);
                }
            #if AMREX_SPACEDIM >= 2
                else if (idim == 1) 
                {
                    rhs(i,j,k) = morinishiFlux<1>(i, j, k, vel_arr, pres_arr, dx, Re);
                }
            #endif
            #if AMREX_SPACEDIM == 3
                else if (idim == 2) 
                {
                    rhs(i,j,k) = morinishiFlux<2>(i, j, k, vel_arr, pres_arr, dx, Re);
                }
            #endif
            });
        }
    }
}

void ProjectionWorkspace::predictVelocity(const FlowField& state_n, FlowField& stage, amrex::Real dt, amrex::Real alpha, amrex::Real beta, amrex::Real gamma)
{
    BL_PROFILE("<Compute> advanceTimeStep(): predictVelocity");
    // use the right hand side to predict velocity at the next stage, before
    // enforcing divergence free condition

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // using amrex's linalg functions for this step not updating any ghost
        // cell data here, those are updated by BCs
        amrex::MultiFab::LinComb(stage.getVel(idim), alpha, state_n.getVel(idim), 0, beta, stage.getVel(idim), 0, 0, stage.getVel(idim).nComp(), 0);
        amrex::Real dt_by_gam = dt / gamma;
        amrex::MultiFab::Saxpy(stage.getVel(idim), dt_by_gam, rhs_vel[idim], 0, 0, stage.getVel(idim).nComp(), 0);
    }

    // update ghost cells and physical BCs
    stage.setBoundary();
}

void ProjectionWorkspace::computePressure(FlowField& stage, amrex::Real source_tag_thresh, amrex::Vector<int>& box_tag_arr, int nChebyshev, int nLookup)
{
    BL_PROFILE("<Compute> advanceTimeStep(): computePressure()");

    // extracting physical dx for computations
    const amrex::Geometry& geom = stage.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // compute divU and store in stage
    for(amrex::MFIter mfi(stage.getDivU(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& divU_arr = stage.getDivU().array(mfi);
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel_arr;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) 
        {
            vel_arr[d] = stage.getVel(d).const_array(mfi);
        }
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            divU_arr(i,j,k) = discreteDivergence(i, j, k, dx, vel_arr);
        });
    }

    // use the custom lgf solver to compute the pressure at the next time step
    // running the tagging algorithmn and obtaining the box tags as an array of
    // 0s and 1s
    box_tag_arr = tagSource(stage.getDivU(), source_tag_thresh);
    
    // write out divU_max_norm
    divU_max_norm = stage.getDivU().norm0(0, 0, false);

    // performing addition of box values addEverySourceBox(stage.getDivU(), corr_pres, geom, box_tag_arr); 
    // using FMM based solver instead
    solveFMM(stage.getDivU(), corr_pres, geom, nChebyshev, nLookup);

    corr_pres.FillBoundary(geom.periodicity());
}

void ProjectionWorkspace::computeVelocityCorrection(FlowField& stage)
{
    BL_PROFILE("<Compute> advanceTimeStep(): computeVelocityCorrection");

    const amrex::Geometry& geom = stage.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // for each velocity direction, vel_corr is computed accordingly
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        for (amrex::MFIter mfi(stage.getVel(idim), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox();
            
            auto const& corr_pres_arr = corr_pres.const_array(mfi);
            auto const& rhs_corr = rhs_vel_corr[idim].array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // evaluating one at a time for template variable idim, because
                // these happen at compile time
                if (idim == 0) 
                {
                    rhs_corr(i,j,k) = discreteGradient<0>(i, j, k, corr_pres_arr, dx);
                }
                #if AMREX_SPACEDIM >= 2
                    else if (idim == 1) 
                    {
                        rhs_corr(i,j,k) = discreteGradient<1>(i, j, k, corr_pres_arr, dx);
                    }
                #endif
                #if AMREX_SPACEDIM == 3
                    else if (idim == 2) 
                    {
                        rhs_corr(i,j,k) = discreteGradient<2>(i, j, k, corr_pres_arr, dx);
                    }
                #endif
            });
        }
    }
}

void ProjectionWorkspace::correctVelocityandPressure(FlowField& stage, amrex::Real gamma, amrex::Real dt)
{
    BL_PROFILE("<Compute> advanceTimeStep(): correctVelocity()");

    // use the updated pressure to correct velocity to a divergence free field
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // updating velocity correctly BCs are not updated here
        amrex::MultiFab::Subtract(stage.getVel(idim), rhs_vel_corr[idim], 0, 0, stage.getVel(idim).nComp(), 0);
    }

    // updating pressure to reflect base state + corrected
    amrex::Real gam_by_dt = gamma/dt;
    amrex::MultiFab::Saxpy(stage.getPres(), gam_by_dt, corr_pres, 0, 0, stage.getPres().nComp(), stage.getPres().nGrow());

    // fill ghost cells and physical BCs
    stage.setBoundary();

    // additional checker for divergence at end of step extracting physical dx
    // for computations
    const amrex::Geometry& geom = stage.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // compute divU_at_end
    for(amrex::MFIter mfi(stage.getDivUAtEnd(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& divU_at_end_arr = stage.getDivUAtEnd().array(mfi);
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel_arr;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) 
        {
            vel_arr[d] = stage.getVel(d).const_array(mfi);
        }
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            divU_at_end_arr(i,j,k) = discreteDivergence(i, j, k, dx, vel_arr);
        });
    }
}

void ProjectionWorkspace::advanceTimeStep(FlowField& state_n, amrex::Real dt, amrex::Real Re, int rk_order, amrex::Real source_tag_thresh, int n_chebyshev, int n_lookup)
{

    // perform low-storage RK method for specified order, which can be reduced
    // to a set of Forward Euler like stages with the final sum having
    // appropriate coefficients alpha, beta, gamma

    BL_PROFILE("<Compute> advanceTimeStep()");

    FlowField stage = state_n;
    amrex::Vector<RKCoeffs> coeffs = getRKCoeffs(rk_order);
    amrex::Vector<int> tag_region;

    for(int k = 0; k < rk_order; ++k)
    {
        // extracting RK coefficients
        amrex::Real alpha = coeffs[k].alp;
        amrex::Real beta = coeffs[k].bet;
        amrex::Real gamma = coeffs[k].gam;

        // performing KE evolution routine
        // compute and store KE fluxes in workspace
        computeKECompFluxes(stage, Re);

        // evolve KE and store back in stage
        evolveKE(state_n, stage, dt, alpha, beta, gamma);

        // compute and store fluxes in workspace
        computeMomentumFluxes(stage, Re);

        // compute predicted velocity without divergence free condition store
        // predicted velocity within stage
        predictVelocity(state_n, stage, dt, alpha, beta, gamma);

        // find divergence of predicted velocity, store in workspace use custom
        // LGF solver to find pressure correction delta update pressure stored
        // in stage
        computePressure(stage, source_tag_thresh, tag_region, n_chebyshev, n_lookup);

        // use pressure to compute velocity correction store correction in
        // workspace
        computeVelocityCorrection(stage);

        // correct stage using correction from workspace
        correctVelocityandPressure(stage, gamma, dt);
        
    }

    // export tagged cells at the end of each time step
    for (MFIter mfi(stage.getTagRegion()); mfi.isValid(); ++mfi) 
    {
        if (tag_region[mfi.LocalIndex()] == 1) 
        {
            // If active, fill the entire box with 1.0 (on the GPU)
            stage.getTagRegion()[mfi].setVal<RunOn::Device>(1.0); 
        } 
        else 
        {
            // If inactive, fill the entire box with 0.0 (on the GPU)
            stage.getTagRegion()[mfi].setVal<RunOn::Device>(0.0); 
        }
    }

    state_n = stage;
}

// function to compute cell-centered vorticity from staggered flowfield for
// plotting
amrex::MultiFab computeCellCenteredVorticity(const FlowField& state)
{
    BL_PROFILE("computeCellCenteredVorticity()");

    // get geometry and grid info using your safely encapsulated getter!
    const amrex::Geometry& geom = state.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    amrex::BoxArray ba = state.getPres().boxArray();
    amrex::DistributionMapping dm = state.getPres().DistributionMap();

    // allocate the cell-centered vorticity MultiFab In 2D: 1 component
    // (omega_z). In 3D: 3 components (omega_x, omega_y, omega_z)
    int ncomp = (AMREX_SPACEDIM == 2) ? 1 : 3;
    amrex::MultiFab vort(ba, dm, ncomp, 0);

    // compute the averaged gradients on the GPU
    for (amrex::MFIter mfi(vort, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();

        auto const& vort_arr = vort.array(mfi);
        auto const& u_arr    = state.getVel(0).const_array(mfi);
        auto const& v_arr    = state.getVel(1).const_array(mfi);
        
        #if AMREX_SPACEDIM == 3
        auto const& w_arr    = state.getVel(2).const_array(mfi);
        #endif

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // Note: A face-centered array at index (i,j,k) represents the
            // 'left' or 'bottom' face. (i+1,j,k) represents the 'right' face,
            // etc.
            
            #if AMREX_SPACEDIM == 2
                // dv/dx averaged from top and bottom y-faces
                amrex::Real dvdx = (v_arr(i+1,j,k) + v_arr(i+1,j+1,k) - v_arr(i-1,j,k) - v_arr(i-1,j+1,k)) / (4.0 * dx[0]);
                // du/dy averaged from left and right x-faces
                amrex::Real dudy = (u_arr(i,j+1,k) + u_arr(i+1,j+1,k) - u_arr(i,j-1,k) - u_arr(i+1,j-1,k)) / (4.0 * dx[1]);
                
                vort_arr(i,j,k) = dvdx - dudy;

            #elif AMREX_SPACEDIM == 3
                // omega_x = dw/dy - dv/dz
                amrex::Real dwdy = (w_arr(i,j+1,k) + w_arr(i,j+1,k+1) - w_arr(i,j-1,k) - w_arr(i,j-1,k+1)) / (4.0 * dx[1]);
                amrex::Real dvdz = (v_arr(i,j,k+1) + v_arr(i,j+1,k+1) - v_arr(i,j,k-1) - v_arr(i,j+1,k-1)) / (4.0 * dx[2]);
                vort_arr(i,j,k,0) = dwdy - dvdz;

                // omega_y = du/dz - dw/dx
                amrex::Real dudz = (u_arr(i,j,k+1) + u_arr(i+1,j,k+1) - u_arr(i,j,k-1) - u_arr(i+1,j,k-1)) / (4.0 * dx[2]);
                amrex::Real dwdx = (w_arr(i+1,j,k) + w_arr(i+1,j,k+1) - w_arr(i-1,j,k) - w_arr(i-1,j,k+1)) / (4.0 * dx[0]);
                vort_arr(i,j,k,1) = dudz - dwdx;

                // omega_z = dv/dx - du/dy
                amrex::Real dvdx = (v_arr(i+1,j,k) + v_arr(i+1,j+1,k) - v_arr(i-1,j,k) - v_arr(i-1,j+1,k)) / (4.0 * dx[0]);
                amrex::Real dudy = (u_arr(i,j+1,k) + u_arr(i+1,j+1,k) - u_arr(i,j-1,k) - u_arr(i+1,j-1,k)) / (4.0 * dx[1]);
                vort_arr(i,j,k,2) = dvdx - dudy;
            #endif
        });
    }

    return vort;
}

amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> computeKEFromState(const FlowField& state)
{

    BL_PROFILE("computeKEFromState()");
    // compute face-centered component-wise KE from velocity field in state

    // computing kecomp_dir one component at a time
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> ke;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // capture the idim'th velocity component
        const amrex::MultiFab& vel_comp = state.getVel(idim);

        // define the KE component using the velocity's exact staggering and layout
        ke[idim].define(vel_comp.boxArray(), vel_comp.DistributionMap(), vel_comp.nComp(), vel_comp.nGrow());

        // initializing to zero to clear garbage values out
        ke[idim].setVal(0.0);

        for (amrex::MFIter mfi(ke[idim], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox();

            auto const& ke_arr = ke[idim].array(mfi);
            auto const& vel_arr = vel_comp.const_array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                ke_arr(i,j,k) = 0.5 * (vel_arr(i,j,k) * vel_arr(i,j,k));
            });
        }
    }

    return ke;
}