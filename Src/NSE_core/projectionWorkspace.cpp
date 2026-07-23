#include <ProjectionWorkspace.H>

ProjectionWorkspace::ProjectionWorkspace(const amrex::Geometry& geom_in, const amrex::BoxArray& ba_in, const amrex::DistributionMapping& dm_in, const SolverConfig& config)
    : stage(geom_in, ba_in, dm_in, config.n_comp, config.n_ghost), lgf_poisson_solver(geom_in, config.n_lookup)
{
    // initializing required solver parameters
    n_lookup = config.n_lookup;
    rk_order = config.rk_order;
    invRe = config.invRe;
    cfl = config.cfl;

    // copying parameters necessary for constructor only
    int n_comp = config.n_comp;
    int n_ghost = config.n_ghost;

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // convert the box array to face centered
        amrex::BoxArray ba_face = amrex::convert(ba_in, amrex::IntVect::TheDimensionVector(idim));

        // declare the specific velocity component
        rhs_vel[idim].define(ba_face, dm_in, n_comp, n_ghost);
        rhs_vel_corr[idim].define(ba_face, dm_in, n_comp, n_ghost);

        rhs_kecomp[idim].define(ba_face, dm_in, n_comp, n_ghost);
        kecomp_dir[idim].define(ba_face, dm_in, n_comp, n_ghost);

        // initialize velocities upon creation
        rhs_vel[idim].setVal(0.0);
        rhs_vel_corr[idim].setVal(0.0);

        rhs_kecomp[idim].setVal(0.0);
        kecomp_dir[idim].setVal(0.0);

        // initialize global ke component storage variables
        global_kecomp[idim] = 0.0;
        global_kecomp_dir[idim] = 0.0;
        global_kecomp_err[idim] = 0.0;
    }

    // initialize pres_corr upon creation
    pres_corr.define(ba_in, dm_in, n_comp, n_ghost);
    pres_corr.setVal(0.0);

    // initialize divU upon creation
    divU.define(ba_in, dm_in, n_comp, n_ghost);
    divU.setVal(0.0);

    divU_max_norm = 0.0;
    divU_at_end_max_norm = 0.0;

    dt = 0.0;
}

amrex::Real ProjectionWorkspace::computeDt(const FlowField& state_n) const
{
    BL_PROFILE("<Compute> computeDt()");

    const amrex::Geometry& geom = state_n.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx = geom.InvCellSizeArray();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx2;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) 
    {
        invdx2[d] = invdx[d] * invdx[d];
    }

    // cfl constraint: advective limit on dt
    amrex::Real u_max = state_n.getVel(0).norm0(0, 0, false);
    amrex::Real adv_metric = u_max * invdx[0];

    amrex::Real v_max = state_n.getVel(1).norm0(0, 0, false);
    adv_metric += v_max * invdx[1];

#if AMREX_SPACEDIM == 3
    amrex::Real w_max = state_n.getVel(2).norm0(0, 0, false);
    adv_metric += w_max * invdx[2];
#endif

    // dt_adv = CFL / ( |u|/dx + |v|/dy + |w|/dz )
    amrex::Real dt_adv = cfl / (adv_metric + 1.0e-12); // epsilon to prevent div-by-zero

    // diffusive limit on dt
    amrex::Real diff_metric = 1.0 * invdx2[0];
    
    diff_metric += 1.0 * invdx2[1];

    #if AMREX_SPACEDIM == 3
        diff_metric += 1.0 * invdx2[2];
    #endif

    // for explicit schemes, Fourier number <= 0.5 dt_diff <= 0.5 * Re / (
    // 1/dx^2 + 1/dy^2 + 1/dz^2 )
    amrex::Real dt_diff = 0.5 / (invRe * diff_metric);

    return amrex::min(dt_adv, dt_diff);
}

amrex::Real ProjectionWorkspace::computeDivUMaxNorm(const FlowField& input_state) const
{
    BL_PROFILE("<Compute> computeDivUMaxNorm()");

    // function to reduce state directly into divUMaxNorm
    // called at appropriate locations to differentiate divU_star
    // and divU_at_end

    amrex::ReduceOps<amrex::ReduceOpMax> reduce_op;
    amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    // grid spacing for the stencil
    const auto invdx = input_state.getGeom().InvCellSizeArray();  // {1/dx, 1/dy, 1/dz}

    // You reduce over CELL-centered boxes (divergence is cell-centered),
    // so iterate something cell-centered — e.g. the pressure MultiFab —
    // to get the right box/loop domain, and read the face velocities into it.
    for (amrex::MFIter mfi(input_state.getPres()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();

        AMREX_D_TERM(auto const& u = input_state.getVel(0).const_array(mfi);,
                    auto const& v = input_state.getVel(1).const_array(mfi);,
                    auto const& w = input_state.getVel(2).const_array(mfi);)

        reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                // discrete divergence at cell (i,j,k) from surrounding faces
                amrex::Real div =
                    AMREX_D_TERM(  (u(i+1,j,k) - u(i,j,k)) * invdx[0],
                                + (v(i,j+1,k) - v(i,j,k)) * invdx[1],
                                + (w(i,j,k+1) - w(i,j,k)) * invdx[2] );
                return { amrex::Math::abs(div) };
            });
    }

    amrex::Real max_div = amrex::get<0>(reduce_data.value());
    amrex::ParallelDescriptor::ReduceRealMax(max_div);
    
    return max_div;
}

void ProjectionWorkspace::initializePresField(FlowField& init_state, const amrex::BoxArray& init_supp_ba)
{
    BL_PROFILE("<Setup> InitializePresField()");
    
    // set value to 0.0 and store fresh
    divU.setVal(0.0);
    init_state.getPres().setVal(0.0);

    computeMomentumFluxes(init_state);

    // compute divergence of rhs_vel and store in divU
    const amrex::Geometry& geom = init_state.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx = geom.InvCellSizeArray();

    for(amrex::MFIter mfi(divU, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& div_arr = divU.array(mfi);
        
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> rhs_arr;
        for(int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            rhs_arr[d] = rhs_vel[d].const_array(mfi);
        }

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) 
        {
            // discreteDivergence works here because rhs_vel is face-centered
            div_arr(i,j,k) = discreteDivergenceF2C(i, j, k, invdx, rhs_arr);
        });
    }

    lgf_poisson_solver.solvePoisson(divU, init_state.getPres(), init_supp_ba);
    init_state.getPres().FillBoundary(init_state.getGeom().periodicity());

    // write out divU_max_norm
    divU_max_norm = divU.norm0(0, 0, false);

    // write out divU_at_end_max_norm
    divU_at_end_max_norm = computeDivUMaxNorm(init_state);
}

void ProjectionWorkspace::computeKECompFluxes(const FlowField& input_state)
{
    BL_PROFILE("<Compute> advanceTimeStep(): computeKEFluxes()");
    // compute the right hand side of the KE evolution equations along x,y,z
    // at the given input_state discretized using a second order finite difference
    // KEP scheme as outlined in Morinish et. al.

    // function's copy of Re to be passed to GPU lambdas
    amrex::Real invRe_temp = invRe;
    
    // extracting physical dx for computations
    const amrex::Geometry& geom = input_state.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx = geom.InvCellSizeArray();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx2;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) 
    {
        invdx2[d] = invdx[d] * invdx[d];
    }

    // for each velocity direction, rhs is computed accordingly
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        for (amrex::MFIter mfi(rhs_kecomp[idim], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox();

            // .........................KEFlux directly from MomentumFlux....................................
            // auto const& rhs_ke_arr  = rhs_kecomp[idim].array(mfi);
            // auto const& rhs_vel_arr = rhs_vel[idim].const_array(mfi);
            // auto const& vel_arr     = input_state.getVel(idim).const_array(mfi);

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
                vel_arr[d] = input_state.getVel(d).const_array(mfi);
                kecomp_arr[d] = input_state.getKEComp(d).const_array(mfi);
            }
            auto const& pres_arr = input_state.getPres().const_array(mfi);
            auto const& rhs_ke_arr  = rhs_kecomp[idim].array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // evaluating one at a time for template variable idim, because
                // these happen at compile time
                if (idim == 0) 
                {
                    rhs_ke_arr(i,j,k) = morinishiKEFlux<0>(i, j, k, invRe_temp, invdx, invdx2, vel_arr, kecomp_arr, pres_arr);
                }
            #if AMREX_SPACEDIM >= 2
                else if (idim == 1) 
                {
                    rhs_ke_arr(i,j,k) = morinishiKEFlux<1>(i, j, k, invRe_temp, invdx, invdx2, vel_arr, kecomp_arr, pres_arr);
                }
            #endif
            #if AMREX_SPACEDIM == 3
                else if (idim == 2) 
                {
                    rhs_ke_arr(i,j,k) = morinishiKEFlux<2>(i, j, k, invRe_temp, invdx, invdx2, vel_arr, kecomp_arr, pres_arr);
                }
            #endif
            });
            // ............................................END.................................................
        }
    }
}

void ProjectionWorkspace::evolveKE(const FlowField& state_n, amrex::Real alpha, amrex::Real beta, amrex::Real gamma)
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

void ProjectionWorkspace::computeKEFromState(const FlowField& state)
{
    BL_PROFILE("computeKEFromState()");
    // compute face-centered component-wise KE from velocity field in state

    // computing kecomp_dir one component at a time
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // capture the idim'th velocity component
        const amrex::MultiFab& vel_comp = state.getVel(idim);

        // initializing to zero to clear garbage values out
        kecomp_dir[idim].setVal(0.0);

        for (amrex::MFIter mfi(kecomp_dir[idim], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox();

            auto const& ke_arr = kecomp_dir[idim].array(mfi);
            auto const& vel_arr = vel_comp.const_array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                ke_arr(i,j,k) = 0.5 * (vel_arr(i,j,k) * vel_arr(i,j,k));
            });
        }
    }
}

void ProjectionWorkspace::compareKE(const FlowField& state_n)
{
    BL_PROFILE("<Compute> advanceTimeStep(): compareKE()");
    // compute KE components directly from velocity fields; global reduce both
    // KE and KEdir into component-wise sums and compare/writeout/print

    computeKEFromState(state_n);

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // summing up across all MPI ranks
        global_kecomp[idim] = state_n.getKEComp(idim).sum(0, false);
        global_kecomp_dir[idim] = kecomp_dir[idim].sum(0, false);

        // computing and storing error
        global_kecomp_err[idim] = global_kecomp_dir[idim] - global_kecomp[idim];
    }
}

void ProjectionWorkspace::computeMomentumFluxes(const FlowField& input_state)
{
    BL_PROFILE("<Compute> advanceTimeStep(): computeMomentumFluxes()");
    // compute the right hand side which is of the form 1/Re(laplacian(u)) -
    // grad(P) - u.divergence(u) all taken at the given input_state discretized
    // using a second order finite difference KEP scheme as outlined in
    // Morinishi et. al.

    // extracting physical dx for computations
    const amrex::Geometry& geom = input_state.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx = geom.InvCellSizeArray();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx2;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) 
    {
        invdx2[d] = invdx[d] * invdx[d];
    }

    // local copy of Re for passing to GPU lambdas
    amrex::Real invRe_temp = invRe;

    // for each velocity direction, rhs is computed accordingly
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        for (amrex::MFIter mfi(input_state.getVel(idim), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox();
            amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel_arr;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) 
            {
                vel_arr[d] = input_state.getVel(d).const_array(mfi);
            }
            auto const& pres_arr = input_state.getPres().const_array(mfi);
            auto const& rhs = rhs_vel[idim].array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // evaluating one at a time for template variable idim, because
                // these happen at compile time
                if (idim == 0) 
                {
                    rhs(i,j,k) = morinishiFlux<0>(i, j, k, invRe_temp, invdx, invdx2, vel_arr, pres_arr);
                }
            #if AMREX_SPACEDIM >= 2
                else if (idim == 1) 
                {
                    rhs(i,j,k) = morinishiFlux<1>(i, j, k, invRe_temp, invdx, invdx2, vel_arr, pres_arr);
                }
            #endif
            #if AMREX_SPACEDIM == 3
                else if (idim == 2) 
                {
                    rhs(i,j,k) = morinishiFlux<2>(i, j, k, invRe_temp, invdx, invdx2, vel_arr, pres_arr);
                }
            #endif
            });
        }
    }
}

void ProjectionWorkspace::predictVelocity(const FlowField& state_n, amrex::Real alpha, amrex::Real beta, amrex::Real gamma)
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

void ProjectionWorkspace::computePressure()
{
    BL_PROFILE("<Compute> advanceTimeStep(): computePressure()");

    // compute divU and store back into stage
    computeDivU(divU, stage);

    // performing addition of box values 
    lgf_poisson_solver.solvePoisson(divU, pres_corr, tag_ba);
    pres_corr.FillBoundary(stage.getGeom().periodicity());
    
    // write out divU_max_norm
    divU_max_norm = divU.norm0(0, 0, false);
}

void ProjectionWorkspace::computeVelocityCorrection()
{
    BL_PROFILE("<Compute> advanceTimeStep(): computeVelocityCorrection");

    const amrex::Geometry& geom = stage.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx = geom.InvCellSizeArray();

    // for each velocity direction, vel_corr is computed accordingly
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        for (amrex::MFIter mfi(stage.getVel(idim), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox();
            
            auto const& pres_corr_arr = pres_corr.const_array(mfi);
            auto const& rhs_corr = rhs_vel_corr[idim].array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // evaluating one at a time for template variable idim, because
                // these happen at compile time
                if (idim == 0) 
                {
                    rhs_corr(i,j,k) = discreteGradientC2F<0>(i, j, k, invdx, pres_corr_arr);
                }
                else if (idim == 1) 
                {
                    rhs_corr(i,j,k) = discreteGradientC2F<1>(i, j, k, invdx, pres_corr_arr);
                }
                #if AMREX_SPACEDIM == 3
                    else if (idim == 2) 
                    {
                        rhs_corr(i,j,k) = discreteGradientC2F<2>(i, j, k, invdx, pres_corr_arr);
                    }
                #endif
            });
        }
    }
}

void ProjectionWorkspace::correctVelocityandPressure(amrex::Real gamma)
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
    amrex::MultiFab::Saxpy(stage.getPres(), gam_by_dt, pres_corr, 0, 0, stage.getPres().nComp(), stage.getPres().nGrow());

    // fill ghost cells and physical BCs
    stage.setBoundary();

    // write out divU_at_end_max_norm
    divU_at_end_max_norm = computeDivUMaxNorm(stage);
}

void ProjectionWorkspace::advanceTimeStep(FlowField& state_n, const amrex::Real dt_in, const amrex::BoxArray& supp_ba)
{

    // perform low-storage RK method for specified order, which can be reduced
    // to a set of Forward Euler like stages with the final sum having
    // appropriate coefficients alpha, beta, gamma

    BL_PROFILE("<Compute> advanceTimeStep()");

    // set member value using incoming dt and supp_ba
    dt = dt_in;
    tag_ba = supp_ba;

    stage = state_n;
    amrex::Vector<RKCoeffs> coeffs = getRKCoeffs(rk_order);

    for(int k = 0; k < rk_order; ++k)
    {
        // extracting RK coefficients
        amrex::Real alpha = coeffs[k].alp;
        amrex::Real beta = coeffs[k].bet;
        amrex::Real gamma = coeffs[k].gam;

        // performing KE evolution routine
        // compute and store KE fluxes in workspace
        computeKECompFluxes(stage);

        // evolve KE and store back in stage
        evolveKE(state_n, alpha, beta, gamma);

        // compute and store fluxes in workspace
        computeMomentumFluxes(stage);

        // compute predicted velocity without divergence free condition store
        // predicted velocity within stage
        predictVelocity(state_n, alpha, beta, gamma);

        // find divergence of predicted velocity, store in workspace use custom
        // LGF solver to find pressure correction delta update pressure stored
        // in stage
        computePressure();

        // use pressure to compute velocity correction store correction in
        // workspace
        computeVelocityCorrection();

        // correct stage using correction from workspace
        correctVelocityandPressure(gamma);
        
    }

    state_n = stage;
}

void ProjectionWorkspace::regridOnto(const amrex::Geometry& new_geom, const amrex::BoxArray& new_ba, const amrex::DistributionMapping& new_dm)
{
    // update the Poisson solver
    lgf_poisson_solver.regridOnto(new_geom, new_ba, new_dm);

    // update stage without worrying about data and so on
    stage.redefine(new_geom, new_ba, new_dm);

    // update all scratch MultiFabs
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        amrex::BoxArray new_ba_face = amrex::convert(new_ba, amrex::IntVect::TheDimensionVector(idim));
        
        rhs_vel[idim].define(new_ba_face, new_dm, rhs_vel[idim].nComp(), rhs_vel[idim].nGrow());
        rhs_vel_corr[idim].define(new_ba_face, new_dm, rhs_vel_corr[idim].nComp(), rhs_vel_corr[idim].nGrow());
        rhs_kecomp[idim].define(new_ba_face, new_dm, rhs_kecomp[idim].nComp(), rhs_kecomp[idim].nGrow());
        kecomp_dir[idim].define(new_ba_face, new_dm, kecomp_dir[idim].nComp(), kecomp_dir[idim].nGrow());
    }

    // reallocate cell-centered arrays
    pres_corr.define(new_ba, new_dm, pres_corr.nComp(), pres_corr.nGrow());
    divU.define(new_ba, new_dm, divU.nComp(), divU.nGrow());
}