#include <ProjectionWorkspace.H>

ProjectionWorkspace::ProjectionWorkspace(const amrex::Geometry& geom_in, const amrex::BoxArray& ba_in, const amrex::DistributionMapping& dm_in, const SolverConfig& config)
    : stage(geom_in, ba_in, dm_in, config.n_comp, config.n_ghost), lgf_poisson_solver(geom_in, config.n_lookup)
{
    // initializing required solver parameters
    n_lookup = config.n_lookup;
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

        // initialize velocities upon creation
        rhs_vel[idim].setVal(0.0);
        rhs_vel_corr[idim].setVal(0.0);
    }

    // initialize pres_corr upon creation
    pres_corr.define(ba_in, dm_in, n_comp, n_ghost);
    pres_corr.setVal(0.0);

    // initialize divU upon creation
    divU.define(ba_in, dm_in, n_comp, n_ghost);
    divU.setVal(0.0);

    divU_max_norm = 0.0;
    divU_at_end_max_norm = 0.0;
    divU_at_end_max_norm_support = 0.0;

    dt = 0.0;

    // compute shifted RK coeffs 
    setupRKCoeffs(getRKButcher());

    // pre-computing IF kernels for given RK tableau
    precomputeIFs();
}

void ProjectionWorkspace::setupRKCoeffs(const RKButcher& rkbt)
{
    BL_PROFILE("<Setup> setupRKCoeffs()");

    // set value of rk_stages member
    rk_stages = rkbt.n_stages;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(rk_stages >= 3 && rk_stages <= RKButcher::MAX_STAGES,
        "incompatible rk_stages and MAX_STAGES!");

    // check if tableau is valid
    const amrex::Real eps = 1.0e-14;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(rkbt.c[0]) < eps,
        "c_1 must be 0 (explicit first stage)");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(rkbt.c[rk_stages-1] - 1.0) < eps,
        "c_s must be 1: required for second-order constraints, and what makes "
        "the final integrating factor the identity (footnote 16)");

    amrex::Real bsum = 0.0;
    for (int j = 0; j < rk_stages; ++j) { bsum += rkbt.b[j]; }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(bsum - 1.0) < eps, "sum(b) != 1");

    for (int i = 0; i < rk_stages; ++i)
    {
        amrex::Real rowsum = 0.0;
        for (int j = 0; j < rk_stages; ++j)
        {// pre-computing IF kernels for given RK tableau
    precomputeIFs();
            if (j >= i) { AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(rkbt.A[i][j]) < eps,
                "A must be strictly lower triangular (explicit scheme)"); }
            rowsum += rkbt.A[i][j];
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(rowsum - rkbt.c[i]) < eps,
            "row sums of A must equal c (consistency)");
    }

    // initialize values to be computed to zero
    for (int i = 1; i <= RKButcher::MAX_STAGES; ++i)
    {
        rk_ct(i) = 0.0;  rk_gap(i) = 0.0;  rk_if_idx(i) = -1;
        for (int j = 1; j <= RKButcher::MAX_STAGES; ++j) { rk_at(i,j) = 0.0; }
    }

    // compute and store shifted coefficients
    for (int i = 1; i < rk_stages; ++i)
    {
        rk_ct(i) = rkbt.c[i];
        for (int j = 1; j <= rk_stages; ++j)
        {
            rk_at(i,j) = rkbt.A[i][j-1];
        }
    }
    rk_ct(rk_stages) = 1.0;
    for (int j = 1; j <= rk_stages; ++j)
    {
        rk_at(rk_stages,j) = rkbt.b[j-1];
    }

    // compute ct_i - ct_i-1 values
    amrex::Real prev = 0.0;
    for (int i = 1; i <= rk_stages; ++i) { rk_gap(i) = rk_ct(i) - prev; prev = rk_ct(i); }

    // validate the shifted tableau
    for (int i = 1; i <= rk_stages; ++i)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(rk_at(i,i)) > eps,
            "a~_{i,i} == 0: Eq. (30) divides by it when forming w^{i,i}");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(rk_gap(i) > -eps,
            "negative sub-step width: H^i would be E(-alpha), which amplifies the "
            "grid-scale mode. Tableau is IF-incompatible -- this is what rules out "
            "SSP-RK3, whose c = [0, 1, 1/2]");
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(rk_gap(rk_stages) < eps,
        "final sub-step width should vanish when c_s == 1");

    // make sub-vector of unique intervals, while updating a stage map
    amrex::Vector<amrex::Real> rk_unique_gaps;
    rk_unique_gaps.clear();

    for (int i = 1; i <= rk_stages; ++i)
    {
        if (rk_gap(i) < 1.0e-14) { rk_if_idx(i) = -1; continue; }   // H^i = I, skip it

        int found = -1;
        for (int m = 0; m < rk_unique_gaps.size(); ++m)
        {
            if (std::abs(rk_gap(i) - rk_unique_gaps[m]) < 1.0e-14) { found = m; break; }
        }
        if (found < 0)
        {
            rk_unique_gaps.push_back(rk_gap(i));
            found = static_cast<int>(rk_unique_gaps.size()) - 1;
        }
        rk_if_idx(i) = found;
    }
}

void ProjectionWorkspace::precomputeIFs()
{
    BL_PROFILE("<Setup> precomputeIFs()");

    // compute n required for desired eps tolerance
    amrex::Real max_gap = 0.0;
    for (auto g : rk_unique_gaps) { max_gap = std::max(max_gap, g); }

    // compute modified Bessel in 1D and store in array
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
    divU_at_end_max_norm_support = computeDivUMaxNorm(init_state, &init_supp_ba);
}

void ProjectionWorkspace::applyIF(amrex::MultiFab& phi_fab)
{}

void ProjectionWorkspace::computeRHSr(const FlowField& stage)
{}

void ProjectionWorkspace::computePressure(FlowField& stage)
{}

void ProjectionWorkspace::computeVelocity(FlowField& stage)
{}

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
    divU_at_end_max_norm_support = computeDivUMaxNorm(stage, &tag_ba);
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
    amrex::Vector<RKCoeffs> coeffs = getRKCoeffs(rk_stages);

    for(int k = 0; k < rk_stages; ++k)
    {
        // extracting RK coefficients
        amrex::Real alpha = coeffs[k].alp;
        amrex::Real beta = coeffs[k].bet;
        amrex::Real gamma = coeffs[k].gam;

        computeRHSr();
        computePressure();
        computeVelocity();
        
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
    // update stage without worrying about data and so on
    stage.redefine(new_geom, new_ba, new_dm);

    // update all scratch MultiFabs
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        amrex::BoxArray new_ba_face = amrex::convert(new_ba, amrex::IntVect::TheDimensionVector(idim));
        
        rhs_vel[idim].define(new_ba_face, new_dm, rhs_vel[idim].nComp(), rhs_vel[idim].nGrow());
        rhs_vel_corr[idim].define(new_ba_face, new_dm, rhs_vel_corr[idim].nComp(), rhs_vel_corr[idim].nGrow());

        rhs_vel[idim].setVal(0.0);
        rhs_vel_corr[idim].setVal(0.0);
    }

    // reallocate cell-centered arrays
    pres_corr.define(new_ba, new_dm, pres_corr.nComp(), pres_corr.nGrow());
    pres_corr.setVal(0.0);
    divU.define(new_ba, new_dm, divU.nComp(), divU.nGrow());
    divU.setVal(0.0);

    // update the Poisson solver
    lgf_poisson_solver.regridOnto(new_geom, new_ba, new_dm);  
}