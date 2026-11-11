#include <ProjectionWorkspace.H>
#include <FreestreamVelocity.H>

ProjectionWorkspace::ProjectionWorkspace(const amrex::Geometry& geom_in, const amrex::BoxArray& ba_in, const amrex::DistributionMapping& dm_in, const SolverConfig& config)
    : n_comp(config.n_comp), n_ghost(config.n_ghost),
      ifherk(getRKButcher(), geom_in, config.invRe, config.set_dt, config.n_IF, config.int_fact_eps),
      stage(geom_in, ba_in, dm_in, config.n_comp, config.n_ghost),
      lgf_poisson_solver(geom_in, ba_in, config.n_lookup, config.n_ghost_max)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_ghost >= ifherk.nGhostRequired(),
        "n_ghost must be at least the IF stencil half-width");
    ifherk.printSummary();

    defineScratch(ba_in, dm_in);

    divU_max_norm = 0.0;
    divU_at_end_max_norm = 0.0;
    divU_at_end_max_norm_support = 0.0;
}

void ProjectionWorkspace::defineScratch(const amrex::BoxArray& ba_in, const amrex::DistributionMapping& dm_in)
{
    // the zero fill is needed, not cosmetic: r^i is assembled with ghost cells included, so
    // the ghost cells of a w register are read (into ghost cells of r that nothing uses)
    // before any applyIF() has filled them. Left uninitialized, a debug build traps there on
    // AMReX's signaling-NaN fill
    w.resize(ifherk.nStages() + 1);               // w[0] left undefined

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // convert the box array to face centered
        const amrex::BoxArray ba_face = amrex::convert(ba_in, amrex::IntVect::TheDimensionVector(idim));

        rhs_vel[idim].define(ba_face, dm_in, n_comp, n_ghost);
        q[idim]      .define(ba_face, dm_in, n_comp, n_ghost);
        r[idim]      .define(ba_face, dm_in, n_comp, n_ghost);
        IF_buff[idim].define(ba_face, dm_in, n_comp, n_ghost);
        rhs_vel[idim].setVal(0.0);
        q[idim].setVal(0.0);  r[idim].setVal(0.0);  IF_buff[idim].setVal(0.0);

        for (int j = 1; j <= ifherk.nStages(); ++j)
        {
            w[j][idim].define(ba_face, dm_in, n_comp, n_ghost);
            w[j][idim].setVal(0.0);
        }
    }

    // cell-centered scratch
    divU.define(ba_in, dm_in, n_comp, n_ghost);
    divU.setVal(0.0);
}


// ---------------------------------------------------------------------------
// g^i = -a~_{i,i} dt Ntilde(u^{i-1})   ->  slot i
// ---------------------------------------------------------------------------

void ProjectionWorkspace::computeGStage(const FlowField& st, int i, amrex::Real t_stage)
{
    BL_PROFILE("<Compute> computeGStage()");

    const amrex::Geometry& geom = st.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx = geom.InvCellSizeArray();
    const amrex::Real coef = -ifherk.aT(i,i) * ifherk.timeStep();
    const auto uinf = freestreamVelocity(t_stage);     // uniform: evaluated once, on the host

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        LGF_NSE_OMP_MFITER
        for (amrex::MFIter mfi(w[i][idim], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox();
            auto const& g_arr = w[i][idim].array(mfi);

            amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel{
                AMREX_D_DECL(st.getVel(0).const_array(mfi),
                             st.getVel(1).const_array(mfi),
                             st.getVel(2).const_array(mfi))};

            if (idim == 0) {
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i2,int j2,int k2)
                { g_arr(i2,j2,k2) = coef * nonLinearTerm<0>(i2,j2,k2, invdx, vel, uinf); });
            } else if (idim == 1) {
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i2,int j2,int k2)
                { g_arr(i2,j2,k2) = coef * nonLinearTerm<1>(i2,j2,k2, invdx, vel, uinf); });
            }
#if AMREX_SPACEDIM == 3
            else {
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i2,int j2,int k2)
                { g_arr(i2,j2,k2) = coef * nonLinearTerm<2>(i2,j2,k2, invdx, vel, uinf); });
            }
#endif
        }
    }
}

void ProjectionWorkspace::maskToSupport(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& fld,
                                        const amrex::BoxArray& supp)
{
    BL_PROFILE("<Compute> maskToSupport()");
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        const amrex::BoxArray keep = amrex::convert(supp, fld[d].ixType());

        LGF_NSE_OMP_MFITER
        for (amrex::MFIter mfi(fld[d], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::BoxList outside = keep.complementIn(mfi.tilebox());
            for (const amrex::Box& b : outside)
            {
                fld[d][mfi].setVal<amrex::RunOn::Device>(0.0, b);
            }
        }
    }
}

void ProjectionWorkspace::initializePresField(FlowField& init_state,
                                              const amrex::BoxArray& init_supp_ba,
                                              amrex::Real time)
{
    BL_PROFILE("<Setup> initializePresField()");

    // DIAGNOSTIC ONLY under IF-HERK: dhat^i is solved fresh every stage and
    // the previous value of pres is never read.  This supplies d at t=0 for
    // the initial plotfile and smoke-tests the Poisson path.
    init_state.setBoundary();

    computeGStage(init_state, 1, time);    // w[1] = -a~_11 dt Ntilde(u_0, t_0)

    divU.setVal(0.0);
    const auto invdx = init_state.getGeom().InvCellSizeArray();
    LGF_NSE_OMP_MFITER
    for (amrex::MFIter mfi(divU, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& div_arr = divU.array(mfi);
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> g_arr{
            AMREX_D_DECL(w[1][0].const_array(mfi),
                         w[1][1].const_array(mfi),
                         w[1][2].const_array(mfi))};
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i,int j,int k)
        { div_arr(i,j,k) = discreteDivergenceF2C(i,j,k, invdx, g_arr); });
    }

    divU.FillBoundary(init_state.getGeom().periodicity());
    lgf_poisson_solver.solvePoisson(divU, init_state.getPres(), init_supp_ba);
    init_state.getPres().mult(1.0 / (ifherk.aT(1,1) * ifherk.timeStep()), 0);      // dhat -> d, Eq. (31)
    init_state.getPres().FillBoundary(init_state.getGeom().periodicity());

    divU_max_norm                = divU.norm0(0, 0, false);
    divU_at_end_max_norm         = computeDivUMaxNorm(init_state);
    divU_at_end_max_norm_support = computeDivUMaxNorm(init_state, &init_supp_ba);
}


// ---------------------------------------------------------------------------
// advanceTimeStep -- Eqs. (25)-(31) with Eq. (35) for the stage solve
// ---------------------------------------------------------------------------

void ProjectionWorkspace::advanceTimeStep(FlowField& state_n, const amrex::BoxArray& supp_ba,
                                          amrex::Real time_n, InterfaceTracker* tracker)
{
    BL_PROFILE("<Compute> advanceTimeStep()");

    const int         rk_stages = ifherk.nStages();
    const amrex::Real dt        = ifherk.timeStep();

    stage = state_n;                                         // u^0 = u_k

    const amrex::Geometry& geom = state_n.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx = geom.InvCellSizeArray();
    const int ng = q[0].nGrow();

    for (int d = 0; d < AMREX_SPACEDIM; ++d)                 // q^1 = u_k, Eq. (29)
    { amrex::MultiFab::Copy(q[d], state_n.getVel(d), 0, 0, 1, ng); }

    for (int i = 1; i <= rk_stages; ++i)
    {
        // --- g^i into slot i.  Fresh; never aged.  Eq. (28) ----------------
        // u^{i-1} lives at t_n + c~_{i-1} dt (c~_0 = 0): N is evaluated there
        const amrex::Real t_stage = time_n + ((i == 1) ? 0.0 : ifherk.cT(i-1)) * dt;
        stage.setBoundary();
        computeGStage(stage, i, t_stage);
        maskToSupport(w[i], supp_ba);

        // --- psi^i from the same u^{i-1}, t_stage.  No IF: psi does not diffuse
        if (tracker) { tracker->advanceStage(i, ifherk, stage.getVelArr(), freestreamVelocity(t_stage)); }

        // --- age q and w[1..i-1] by H^{i-1}.  Eqs. (29), (30) --------------
        if (i > 1)
        {
            const int idx = ifherk.ifIdx(i-1);
            ifherk.applyIF(q, IF_buff, idx);
            for (int j = 1; j <= i-1; ++j) { ifherk.applyIF(w[j], IF_buff, idx); }
        }

        // --- r^i = q^i + dt sum_{j<i} a~_{i,j} w^{i,j} + g^i.  Eq. (27) ----
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            amrex::MultiFab::Copy(r[d], q[d], 0, 0, 1, ng);
            for (int j = 1; j <= i-1; ++j)
            { amrex::MultiFab::Saxpy(r[d], dt * ifherk.aT(i,j), w[j][d], 0, 0, 1, ng); }
            amrex::MultiFab::Saxpy(r[d], 1.0, w[i][d], 0, 0, 1, ng);   // g^i, coeff 1
            r[d].FillBoundary(geom.periodicity());
        }

        // --- dhat: L_C dhat = D r      (D = -G^dagger, so this is Eq. 35) --
        divU.setVal(0.0);
        LGF_NSE_OMP_MFITER
        for (amrex::MFIter mfi(divU, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox();
            auto const& div_arr = divU.array(mfi);
            amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> r_arr{
                AMREX_D_DECL(r[0].const_array(mfi), r[1].const_array(mfi), r[2].const_array(mfi))};
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i2,int j2,int k2)
            { div_arr(i2,j2,k2) = discreteDivergenceF2C(i2,j2,k2, invdx, r_arr); });
        }
        divU.FillBoundary(geom.periodicity());
        lgf_poisson_solver.solvePoisson(divU, stage.getPres(), supp_ba);
        stage.getPres().FillBoundary(geom.periodicity());

        // --- rhs_vel = G dhat  (used twice: for u^i and for w^{i,i}) -------
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            LGF_NSE_OMP_MFITER
            for (amrex::MFIter mfi(rhs_vel[d], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const amrex::Box& bx = mfi.tilebox();
                auto const& gd  = rhs_vel[d].array(mfi);
                auto const& phi = stage.getPres().const_array(mfi);
                if (d == 0) {
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i2,int j2,int k2)
                    { gd(i2,j2,k2) = discreteGradientC2F<0>(i2,j2,k2, invdx, phi); });
                } else if (d == 1) {
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i2,int j2,int k2)
                    { gd(i2,j2,k2) = discreteGradientC2F<1>(i2,j2,k2, invdx, phi); });
                }
#if AMREX_SPACEDIM == 3
                else {
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i2,int j2,int k2)
                    { gd(i2,j2,k2) = discreteGradientC2F<2>(i2,j2,k2, invdx, phi); });
                }
#endif
            }
        }

        // --- u^i = H^i (r^i - G dhat).  Eq. (35) ---------------------------
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            amrex::MultiFab::Copy (stage.getVel(d), r[d],       0, 0, 1, 0);
            amrex::MultiFab::Saxpy(stage.getVel(d), -1.0, rhs_vel[d], 0, 0, 1, 0);
        }
        ifherk.applyIF(stage.getVelArr(), IF_buff, ifherk.ifIdx(i));

        // --- w^{i,i} = (g^i - G dhat) / (a~_{i,i} dt).  Eq. (30) -----------
        const amrex::Real inv = 1.0 / (ifherk.aT(i,i) * dt);
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            amrex::MultiFab::Saxpy(w[i][d], -1.0, rhs_vel[d], 0, 0, 1, 0);
            w[i][d].mult(inv, 0);
        }
    }

    // --- Eq. (31): d_{k+1} = (a~_{s,s} dt)^{-1} dhat^s ---------------------
    stage.getPres().mult(1.0 / (ifherk.aT(rk_stages, rk_stages) * dt), 0);

    stage.setBoundary();
    state_n = stage;

    // psi_{k+1} = psi^s
    if (tracker) { tracker->finishStep(); }

    divU_max_norm                = divU.norm0(0, 0, false);
    divU_at_end_max_norm         = computeDivUMaxNorm(state_n);
    divU_at_end_max_norm_support = computeDivUMaxNorm(state_n, &supp_ba);
}

void ProjectionWorkspace::regridOnto(const amrex::Geometry& new_geom, const amrex::BoxArray& new_ba, const amrex::DistributionMapping& new_dm)
{
    // update stage without worrying about data and so on
    stage.redefine(new_geom, new_ba, new_dm);

    // update all scratch MultiFabs
    defineScratch(new_ba, new_dm);

    // the integrator keeps its own copy of the geometry (tables are rebuilt only if dx changed)
    ifherk.setGeometry(new_geom);

    // update the Poisson solver
    lgf_poisson_solver.regridOnto(new_geom, new_ba, new_dm);

    // the interface registers live in InterfaceTracker, which drops them in its own regrid()
}