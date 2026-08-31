#include <ProjectionWorkspace.H>

ProjectionWorkspace::ProjectionWorkspace(const amrex::Geometry& geom_in, const amrex::BoxArray& ba_in, const amrex::DistributionMapping& dm_in, const SolverConfig& config)
    : stage(geom_in, ba_in, dm_in, config.n_comp, config.n_ghost), lgf_poisson_solver(geom_in, config.n_lookup)
{
    // initializing required solver parameters
    n_lookup = config.n_lookup;
    invRe = config.invRe;
    IF_eps = config.int_fact_eps;
    n_IF = config.n_IF;
    dt = config.set_dt;
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

    // // compute shifted RK coeffs 
    // setupRKCoeffs(getRKButcher());

    setupRKCoeffs(getRKButcher());
    precomputeIFs(geom_in);                       // sets n_IF

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(config.n_ghost >= n_IF,
        "n_ghost must be at least the IF stencil half-width");

    w.resize(rk_stages + 1);                      // w[0] left undefined
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        amrex::BoxArray bf = amrex::convert(ba_in, amrex::IntVect::TheDimensionVector(idim));
        q[idim]      .define(bf, dm_in, n_comp, n_ghost);
        r[idim]      .define(bf, dm_in, n_comp, n_ghost);
        IF_buff[idim].define(bf, dm_in, n_comp, n_ghost);
        q[idim].setVal(0.0);  r[idim].setVal(0.0);  IF_buff[idim].setVal(0.0);

        for (int j = 1; j <= rk_stages; ++j)
        {
            w[j][idim].define(bf, dm_in, n_comp, n_ghost);
            w[j][idim].setVal(0.0);
        }
    }
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
        {
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

// void ProjectionWorkspace::precomputeIFs(const amrex::Geometry& geom)
// {
//     BL_PROFILE("<Setup> precomputeIFs()");

//     // IMPORTANT ASSUMPTION: dx = dy = dz
//     // find cell sizes, computing invdx2 using dx only
//     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx = geom.InvCellSizeArray();
//     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx2 = invdx[0] * invdx[0];

//     // compute n required for desired eps tolerance
//     amrex::Real max_gap = 0.0;
//     for (auto g : rk_unique_gaps)
//     {
//         max_gap = std::max(max_gap, g);
//     }

//     // checking what stencil size is needed for desired error
//     int n_IF = 14;
//     // amrex::Real max_alpha = max_gap * dt * invRe * invdx2;
//     // while(max_alpha^n_IF)
//     // n_IF = IFSupportSize((max_gap * dt * invRe * invdx2), IF_eps)

//     // compute modified Bessel in 1D and store in array

// }

// amrex::Real ProjectionWorkspace::computeDt(const FlowField& state_n) const
// {
//     BL_PROFILE("<Compute> computeDt()");

//     const amrex::Geometry& geom = state_n.getGeom();
//     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx = geom.InvCellSizeArray();
//     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx2;
//     for (int d = 0; d < AMREX_SPACEDIM; ++d) 
//     {
//         invdx2[d] = invdx[d] * invdx[d];
//     }

//     // cfl constraint: advective limit on dt
//     amrex::Real u_max = state_n.getVel(0).norm0(0, 0, false);
//     amrex::Real adv_metric = u_max * invdx[0];

//     amrex::Real v_max = state_n.getVel(1).norm0(0, 0, false);
//     adv_metric += v_max * invdx[1];

// #if AMREX_SPACEDIM == 3
//     amrex::Real w_max = state_n.getVel(2).norm0(0, 0, false);
//     adv_metric += w_max * invdx[2];
// #endif

//     // dt_adv = CFL / ( |u|/dx + |v|/dy + |w|/dz )
//     amrex::Real dt_adv = cfl / (adv_metric + 1.0e-12); // epsilon to prevent div-by-zero

//     // diffusive limit on dt
//     amrex::Real diff_metric = 1.0 * invdx2[0];
    
//     diff_metric += 1.0 * invdx2[1];

//     #if AMREX_SPACEDIM == 3
//         diff_metric += 1.0 * invdx2[2];
//     #endif

//     // for explicit schemes, Fourier number <= 0.5 dt_diff <= 0.5 * Re / (
//     // 1/dx^2 + 1/dy^2 + 1/dz^2 )
//     amrex::Real dt_diff = 0.5 / (invRe * diff_metric);

//     return amrex::min(dt_adv, dt_diff);
// }

void ProjectionWorkspace::initializePresField(FlowField& init_state,
                                              const amrex::BoxArray& init_supp_ba)
{
    BL_PROFILE("<Setup> initializePresField()");

    // DIAGNOSTIC ONLY under IF-HERK: dhat^i is solved fresh every stage and
    // the previous value of pres is never read.  This supplies d at t=0 for
    // the initial plotfile and smoke-tests the Poisson path.
    init_state.setBoundary();

    computeGStage(init_state, 1);          // w[1] = -a~_11 dt Ntilde(u_0)

    divU.setVal(0.0);
    const auto invdx = init_state.getGeom().InvCellSizeArray();
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

    lgf_poisson_solver.solvePoisson(divU, init_state.getPres(), init_supp_ba);
    init_state.getPres().mult(1.0 / (aT(1,1) * dt), 0);      // dhat -> d, Eq. (31)
    init_state.getPres().FillBoundary(init_state.getGeom().periodicity());

    divU_max_norm                = divU.norm0(0, 0, false);
    divU_at_end_max_norm         = computeDivUMaxNorm(init_state);
    divU_at_end_max_norm_support = computeDivUMaxNorm(init_state, &init_supp_ba);
}

// ---------------------------------------------------------------------------
// 3.  precomputeIFs -- 1D tables  g_a(n) = exp(-2a) I_n(2a),  n = 0..n_IF
// ---------------------------------------------------------------------------

void ProjectionWorkspace::precomputeIFs(const amrex::Geometry& geom)
{
    BL_PROFILE("<Setup> precomputeIFs()");

    // IMPORTANT ASSUMPTION: dx == dy == dz
    const amrex::Real dx  = geom.CellSize(0);
    const amrex::Real dx2 = dx * dx;
    AMREX_ALWAYS_ASSERT(std::abs(geom.CellSize(1) - dx) < 1.0e-12 * dx);

    n_IF = 14;   // TODO: size from IF_eps once the chassis runs

    if_table.resize(rk_unique_gaps.size());

    for (std::size_t m = 0; m < rk_unique_gaps.size(); ++m)
    {
        const amrex::Real alpha = rk_unique_gaps[m] * dt * invRe / dx2;

        // ascending series: I_n(z) = sum_k (z/2)^{n+2k} / (k! (n+k)!)
        amrex::Vector<amrex::Real> h(n_IF + 1);
        const amrex::Real z = 2.0 * alpha;
        for (int n = 0; n <= n_IF; ++n)
        {
            amrex::Real term = 1.0;
            for (int p = 1; p <= n; ++p) { term *= (0.5 * z) / amrex::Real(p); }
            amrex::Real sum = term;
            for (int k = 0; k < 200; ++k)
            {
                term *= (0.25 * z * z) / (amrex::Real(k + 1) * amrex::Real(n + k + 1));
                sum  += term;
                if (term < 1.0e-300) { break; }
            }
            h[n] = std::exp(-z) * sum;
        }

        amrex::Real mass = h[0];
        for (int n = 1; n <= n_IF; ++n) { mass += 2.0 * h[n]; }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(mass - 1.0) < 1.0e-8,
            "IF kernel mass far from 1: n_IF too small for this alpha, or bad table");

        if_table[m].resize(n_IF + 1);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h.begin(), h.end(), if_table[m].begin());

        amrex::Print() << "  IF kernel " << m << ": alpha = " << alpha
                       << ", n_IF = " << n_IF << ", mass deficit = " << 1.0 - mass << "\n";
    }
    amrex::Gpu::streamSynchronize();
}


// ---------------------------------------------------------------------------
// 4.  applyIF -- d separable 1D sweeps.  Ping-pongs through IF_buff.
// ---------------------------------------------------------------------------

void ProjectionWorkspace::applyIF(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& fld,
                                  int if_idx)
{
    BL_PROFILE("<Compute> applyIF()");

    if (if_idx < 0) { return; }                       // H = I, nothing to do

    const amrex::Real* tab = if_table[if_idx].dataPtr();
    const int          n   = n_IF;
    const auto&        per = stage.getGeom().periodicity();

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        for (int sd = 0; sd < AMREX_SPACEDIM; ++sd)
        {
            fld[idim].FillBoundary(per);

            const int di = (sd == 0), dj = (sd == 1), dk = (sd == 2);

            for (amrex::MFIter mfi(fld[idim], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const amrex::Box& bx = mfi.tilebox();
                auto const& src = fld[idim].const_array(mfi);
                auto const& dst = IF_buff[idim].array(mfi);

                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    amrex::Real s = tab[0] * src(i,j,k);
                    for (int m = 1; m <= n; ++m)
                    {
                        s += tab[m] * ( src(i + m*di, j + m*dj, k + m*dk)
                                      + src(i - m*di, j - m*dj, k - m*dk) );
                    }
                    dst(i,j,k) = s;
                });
            }
            std::swap(fld[idim], IF_buff[idim]);      // result back in fld
        }
    }
}


// ---------------------------------------------------------------------------
// 5.  g^i = -a~_{i,i} dt Ntilde(u^{i-1})   ->  slot i
// ---------------------------------------------------------------------------

void ProjectionWorkspace::computeGStage(const FlowField& st, int i)
{
    BL_PROFILE("<Compute> computeGStage()");

    const amrex::Geometry& geom = st.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx = geom.InvCellSizeArray();
    const amrex::Real coef = -aT(i,i) * dt;

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
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
                { g_arr(i2,j2,k2) = coef * nonLinearTerm<0>(i2,j2,k2, invdx, vel); });
            } else if (idim == 1) {
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i2,int j2,int k2)
                { g_arr(i2,j2,k2) = coef * nonLinearTerm<1>(i2,j2,k2, invdx, vel); });
            }
#if AMREX_SPACEDIM == 3
            else {
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i2,int j2,int k2)
                { g_arr(i2,j2,k2) = coef * nonLinearTerm<2>(i2,j2,k2, invdx, vel); });
            }
#endif
        }
    }
}


// ---------------------------------------------------------------------------
// 6.  advanceTimeStep -- Eqs. (25)-(31) with Eq. (35) for the stage solve
// ---------------------------------------------------------------------------

void ProjectionWorkspace::advanceTimeStep(FlowField& state_n, const amrex::Real dt_in,
                                          const amrex::BoxArray& supp_ba)
{
    BL_PROFILE("<Compute> advanceTimeStep()");

    tag_ba = supp_ba;
    stage  = state_n;                                        // u^0 = u_k

    const amrex::Geometry& geom = state_n.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx = geom.InvCellSizeArray();
    const int ng = q[0].nGrow();

    for (int d = 0; d < AMREX_SPACEDIM; ++d)                 // q^1 = u_k, Eq. (29)
    { amrex::MultiFab::Copy(q[d], state_n.getVel(d), 0, 0, 1, ng); }

    for (int i = 1; i <= rk_stages; ++i)
    {
        // --- g^i into slot i.  Fresh; never aged.  Eq. (28) ----------------
        stage.setBoundary();
        computeGStage(stage, i);

        // --- age q and w[1..i-1] by H^{i-1}.  Eqs. (29), (30) --------------
        if (i > 1)
        {
            const int idx = rk_if_idx(i-1);
            applyIF(q, idx);
            for (int j = 1; j <= i-1; ++j) { applyIF(w[j], idx); }
        }

        // --- r^i = q^i + dt sum_{j<i} a~_{i,j} w^{i,j} + g^i.  Eq. (27) ----
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            amrex::MultiFab::Copy(r[d], q[d], 0, 0, 1, ng);
            for (int j = 1; j <= i-1; ++j)
            { amrex::MultiFab::Saxpy(r[d], dt * aT(i,j), w[j][d], 0, 0, 1, ng); }
            amrex::MultiFab::Saxpy(r[d], 1.0, w[i][d], 0, 0, 1, ng);   // g^i, coeff 1
            r[d].FillBoundary(geom.periodicity());
        }

        // --- dhat: L_C dhat = D r      (D = -G^dagger, so this is Eq. 35) --
        divU.setVal(0.0);
        for (amrex::MFIter mfi(divU, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox();
            auto const& div_arr = divU.array(mfi);
            amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> r_arr{
                AMREX_D_DECL(r[0].const_array(mfi), r[1].const_array(mfi), r[2].const_array(mfi))};
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i2,int j2,int k2)
            { div_arr(i2,j2,k2) = discreteDivergenceF2C(i2,j2,k2, invdx, r_arr); });
        }
        lgf_poisson_solver.solvePoisson(divU, stage.getPres(), tag_ba);
        stage.getPres().FillBoundary(geom.periodicity());

        // --- rhs_vel = G dhat  (used twice: for u^i and for w^{i,i}) -------
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
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
        applyIF(stage.getVelArr(), rk_if_idx(i));   // TODO: add getVelArr() to FlowField

        // --- w^{i,i} = (g^i - G dhat) / (a~_{i,i} dt).  Eq. (30) -----------
        const amrex::Real inv = 1.0 / (aT(i,i) * dt);
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            amrex::MultiFab::Saxpy(w[i][d], -1.0, rhs_vel[d], 0, 0, 1, 0);
            w[i][d].mult(inv, 0);
        }
    }

    // --- Eq. (31): d_{k+1} = (a~_{s,s} dt)^{-1} dhat^s ---------------------
    stage.getPres().mult(1.0 / (aT(rk_stages, rk_stages) * dt), 0);

    state_n = stage;

    divU_at_end_max_norm         = computeDivUMaxNorm(state_n);
    divU_at_end_max_norm_support = computeDivUMaxNorm(state_n, &tag_ba);
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