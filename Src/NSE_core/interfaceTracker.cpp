#include <InterfaceTracker.H>

InterfaceTracker::InterfaceTracker(const amrex::Geometry& geom_in, const amrex::BoxArray& ba_in,
                                   const amrex::DistributionMapping& dm_in, const SolverConfig& config)
    : InterfaceTracker(geom_in, ba_in, dm_in, advectionSchemeFromName(config.interface_scheme),
                       config.interface_psi_far)
{
    setReinitialization(config.interface_reinit_int, config.interface_reinit_drift, config.interface_reinit_degree);
    amrex::Print() << "Interface tracker: " << n_interface << " interface(s) from InitialSDF.H\n";
}

InterfaceTracker::InterfaceTracker(const amrex::Geometry& geom_in, const amrex::BoxArray& ba_in,
                                   const amrex::DistributionMapping& dm_in,
                                   AdvectionScheme scheme_in, amrex::Real psi_far_in)
    : geom(geom_in), ba(ba_in), dm(dm_in),
      n_interface(N_INTERFACE), scheme(scheme_in), psi_far(psi_far_in)
{
    // the grid is expected to be the snug domain (DomainManager after
    // initializeSnugDomain() or restartSnugDomain()); one psi component per
    // interface of the active case in InitialSDF.H

    // rules on what can be initialized and what can't be
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_interface >= 1, "InterfaceTracker: N_INTERFACE must be at least 1");

    // the kink at |psi| = psi_far must stay out of the stencils of cells near the interface
    const amrex::Real dx_max = *std::max_element(geom.CellSize(), geom.CellSize() + AMREX_SPACEDIM);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(psi_far > 2 * LS_N_GHOST * dx_max,
        "InterfaceTracker: psi_far must exceed 2*LS_N_GHOST cell widths");

    defineLikePsi(psi, LS_N_GHOST);
    psi.setVal(-psi_far);
}

void InterfaceTracker::defineLikePsi(amrex::MultiFab& mf, int n_grow) const
{
    // for stage and right-hand-side storage in the time integrator
    mf.define(ba, dm, n_interface, n_grow);
}

void InterfaceTracker::initializeSDF()
{
    BL_PROFILE("<Compute> InterfaceTracker::initializeSDF()");

    const auto problo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();
    const amrex::Real clip = psi_far;

    // the active case of InitialSDF.H, clipped to +-psi_far, on valid and ghost
    // cells alike, so no boundary fill is needed here
    LGF_NSE_OMP_MFITER
    for (amrex::MFIter mfi(psi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box bx = mfi.growntilebox();
        auto const& p = psi.array(mfi);

        amrex::ParallelFor(bx, n_interface, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            // cell centre; the 2D build passes z = 0, as for initialVorticity
            const int ijk[3] = {i, j, k};
            amrex::Real x[3] = {0.0, 0.0, 0.0};
            for (int d = 0; d < AMREX_SPACEDIM; ++d) { x[d] = problo[d] + (ijk[d] + 0.5) * dx[d]; }

            p(i, j, k, n) = amrex::Clamp(initialSDF(n, x[0], x[1], x[2]), -clip, clip);
        });
    }

    // the band of every interface must start inside the snug domain
    checkBandInside(ba, "InterfaceTracker::initializeSDF");
}

void InterfaceTracker::regrid(const amrex::Geometry& new_geom, const amrex::BoxArray& new_ba,
                              const amrex::DistributionMapping& new_dm)
{
    BL_PROFILE("<Compute> InterfaceTracker::regrid()");

    // call whenever DomainManager::did_snug_domain_change is true. Cells new to
    // the snug domain take the far-field value and dropped cells are discarded;
    // checkBandInside() aborts if either would touch the band. All other cells
    // are copied over
    checkBandInside(new_ba, "InterfaceTracker::regrid");

    amrex::MultiFab new_psi(new_ba, new_dm, n_interface, LS_N_GHOST);
    new_psi.setVal(-psi_far);
    new_psi.ParallelCopy(psi, 0, 0, n_interface, 0, 0);

    // update members
    geom = new_geom;
    ba = new_ba;
    dm = new_dm;
    psi = std::move(new_psi);

    // the stage registers hold no state between steps: drop them, and the next
    // advanceStage() defines them on the new grid
    psi_stage.clear();
    psi_rhs.clear();

    fillPsiBoundary(psi);
}

void InterfaceTracker::checkBandInside(const amrex::BoxArray& next_ba, const std::string& caller) const
{
    BL_PROFILE("<Compute> InterfaceTracker::checkBandInside()");

    // the band (|psi| < psi_far) must stay inside the snug domain: regrid() gives
    // cells new to the domain -psi_far and discards the cells it drops, and ghost
    // cells outside the domain are extrapolated. So check the largest psi over two
    // sets of cells of the current grid:
    //   (a) cells that next_ba no longer covers (none when next_ba is the current
    //       grid, as after initialization), and
    //   (b) cells on the edge of the current grid (touching a cell outside it,
    //       diagonals included), which neighbour any new cells.
    // The clip contour may come within the stencil reach of either set; anything
    // closer means the band itself would be cut, or would border a jump in psi

    using ReduceTuple = amrex::GpuTuple<amrex::Real, amrex::Real>;
    amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax> reduce_op;
    amrex::ReduceData<amrex::Real, amrex::Real> reduce_data(reduce_op);

    const amrex::Real lowest = std::numeric_limits<amrex::Real>::lowest();

    LGF_NSE_OMP_MFITER
    for (amrex::MFIter mfi(psi); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        auto const& p = psi.const_array(mfi);

        // (a) parts of this box the next grid drops
        for (const amrex::Box& b : next_ba.complementIn(vbx))
        {
            reduce_op.eval(b, n_interface, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept -> ReduceTuple
            {
                return {p(i, j, k, n), lowest};
            });
        }

        // (b) cells of this box that touch a cell outside the current grid
        for (const amrex::Box& outside : ba.complementIn(amrex::grow(vbx, 1)))
        {
            const amrex::Box b = amrex::grow(outside, 1) & vbx;
            if (!b.ok()) { continue; }

            reduce_op.eval(b, n_interface, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept -> ReduceTuple
            {
                return {lowest, p(i, j, k, n)};
            });
        }
    }

    const ReduceTuple local_max = reduce_data.value(reduce_op);
    amrex::Real psi_max[2] = {amrex::get<0>(local_max), amrex::get<1>(local_max)};
    amrex::ParallelDescriptor::ReduceRealMax(psi_max, 2);

    const amrex::Real dx_max = *std::max_element(geom.CellSize(), geom.CellSize() + AMREX_SPACEDIM);
    const amrex::Real psi_allowed = -psi_far + LS_N_GHOST * dx_max;
    const char* region[2] = {"cells the new grid drops", "cells on the edge of the snug domain"};

    for (int r = 0; r < 2; ++r)
    {
        if (psi_max[r] > psi_allowed)
        {
            amrex::Print(amrex::ErrorStream())
                << caller << ": max psi over " << region[r] << " is " << psi_max[r]
                << ", but must not exceed -psi_far + LS_N_GHOST*dx = " << psi_allowed << "\n";
            amrex::Abort(caller + ": the interface band (|psi| < psi_far) reaches the edge of the snug domain"
                         " or cells a regrid drops; it must stay inside (band-based tagging is not"
                         " implemented yet). Enlarge the domain/buffer, or reduce psi_far");
        }
    }
}

void InterfaceTracker::fillPsiBoundary(amrex::MultiFab& psi_in) const
{
    // no flow information (initialization, regrid, restart): extrapolate every
    // outside ghost. The next right-hand side refills them with the flow
    fillPsiBoundaryImpl(psi_in, nullptr, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>{});
}

void InterfaceTracker::fillPsiBoundary(amrex::MultiFab& psi_in, const amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& vel,
                                       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& uinf) const
{
    fillPsiBoundaryImpl(psi_in, &vel, uinf);
}

void InterfaceTracker::fillPsiBoundaryImpl(amrex::MultiFab& psi_in, const amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>* vel,
                                           amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& uinf) const
{
    BL_PROFILE("<Compute> InterfaceTracker::fillPsiBoundary()");

    // psi is not compact, so the flow solver's "zero outside the snug domain"
    // convention cannot be used (zero is the interface itself). Ghost cells
    // outside the snug domain follow the characteristics, as the legacy tube
    // edge does with first-order upwinding:
    //   inflow  (flow enters through the box face the ghost lies beyond): the
    //           far-field value -psi_far, the only correct data from outside;
    //   outflow (or no velocity given): linear extrapolation from the nearest
    //           valid cell, psi_c + s (psi_c - psi_{c-1}) for the ghost s cells out.
    // Extrapolating at an inflow face would re-impose the interior slope every
    // step, so a disturbance there would grow instead of being washed out

    const bool has_vel = (vel != nullptr);
    const amrex::Real far = psi_far;

    // 1. every ghost cell is filled from its own box. Reads valid cells and
    //    boundary faces only, writes ghosts only, so there is no race
    LGF_NSE_OMP_MFITER
    for (amrex::MFIter mfi(psi_in, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box gbx = mfi.growntilebox();
        const amrex::Box vbx = mfi.validbox();
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);

        auto const& p = psi_in.array(mfi);
        auto const& pc = psi_in.const_array(mfi);
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> u;
        if (has_vel)
        {
            for (int d = 0; d < AMREX_SPACEDIM; ++d) { u[d] = (*vel)[d].const_array(mfi); }
        }

        amrex::ParallelFor(gbx, psi_in.nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            if (vbx.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))) { return; }

            // nearest valid cell
            const int ic = amrex::Clamp(i, lo.x, hi.x);
            const int jc = amrex::Clamp(j, lo.y, hi.y);
            const int kc = amrex::Clamp(k, lo.z, hi.z);

            // inflow through any face the ghost lies beyond (edges and corners: either face)
            const bool inflow = has_vel && (AMREX_D_TERM(ghostInflow<0>(u[0], ic, jc, kc, i - ic, uinf[0]),
                                                      || ghostInflow<1>(u[1], ic, jc, kc, j - jc, uinf[1]),
                                                      || ghostInflow<2>(u[2], ic, jc, kc, k - kc, uinf[2])));

            p(i, j, k, n) = inflow ? -far
                          : pc(ic, jc, kc, n)
                            AMREX_D_TERM(+ ghostIncrement<0>(pc, ic, jc, kc, n, i - ic, hi.x > lo.x),
                                         + ghostIncrement<1>(pc, ic, jc, kc, n, j - jc, hi.y > lo.y),
                                         + ghostIncrement<2>(pc, ic, jc, kc, n, k - kc, hi.z > lo.z));
        });
    }

    // 2. ghost cells that overlap a neighbouring box take its valid values
    psi_in.FillBoundary(geom.periodicity());
}

template <AdvectionScheme scheme_t>
void InterfaceTracker::computeAdvectionRHSImpl(const amrex::MultiFab& psi_in, amrex::MultiFab& rhs,
                                               const amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& vel,
                                               amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& uinf) const
{
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx = geom.InvCellSizeArray();

    LGF_NSE_OMP_MFITER
    for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box bx = mfi.tilebox();

        auto const& p = psi_in.const_array(mfi);
        auto const& r = rhs.array(mfi);
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> u;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            u[d] = vel[d].const_array(mfi);
        }

        amrex::ParallelFor(bx, n_interface, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            r(i, j, k, n) = advectionRHS<scheme_t>(i, j, k, n, invdx, p, u, uinf,
                                                   std::make_integer_sequence<int, AMREX_SPACEDIM>{});
        });
    }
}

void InterfaceTracker::computeAdvectionRHS(amrex::MultiFab& psi_in, amrex::MultiFab& rhs,
                                           const amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& vel,
                                           amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& uinf) const
{
    BL_PROFILE("<Compute> InterfaceTracker::computeAdvectionRHS()");

    // rhs = -(u + u_inf) . grad(psi) on the valid cells of psi_in's grid. vel holds
    // the face perturbation velocity of the IF-HERK stage (the same one passed
    // to the nonlinear term) and uinf the freestream at that stage's time.
    // psi_in may be psi itself or a stage copy; its ghost cells are refilled here,
    // with inflow faces of the snug domain taking the far-field value

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(psi_in.nGrow() >= LS_N_GHOST,
        "InterfaceTracker::computeAdvectionRHS: psi needs LS_N_GHOST ghost cells");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(psi_in.nComp() == n_interface && rhs.nComp() == n_interface,
        "InterfaceTracker::computeAdvectionRHS: psi and rhs need one component per interface");
    AMREX_ASSERT_WITH_MESSAGE(psi_in.boxArray() == ba && rhs.boxArray() == ba && psi_in.DistributionMap() == dm,
        "InterfaceTracker::computeAdvectionRHS: grid mismatch; was regrid() called after the snug domain changed?");
#ifdef AMREX_DEBUG
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        AMREX_ASSERT_WITH_MESSAGE(vel[d].boxArray().CellEqual(ba) && vel[d].DistributionMap() == dm,
            "InterfaceTracker::computeAdvectionRHS: velocity lives on a different grid than psi");
    }
#endif

    fillPsiBoundary(psi_in, vel, uinf);

    switch (scheme)
    {
        case AdvectionScheme::WENO5:
            computeAdvectionRHSImpl<AdvectionScheme::WENO5>(psi_in, rhs, vel, uinf);
            break;
        case AdvectionScheme::HOUC5:
            computeAdvectionRHSImpl<AdvectionScheme::HOUC5>(psi_in, rhs, vel, uinf);
            break;
    }
}

amrex::Real InterfaceTracker::computeCFL(const amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& vel,
                                         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& uinf,
                                         amrex::Real dt) const
{
    BL_PROFILE("<Compute> InterfaceTracker::computeCFL()");

    // advective CFL number of the psi transport in sum form, max over the snug
    // domain of sum_dim |u_dim| dt / dx_dim, with u_dim the same face-to-centre
    // average (plus freestream) that the right-hand side uses
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx = geom.InvCellSizeArray();

    amrex::ReduceOps<amrex::ReduceOpMax> reduce_op;
    amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    LGF_NSE_OMP_MFITER
    for (amrex::MFIter mfi(psi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> u;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { u[d] = vel[d].const_array(mfi); }

        reduce_op.eval(mfi.tilebox(), reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
        {
            amrex::Real rate = 0.0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                const amrex::Real uc = 0.5 * (u[d](i, j, k) + u[d](i + (d == 0), j + (d == 1), k + (d == 2))) + uinf[d];
                rate += amrex::Math::abs(uc) * invdx[d];
            }
            return {rate};
        });
    }

    amrex::Real rate_max = amrex::get<0>(reduce_data.value(reduce_op));
    amrex::ParallelDescriptor::ReduceRealMax(rate_max);
    return rate_max * dt;
}

void InterfaceTracker::defineStageRegisters(int n_stages)
{
    // psi_stage is fed back to the right-hand side, which needs LS_N_GHOST ghost cells
    defineLikePsi(psi_stage, LS_N_GHOST);
    psi_stage.setVal(0.0);

    psi_rhs.clear();
    psi_rhs.resize(n_stages + 1);                 // psi_rhs[0] left undefined
    for (int j = 1; j <= n_stages; ++j)
    {
        defineLikePsi(psi_rhs[j], 0);
        psi_rhs[j].setVal(0.0);
    }
}

void InterfaceTracker::advanceStage(int i, const IFHERK& integrator,
                                    const amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& vel,
                                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& uinf)
{
    BL_PROFILE("<Compute> InterfaceTracker::advanceStage()");

    // psi^i = psi_k + dt sum_{j<=i} a~_{i,j} L^j,  L^i = L(psi^{i-1}, u^{i-1})
    //
    // plain explicit RK with the shifted tableau: vel and uinf are the same u^{i-1} and
    // u_inf(t_stage) that the flow solver feeds to N, and psi^{i-1} is psi_k itself on the
    // first stage. psi is only written by finishStep(), so it is psi_k for the whole step
    // (the right-hand side refills its ghost cells, nothing more)
    const int n_stages = integrator.nStages();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(i >= 1 && i <= n_stages, "InterfaceTracker::advanceStage: stage out of range");

    if (static_cast<int>(psi_rhs.size()) != n_stages + 1) { defineStageRegisters(n_stages); }

    amrex::MultiFab& psi_prev = (i == 1) ? psi : psi_stage;
    computeAdvectionRHS(psi_prev, psi_rhs[i], vel, uinf);

    const amrex::Real dt = integrator.timeStep();
    amrex::MultiFab::Copy(psi_stage, psi, 0, 0, n_interface, 0);
    for (int j = 1; j <= i; ++j)
    { amrex::MultiFab::Saxpy(psi_stage, dt * integrator.aT(i,j), psi_rhs[j], 0, 0, n_interface, 0); }
}

void InterfaceTracker::finishStep()
{
    // psi_{k+1} = psi^s, since a~_{s,j} = b_j
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!psi_stage.empty(), "InterfaceTracker::finishStep: no stage was advanced");
    amrex::MultiFab::Copy(psi, psi_stage, 0, 0, n_interface, 0);
}

// ------------------------------------------------------------------------------------------------
// reinitialization: psi reset to the distance to its own zero level set, rebuilt by local
// interpolation (closest points, InterfaceReinit.H). Only the band within psi_far of the
// interface is computed; everything beyond is the far-field value

void InterfaceTracker::setReinitialization(int interval, amrex::Real drift, int degree)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(interval >= 0, "InterfaceTracker: interface_reinit_int must be >= 0");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(drift >= 0.0, "InterfaceTracker: interface_reinit_drift must be >= 0");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(degree == 3 || degree == 5, "InterfaceTracker: interface_reinit_degree must be 3 or 5");

    reinit_int = interval;
    reinit_drift = drift;
    reinit_degree = degree;
    if (!reinitEnabled()) { return; }

#if (AMREX_SPACEDIM != 3)
    amrex::Abort("InterfaceTracker: reinitialization is implemented for 3D builds only");
#endif

    // every band cell reads psi up to psi_far + a few cells away, so psi is copied with that
    // many ghost cells; a wide band would make the copy (and the work) large
    const amrex::Real h = *std::min_element(geom.CellSize(), geom.CellSize() + AMREX_SPACEDIM);
    if (psi_far > REINIT_MAX_BAND_CELLS * h)
    {
        amrex::Print(amrex::ErrorStream())
            << "InterfaceTracker: psi_far = " << psi_far << " is " << psi_far / h << " cells; reinitialization"
            << " handles a band of at most " << REINIT_MAX_BAND_CELLS << " cells. With regular reinitialization"
            << " the band only needs to cover the transport stencils: about 8 cells (" << 8.0 * h << ") is plenty\n";
        amrex::Abort("InterfaceTracker: interface_psi_far too large for reinitialization");
    }

    amrex::Print() << "Interface reinitialization: closest point, degree " << reinit_degree;
    if (reinit_int > 0) { amrex::Print() << " | every " << reinit_int << " steps"; }
    if (reinit_drift > 0.0)
    {
        amrex::Print() << " | when more than " << 100.0 * REINIT_DRIFT_FRACTION
                       << "% of cells next to the interface have ||grad psi| - 1| > " << reinit_drift;
    }
    amrex::Print() << " | band " << psi_far / h << " cells, " << reinitGhost() << " ghost cells for the copy\n";
}

int InterfaceTracker::reinitBandCells() const
{
    // cells within psi_far of the interface lie within this many cells (in every direction)
    // of a cell next to a sign change of psi
    const amrex::Real h = *std::min_element(geom.CellSize(), geom.CellSize() + AMREX_SPACEDIM);
    return static_cast<int>(std::ceil(psi_far / h - 1.0e-9)) + 1;
}

int InterfaceTracker::reinitGhost() const
{
    // the deep ghost copy must hold (a) the sign-change test around the band and (b) the
    // interpolation stencil (P = degree + 1 cells per direction) of every iterate, which may
    // stray up to psi_far + REINIT_REACH_MARGIN cells from its own cell
    const amrex::Real h = *std::min_element(geom.CellSize(), geom.CellSize() + AMREX_SPACEDIM);
    const int reach = static_cast<int>(std::ceil(psi_far / h + 2.0 - 1.0e-9));   // REINIT_REACH_MARGIN = 2
    const int half_stencil = (reinit_degree + 1) / 2;
    return std::max(reinitBandCells() + 1, reach + half_stencil + 1);
}

template <int P>
InterfaceTracker::ReinitStats InterfaceTracker::reinitializeComponent(int n)
{
    ReinitStats stats;
#if (AMREX_SPACEDIM == 3)
    const auto problo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();
    const auto invdx = geom.InvCellSizeArray();
    const amrex::Real h = amrex::min(dx[0], amrex::min(dx[1], dx[2]));
    const amrex::Real far = psi_far;
    const amrex::Real reach = psi_far + REINIT_REACH_MARGIN * h;
    const int m = reinitBandCells();
    const int ng = reinitGhost();

    // 0. psi_old of this interface with deep ghost cells: other boxes' valid values where they
    //    overlap, the far-field value elsewhere (outside the snug domain psi is -psi_far, which
    //    checkBandInside() guarantees near its edges)
    amrex::MultiFab phi(ba, dm, 1, ng);
    phi.setVal(-far);
    amrex::MultiFab::Copy(phi, psi, n, 0, 1, 0);
    phi.FillBoundary(geom.periodicity());

    // closest point of every cell (components 0-2) and its status (component 3), with one
    // ghost cell for the neighbour passes; double-buffered so each pass reads a fixed state
    amrex::MultiFab cp(ba, dm, 4, 1), cp_next(ba, dm, 4, 1);
    for (amrex::MultiFab* mf : {&cp, &cp_next})
    {
        mf->setVal(0.0);
        mf->setVal(REINIT_OUTSIDE, 3, 1, 1);
    }

    // 1. the band, and the first pass: projection from the cell centre, then Newton. The band
    //    is every cell within m cells (in each direction) of a cell whose face neighbour has
    //    the opposite sign: the sign-change flags are widened by m with three separable max
    //    filters, on boxes grown so that the valid box ends up exact. Whole boxes, no tiling:
    //    the filters need the flags around the box
    LGF_NSE_OMP_MFITER
    for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi)
    {
        const amrex::Box vbx = mfi.validbox();
        auto const& p = phi.const_array(mfi);
        auto const& c = cp.array(mfi);

        const amrex::Box b0 = amrex::grow(vbx, m);
        const amrex::Box b1 = amrex::grow(vbx, amrex::IntVect(0, m, m));
        const amrex::Box b2 = amrex::grow(vbx, amrex::IntVect(0, 0, m));
        amrex::IArrayBox f0(b0, 1, amrex::The_Async_Arena());
        amrex::IArrayBox f1(b1, 1, amrex::The_Async_Arena());
        amrex::IArrayBox f2(b2, 1, amrex::The_Async_Arena());
        auto const& a0 = f0.array();
        auto const& a1 = f1.array();
        auto const& a2 = f2.array();

        amrex::ParallelFor(b0, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const bool s = p(i, j, k) > 0.0;
            a0(i, j, k) = ((p(i-1, j, k) > 0.0) != s || (p(i+1, j, k) > 0.0) != s ||
                           (p(i, j-1, k) > 0.0) != s || (p(i, j+1, k) > 0.0) != s ||
                           (p(i, j, k-1) > 0.0) != s || (p(i, j, k+1) > 0.0) != s) ? 1 : 0;
        });
        amrex::ParallelFor(b1, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int f = 0;
            for (int d = -m; d <= m; ++d) { f |= a0(i + d, j, k); }
            a1(i, j, k) = f;
        });
        amrex::ParallelFor(b2, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int f = 0;
            for (int d = -m; d <= m; ++d) { f |= a1(i, j + d, k); }
            a2(i, j, k) = f;
        });
        amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int band = 0;
            for (int d = -m; d <= m; ++d) { band |= a2(i, j, k + d); }
            if (!band) { return; }   // stays REINIT_OUTSIDE

            const amrex::Real x[3] = {problo[0] + (i + 0.5) * dx[0], problo[1] + (j + 0.5) * dx[1],
                                      problo[2] + (k + 0.5) * dx[2]};
            amrex::Real y[3];
            const bool ok = closestPoint<P>(p, x, y, true, problo, invdx, h, reach);
            c(i, j, k, 0) = y[0];
            c(i, j, k, 1) = y[1];
            c(i, j, k, 2) = y[2];
            c(i, j, k, 3) = ok ? REINIT_CONVERGED : REINIT_FAILED;
        });
    }

    {
        amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<amrex::Long> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        LGF_NSE_OMP_MFITER
        for (amrex::MFIter mfi(cp, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto const& c = cp.const_array(mfi);
            reduce_op.eval(mfi.tilebox(), reduce_data, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
            {
                return {c(i, j, k, 3) == REINIT_FAILED ? amrex::Long(1) : amrex::Long(0)};
            });
        }
        stats.n_first_pass_failed = amrex::get<0>(reduce_data.value(reduce_op));
    }

    // 2. neighbour passes: a cell takes the nearest of its 26 neighbours' closest points when it
    //    is nearer than its own (or it has none), polishes it with Newton and keeps the result
    //    if it converged and is nearer. This fixes points that are only locally closest beside
    //    a medial surface, and carries closest points outward, one cell per pass, to cells the
    //    first pass could not resolve: when Newton cannot polish the candidate, the cell keeps
    //    it as an estimate (a point on the interface, so an upper bound on the distance)
    const amrex::Real eps_d = 1.0e-9 * h;
    for (int pass = 0; pass < REINIT_NEIGHBOUR_PASSES; ++pass)
    {
        cp.FillBoundary(geom.periodicity());

        LGF_NSE_OMP_MFITER
        for (amrex::MFIter mfi(cp, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box tbx = mfi.tilebox();
            auto const& p = phi.const_array(mfi);
            auto const& c = cp.const_array(mfi);
            auto const& cn = cp_next.array(mfi);

            amrex::ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                for (int q = 0; q < 4; ++q) { cn(i, j, k, q) = c(i, j, k, q); }
                const amrex::Real st = c(i, j, k, 3);
                if (st == REINIT_OUTSIDE) { return; }

                const amrex::Real x[3] = {problo[0] + (i + 0.5) * dx[0], problo[1] + (j + 0.5) * dx[1],
                                          problo[2] + (k + 0.5) * dx[2]};
                amrex::Real d_cur = std::numeric_limits<amrex::Real>::max();
                if (st == REINIT_CONVERGED || st == REINIT_ESTIMATE)
                {
                    const amrex::Real yc[3] = {c(i, j, k, 0), c(i, j, k, 1), c(i, j, k, 2)};
                    d_cur = std::sqrt(distance2(x, yc));
                }

                amrex::Real best = d_cur;
                amrex::Real yb[3] = {0.0, 0.0, 0.0};
                for (int dk = -1; dk <= 1; ++dk) {
                for (int dj = -1; dj <= 1; ++dj) {
                for (int di = -1; di <= 1; ++di)
                {
                    const amrex::Real sn = c(i + di, j + dj, k + dk, 3);
                    if ((di == 0 && dj == 0 && dk == 0) || (sn != REINIT_CONVERGED && sn != REINIT_ESTIMATE)) { continue; }
                    const amrex::Real yn[3] = {c(i + di, j + dj, k + dk, 0), c(i + di, j + dj, k + dk, 1),
                                               c(i + di, j + dj, k + dk, 2)};
                    const amrex::Real dn = std::sqrt(distance2(x, yn));
                    if (dn < best - eps_d) { best = dn; yb[0] = yn[0]; yb[1] = yn[1]; yb[2] = yn[2]; }
                }}}
                if (!(best < d_cur - eps_d)) { return; }

                amrex::Real y[3] = {yb[0], yb[1], yb[2]};
                if (closestPoint<P>(p, x, y, false, problo, invdx, h, reach) && std::sqrt(distance2(x, y)) < d_cur - eps_d)
                {
                    cn(i, j, k, 0) = y[0];
                    cn(i, j, k, 1) = y[1];
                    cn(i, j, k, 2) = y[2];
                    cn(i, j, k, 3) = REINIT_CONVERGED;
                }
                else if (st != REINIT_CONVERGED)
                {
                    cn(i, j, k, 0) = yb[0];
                    cn(i, j, k, 1) = yb[1];
                    cn(i, j, k, 2) = yb[2];
                    cn(i, j, k, 3) = REINIT_ESTIMATE;
                }
            });
        }
        std::swap(cp, cp_next);
    }

    // 3. psi = sign(psi_old) min(|x - y|, psi_far), from a closest point or an estimate. A cell
    //    with neither is beyond the reach of the passes, hence far from the interface, unless
    //    its own value says it is next to it (then it keeps that value: the safe choice where
    //    the interpolant gave no answer). Cells outside the band take the far-field value
    {
        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpMax> reduce_op;
        amrex::ReduceData<amrex::Long, amrex::Long, amrex::Long, amrex::Long, amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        LGF_NSE_OMP_MFITER
        for (amrex::MFIter mfi(psi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto const& p = phi.const_array(mfi);
            auto const& c = cp.const_array(mfi);
            auto const& q = psi.array(mfi);

            reduce_op.eval(mfi.tilebox(), reduce_data, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
            {
                const amrex::Real old = p(i, j, k);
                const amrex::Real sgn = (old >= 0.0) ? 1.0 : -1.0;
                const amrex::Real st = c(i, j, k, 3);
                amrex::Real val = sgn * far;
                amrex::Long in_band = 0, estimated = 0, beyond = 0, unconverged = 0;
                if (st == REINIT_CONVERGED || st == REINIT_ESTIMATE)
                {
                    const amrex::Real x[3] = {problo[0] + (i + 0.5) * dx[0], problo[1] + (j + 0.5) * dx[1],
                                              problo[2] + (k + 0.5) * dx[2]};
                    const amrex::Real y[3] = {c(i, j, k, 0), c(i, j, k, 1), c(i, j, k, 2)};
                    val = sgn * amrex::min(std::sqrt(distance2(x, y)), far);
                    in_band = 1;
                    estimated = (st == REINIT_ESTIMATE) ? 1 : 0;
                }
                else if (st == REINIT_FAILED)
                {
                    const bool next_to = amrex::Math::abs(old) < 2.0 * h;
                    val = next_to ? old : sgn * far;
                    in_band = 1;
                    beyond = next_to ? 0 : 1;
                    unconverged = next_to ? 1 : 0;
                }
                q(i, j, k, n) = val;
                const amrex::Real change = (amrex::Math::abs(old) < h) ? amrex::Math::abs(val - old) / h : 0.0;
                return {in_band, estimated, beyond, unconverged, change};
            });
        }
        const ReduceTuple r = reduce_data.value(reduce_op);
        stats.n_band = amrex::get<0>(r);
        stats.n_estimated = amrex::get<1>(r);
        stats.n_beyond = amrex::get<2>(r);
        stats.n_unconverged = amrex::get<3>(r);
        stats.max_change_near = amrex::get<4>(r);
    }

    // 4. zero-set correction. The new psi is the distance to the zero set of the old psi's
    //    interpolant, but the zero set of the NEW psi's interpolant sits slightly off it (the
    //    interpolation error of a distance function, about (dx/r)^3 dx for degree 3 at a radius
    //    of curvature r), always to the same side: repeated reinitialization walks the
    //    interface (measured: 3e-5 dx per application, mean, at 8 cells per tube radius). The
    //    new interpolant at a cell's closest point, which lies on the old zero set, is that
    //    offset; subtracting it removes its smooth (curvature-driven) part, which is what
    //    accumulates (measured: 5e-7 dx per application). The part that oscillates with the
    //    position relative to the grid is not representable by the interpolant and stays, but
    //    does not accumulate. The correction is constant along normals, so |grad psi| = 1 is
    //    kept, and it never changes a cell's sign (at worst it puts the cell on the interface)
    if (reinit_correct)
    {
    phi.setVal(-far);
    amrex::MultiFab::Copy(phi, psi, n, 0, 1, 0);
    phi.FillBoundary(geom.periodicity());
    {
        amrex::ReduceOps<amrex::ReduceOpMax> reduce_op;
        amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        LGF_NSE_OMP_MFITER
        for (amrex::MFIter mfi(psi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto const& pn = phi.const_array(mfi);
            auto const& c = cp.const_array(mfi);
            auto const& q = psi.array(mfi);

            reduce_op.eval(mfi.tilebox(), reduce_data, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
            {
                const amrex::Real st = c(i, j, k, 3);
                const amrex::Real v = q(i, j, k, n);
                if ((st != REINIT_CONVERGED && st != REINIT_ESTIMATE) || amrex::Math::abs(v) >= far) { return {0.0}; }

                const amrex::Real y[3] = {c(i, j, k, 0), c(i, j, k, 1), c(i, j, k, 2)};
                InterpValue f;
                if (!interpolatePsi<P>(pn, y, problo, invdx, f)) { return {0.0}; }
                amrex::Real w = v - f.v;
                if (w * v < 0.0) { w = 0.0; }
                q(i, j, k, n) = amrex::Clamp(w, -far, far);
                return {amrex::Math::abs(f.v) / h};
            });
        }
        stats.max_correction = amrex::get<0>(reduce_data.value(reduce_op));
    }
    }
#else
    amrex::ignore_unused(n);
#endif
    return stats;
}

InterfaceTracker::ReinitStats InterfaceTracker::reinitialize()
{
    BL_PROFILE("<Compute> InterfaceTracker::reinitialize()");

    // valid cells of every component are rewritten; the ghost cells are refilled at the end
    ReinitStats total;
    for (int n = 0; n < n_interface; ++n)
    {
        const ReinitStats s = (reinit_degree == 5) ? reinitializeComponent<6>(n) : reinitializeComponent<4>(n);
        total.n_band += s.n_band;
        total.n_first_pass_failed += s.n_first_pass_failed;
        total.n_estimated += s.n_estimated;
        total.n_beyond += s.n_beyond;
        total.n_unconverged += s.n_unconverged;
        total.max_change_near = amrex::max(total.max_change_near, s.max_change_near);
        total.max_correction = amrex::max(total.max_correction, s.max_correction);
    }

    amrex::Long counts[5] = {total.n_band, total.n_first_pass_failed, total.n_estimated, total.n_beyond, total.n_unconverged};
    amrex::ParallelDescriptor::ReduceLongSum(counts, 5);
    amrex::Real maxima[2] = {total.max_change_near, total.max_correction};
    amrex::ParallelDescriptor::ReduceRealMax(maxima, 2);
    total.n_band = counts[0];
    total.n_first_pass_failed = counts[1];
    total.n_estimated = counts[2];
    total.n_beyond = counts[3];
    total.n_unconverged = counts[4];
    total.max_change_near = maxima[0];
    total.max_correction = maxima[1];

    fillPsiBoundary(psi);
    return total;
}

InterfaceTracker::GradientDrift InterfaceTracker::computeGradientDrift(amrex::Real tol)
{
    BL_PROFILE("<Compute> InterfaceTracker::computeGradientDrift()");

    // ||grad psi| - 1| by central differences, over cells within LS_N_GHOST cells of the
    // interface: the cells the transport stencils of the interface read. Thin features put a
    // medial surface (a kink of the distance) this close to the interface, so the trigger
    // counts cells over the tolerance rather than taking the maximum
    fillPsiBoundary(psi);

    const auto invdx = geom.InvCellSizeArray();
    const amrex::Real h = *std::min_element(geom.CellSize(), geom.CellSize() + AMREX_SPACEDIM);
    const amrex::Real near = LS_N_GHOST * h;

    amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
    amrex::ReduceData<amrex::Real, amrex::Long, amrex::Long> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    LGF_NSE_OMP_MFITER
    for (amrex::MFIter mfi(psi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const& p = psi.const_array(mfi);
        reduce_op.eval(mfi.tilebox(), n_interface, reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept -> ReduceTuple
        {
            if (amrex::Math::abs(p(i, j, k, n)) >= near) { return {0.0, 0, 0}; }
            amrex::Real g2 = 0.0;
            AMREX_D_TERM(g2 += amrex::Math::powi<2>(0.5 * (p(i+1, j, k, n) - p(i-1, j, k, n)) * invdx[0]);,
                         g2 += amrex::Math::powi<2>(0.5 * (p(i, j+1, k, n) - p(i, j-1, k, n)) * invdx[1]);,
                         g2 += amrex::Math::powi<2>(0.5 * (p(i, j, k+1, n) - p(i, j, k-1, n)) * invdx[2]);)
            const amrex::Real drift = amrex::Math::abs(std::sqrt(g2) - 1.0);
            return {drift, amrex::Long(1), drift > tol ? amrex::Long(1) : amrex::Long(0)};
        });
    }

    const ReduceTuple r = reduce_data.value(reduce_op);
    GradientDrift gd;
    gd.max_drift = amrex::get<0>(r);
    amrex::Long counts[2] = {amrex::get<1>(r), amrex::get<2>(r)};
    amrex::ParallelDescriptor::ReduceRealMax(gd.max_drift);
    amrex::ParallelDescriptor::ReduceLongSum(counts, 2);
    gd.n_near = counts[0];
    gd.n_over = counts[1];
    return gd;
}

bool InterfaceTracker::reinitializeIfDue(int step)
{
    // call after each time step (with the new step count): reinitializes on the interval, or
    // when psi has drifted from a distance function next to the interface
    if (!reinitEnabled()) { return false; }

    std::string trigger;
    if (reinit_int > 0 && step % reinit_int == 0)
    {
        trigger = "every " + std::to_string(reinit_int) + " steps";
    }
    else if (reinit_drift > 0.0)
    {
        const GradientDrift gd = computeGradientDrift(reinit_drift);
        const amrex::Real share = (gd.n_near > 0) ? amrex::Real(gd.n_over) / amrex::Real(gd.n_near) : 0.0;
        if (share > REINIT_DRIFT_FRACTION)
        {
            std::ostringstream os;
            os << 100.0 * share << "% of cells next to the interface drift > " << reinit_drift
               << ", max " << gd.max_drift;
            trigger = os.str();
        }
    }
    if (trigger.empty()) { return false; }

    reinitializeAndReport(step, trigger);
    return true;
}

void InterfaceTracker::reinitializeAndReport(int step, const std::string& trigger)
{
    amrex::Gpu::streamSynchronize();
    const amrex::Real t0 = amrex::second();

    const ReinitStats s = reinitialize();

    amrex::Gpu::streamSynchronize();
    amrex::Real wall = amrex::second() - t0;
    amrex::ParallelDescriptor::ReduceRealMax(wall);

    amrex::Print() << "Reinitialized psi at step " << step << " (" << trigger << ") | band cells " << s.n_band
                   << " | first pass unresolved " << s.n_first_pass_failed
                   << " | from a neighbour's closest point " << s.n_estimated
                   << " | beyond reach (far field) " << s.n_beyond
                   << " | unresolved next to the interface (kept) " << s.n_unconverged
                   << " | max |dpsi| next to the interface " << s.max_change_near << " dx"
                   << " | max zero-set correction " << s.max_correction << " dx"
                   << " | WallTime: " << wall << "s\n";
}
