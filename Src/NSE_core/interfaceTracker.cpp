#include <InterfaceTracker.H>

InterfaceTracker::InterfaceTracker(const amrex::Geometry& geom_in, const amrex::BoxArray& ba_in,
                                   const amrex::DistributionMapping& dm_in, const SolverConfig& config)
    : InterfaceTracker(geom_in, ba_in, dm_in, advectionSchemeFromName(config.interface_scheme),
                       config.interface_psi_far)
{
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
    BL_PROFILE("<Compute> InterfaceTracker::fillPsiBoundary()");

    // psi is not compact, so the flow solver's "zero outside the snug domain"
    // convention cannot be used (zero is the interface itself)

    // 1. every ghost cell gets a linear extrapolation from its own box; those
    //    outside the snug domain keep it. Reads valid cells only, writes ghosts
    //    only, so there is no race between the two
    for (amrex::MFIter mfi(psi_in, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box gbx = mfi.growntilebox();
        const amrex::Box vbx = mfi.validbox();
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);

        auto const& p = psi_in.array(mfi);
        auto const& pc = psi_in.const_array(mfi);

        amrex::ParallelFor(gbx, psi_in.nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            if (vbx.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))) { return; }

            // nearest valid cell
            const int ic = amrex::Clamp(i, lo.x, hi.x);
            const int jc = amrex::Clamp(j, lo.y, hi.y);
            const int kc = amrex::Clamp(k, lo.z, hi.z);

            p(i, j, k, n) = pc(ic, jc, kc, n)
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
    // psi_in may be psi itself or a stage copy; its ghost cells are refilled here

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

    fillPsiBoundary(psi_in);

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

    for (amrex::MFIter mfi(psi); mfi.isValid(); ++mfi)
    {
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> u;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { u[d] = vel[d].const_array(mfi); }

        reduce_op.eval(mfi.validbox(), reduce_data,
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
