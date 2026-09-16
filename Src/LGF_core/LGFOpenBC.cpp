#include <LGFOpenBC.H>

LGFOpenBC::LGFOpenBC(const amrex::Geometry& geom_in, int n_look_in)
    : geom(geom_in), n_lookup(n_look_in),
      cached_cc_domain(amrex::IntVect(AMREX_D_DECL(0,0,0)), amrex::IntVect(AMREX_D_DECL(-1,-1,-1))),
      cached_nd_domain(amrex::IntVect(AMREX_D_DECL(0,0,0)), amrex::IntVect(AMREX_D_DECL(-1,-1,-1)), amrex::IndexType::TheNodeType())
{
    // cached_domains are set to be empty (high < low);
    // first solve always builds and stores
}


amrex::Box LGFOpenBC::makeDomain(const amrex::MultiFab& target, const amrex::BoxArray& tag_ba) const
{
    // the domain fed to the FFT machinery must include the tagged source boxes
    // as well as all target boxes
    amrex::Box bx = amrex::grow(target.boxArray().minimalBox(), target.nGrowVect());
    bx.minBox(tag_ba.minimalBox());

    return bx;
}


amrex::FFT::OpenBCSolver<amrex::Real>& LGFOpenBC::buildSolver(const amrex::Box& domain)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(domain.ixType() == amrex::IndexType::TheCellType() ||
                                        domain.ixType() == amrex::IndexType::TheNodeType(),
                                        "LGFOpenBC: cache has slots for cell-centered and fully-nodal only");

    // check which solver to update
    const bool is_nodal = domain.ixType().nodeCentered();
    auto& solver_ptr = is_nodal ? nd_solver : cc_solver;
    auto& cached_domain_ptr = is_nodal ? cached_nd_domain : cached_cc_domain;

    // If the solver object already has a pointer and the cached domain is the
    // same as present, don't build. Note that box comparison does not require
    // two-way .contains(), simple == suffices.
    if (solver_ptr && domain == cached_domain_ptr) { return *solver_ptr; }

    BL_PROFILE("<Setup> LGFOpenBC::buildSolver");

    amrex::FFT::Info info;
    // default padding; rounding to nextFastLen()
    info.setOpenBCPadding(true);
    solver_ptr.reset();
    solver_ptr = std::make_unique<amrex::FFT::OpenBCSolver<amrex::Real>>(domain, info);
    cached_domain_ptr = domain;

    // AI COMMENT <yet to be fully interpreted>:
    // setGreensFunction calls f(ii+lo.x, jj+lo.y, kk+lo.z) where ii,jj,kk are
    // ZERO-BASED offsets into the (padded, doubled) domain. So the lattice
    // offset is (arg - domain.smallEnd()). AMReX handles the mirroring and the
    // zeroed middle planes internally -- we never touch the wrap-around layout.
    
    // plain data for passing into GPU lambda
    const auto  glo = domain.smallEnd().dim3();
    const auto  dx  = geom.CellSizeArray();
    const amrex::Real  h2  = dx[0] * dx[1];
    const int   nlk = n_lookup;

    solver_ptr->setGreensFunction(
        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> amrex::Real
        {
            AMREX_D_TERM(const amrex::Real xt = amrex::Real(i - glo.x) * dx[0];,
                         const amrex::Real yt = amrex::Real(j - glo.y) * dx[1];,
                         const amrex::Real zt = amrex::Real(k - glo.z) * dx[2];)

            return h2 * computeLGF(nlk, AMREX_D_DECL(xt, yt, zt),
                                        AMREX_D_DECL(amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0)),
                                        AMREX_D_DECL(dx[0], dx[1], dx[2]));
        });

    amrex::Print() << "LGFOpenBC: built solver on " << domain
                   << ", padded one-sided length " << solver_ptr->PaddedLength() << "\n";
    
    return *solver_ptr;
}


void LGFOpenBC::doSolve(const amrex::MultiFab& source, amrex::MultiFab& target,
                        const amrex::BoxArray& tag_ba)
{
    BL_PROFILE("<Compute> LGFOpenBC::solve()");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(source.ixType() == target.ixType(),
        "LGFOpenBC: source and target must share an index type");

    // backward_doit writes only where phi overlaps the solver domain, so
    // anything outside would otherwise retain stale values.
    target.setVal(0.0);

    if (tag_ba.empty()) { return; }   // no sources => u == 0 everywhere

    // restrict the source field to the support, specified by tag_ba
    if (!rhs.ok() || rhs_tag_ba != tag_ba)
    {
        // create appropriate nodal/cell-centered rhs_ba and corresponding dm
        amrex::BoxArray rhs_ba = amrex::convert(tag_ba, source.ixType());
        amrex::DistributionMapping rhs_dm(rhs_ba);

        // update rhs multifab
        rhs.clear();
        rhs.define(rhs_ba, rhs_dm, 1, 0);
        rhs_tag_ba = tag_ba;
    }
    rhs.ParallelCopy(source, 0, 0, 1);

    buildSolver(makeDomain(target, tag_ba)).solve(target, rhs);
}

void LGFOpenBC::solvePoisson(const amrex::MultiFab& source, amrex::MultiFab& target,
                              const amrex::BoxArray& tag_ba)
{
    doSolve(source, target, tag_ba);
}


void LGFOpenBC::solveNodalPoisson(const amrex::MultiFab& source, amrex::MultiFab& target,
                                   const amrex::BoxArray& tag_ba)
{
    // OpenBCSolver preserves domain.ixType() through make_grown_domain, and the
    // lattice offsets are node-to-node exactly as they are cell-to-cell, so the
    // same path serves both. No OwnerMask is needed: ParallelCopy resolves
    // shared seam nodes, unlike the manual packing in consolidateMultiFab.
    doSolve(source, target, tag_ba);
}


void LGFOpenBC::regridOnto(const amrex::Geometry& new_geom, const amrex::BoxArray& new_ba, const amrex::DistributionMapping& new_dm)
{
    amrex::ignore_unused(new_ba, new_dm);
    geom = new_geom;

    // clear cache
    cc_solver.reset();
    nd_solver.reset();
    cached_cc_domain = amrex::Box(amrex::IntVect(AMREX_D_DECL(0,0,0)), amrex::IntVect(AMREX_D_DECL(-1,-1,-1)));
    cached_nd_domain = amrex::Box(amrex::IntVect(AMREX_D_DECL(0,0,0)), amrex::IntVect(AMREX_D_DECL(-1,-1,-1)), amrex::IndexType::TheNodeType());
}
