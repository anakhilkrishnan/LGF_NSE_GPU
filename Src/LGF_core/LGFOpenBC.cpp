#include <LGFOpenBC.H>

LGFOpenBC::LGFOpenBC(const amrex::Geometry& geom_in, const amrex::BoxArray xsoln_ba_in, int n_look_in, int n_ghost_max_in)
    : geom(geom_in), n_lookup(n_look_in), n_ghost_max(n_ghost_max_in),
      cached_domain(amrex::IntVect(AMREX_D_DECL(0,0,0)), amrex::IntVect(AMREX_D_DECL(-1,-1,-1)))
{
    // cached_domains are set to be empty (high < low);
    // first solve always builds and stores
    xsoln_ba_ref = xsoln_ba_in;
}

amrex::BoxArray LGFOpenBC::retypeBoxArray(const amrex::BoxArray& ba)
{
    // create exact same boxArray, but cell-centered
    // amrex::convert() doesn't work here, as it drops an index

    amrex::BoxList bl;
    bl.reserve(ba.size());
    for (int i = 0; i < ba.size(); ++i)
    {
        const amrex::Box& b = ba[i];
        bl.push_back(amrex::Box(b.smallEnd(), b.bigEnd(), amrex::IndexType::TheCellType()));
    }

    return amrex::BoxArray(std::move(bl));
}

amrex::MultiFab& LGFOpenBC::retypedMultiFab(retypeSlot& slot, const amrex::MultiFab& in)
{
    const amrex::BoxArray& src_ba = in.boxArray();
 
    const bool stale = (!slot.cc_copy.ok()
              || slot.original_ba != src_ba
              || slot.cc_copy.DistributionMap() != in.DistributionMap()
              || slot.cc_copy.nGrow()           != in.nGrow());
 
    if (stale)
    {
        slot.cc_copy.clear();                       // FabArray::define does not clear
        slot.original_ba = src_ba;
        slot.cc_copy.define(retypeBoxArray(src_ba), in.DistributionMap(), 1, in.nGrow());
    }

    return slot.cc_copy;
}

void LGFOpenBC::retypeCopy(amrex::MultiFab& dst, const amrex::MultiFab& src)
{
    BL_PROFILE("<Compute> LGFOpenBC::retypeCopy()");
 
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dst.size() == src.size(),
        "retypeCopy: box counts differ -- the cc_copy was not built from this BoxArray");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dst.DistributionMap() == src.DistributionMap(),
        "retypeCopy: DistributionMaps differ -- MFIter would pair the wrong fabs");
 
    const amrex::IntVect ng = amrex::min(dst.nGrowVect(), src.nGrowVect());
 
    for (amrex::MFIter mfi(dst, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        // copy ghost data as well
        const amrex::Box& bx = mfi.growntilebox(ng);
 
        auto const& d = dst.array(mfi);
        auto const& s = src.const_array(mfi);
 
        // by construction, the cc BoxArray was created such that i,j,k is the
        // same in each MultiFab
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            d(i,j,k) = s(i,j,k);
        });
    }
}

amrex::Box LGFOpenBC::makeDomain(const amrex::BoxArray& xsoln_ba, const amrex::BoxArray& tag_ba) const
{
    // the domain fed to the FFT machinery must include the tagged source boxes
    // as well as all target boxes; the target's boundary ghost cells need to be
    // filled, but instead of taking n_ghost from the target itself, it will be
    // max(n_ghost) across all multifabs in the solver; it depends on n_IF
    amrex::Box bx = amrex::grow(xsoln_ba.minimalBox(), n_ghost_max);
    bx.minBox(tag_ba.minimalBox());

    return bx;
}


void LGFOpenBC::buildSolver(const amrex::Box& domain)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(domain.ixType() == amrex::IndexType::TheCellType(),
                                        "LGFOpenBC: why is a nodal Box here?");

    // If the solver object already has a pointer and the cached domain is the
    // same as present, don't build. Note that box comparison does not require
    // two-way .contains(), simple == suffices.
    if (solver && domain == cached_domain) { return;}

    BL_PROFILE("<Setup> LGFOpenBC::buildSolver");

    amrex::FFT::Info info;
    // default padding; rounding to nextFastLen()
    info.setOpenBCPadding(true);
    solver.reset();
    solver = std::make_unique<amrex::FFT::OpenBCSolver<amrex::Real>>(domain, info);
    cached_domain = domain;

    // AI COMMENT <yet to be fully interpreted>:
    // setGreensFunction calls f(ii+lo.x, jj+lo.y, kk+lo.z) where ii,jj,kk are
    // ZERO-BASED offsets into the (padded, doubled) domain. So the lattice
    // offset is (arg - domain.smallEnd()). AMReX handles the mirroring and the
    // zeroed middle planes internally -- we never touch the wrap-around layout.
    
    // plain data for passing into GPU lambda
    const auto  glo = domain.smallEnd().dim3();
    const auto  dx  = geom.CellSizeArray();
    const amrex::Real  h2  = dx[0] * dx[0]; // IMPORTANT: Assumes dx[0] = dx[1] = dx[2]
    const int   nlk = n_lookup;

    solver->setGreensFunction(
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
                   << ", padded one-sided length " << solver->PaddedLength() << "\n";
}


void LGFOpenBC::doSolve(const amrex::MultiFab& source, amrex::MultiFab& target,
                        const amrex::BoxArray& tag_ba)
{
    BL_PROFILE("<Compute> LGFOpenBC::solve()");

    // backward_doit writes only where phi overlaps the solver domain, so
    // anything outside would otherwise retain stale values.
    target.setVal(0.0);

    if (tag_ba.empty()) { return; }   // no sources => u == 0 everywhere

    // check if incoming source is cell-centered
    AMREX_ALWAYS_ASSERT(source.ixType() == amrex::IndexType::TheCellType());

    // restrict the source field to the support, specified by tag_ba
    amrex::BoxArray rhs_ba = tag_ba;
    if (!rhs.ok() || rhs.boxArray() != rhs_ba)
    {
        rhs.clear();
        rhs.define(rhs_ba, amrex::DistributionMapping(rhs_ba), 1, 0);
    }
    rhs.setVal(0.0);

    // assert before copying
    rhs.ParallelCopy(source, 0, 0, 1);

    // WARNING: This is not there in the original code, purely for testing
    const amrex::Box dom = makeDomain(xsoln_ba_ref, tag_ba);
    buildSolver(dom);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(solver && solver->Domain() == dom,
        "LGFOpenBC: cached solver was built for a different domain");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dom.contains(
        amrex::grow(retypeBoxArray(target.boxArray()).minimalBox(), target.nGrowVect())),
        "LGFOpenBC: n_ghost_max too small for this target's extent + ghosts");

    solver->solve(target, rhs);
}

void LGFOpenBC::solvePoisson(const amrex::MultiFab& source, amrex::MultiFab& target,
                              const amrex::BoxArray& tag_ba)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(source.ixType() == target.ixType(),
        "LGFOpenBC: source and target must share an index type");

    if (source.ixType() == amrex::IndexType::TheCellType())
    {
        doSolve(source, target, tag_ba);
        return;
    }

    amrex::MultiFab& intermediate_src = retypedMultiFab(retype_source, source);
    amrex::MultiFab& intermediate_tar = retypedMultiFab(retype_target, target);

    retypeCopy(intermediate_src, source);
    doSolve(intermediate_src, intermediate_tar, tag_ba);
    retypeCopy(target, intermediate_tar);
}

void LGFOpenBC::regridOnto(const amrex::Geometry& new_geom, const amrex::BoxArray& new_ba, const amrex::DistributionMapping& new_dm)
{
    amrex::ignore_unused(new_dm);
    geom = new_geom;
    xsoln_ba_ref = new_ba;

    // clear cache
    rhs.clear();
    solver.reset();
    cached_domain = amrex::Box(amrex::IntVect(AMREX_D_DECL(0,0,0)), amrex::IntVect(AMREX_D_DECL(-1,-1,-1)));
}
