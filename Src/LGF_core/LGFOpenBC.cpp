#include <LGFOpenBC.H>

// one entry per (fab, copy region); all regions of a retypeCopy go out in a
// single fused launch. Own struct rather than amrex::Array4CopyTag, whose
// members have changed across AMReX releases.
namespace lgf_detail
{
    struct RetypeCopyTag
    {
        amrex::Array4<amrex::Real>       dfab;
        amrex::Array4<amrex::Real const> sfab;
        amrex::Box                       dbox;    // region, in dst indices
        amrex::Dim3                      offset;  // src index - dst index

        [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Box const& box () const noexcept { return dbox; }
    };
}

LGFOpenBC::LGFOpenBC(const amrex::Geometry& geom_in, const amrex::BoxArray& xsoln_ba_in,
                     int n_look_in, int n_ghost_max_in, int len_quantum_in)
    : geom(geom_in), xsoln_ba_ref(xsoln_ba_in), n_lookup(n_look_in),
      n_ghost_max(n_ghost_max_in), len_quantum(len_quantum_in),
      cached_len(amrex::IntVect(0)), cached_dx{}
{
    AMREX_ALWAYS_ASSERT(n_ghost_max >= 0 && len_quantum >= 1);

    // no solver yet; the first solve builds it
    updateGeometry();
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

amrex::BoxArray LGFOpenBC::canonicalBoxArray(const amrex::BoxArray& ba)
{
    // convert() only ever increments hi, so going to nodal first loses nothing;
    // every index type built from the same cell ba lands on the same boxes
    return retypeBoxArray(amrex::convert(ba, amrex::IndexType::TheNodeType()));
}

void LGFOpenBC::updateGeometry()
{
    // canonical boxes in physical indices; their bounding box is the xsoln
    // bounding box with hi+1, and n_ghost_max on top of that covers any target
    amrex::BoxArray canon_phys = canonicalBoxArray(xsoln_ba_ref);
    domain    = amrex::grow(canon_phys.minimalBox(), n_ghost_max);
    domain_lo = domain.smallEnd();

    // scratch lives at the origin; see the translation-invariance note in the header
    canon_ba = std::move(canon_phys);
    canon_ba.shift(-domain_lo);
}

amrex::MultiFab& LGFOpenBC::scratchMultiFab(amrex::MultiFab& slot, const amrex::DistributionMapping& dm,
                                            const amrex::IntVect& ng)
{
    // both comparisons are pointer checks when the refs are shared
    const bool same_layout = slot.ok()
                          && slot.boxArray()        == canon_ba
                          && slot.DistributionMap() == dm;

    if (same_layout && slot.nGrowVect().allGE(ng)) { return slot; }

    // ghosts only ever grow within a layout, so alternating targets don't thrash
    const amrex::IntVect ng_new = same_layout ? amrex::elemwiseMax(slot.nGrowVect(), ng) : ng;

    slot.clear();                                   // FabArray::define does not clear
    slot.define(canon_ba, dm, 1, ng_new);

    return slot;
}

void LGFOpenBC::retypeCopy(amrex::MultiFab& dst, const amrex::MultiFab& src,
                           const amrex::IntVect& offset, const amrex::BoxArray* restrict_ba)
{
    BL_PROFILE("<Compute> LGFOpenBC::retypeCopy()");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dst.size() == src.size(),
        "retypeCopy: box counts differ -- dst was not built from src's BoxArray");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dst.DistributionMap() == src.DistributionMap(),
        "retypeCopy: DistributionMaps differ -- MFIter would pair the wrong fabs");

    // restriction set, in src's own index type and indices; built once per call
    amrex::BoxArray keep;
    if (restrict_ba != nullptr)
    {
        AMREX_ALWAYS_ASSERT(restrict_ba->ixType().cellCentered());
        keep = retypeBoxArray(amrex::convert(*restrict_ba, src.ixType()));

        // the complement must be zero, not stale
        dst.setVal(0.0);
        if (keep.empty()) { return; }
    }

    amrex::Vector<lgf_detail::RetypeCopyTag> tags;
    std::vector<std::pair<int, amrex::Box>> isects;

    for (amrex::MFIter mfi(dst); mfi.isValid(); ++mfi)
    {
        const int K = mfi.index();

        // src fab (with ghosts) moved into dst indices, intersected with the
        // dst fab (with ghosts); this is min(nGrow) generalised to fabs whose
        // extents and positions differ
        const amrex::Box& sb = src.fabbox(K);
        const amrex::Box  s_in_d = amrex::shift(amrex::Box(sb.smallEnd(), sb.bigEnd()), -offset);
        const amrex::Box  region = dst.fabbox(K) & s_in_d;
        if (region.isEmpty()) { continue; }

        auto const& d = dst.array(mfi);
        auto const& s = src.const_array(mfi);

        if (restrict_ba == nullptr)
        {
            tags.push_back({d, s, region, offset.dim3()});
            continue;
        }

        // pieces of the region inside the restriction, back into dst indices;
        // pieces may overlap (nodal keep boxes share faces) -- harmless for a copy
        keep.intersections(amrex::shift(region, offset), isects);
        for (auto const& is : isects)
        {
            tags.push_back({d, s, amrex::shift(is.second, -offset), offset.dim3()});
        }
    }

    if (tags.empty()) { return; }

    amrex::ParallelFor(tags,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, lgf_detail::RetypeCopyTag const& t) noexcept
        {
            t.dfab(i, j, k) = t.sfab(i + t.offset.x, j + t.offset.y, k + t.offset.z);
        });
}

void LGFOpenBC::buildSolver()
{
    // the kernel depends only on the solver lengths and dx
    amrex::IntVect need = domain.length();
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        need[d] = ((need[d] + len_quantum - 1) / len_quantum) * len_quantum;
    }

    const auto dx = geom.CellSizeArray();
    bool same_dx = true;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { same_dx = same_dx && (dx[d] == cached_dx[d]); }

    // a larger free-space domain is valid for a smaller problem: only grow
    if (solver && same_dx && cached_len.allGE(need)) { return; }

    BL_PROFILE("<Setup> LGFOpenBC::buildSolver");

    cached_len = (solver && same_dx) ? amrex::elemwiseMax(cached_len, need) : need;
    cached_dx  = dx;

    amrex::FFT::Info info;
    // default padding; rounding to nextFastLen()
    info.setOpenBCPadding(true);
    solver.reset();
    solver = std::make_unique<amrex::FFT::OpenBCSolver<amrex::Real>>(
                 amrex::Box(amrex::IntVect(0), cached_len - amrex::IntVect(1)), info);

    // setGreensFunction calls f(ii+lo.x, jj+lo.y, kk+lo.z) with ii,jj,kk the
    // zero-based lattice offsets; the solver box sits at the origin, so the
    // arguments ARE the offsets. AMReX handles the mirroring and the zeroed
    // middle planes internally.

    // plain data for passing into GPU lambda
    const amrex::Real  h2  = dx[0] * dx[0]; // IMPORTANT: Assumes dx[0] = dx[1] = dx[2]
    const int   nlk = n_lookup;

    solver->setGreensFunction(
        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> amrex::Real
        {
            amrex::ignore_unused(i, j, k);
            AMREX_D_TERM(const amrex::Real xt = amrex::Real(i) * dx[0];,
                         const amrex::Real yt = amrex::Real(j) * dx[1];,
                         const amrex::Real zt = amrex::Real(k) * dx[2];)

            return h2 * computeLGF(nlk, AMREX_D_DECL(xt, yt, zt),
                                        AMREX_D_DECL(amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0)),
                                        AMREX_D_DECL(dx[0], dx[1], dx[2]));
        });

    amrex::Print() << "LGFOpenBC: built solver for lengths " << cached_len
                   << " (needed " << domain.length() << ")"
                   << ", padded one-sided length " << solver->PaddedLength() << "\n";
}

void LGFOpenBC::solvePoisson(const amrex::MultiFab& source, amrex::MultiFab& target,
                              const amrex::BoxArray& tag_ba)
{
    BL_PROFILE("<Compute> LGFOpenBC::solvePoisson()");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(source.ixType() == target.ixType(),
        "LGFOpenBC: source and target must share an index type");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(source.boxArray().CellEqual(xsoln_ba_ref)
                                  && target.boxArray().CellEqual(xsoln_ba_ref),
        "LGFOpenBC: source/target not built from xsoln_ba_ref -- was regridOnto() called?");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(source.DistributionMap() == target.DistributionMap(),
        "LGFOpenBC: source and target DistributionMaps differ");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(target.nGrowVect().allLE(n_ghost_max),
        "LGFOpenBC: target has more ghosts than n_ghost_max");
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!source.ixType().cellCentered(d) || source.nGrow(d) >= 1,
            "LGFOpenBC: source needs >= 1 (filled) ghost in each cell-centred direction");
    }

    if (tag_ba.empty()) { target.setVal(0.0); return; }   // no sources => u == 0 everywhere

    // keeps the restriction off the fabricated hi+1 planes, which have no source data
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(xsoln_ba_ref.contains(tag_ba, true),
        "LGFOpenBC: tag_ba reaches outside xsoln_ba");

    buildSolver();

    amrex::MultiFab& src = scratchMultiFab(src_cc, source.DistributionMap(), amrex::IntVect(0));
    amrex::MultiFab& tar = scratchMultiFab(tar_cc, target.DistributionMap(), target.nGrowVect());

    // physical -> canonical, restricted to the tagged support
    retypeCopy(src, source, domain_lo, &tag_ba);

    solver->solve(tar, src);

    // canonical -> physical; tar covers every target cell, ghosts included
    retypeCopy(target, tar, -domain_lo);
}

void LGFOpenBC::regridOnto(const amrex::Geometry& new_geom, const amrex::BoxArray& new_ba, const amrex::DistributionMapping& new_dm)
{
    amrex::ignore_unused(new_dm);
    geom = new_geom;
    xsoln_ba_ref = new_ba;

    updateGeometry();

    // scratch is redefined lazily on the next solve; the solver is kept, and
    // buildSolver() decides from (length, dx) whether the kernel is still valid
    src_cc.clear();
    tar_cc.clear();
}
