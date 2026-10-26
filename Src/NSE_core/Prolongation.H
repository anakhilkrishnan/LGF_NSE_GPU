#ifndef PROLONGATION_H_
#define PROLONGATION_H_

#include <AMReX_MultiFab.H>
#include <AMReX_Gpu.H>
#include <AMReX_Print.H>
#include <AMReX_BLProfiler.H>

#include <SpatialDiscretization.H>   // TheVortEdgeVector (tests only)

#include <string>
#include <utility>

// Tensor-product cubic Lagrange prolongation by an integer ratio r, for any index type
// (cell, face, edge, node; 2D or 3D). Each direction is handled on its own:
//   nodal direction:         fine point I sits at coarse index coordinate  x = I / r
//   cell-centred direction:  fine point I sits at coarse index coordinate  x = (2I + 1 - r) / (2r)
// With x = ic + t, ic = floor(x), t in [0, 1), the stencil is coarse points ic-1 .. ic+2.
// t is formed from integers, so a fine point on a nodal plane shared by two boxes gets
// bitwise-identical weights in both, and r = 1 is an exact copy.

namespace prolong_detail
{
    // floor(a / b) for b > 0, correct for negative a (ghost indices)
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int floorDiv (int a, int b) noexcept
    {
        return (a >= 0) ? a / b : -((-a + b - 1) / b);
    }

    // cubic Lagrange weights on nodes {-1, 0, 1, 2}, evaluated at t; coarse base index b = ic - 1
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void cubicWeights1D (bool nodal, int r, int I, int& b, amrex::GpuArray<amrex::Real, 4>& w) noexcept
    {
        const int num = nodal ? I : 2 * I + 1 - r;
        const int den = nodal ? r : 2 * r;
        const int ic  = floorDiv(num, den);
        const amrex::Real t = amrex::Real(num - ic * den) / amrex::Real(den);

        b    = ic - 1;
        w[0] = -t * (t - 1) * (t - 2) / 6;
        w[1] =  (t + 1) * (t - 1) * (t - 2) / 2;
        w[2] = -(t + 1) * t * (t - 2) / 2;
        w[3] =  (t + 1) * t * (t - 1) / 6;
    }
}

// fine must be built on refine(crse.boxArray(), r) with crse's DistributionMapping, so box K of
// fine lies over box K of crse and no communication is needed. crse needs >= 2 filled ghost
// cells (the stencil reaches exactly 2 beyond the valid region). Only fine's valid region is written.
inline void prolongCubic (const amrex::MultiFab& crse, amrex::MultiFab& fine, int r)
{
    BL_PROFILE("<Compute> prolongCubic()");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(r >= 1, "prolongCubic: ratio must be >= 1");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(fine.ixType() == crse.ixType(), "prolongCubic: index types differ");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(fine.nComp() == crse.nComp(), "prolongCubic: component counts differ");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(crse.nGrowVect().allGE(2),
        "prolongCubic: coarse data needs >= 2 filled ghost cells");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(fine.DistributionMap() == crse.DistributionMap(),
        "prolongCubic: fine must use the coarse DistributionMapping");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(fine.boxArray().CellEqual(
        amrex::refine(amrex::convert(crse.boxArray(), amrex::IndexType::TheCellType()), r)),
        "prolongCubic: fine BoxArray is not the coarse one refined by r");

    const amrex::IndexType typ = crse.ixType();
    const amrex::GpuArray<bool, 3> nodal = {AMREX_D_DECL(typ.nodeCentered(0),
                                                         typ.nodeCentered(1),
                                                         typ.nodeCentered(2))};
    const int ncomp = crse.nComp();

    for (amrex::MFIter mfi(fine, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box bx = mfi.tilebox();
        auto const& f = fine.array(mfi);
        auto const& c = crse.const_array(mfi);

        amrex::ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            constexpr int nk = (AMREX_SPACEDIM == 3) ? 4 : 1;   // 2D: the single plane k = 0

            int bi, bj, bk = 0;
            amrex::GpuArray<amrex::Real, 4> wi, wj;
            amrex::GpuArray<amrex::Real, 4> wk = {1.0, 0.0, 0.0, 0.0};
            prolong_detail::cubicWeights1D(nodal[0], r, i, bi, wi);
            prolong_detail::cubicWeights1D(nodal[1], r, j, bj, wj);
#if AMREX_SPACEDIM == 3
            prolong_detail::cubicWeights1D(nodal[2], r, k, bk, wk);
#endif
            amrex::Real s = 0.0;
            for (int c3 = 0; c3 < nk; ++c3) {
                for (int c2 = 0; c2 < 4; ++c2) {
                    for (int c1 = 0; c1 < 4; ++c1) {
                        s += wi[c1] * wj[c2] * wk[c3] * c(bi + c1, bj + c2, bk + c3, n);
                    }
                }
            }
            f(i, j, k, n) = s;
        });
    }
}

// ---------------------------------------------------------------------------------------------
// TESTS (plan items 15-16): the tensor-product cubic must reproduce any polynomial of degree <= 3
// in each direction to round-off, for every index type and ratio, and r = 1 must be an exact copy.
// Positions are physical (coarse spacing 1), independent of the kernel's index mapping, so an
// offset error in any direction fails. Run once in a DEBUG build too: Array4 bounds checking then
// also verifies that the stencil never reaches past the 2 coarse ghost cells.

namespace prolong_detail
{
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real testPoly (amrex::Real x, amrex::Real y, amrex::Real z) noexcept
    {
        return (1.0 + x - 0.5*x*x + 0.1*x*x*x) * (0.3 - y + 0.2*y*y*y)
             + (2.0 - 0.7*y*y) * (1.0 + 0.5*z - 0.05*z*z*z)
             + 0.03*x*x*x*z*z;
    }
}

// returns max |prolonged - exact| / max |exact| over the fine valid region
inline amrex::Real testProlongCubic (int r, const amrex::IntVect& ixtype)
{
    const amrex::IndexType typ(ixtype);
    amrex::BoxArray ba_c(amrex::Box(amrex::IntVect(0), amrex::IntVect(15)));
    ba_c.maxSize(8);                                            // several boxes: exercises box seams
    const amrex::DistributionMapping dm(ba_c);

    amrex::MultiFab crse(amrex::convert(ba_c, typ), dm, 1, 2);
    amrex::MultiFab fine(amrex::convert(amrex::refine(ba_c, r), typ), dm, 1, 0);
    amrex::MultiFab exact(fine.boxArray(), dm, 1, 0);

    const amrex::GpuArray<amrex::Real, 3> off = {AMREX_D_DECL(typ.cellCentered(0) ? 0.5 : 0.0,
                                                              typ.cellCentered(1) ? 0.5 : 0.0,
                                                              typ.cellCentered(2) ? 0.5 : 0.0)};
    const amrex::Real rr = amrex::Real(r);

    for (amrex::MFIter mfi(crse); mfi.isValid(); ++mfi)         // ghosts included: no boundary fill needed
    {
        auto const& a = crse.array(mfi);
        amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            a(i, j, k) = prolong_detail::testPoly(i + off[0], j + off[1], k + off[2]);
        });
    }
    for (amrex::MFIter mfi(exact); mfi.isValid(); ++mfi)
    {
        auto const& a = exact.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            a(i, j, k) = prolong_detail::testPoly((i + off[0]) / rr, (j + off[1]) / rr, (k + off[2]) / rr);
        });
    }

    prolongCubic(crse, fine, r);

    const amrex::Real scale = exact.norm0();
    amrex::MultiFab::Subtract(fine, exact, 0, 0, 1, 0);
    return fine.norm0() / scale;
}

inline void runProlongCubicTests (amrex::Real tol = 1.0e-12)
{
    amrex::Vector<std::pair<std::string, amrex::IntVect>> types = {{"cell", amrex::IntVect(0)},
                                                                   {"node", amrex::IntVect(1)}};
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        types.push_back({"face" + std::to_string(d), amrex::IntVect::TheDimensionVector(d)});
#if AMREX_SPACEDIM == 3
        types.push_back({"edge" + std::to_string(d), TheVortEdgeVector(d)});
#endif
    }

    bool all_ok = true;
    for (int r = 1; r <= 4; ++r)
    {
        for (auto const& [name, t] : types)
        {
            const amrex::Real err = testProlongCubic(r, t);
            const bool ok = (r == 1) ? (err == 0.0) : (err < tol);   // r = 1 must be an exact copy
            all_ok = all_ok && ok;
            amrex::Print() << "prolongCubic r=" << r << " " << name << ": rel err " << err
                           << (ok ? "  ok" : "  FAIL") << "\n";
        }
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(all_ok, "prolongCubic: polynomial exactness test failed");
}

#endif
