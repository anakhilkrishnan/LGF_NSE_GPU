#ifndef LGFOPENBC_H_
#define LGFOPENBC_H_

#include <directSumKernel.H>

#include <AMReX_FFT.H>

#include <memory>

/**
 * FFT-based free-space LGF Poisson solver.
 *
 *   u(n) = h^2 * sum_m G(n-m) f(m)
 *
 * evaluated as a single zero-padded convolution using amrex::FFT::OpenBCSolver.
 * This is the "FFT Conv." case of Liska & Colonius (2014) -- the special case
 * where the entire active grid is one block and the FLGF method reduces to a
 * single FFT-based convolution. Their Table 2 puts it at 3.54e5 pts/s against
 * 1.19e5 for the full multilevel FMM at 1e8 points, i.e. FASTER than the tree
 * on a box, so this is not a stepping stone -- it is the fastest configuration
 * for boxed domains, and O(N log N) rather than O(N^2).
 *
 * ONE CANONICAL CELL-CENTRED LATTICE
 *
 *   Every MultiFab reaching solvePoisson() is built by amrex::convert() from
 *   the same cell-centred xsoln_ba. So for ANY index type,
 *       retypeBoxArray(convert(ba, nodal))
 *   is the same BoxArray: each cell box [lo, hi] becomes [lo, hi+1], cell
 *   typed. Sources and targets of all types are copied into scratch
 *   MultiFabs on this one canonical BoxArray, so there is exactly one source
 *   scratch and one target scratch, reallocated only on regrid.
 *
 *   Neighbouring canonical boxes share their hi+1 plane. The backend's
 *   inmf.ParallelCopy(rho) resolves that overlap in no particular order, so
 *   duplicated indices must agree:
 *     - staggered directions: both copies are valid data, identical;
 *     - cell-centred directions: the lower box's copy comes from the
 *       source's GHOST cell, so the source needs >= 1 ghost there, FILLED
 *       (FillBoundary) before the solve. This is a caller contract; nothing
 *       here can detect a stale ghost.
 *   The outermost hi+1 plane in a cell-centred direction has no source data
 *   anywhere; it is zeroed, and the tag restriction never reaches it.
 *
 * TRANSLATION-INVARIANT KERNEL CACHE
 *
 *   The solver is built on a box at the origin, [0, L-1], and data is shifted
 *   by -domain_lo on the way in and +domain_lo on the way out (folded into
 *   retypeCopy, so no FabArray::shift). The spectral kernel therefore depends
 *   only on (L, dx). L is rounded up to a multiple of len_quantum and kept as
 *   a high-water mark: a regrid only rebuilds the kernel when the needed
 *   length exceeds what is cached, since a larger free-space domain is
 *   always valid for a smaller problem.
 *
 *   domain = grow(canonical minimalBox, n_ghost_max), i.e. the xsoln bounding
 *   box grown by n_ghost_max low and n_ghost_max+1 high. Any target with
 *   nGrow <= n_ghost_max fits, so the kernel never depends on a target's
 *   ghost count.
 *
 * WHAT THE BACKEND DOES
 *
 *   OpenBCSolver::solve() does
 *       inmf.setVal(0); inmf.ParallelCopy(rho, 0, 0, 1);   ... forward
 *       ... multiply by cached spectral G ...
 *       m_r2c.backward_doit(phi, phi.nGrowVect());
 *   phi cells below the domain are left untouched and cells above it read
 *   the wrap-around half, so every target cell must lie inside the domain.
 */
class LGFOpenBC
{
private:

    amrex::Geometry geom;
    amrex::BoxArray xsoln_ba_ref; // local reference copy of xsoln_ba from DomainManager
    int n_lookup;
    int n_ghost_max;
    int len_quantum;              // solver lengths are rounded up to a multiple of this

    // geometry derived from xsoln_ba_ref alone; recomputed only in regridOnto()
    amrex::Box      domain;       // index-space box the solve must cover (physical indices)
    amrex::IntVect  domain_lo;    // domain.smallEnd(): physical index = canonical index + domain_lo
    amrex::BoxArray canon_ba;     // canonical scratch BoxArray, in canonical (shifted) indices

    // scratch MultiFabs on canon_ba; source holds valid data only, target
    // carries as many ghosts as the largest target seen since the last regrid
    amrex::MultiFab src_cc;
    amrex::MultiFab tar_cc;

    // creating object for AMReX's FFT class OpenBCSolver, plus its cache key
    std::unique_ptr<amrex::FFT::OpenBCSolver<amrex::Real>> solver;
    amrex::IntVect cached_len;                              // high-water solver lengths
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> cached_dx; // dx the kernel was built with

    // HELPER functions for handling the retype
    // function that substitutes amrex::convert without the index drop
    static amrex::BoxArray retypeBoxArray(const amrex::BoxArray& ba);
    // the single cell-centred BoxArray shared by all index types: retype(convert(ba, nodal))
    static amrex::BoxArray canonicalBoxArray(const amrex::BoxArray& ba);
    // recompute domain, domain_lo and canon_ba from xsoln_ba_ref
    void updateGeometry();
    // fetch a scratch MultiFab on canon_ba, (re)defining it only when stale
    amrex::MultiFab& scratchMultiFab(amrex::MultiFab& slot, const amrex::DistributionMapping& dm,
                                     const amrex::IntVect& ng);

public:

    LGFOpenBC(const amrex::Geometry& geom_in, const amrex::BoxArray& xsoln_ba_in,
              int n_look_in, int n_ghost_max_in, int len_quantum_in = 1);

    // HELPER functions, kept public because GPU kernels are launched from them
    // checks the cached kernel against domain.length() and dx; rebuilds if necessary
    void buildSolver();
    // dst(p) = src(p + offset) wherever both fabs (incl. ghosts) overlap. If
    // restrict_ba (cell-centred, src's physical indices) is given, dst is zeroed
    // first and only indices of src inside convert(*restrict_ba, src.ixType())
    // are copied. dst and src must share size and DistributionMapping.
    static void retypeCopy(amrex::MultiFab& dst, const amrex::MultiFab& src,
                           const amrex::IntVect& offset,
                           const amrex::BoxArray* restrict_ba = nullptr);

    // WRAPPER functions for external calling
    // source and target: same index type, built from xsoln_ba_ref, same dm;
    // source ghosts filled in its cell-centred directions (see above)
    void solvePoisson(const amrex::MultiFab& source, amrex::MultiFab& target, const amrex::BoxArray& tag_ba);
    void regridOnto(const amrex::Geometry& new_geom, const amrex::BoxArray& new_ba, const amrex::DistributionMapping& new_dm);

};

#endif
