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
 * WHY THIS NEEDS ALMOST NO PLUMBING
 *
 *   OpenBCSolver::solve() does
 *       inmf.setVal(0); inmf.ParallelCopy(rho, 0, 0, 1);   ... forward
 *       ... multiply by cached spectral G ...
 *       m_r2c.backward_doit(phi, phi.nGrowVect());
 *   so rho and phi may live on ANY BoxArray overlapping the solver domain.
 *   The snug BoxArray goes straight in and out -- no marshalling layer, no
 *   intermediate bounding-box MultiFabs, nothing like consolidateMultiFab().
 *
 * THE EXPENSIVE PART IS CACHED
 *
 *   setGreensFunction() evaluates G over the doubled domain, forward-transforms
 *   it, and stores the result in the solver's m_G_fft. solve() only reads it.
 *   So the kernel transform must happen once per DOMAIN SIZE, not once per
 *   solve. Since G is translation invariant, the spectral kernel depends only
 *   on the box LENGTHS -- a snug domain that drifts without changing its
 *   bounding-box dimensions needs no recompute at all.
 */
class LGFOpenBC
{
private:

    amrex::Geometry geom;
    int n_lookup;

    // the domain that the spectral Green's function is built on, for
    // cell-centered and nodal MultiFabs respectively; these are computed once,
    // stored and recomputed only when regridOnto() is called
    amrex::Box cached_cc_domain;
    amrex::Box cached_nd_domain;

    // creating object for AMReX's FFT class OpenBCSolver
    std::unique_ptr<amrex::FFT::OpenBCSolver<amrex::Real>> cc_solver;
    std::unique_ptr<amrex::FFT::OpenBCSolver<amrex::Real>> nd_solver;

    // scratch multifab for storing tagged source alone
    amrex::MultiFab cc_rhs;
    amrex::MultiFab nd_rhs;

    // function for finding bounding-box domain that fully covers all target
    // cells and all tagged source boxes
    amrex::Box makeDomain(const amrex::MultiFab& target, const amrex::BoxArray& tag_ba) const;


public:

    LGFOpenBC(const amrex::Geometry& geom_in, int n_look_in);

    // HELPER functions, kept public because GPU kernel needs to be launched
    // checks if cached domain reflects current domain; rebuilds if necessary
    amrex::FFT::OpenBCSolver<amrex::Real>& buildSolver(const amrex::Box& domain);
    // core solve function; invokes functions from AMReX::FFT
    void doSolve(const amrex::MultiFab& source, amrex::MultiFab& target, const amrex::BoxArray& tag_ba);

    // WRAPPER functions for external calling
    void solvePoisson(const amrex::MultiFab& source, amrex::MultiFab& target, const amrex::BoxArray& tag_ba);
    void solveNodalPoisson(const amrex::MultiFab& source, amrex::MultiFab& target, const amrex::BoxArray& tag_ba);
    void regridOnto(const amrex::Geometry& new_geom, const amrex::BoxArray& new_ba, const amrex::DistributionMapping& new_dm);
      
};

#endif
