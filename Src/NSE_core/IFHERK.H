#ifndef IFHERK_H_
#define IFHERK_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Array.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Vector.H>
#include <AMReX_Print.H>
#include <AMReX_BLProfiler.H>

#include <RKCoefficients.H>

#include <algorithm>
#include <cmath>
#include <utility>

// the problem-agnostic half of the integrating-factor half-explicit Runge-Kutta scheme of
// Liska & Colonius (2016): the shifted tableau, the time step, and the integrating factor
// H = E(alpha) as 1D heat-kernel tables. The stage loop stays with the problem
// (ProjectionWorkspace::advanceTimeStep)

class IFHERK
{
    public:
        static constexpr int MAX_STAGES = RKButcher::MAX_STAGES;

        // constructor. n_IF_in <= 0 picks the stencil half-width automatically: the smallest n
        // whose kernel tail 2*sum_{m>n} h(m) is below IF_eps_in for every sub-step. A positive
        // n_IF_in is used as given and must meet IF_eps_in. invRe_in == 0 (inviscid) makes
        // every H the identity: no tables, no sweeps, nGhostRequired() == 0
        IFHERK (const RKButcher& tableau_in, const amrex::Geometry& geom_in, amrex::Real invRe_in, amrex::Real dt_in, int n_IF_in, amrex::Real IF_eps_in);

        // shifted tableau retrieving functions, 1-indexed
        int nStages() const { return rk_stages; }
        amrex::Real aT(int i, int j) const { return rk_at(i,j); }
        amrex::Real cT(int i) const { return rk_ct(i); }
        amrex::Real gapT(int i) const { return rk_gap(i); }
        int ifIdx(int i) const { return rk_if_idx(i); }   // -1: H = identity

        // helper functions
        int nGhostRequired() const { return n_IF; }   // ghost cells applyIF() needs on field and buffer
        amrex::Real timeStep() const { return dt; }
        amrex::Real invReynolds() const { return invRe; }
        amrex::Real maxKernelTail() const;   // largest tail over the current tables
        void printSummary() const;

        // functions for changing dt/changing grid, not used at present
        void setTimeStep(amrex::Real dt_new);
        void setGeometry(const amrex::Geometry& geom_new);

        // H applied in place, as one 1D sweep per direction that ping-pongs through buffer.
        // Cells outside the grid count as zero (the M^xsoln mask), so every ghost cell that no
        // neighbouring box covers is zeroed before each sweep, and again on return.
        // buffer: scratch with the BoxArray, DistributionMapping and index type of field, and
        // like field at least nGhostRequired() ghost cells. The two MultiFabs trade storage
        // once per sweep, so pass a buffer that holds nothing worth keeping.
        // Public because it launches a device lambda (CUDA extended-lambda rule)
        void applyIF (amrex::MultiFab& field, amrex::MultiFab& buffer, int if_idx) const;

        // vector overload: launches nothing itself, so it can live in the header
        template <std::size_t N>
        void applyIF (amrex::Array<amrex::MultiFab, N>& fld,
                    amrex::Array<amrex::MultiFab, N>& buff, int if_idx) const
        {
            for (std::size_t c = 0; c < N; ++c) { applyIF(fld[c], buff[c], if_idx); }
        }
    
    private:
        // member variables
        amrex::Real IF_eps = 0.0;
        int n_IF      = 0;

        int rk_stages = 0;

        amrex::Geometry geom;
        amrex::Real dt  = 0.0;
        amrex::Real invRe  = 0.0;

        // Shifted RK tableau data members
        amrex::Array2D<amrex::Real, 1, MAX_STAGES, 1, MAX_STAGES> rk_at;   // a~_{i,j}
        amrex::Array1D<amrex::Real, 1, MAX_STAGES> rk_ct;   // c~_i

        // indexing and tracking unique gaps for number of IFs to precompute
        amrex::Array1D<amrex::Real, 1, MAX_STAGES> rk_gap;  // c~_i - c~_{i-1}
        amrex::Vector<amrex::Real> rk_unique_gaps;
        amrex::Array1D<int, 1, MAX_STAGES> rk_if_idx;
        amrex::Vector<amrex::Real> if_tail;    // per unique gap

        // final precomputed IF data member
        amrex::Vector<amrex::Gpu::DeviceVector<amrex::Real>> if_table;   // [gap][0..n_IF]

        // helper functions for generating the IF
        static amrex::Vector<amrex::Real> besselKernel (amrex::Real alpha, int nmax);
        static amrex::Real kernelTail (const amrex::Vector<amrex::Real>& h, int n);
        amrex::Real alphaOf (amrex::Real gap) const;   // gap * dt * invRe / dx^2
        void assertIsotropic () const;                 // dx == dy == dz

        // functions for setup
        void setupRKCoeffs (const RKButcher& tableau);
        void precomputeIFs();
};   

#endif