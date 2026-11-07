#ifndef IFHERK_H_
#define IFHERK_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Array.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Vector.H>

#include <RKCoefficients.H>

class IFHERK
{
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

        // functions for setup
        void setupRKCoeffs (const RKButcher& tableau);
        void precomputeIFs();

    public:
        static constexpr int MAX_STAGES = RKButcher::MAX_STAGES;

        // constructor
        IFHERK (const RKButcher& tableau_in, const amrex::Geometry& geom_in, amrex::Real invRe_in, amrex::Real dt_in, int n_IF_in, amrex::Real IF_eps_in);

        // shifted tableau retrieving functions, 1-indexed
        int nStages() const { return rk_stages; }
        amrex::Real aT(int i, int j) const { return rk_at(i,j); }
        amrex::Real cT(int i) const { return rk_ct(i); }
        amrex::Real gapT(int i) const { return rk_gap(i); }
        int ifIdx(int i) const { return rk_if_idx(i); }   // -1: H = identity

        // helper functions
        int nGhostRequired()const { return n_IF; }
        amrex::Real maxKernelTail() const;   // largest tail over the current tables
        void printSummary() const;
}   

#endif