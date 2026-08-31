#ifndef PROJECTIONWORKSPACE_H_
#define PROJECTIONWORKSPACE_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BCRec.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_Math.H>
#include <AMReX_Array.H>
#include <AMReX_BLProfiler.H>

#include <SolverConfig.H>
#include <DomainManager.H>
#include <FlowField.H>
#include <SpatialDiscretization.H>
#include <RKCoefficients.H>
#include <DirectSumLGF.H>

using namespace amrex;

class ProjectionWorkspace
{
    private:
        static constexpr int MAX_STAGES = RKButcher::MAX_STAGES;
        // brought for declaring scratch for IF-HERK along with shifted tableau

        // variables that are initialized in constructor call and live 
        // throughout solver life cycle
        int n_lookup;
        int rk_stages;
        amrex::Real cfl;

        amrex::Real dt;
        amrex::Real invRe;
        amrex::Real IF_eps;

        FlowField stage;

        // rk shifted coefficients and scratch variables
        amrex::Array2D<amrex::Real, 1, MAX_STAGES, 1, MAX_STAGES> rk_at; // a~_{i,j}
        amrex::Array1D<amrex::Real, 1, MAX_STAGES> rk_ct; // c~_i
        amrex::Array1D<amrex::Real, 1, MAX_STAGES> rk_gap; // c~_i - c~_{i-1}
        amrex::Array1D<int, 1, MAX_STAGES> rk_if_idx;

        // IF scratch
        amrex::Gpu::DeviceVector<amrex::Real> IF_1d_comp;
        
        // scratch fields
        amrex::BoxArray tag_ba; // BoxArray that represents tagged support within ba

        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> rhs_vel;
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> rhs_vel_corr;
        amrex::MultiFab pres_corr;

        // temp
        int  n_IF;                                     // IF stencil half-width
        amrex::Vector<amrex::Real>            rk_unique_gaps;
        amrex::Vector<amrex::Gpu::DeviceVector<amrex::Real>> if_table;  // [gap][0..n_IF]

        // IF-HERK registers.  1-based in j: w[j] holds w^{.,j}; w[0] unused.
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> q;
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> r;
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> IF_buff;   // sweep scratch
        amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> w;

        void setupRKCoeffs(const RKButcher& rkbt);
        void precomputeIFs(const amrex::Geometry& geom);

        // 1-based accessors
        amrex::Real aT(int i,int j) const { return rk_at(i,j); }
        amrex::Real cT(int i)       const { return rk_ct(i);   }
        amrex::Real gapT(int i)     const { return rk_gap(i);  }

    public: 

        // public, so it can be passed to DomainManager when field needs to be updated
        DirectSumLGF lgf_poisson_solver;

        amrex::MultiFab divU; // refers to divU star, the RHS of what the Poisson solver sees

        amrex::Real divU_max_norm;
        amrex::Real divU_at_end_max_norm;
        amrex::Real divU_at_end_max_norm_support;

        ProjectionWorkspace(const amrex::Geometry& geom_in, const amrex::BoxArray& ba_in, const amrex::DistributionMapping& dm_in, const SolverConfig& config);

        // functions for initialization
        void initializePresField(FlowField& init_state, const amrex::BoxArray& init_supp_ba);

        void applyIF(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& fld, int if_idx);
        void computeGStage(const FlowField& st, int i);   // g^i -> w[i]
        
        void advanceTimeStep(FlowField& state_n, const amrex::Real dt_in, const amrex::BoxArray& supp_ba);
        void regridOnto(const amrex::Geometry& new_geom, const amrex::BoxArray& new_ba, const amrex::DistributionMapping& new_dm);
};

#endif