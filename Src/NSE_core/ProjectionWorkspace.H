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
        // variables that are initialized in constructor call and live 
        // throughout solver life cycle
        int n_lookup;
        int rk_stages;
        amrex::Real cfl;

        amrex::Real dt;
        amrex::Real invRe;

        FlowField stage;

        // scratch space type data
        amrex::BoxArray tag_ba; // BoxArray that represents tagged support within ba

        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> rhs_vel;
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> rhs_vel_corr;
        amrex::MultiFab pres_corr;

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

        // functions for flow field computations
        void computeMomentumFluxes(const FlowField& input_state);
        void predictVelocity(const FlowField& state_n, amrex::Real alpha, amrex::Real beta, amrex::Real gamma);
        void computePressure();
        void computeVelocityCorrection();
        void correctVelocityandPressure(amrex::Real gamma);

        amrex::Real computeDt(const FlowField& state_n) const;
        amrex::Real computeDivUMaxNorm(const FlowField&, const amrex::BoxArray* restrict_ba = nullptr) const;
        void advanceTimeStep(FlowField& state_n, const amrex::Real dt_in, const amrex::BoxArray& supp_ba);
        void regridOnto(const amrex::Geometry& new_geom, const amrex::BoxArray& new_ba, const amrex::DistributionMapping& new_dm);
};

#endif