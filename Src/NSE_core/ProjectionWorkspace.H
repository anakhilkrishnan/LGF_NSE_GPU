#ifndef PROJECTIONWORKSPACE_H_
#define PROJECTIONWORKSPACE_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Math.H>
#include <AMReX_Array.H>
#include <AMReX_BLProfiler.H>

#include <SolverConfig.H>
#include <DomainManager.H>
#include <FlowField.H>
#include <SpatialDiscretization.H>
#include <IFHERK.H>
#include <LGFOpenBC.H>
#include <InterfaceTracker.H>

class ProjectionWorkspace
{
    private:
        // layout of the scratch fields, kept for regridOnto()
        int n_comp;
        int n_ghost;

        // time integrator: shifted tableau, time step and integrating factor
        IFHERK ifherk;

        FlowField stage;

        // scratch fields
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> rhs_vel;

        // IF-HERK registers.  1-based in j: w[j] holds w^{.,j}; w[0] unused.
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> q;
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> r;
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> IF_buff;   // sweep scratch
        amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> w;

        // (re)defines rhs_vel, q, r, IF_buff, w and divU on a grid
        void defineScratch(const amrex::BoxArray& ba_in, const amrex::DistributionMapping& dm_in);

    public: 
        // public, so it can be passed to DomainManager when field needs to be updated
        LGFOpenBC lgf_poisson_solver;

        amrex::MultiFab divU; // refers to divU star, the RHS of what the Poisson solver sees

        amrex::Real divU_max_norm;
        amrex::Real divU_at_end_max_norm;
        amrex::Real divU_at_end_max_norm_support;

        ProjectionWorkspace(const amrex::Geometry& geom_in, const amrex::BoxArray& ba_in, const amrex::DistributionMapping& dm_in, const SolverConfig& config);

        // the one source of the time step and the tableau
        const IFHERK& getIntegrator() const { return ifherk; }

        // functions for initialization
        void initializePresField(FlowField& init_state, const amrex::BoxArray& init_supp_ba, amrex::Real time);

        void computeGStage(const FlowField& st, int i, amrex::Real t_stage);  // N(u^{i-1} + u_inf(t_stage))   // g^i -> w[i]
        void maskToSupport(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& fld, const amrex::BoxArray& supp);
        void advanceTimeStep(FlowField& state_n, const amrex::BoxArray& supp_ba, amrex::Real time_n,
                             InterfaceTracker* tracker);    // nullptr when no interface is tracked
        void regridOnto(const amrex::Geometry& new_geom, const amrex::BoxArray& new_ba, const amrex::DistributionMapping& new_dm);
};

#endif