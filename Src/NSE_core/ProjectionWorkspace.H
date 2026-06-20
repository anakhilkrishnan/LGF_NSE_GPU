#ifndef PROJECTIONWORKSPACE_H_
#define PROJECTIONWORKSPACE_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BCRec.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_Math.H>
#include <AMReX_Array.H>
#include <AMReX_BLProfiler.H>

#include <FlowField.H>
#include <SpatialDiscretization.H>
#include <RKCoefficients.H>
#include <LGFCore.H>

using namespace amrex;

// free functions for convenience/plotting
amrex::MultiFab computeCellCenteredVorticity(const FlowField& state);
amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> computeKEFromState(const FlowField& state);


class ProjectionWorkspace
{
    private:
        amrex::Geometry geom;
        amrex::BoxArray ba;
        amrex::DistributionMapping dm;

        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> rhs_vel;
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> rhs_vel_corr;
        amrex::MultiFab corr_pres;

        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> rhs_kecomp;
        
    public: 
        amrex::Real divU_max_norm;
        amrex::Array<amrex::Real, AMREX_SPACEDIM> global_kecomp;
        amrex::Array<amrex::Real, AMREX_SPACEDIM> global_kecomp_dir;
        amrex::Array<amrex::Real, AMREX_SPACEDIM> global_kecomp_err;

        ProjectionWorkspace(const amrex::Geometry& geom_in, const amrex::BoxArray& ba_in, const amrex::DistributionMapping& dm_in, const int n_comp, const int n_ghost);

        amrex::Real computeDt(const FlowField& state, amrex::Real cfl, amrex::Real Re);
        void initializePresField(FlowField& init_state, amrex::Real Re, amrex::Real source_tag_thresh, int nChebyshev, int nLookup);

        void computeKECompFluxes(const FlowField& stage, amrex::Real Re);
        void evolveKE(const FlowField& state_n, FlowField& stage, amrex::Real dt, amrex::Real alpha, amrex::Real beta, amrex::Real gamma);
        void compareKE(const FlowField& state_n);

        void computeMomentumFluxes(const FlowField& stage, amrex::Real Re);
        void predictVelocity(const FlowField& state_n, FlowField& stage, amrex::Real dt, amrex::Real alpha, amrex::Real beta, amrex::Real gamma);
        void computePressure(FlowField& stage, amrex::Real source_tag_thresh, amrex::Vector<int>& box_tag_arr, int nChebyshev, int nLookup);
        void computeVelocityCorrection(FlowField& stage);
        void correctVelocityandPressure(FlowField& stage, amrex::Real gamma, amrex::Real dt);

        void advanceTimeStep(FlowField& state_n, amrex::Real dt, amrex::Real Re, int rk_order, amrex::Real source_tag_thresh, int n_chebyshev, int n_lookup);
};

#endif