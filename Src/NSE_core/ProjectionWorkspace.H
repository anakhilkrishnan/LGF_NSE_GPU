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
#include <FlowField.H>
#include <SpatialDiscretization.H>
#include <RKCoefficients.H>
#include <directSumLGF.H>

using namespace amrex;

// free functions for convenience/plotting
amrex::MultiFab computeCellCenteredVorticity(const FlowField& state);

class ProjectionWorkspace
{
    private:
        // variables that are initialized in constructor call and live 
        // throughout solver life cycle
        int n_lookup;
        int rk_order;
        amrex::Real cfl;
        amrex::Real Re;
        amrex::Real source_tag_thresh;

        directSumLGF lgf_poisson_solver;

        // variables that are updated continouosly 
        amrex::Real dt;

        FlowField stage;
        
        amrex::Geometry geom;
        amrex::BoxArray ba;
        amrex::DistributionMapping dm;
        amrex::BoxArray ba_fine;
        amrex::DistributionMapping dm_fine;

        // scratch space type data

        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> rhs_vel;
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> rhs_vel_corr;
        amrex::MultiFab corr_pres;

        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> rhs_kecomp;

        amrex::Gpu::DeviceVector<int> box_tag_arr;

        amrex::MultiFab divU_fine;
        amrex::MultiFab tagRegion_fine;

    public: 
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> kecomp_dir;

        amrex::Real divU_max_norm;
        amrex::Real divU_at_end_max_norm;
        amrex::Array<amrex::Real, AMREX_SPACEDIM> global_kecomp;
        amrex::Array<amrex::Real, AMREX_SPACEDIM> global_kecomp_dir;
        amrex::Array<amrex::Real, AMREX_SPACEDIM> global_kecomp_err;

        ProjectionWorkspace(const amrex::Geometry& geom_in, const amrex::BoxArray& ba_in, const amrex::DistributionMapping& dm_in, const SolverConfig config);

        // functions for initialization
        void initializePresField(FlowField& init_state);

        // functions for KE computations
        void computeKECompFluxes(const FlowField& input_state);
        void evolveKE(const FlowField& state_n, amrex::Real alpha, amrex::Real beta, amrex::Real gamma);
        void computeKEFromState(const FlowField& state);
        void compareKE(const FlowField& state_n);

        // functions for flow field computations
        void computeMomentumFluxes(const FlowField& input_state);
        void predictVelocity(const FlowField& state_n, amrex::Real alpha, amrex::Real beta, amrex::Real gamma);
        void computePressure();
        void computeVelocityCorrection();
        void correctVelocityandPressure(amrex::Real gamma);

        void computeDt(const FlowField& state);
        void advanceTimeStep(FlowField& state_n);
};

#endif