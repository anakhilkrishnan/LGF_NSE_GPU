#ifndef DOMAINMANAGER_H_
#define DOMAINMANAGER_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_BCRec.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_Math.H>
#include <AMReX_Array.H>
#include <AMReX_BLProfiler.H>

#include <SolverConfig.H>
#include <FlowField.H>
#include <SpatialDiscretization.H>
#include <directSumLGF.H>

// free function: this must be defined in NSE_run
void initializeVelField(FlowField& init_state);

// free functions for convenience/plotting
amrex::MultiFab computePlotVorticity(const FlowField& state);
amrex::MultiFab computeDivNonLinearTerm(const FlowField& state);
void computeDivU(amrex::MultiFab& output_divU, const FlowField& input_state);
amrex::MultiFab computePlotDivU(const FlowField& state);



// IMPORTANT: At present, domainManager is wired up for 2D, see vorticity
// management (in both tagging and vort2vel)
class DomainManager
{
    private:
        // the Geometry, BoxArray and DistributionMapping that refer to Dxsoln
        // along with the box_tag_arr that only represents Dsupp
        amrex::Geometry geom;
        amrex::BoxArray ba;
        amrex::DistributionMapping dm;
        amrex::Gpu::DeviceVector<int> supp_tag_arr;

        // inputs needed from Solver config
        amrex::Real supp_tag_eps;
        int regrid_int;
        int n_buffer;

        // supporting functions
        

    public:
        // scratch type variables
        amrex::MultiFab vort;
        amrex::MultiFab divN;
        amrex::MultiFab DSupp;

        // getter functions
        amrex::BoxArray getBoxArr() const { return ba; }
        amrex::DistributionMapping getDistMap() const { return dm; }
        amrex::Geometry getGeom() const { return geom; }
        const amrex::Gpu::DeviceVector<int>& getSuppTagArr() const { return supp_tag_arr; }

        // primary functionality
        DomainManager(const SolverConfig& config);
        void tagSupportRegion(const FlowField& state);
        void initializeSnugDomain(const SolverConfig& config);
        int computeRegridInterval(const FlowField& state) const ;
        void updateSnugDomain(const FlowField& state);
};

#endif