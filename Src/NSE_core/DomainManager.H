#ifndef DOMAINMANAGER_H_
#define DOMAINMANAGER_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_BoxList.H>
#include <AMReX_BCRec.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_Math.H>
#include <AMReX_Array.H>
#include <AMReX_BLProfiler.H>

#include <SolverConfig.H>
#include <FlowField.H>
#include <SpatialDiscretization.H>
#include <DirectSumLGF.H>

// free function: this must be defined in NSE_run
void initializeVelField(FlowField& init_state);

// free functions for convenience/plotting
amrex::MultiFab computeNodalVorticity(const FlowField& state);
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

        // scratch variables
        amrex::MultiFab DSupp;
        amrex::Vector<int>h_tag_arr;

        // inputs needed from Solver config
        int regrid_int;

        int n_cell, max_grid_size, n_buffer_box, n_shed_box, search_to_fine_ref_ratio, n_comp, n_ghost;
        amrex::Array<amrex::Real, AMREX_SPACEDIM> dom_lo, dom_hi;
        amrex::Real supp_tag_eps;

        // supporting functions
        amrex::BoxArray gatherTaggedBoxArr(const FlowField& state);
        void growBoxArr(amrex::BoxArray& tag_ba, int nBuff, int max_grid_size);
        void updateGeomBaDm(amrex::BoxArray& tag_ba);

    public:
        // scratch type variables
        amrex::MultiFab psi;
        amrex::MultiFab vort;
        amrex::MultiFab divN;
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> vel_refresh_err; // checks if the velocity refresh is working
        amrex::Array<amrex::Real, AMREX_SPACEDIM> refresh_err_max_norm;

        // getter functions
        amrex::BoxArray getBoxArr() const { return ba; }
        amrex::DistributionMapping getDistMap() const { return dm; }
        amrex::Geometry getGeom() const { return geom; }
        const amrex::Gpu::DeviceVector<int>& getSuppTagArr() const { return supp_tag_arr; }

        // primary functionality
        DomainManager(const SolverConfig& config);
        const amrex::MultiFab& refreshAndGetDSuppFab();
        void tagSupportRegion(const FlowField& state);
        void initializeSnugDomain();
        int computeRegridInterval(const FlowField& state) const;
        void updateSnugDomain(const FlowField& state);

        // supporting functions
        void vor2vel(FlowField& state, DirectSumLGF& lgf_nodal_poisson_solver);
        void regridFlowFieldOntoNewSnugDomain(FlowField& state, DirectSumLGF& lgf_nodal_poisson_solver);
};  

#endif