#ifndef DOMAINMANAGER_H_
#define DOMAINMANAGER_H_

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
#include <directSumLGF.H>

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
        // scratch type variables

        // supporting functions
        

    public:
        // information about regridding that is open to the solver

        // getter functions
        amrex::BoxArray getBoxArr() const { return ba; }
        amrex::DistributionMapping getDistMap() const { return dm; }
        amrex::Geometry getGeom() const { return geom; }
        const amrex::Gpu::DeviceVector<int>& getSuppTagArr() const { return supp_tag_arr; }

        // primary functionality
        DomainManager(const SolverConfig& config);
        void tagSupportRegion(const FlowField& state, const MultiFab& divU_fine);
        void initializeSnugDomain();
        int computeRegridInterval(const FlowField& state) const ;
        void updateSnugDomain(const FlowField& state);
};

#endif