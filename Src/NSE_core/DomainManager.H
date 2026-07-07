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
#include <DirectSumLGF.H>
#include <InitializeFlow.H>

class DomainManager
{
    private:
        // input variables
        int n_cell, max_grid_size;
        int n_cell_init, max_grid_size_init;
        amrex::Real eps_star; 
        amrex::Array<AMREX_SPACEDIM, int> periodicity;
        amrex::Array<AMREX_SPACEDIM, amrex::Real> dom_lo, dom_hi;

        // reference variables at any given instant
        amrex::Geometry geom;
        amrex::BoxArray ba;
        amrex::DistributionMapping dm;

        // scratch space variables
        amrex::MultiFab vort_mag;

        // helper functions
        void initializeCoarseVort();
        void tagDomain();
        void gatherBoxes();

    public:
        // general constructor
        DomainManager(const SolverConfig& config);

        // caller functions
        const amrex::Geometry& getGeom() const;
        const amrex::BoxArray& getBoxArr() const;
        const amrex::DistributionMapping& getDistMap() const;

        // class functions
        void initializeBoxArray();
        void updateBoxArray();
};

#endif