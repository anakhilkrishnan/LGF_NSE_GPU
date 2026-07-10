#ifndef FLOWFIELD_H_
#define FLOWFIELD_H_

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_BCRec.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_Math.H>
#include <AMReX_Array.H>

#include <SolverConfig.H>

using namespace amrex;

class FlowField 
{
    private:

        amrex::Geometry globalgeom;
        
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> vel;
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> kecomp;
        amrex::MultiFab pres;

    public:
        FlowField(const amrex::Geometry& geom, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, const SolverConfig& config);
        FlowField(const FlowField& other);
        FlowField& operator=(const FlowField& other);
    
        // fetch functions for private data
        const amrex::Geometry& getGeom() const { return globalgeom; }
        amrex::MultiFab& getVel(int idim) { return vel[idim]; }
        const amrex::MultiFab& getVel(int idim) const { return vel[idim]; }
        amrex::MultiFab& getKEComp(int idim) { return kecomp[idim]; }
        const amrex::MultiFab& getKEComp(int idim) const { return kecomp[idim]; }
        amrex::MultiFab& getPres() { return pres; }
        const amrex::MultiFab& getPres() const { return pres; }

        // function to enforce physical BC's and copy across ghost cells, in one shot
        void setBoundary();
        void regridOnto(const amrex::Geometry& new_geom, const amrex::BoxArray& new_ba, const amrex::DistributionMapping& new_dm);
};

#endif