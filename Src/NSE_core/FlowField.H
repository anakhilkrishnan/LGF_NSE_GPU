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
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> vel;
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> kecomp;
        amrex::MultiFab pres;
        amrex::MultiFab divU; // refers to divU star, the RHS of what the Poisson solver sees
        amrex::MultiFab divU_at_end; // refers to divU at the end of a time step
        amrex::MultiFab tagRegion;
        amrex::Geometry globalgeom;

    public:
        FlowField(const amrex::Geometry& geom, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, const SolverConfig config);
        FlowField(const FlowField& other);
        FlowField& operator=(const FlowField& other);
    
        // fetch functions for private data
        amrex::MultiFab& getVel(int idim);
        const amrex::MultiFab& getVel(int idim) const;
        amrex::MultiFab& getKEComp(int idim);
        const amrex::MultiFab& getKEComp(int idim) const;
        amrex::MultiFab& getPres();
        const amrex::MultiFab& getPres() const;
        amrex::MultiFab& getTagRegion();
        const amrex::MultiFab& getTagRegion() const;
        amrex::MultiFab& getDivU();
        const amrex::MultiFab& getDivU() const;
        amrex::MultiFab& getDivUAtEnd();
        const amrex::MultiFab& getDivUAtEnd() const;
        const amrex::Geometry& getGeom() const;

        void setBoundary();
};

#endif