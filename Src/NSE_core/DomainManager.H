#ifndef DOMAINMANAGER_H_
#define DOMAINMANAGER_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_BoxList.H>
#include <AMReX_BoxArray.H>
#include <AMReX_BCRec.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_Math.H>
#include <AMReX_Array.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_GpuContainers.H>

#include <limits>
#include <vector>

#include <SolverConfig.H>
#include <FlowField.H>
#include <SpatialDiscretization.H>
#include <LGFOpenBC.H>

class DomainManager
{
    private:
        // the Geometry, BoxArray and DistributionMapping that refer to Dxsoln
        // along with the box_tag_arr that only represents Dsupp
        amrex::Geometry geom;
        amrex::BoxArray ba;
        amrex::BoxArray supp_ba;
        amrex::DistributionMapping dm;

        // tagging fields
        amrex::MultiFab tag_vort;
        amrex::MultiFab tag_divN;

        // velocity refresh fields
        amrex::Array<amrex::MultiFab, N_VORT> vort;
        amrex::Array<amrex::MultiFab, N_VORT> psi;

        // scratch variables
        amrex::BoxArray old_supp_ba;
        amrex::Gpu::DeviceVector<int> supp_tag_arr;
        amrex::Vector<int>h_tag_arr;

        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> vel_refresh_err; // checks if the velocity refresh is working

        // inputs needed from Solver config
        int regrid_int;

        int n_cell, max_grid_size, n_buffer_box, n_shed_box, search_to_fine_ref_ratio, n_lookup, n_comp, n_ghost, n_ghost_max;
        amrex::Array<amrex::Real, AMREX_SPACEDIM> dom_lo, dom_hi;
        amrex::Real supp_tag_eps;

        // supporting functions
        void growBoxArr(amrex::BoxArray& tag_ba, int nBuff, int max_grid_size);
        void updateGeomBaDm(const amrex::BoxArray& new_ba);
        int computeRegridInterval(const FlowField& state) const;

    public:
        
        // public scratch variables
        bool did_snug_domain_change;
        amrex::Array<amrex::Real, AMREX_SPACEDIM> refresh_err_max_norm;

        // getter functions
        amrex::BoxArray getBoxArr() const { return ba; }
        amrex::BoxArray getSuppBoxArr() const { return supp_ba; }
        amrex::DistributionMapping getDistMap() const { return dm; }
        amrex::Geometry getGeom() const { return geom; }

        const amrex::MultiFab& getTagVort() const { return tag_vort; }
        const amrex::MultiFab& getTagDivN() const { return tag_divN; }

        const amrex::Array<amrex::MultiFab, N_VORT>& getVort() const { return vort; }
        const amrex::Array<amrex::MultiFab, N_VORT>& getPsi() const { return psi; }

        const amrex::MultiFab& getVelRefreshErrComp(int idim) const { return vel_refresh_err[idim]; }

        // primary functionality
        DomainManager(const SolverConfig& config);
        void defineScratch();
        void computeSuppBoxArr(const FlowField& state, bool shedOuterLayer = true); // tagging function
        void initializeSnugDomain();
        void initializeVelField(FlowField& state, LGFOpenBC& lgf_poisson_solver);
        void restartSnugDomain(const amrex::BoxArray& chk_ba, const amrex::BoxArray& chk_supp_ba);
        void checkAndUpdateSnugDomain(const FlowField& state);
        void checkAndRefreshVelocity(FlowField& state, LGFOpenBC& lgf_poisson_solver, int step);
        

        // supporting functions
        template <int... dim>
        void fillVelocity(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& vel,
                        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& vel_refresh_err,
                        const amrex::Array<amrex::MultiFab, N_VORT>& psi,
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& invdx,
                        std::integer_sequence<int, dim...>);
        template <int dim>
        void fillVelocityComponent(amrex::MultiFab& vel_comp,
                                                amrex::MultiFab& vel_refresh_err_comp,
                                                const amrex::Array<amrex::MultiFab, N_VORT>& psi,
                                                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& invdx);
        void vor2vel(amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& vel_arr, amrex::Array<amrex::MultiFab, N_VORT>& vort, LGFOpenBC& lgf_poisson_solver);
};  

#endif