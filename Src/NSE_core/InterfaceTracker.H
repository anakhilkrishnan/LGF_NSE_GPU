#ifndef INTERFACETRACKER_H_
#define INTERFACETRACKER_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Array.H>
#include <AMReX_Math.H>
#include <AMReX_Reduce.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_BLProfiler.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include <InterfaceDiscretization.H>
#include <InitialSDF.H>
#include <SolverConfig.H>

// SIGN CONVENTION: psi > 0 inside an interface, psi < 0 outside (see InitialSDF.H, where the
// initial interfaces are defined). Interfaces are closed surfaces, so everything far away is
// "outside" (psi = -psi_far).

// linear-extrapolation increment along direction dim for a ghost cell that lies
// `steps` cells outside its box, measured from the nearest valid cell
// (ic,jc,kc) with the inward one-sided difference; zero when the box is one
// cell wide in that direction (no inward neighbour to difference with)
template <int dim>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real ghostIncrement(amrex::Array4<amrex::Real const> const& p, int ic, int jc, int kc, int n,
                           int steps, bool has_inner)
{
    if (steps == 0 || !has_inner) { return 0.0; }
    const int s = (steps > 0) ? 1 : -1;
    return amrex::Math::abs(steps) * (p(ic, jc, kc, n) - valueAlong<dim>(p, ic, jc, kc, n, -s));
}

// the scheme named in the inputs (SolverConfig::interface_scheme)
inline AdvectionScheme advectionSchemeFromName(const std::string& name)
{
    if (name == "WENO5") { return AdvectionScheme::WENO5; }
    if (name == "HOUC5") { return AdvectionScheme::HOUC5; }
    amrex::Abort("advectionSchemeFromName: unknown scheme '" + name + "' (WENO5 or HOUC5)");
    return AdvectionScheme::WENO5;
}

class InterfaceTracker
{
    private:

        // grid of the snug domain (Dxsoln), kept in sync with DomainManager
        // through regrid()
        amrex::Geometry geom;
        amrex::BoxArray ba;
        amrex::DistributionMapping dm;

        // signed distance functions, one component per interface
        amrex::MultiFab psi;

        // interface parameters; the interfaces themselves are the active case in InitialSDF.H
        int n_interface;
        AdvectionScheme scheme;
        amrex::Real psi_far;    // clip magnitude of psi: the far-field value, and the value of cells new to the domain

        // supporting functions
        template <AdvectionScheme scheme_t>
        void computeAdvectionRHSImpl(const amrex::MultiFab& psi_in, amrex::MultiFab& rhs,
                                     const amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& vel,
                                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& uinf) const;

    public:

        // getter functions
        amrex::MultiFab& getPsi() { return psi; }
        const amrex::MultiFab& getPsi() const { return psi; }
        int getNInterface() const { return n_interface; }
        AdvectionScheme getScheme() const { return scheme; }
        amrex::Real getPsiFar() const { return psi_far; }

        // primary functionality
        InterfaceTracker(const amrex::Geometry& geom_in, const amrex::BoxArray& ba_in,
                         const amrex::DistributionMapping& dm_in, const SolverConfig& config);
        InterfaceTracker(const amrex::Geometry& geom_in, const amrex::BoxArray& ba_in,
                         const amrex::DistributionMapping& dm_in, AdvectionScheme scheme_in, amrex::Real psi_far_in);
        void initializeSDF();
        void regrid(const amrex::Geometry& new_geom, const amrex::BoxArray& new_ba,
                    const amrex::DistributionMapping& new_dm);
        void computeAdvectionRHS(amrex::MultiFab& psi_in, amrex::MultiFab& rhs,
                                 const amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& vel,
                                 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& uinf) const;
        amrex::Real computeCFL(const amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& vel,
                               amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& uinf, amrex::Real dt) const;

        // supporting functions
        void defineLikePsi(amrex::MultiFab& mf, int n_grow) const;
        void fillPsiBoundary(amrex::MultiFab& psi_in) const;
        void checkBandInside(const amrex::BoxArray& next_ba, const std::string& caller) const;
};

#endif
