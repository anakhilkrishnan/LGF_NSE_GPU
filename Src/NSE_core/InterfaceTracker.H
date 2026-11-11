#ifndef INTERFACETRACKER_H_
#define INTERFACETRACKER_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_IArrayBox.H>
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
#include <sstream>
#include <string>
#include <utility>

#include <SolverConfig.H>
#include <InterfaceDiscretization.H>
#include <InterfaceReinit.H>
#include <InitialSDF.H>
#include <IFHERK.H>

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

// host-side settings of the reinitialization (the device constants are in InterfaceReinit.H)
inline constexpr int REINIT_NEIGHBOUR_PASSES = 3;           // passes retrying cells from their neighbours' closest points
inline constexpr amrex::Real REINIT_DRIFT_FRACTION = 0.01;  // adaptive trigger: share of near-interface cells over the tolerance
inline constexpr int REINIT_MAX_BAND_CELLS = 16;            // largest psi_far, in cells, the deep ghost copy allows

// the scheme named in the inputs (SolverConfig::interface_scheme)
inline AdvectionScheme advectionSchemeFromName(const std::string& name)
{
    if (name == "WENO5") { return AdvectionScheme::WENO5; }
    if (name == "HOUC5") { return AdvectionScheme::HOUC5; }
    amrex::Abort("advectionSchemeFromName: unknown scheme '" + name + "' (WENO5 or HOUC5)");
    return AdvectionScheme::WENO5;
}

// true when the flow enters the box through its dim-face on the side of a ghost cell lying
// `steps` cells beyond the nearest valid cell (ic,jc,kc); steps = 0 means the ghost is not
// outside in this direction. The face is the box's own boundary face, so no velocity ghost
// cells are needed: the high face of cell ic is face ic + 1, its low face is face ic
template <int dim>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool ghostInflow(amrex::Array4<amrex::Real const> const& u_face, int ic, int jc, int kc, int steps, amrex::Real uinf)
{
    if (steps == 0) { return false; }
    const int shift = (steps > 0) ? 1 : 0;
    const amrex::Real un = u_face(ic + (dim == 0 ? shift : 0), jc + (dim == 1 ? shift : 0),
                                  kc + (dim == 2 ? shift : 0)) + uinf;
    return (steps > 0) ? (un < 0.0) : (un > 0.0);
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

        // geometric (closest-point) reinitialization, InterfaceReinit.H; off unless set
        int reinit_int = 0;             // reinitialize every reinit_int steps (0: not on an interval)
        amrex::Real reinit_drift = 0.0; // reinitialize when more than REINIT_DRIFT_FRACTION of the cells
                                        // next to the interface have ||grad psi| - 1| above this (0: off)
        int reinit_degree = 3;          // interpolant: 3 (tricubic, 4^3 cells) or 5 (triquintic, 6^3)
        bool reinit_correct = true;     // zero-set correction (step 4 of reinitializeComponent); off only in tests

        // Runge-Kutta stage storage for the transport of psi. psi has no viscous term, so no
        // integrating factor: psi_stage = psi^i and psi_rhs[j] = L^j, 1-based like the flow
        // solver's w registers (psi_rhs[0] unused). psi itself stays at psi(t_k) until
        // finishStep(). Defined by the first advanceStage() after construction or regrid()
        amrex::MultiFab psi_stage;
        amrex::Vector<amrex::MultiFab> psi_rhs;

        void defineStageRegisters(int n_stages);

    public:

        // outcome of one reinitialization, summed (counts) or maximized over the interfaces
        struct ReinitStats
        {
            amrex::Long n_band = 0;               // cells whose closest point was sought
            amrex::Long n_first_pass_failed = 0;  // ... that the first pass did not resolve
            amrex::Long n_estimated = 0;          // ... set from a neighbour's closest point Newton could not polish
            amrex::Long n_beyond = 0;             // ... with neither, beyond the passes' reach: far-field value
            amrex::Long n_unconverged = 0;        // ... with neither, within 2 cells of the interface: kept psi
            amrex::Real max_change_near = 0.0;    // max |psi_new - psi_old| over cells with |psi_old| < dx, in cells
            amrex::Real max_correction = 0.0;     // largest zero-set correction applied (step 4), in cells
        };

        // how far psi is from a distance function next to the interface (|psi| < LS_N_GHOST dx)
        struct GradientDrift
        {
            amrex::Real max_drift = 0.0;   // max ||grad psi| - 1|, central differences
            amrex::Long n_near = 0;        // cells next to the interface
            amrex::Long n_over = 0;        // ... with drift above the tolerance asked for
        };

        // getter functions
        amrex::MultiFab& getPsi() { return psi; }
        const amrex::MultiFab& getPsi() const { return psi; }
        int getNInterface() const { return n_interface; }
        AdvectionScheme getScheme() const { return scheme; }
        amrex::Real getPsiFar() const { return psi_far; }
        bool reinitEnabled() const { return reinit_int > 0 || reinit_drift > 0.0; }
        void setReinitCorrection(bool on) { reinit_correct = on; }   // for tests: the correction belongs on

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

        // reinitialization (closest point; see InterfaceReinit.H)
        void setReinitialization(int interval, amrex::Real drift, int degree);
        ReinitStats reinitialize();
        bool reinitializeIfDue(int step);
        void reinitializeAndReport(int step, const std::string& trigger);
        GradientDrift computeGradientDrift(amrex::Real tol);

        // time integration, driven by the flow solver's stage loop: advanceStage(i, ...) for
        // i = 1 ... integrator.nStages() with the velocity u^{i-1} and freestream that stage's
        // nonlinear term uses, then finishStep() once to accept psi^s as psi(t_{k+1})
        void advanceStage(int i, const IFHERK& integrator,
                          const amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& vel,
                          amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& uinf);
        void finishStep();

        // supporting functions
        template <AdvectionScheme scheme_t>
        void computeAdvectionRHSImpl(const amrex::MultiFab& psi_in, amrex::MultiFab& rhs,
                                     const amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& vel,
                                     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& uinf) const;
        void fillPsiBoundaryImpl(amrex::MultiFab& psi_in, const amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>* vel,
                                 amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& uinf) const;
        void defineLikePsi(amrex::MultiFab& mf, int n_grow) const;
        void fillPsiBoundary(amrex::MultiFab& psi_in) const;
        void fillPsiBoundary(amrex::MultiFab& psi_in, const amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& vel,
                             amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& uinf) const;
        void checkBandInside(const amrex::BoxArray& next_ba, const std::string& caller) const;
        template <int P>
        ReinitStats reinitializeComponent(int n);
        int reinitBandCells() const;
        int reinitGhost() const;
};

#endif