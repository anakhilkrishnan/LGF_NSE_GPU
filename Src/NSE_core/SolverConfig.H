#ifndef SOLVERCONFIG_H_
#define SOLVERCONFIG_H_

#include <AMReX_ParmParse.H>
#include <AMReX_Array.H>
#include <string>

struct SolverConfig
{
    // grid parameters
    int n_cell;
    int max_grid_size;
    int max_grid_size_tagging;
    int n_ghost;
    int n_comp = 1;
    amrex::Array<amrex::Real, AMREX_SPACEDIM> dom_lo, dom_hi;

    // time stepping parameters
    int rk_order;
    int max_steps;
    amrex::Real t_start, t_stop;
    amrex::Real cfl;

    // physical parameters
    amrex::Real Re;
    amrex::Real supp_tag_eps;

    // LGF kernel parameters
    int n_lookup;

    void readInputs()
    {
        amrex::ParmParse pp;

        pp.get("n_cell", n_cell);
        pp.get("max_grid_size", max_grid_size);
        pp.get("max_grid_size_tagging",max_grid_size_tagging);
        pp.get("n_ghost", n_ghost);
        pp.query("n_comp", n_comp);
        pp.get("domain_lo", dom_lo);
        pp.get("domain_hi", dom_hi);

        pp.get("rk_order", rk_order);
        pp.get("max_steps", max_steps);
        pp.get("t_start", t_start);
        pp.get("t_stop", t_stop);
        pp.get("cfl", cfl);  

        pp.get("Re", Re);
        pp.get("support_threshold", supp_tag_eps);

        pp.get("n_lookup", n_lookup);
    }
};

#endif