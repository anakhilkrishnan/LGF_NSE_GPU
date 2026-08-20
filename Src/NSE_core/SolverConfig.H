#ifndef SOLVERCONFIG_H_
#define SOLVERCONFIG_H_

#include <AMReX_ParmParse.H>
#include <AMReX_Array.H>
#include <string>

struct SolverConfig
{
    // grid parameters
    // initial coarse search grid parameters
    int n_cell;
    int max_grid_size;

    // fine grid parameters
    int search_to_fine_ref_ratio;
    int n_cell_search;
    int max_grid_size_search;
    int n_ghost;
    int n_comp = 1;
    int regrid_int = 5;
    int n_buffer_box = 2;
    int n_shed_box = 1;
    amrex::Array<amrex::Real, AMREX_SPACEDIM> dom_lo, dom_hi;

    // time stepping parameters
    int rk_stages;
    int max_steps;
    amrex::Real t_start, t_stop;
    amrex::Real cfl;

    // physical parameters
    amrex::Real invRe;
    amrex::Real supp_tag_eps;

    // LGF kernel parameters
    int n_lookup;

    void readInputs()
    {
        amrex::ParmParse pp;

        pp.get("n_cell", n_cell);
        pp.get("max_grid_size", max_grid_size);

        pp.get("search_to_fine_ref_ratio", search_to_fine_ref_ratio);
        pp.get("n_ghost", n_ghost);
        pp.query("n_comp", n_comp);
        pp.query("regrid_int", regrid_int);
        pp.query("n_buffer_box", n_buffer_box);
        pp.query("n_shed_box", n_shed_box);
        pp.get("domain_lo", dom_lo);
        pp.get("domain_hi", dom_hi);

        pp.get("rk_stages", rk_stages);
        pp.get("max_steps", max_steps);
        pp.get("t_start", t_start);
        pp.get("t_stop", t_stop);
        pp.get("cfl", cfl);  

        pp.get("invRe", invRe);
        pp.get("support_threshold", supp_tag_eps);

        pp.get("n_lookup", n_lookup);

        n_cell_search = n_cell / search_to_fine_ref_ratio;
        max_grid_size_search = max_grid_size / search_to_fine_ref_ratio;
    }
};

#endif