#ifndef SOLVERCONFIG_H_
#define SOLVERCONFIG_H_

#include <AMReX_ParmParse.H>
#include <AMReX_Array.H>
#include <string>

struct SolverConfig
{
    // grid parameters
    // initial coarse search grid parameters
    amrex::IntVect n_cell;
    amrex::IntVect max_grid_size;

    // fine grid parameters
    int search_to_fine_ref_ratio;
    amrex::IntVect n_cell_search;
    amrex::IntVect max_grid_size_search;
    int n_ghost;
    int n_ghost_max;
    int n_IF = 4;
    int n_comp = 1;
    int regrid_int = 5;
    int n_buffer_box = 2;
    int n_shed_box = 1;
    amrex::RealVect dom_lo, dom_hi;

    // time stepping parameters
    int max_steps;
    amrex::Real t_start, t_stop;
    amrex::Real set_dt, cfl;

    // physical parameters
    amrex::Real invRe;
    amrex::Real supp_tag_eps;
    amrex::Real int_fact_eps;

    // LGF kernel parameters
    int n_lookup;

    void readInputs()
    {
        amrex::ParmParse pp;

        pp.getarr("n_cell", n_cell);
        pp.getarr("max_grid_size", max_grid_size);

        pp.get("search_to_fine_ref_ratio", search_to_fine_ref_ratio);
        pp.get("n_ghost", n_ghost);
        pp.get("n_ghost_max", n_ghost_max);
        pp.query("n_IF", n_IF);
        pp.query("n_comp", n_comp);
        pp.query("regrid_int", regrid_int);
        pp.query("n_buffer_box", n_buffer_box);
        pp.query("n_shed_box", n_shed_box);
        pp.getarr("search_domain_lo", dom_lo);
        pp.getarr("search_domain_hi", dom_hi);

        pp.get("max_steps", max_steps);
        pp.get("t_start", t_start);
        pp.get("t_stop", t_stop);
        pp.get("set_dt", set_dt);
        pp.get("cfl", cfl);  

        pp.get("invRe", invRe);
        pp.get("support_threshold", supp_tag_eps);
        pp.get("int_fact_threshold", int_fact_eps);

        pp.get("n_lookup", n_lookup);

        n_cell_search = n_cell / search_to_fine_ref_ratio;
        max_grid_size_search = max_grid_size / search_to_fine_ref_ratio;

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_cell_search * search_to_fine_ref_ratio == n_cell, "Search space refinement doesn't divide nicely!");
        
    }
};

#endif