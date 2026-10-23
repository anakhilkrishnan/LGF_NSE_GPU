#ifndef SOLVERCONFIG_H_
#define SOLVERCONFIG_H_

#include <AMReX_ParmParse.H>
#include <AMReX_Array.H>
#include <string>

struct SolverConfig
{
    // refinement parameters
    int refine_ratio;
    amrex::IntVect n_cell;
    amrex::Real set_dt;

    // grid parameters
    // initial coarse search grid parameters
    int search_to_base_refine_ratio;
    amrex::IntVect n_cell_search;
    amrex::IntVect max_grid_size_search;

    // base grid parameters
    amrex::IntVect n_cell_base;
    amrex::IntVect max_grid_size;
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
    amrex::Real set_dt_base, cfl;

    // physical parameters
    amrex::Real invRe;
    amrex::Real supp_tag_eps;
    amrex::Real int_fact_eps;

    // LGF kernel parameters
    int n_lookup;

    void readInputs()
    {
        amrex::ParmParse pp;
        pp.get("refine_ratio", refine_ratio);
        pp.getarr("n_cell", n_cell);
        pp.get("set_dt", set_dt);

        pp.get("search_to_base_refine_ratio", search_to_base_refine_ratio);

        pp.getarr("n_cell_base", n_cell_base);
        pp.getarr("max_grid_size", max_grid_size);
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
        pp.get("set_dt_base", set_dt_base);
        pp.get("cfl", cfl);  

        pp.get("invRe", invRe);
        pp.get("support_threshold", supp_tag_eps);
        pp.get("int_fact_threshold", int_fact_eps);

        pp.get("n_lookup", n_lookup);

        // alter base grid to match refinement ratio 
        if (n_cell != (n_cell_base * refine_ratio))
        {
            amrex::Print() << "n_cell from inputs doesn't match requested refine ratio! Going ahead with n_cell = n_cell_base * refine_ratio, please update inputs file";
            n_cell = n_cell_base * refine_ratio;
        }
        if (set_dt != (set_dt_base / refine_ratio))
        {
            amrex::Print() << "set_dt from inputs doesn't match requested refine ratio! Going ahead with set_dt = set_dt_base * refine_ratio, please update inputs file";
            set_dt = set_dt_base / refine_ratio;
        }
        max_steps *= refine_ratio;
        
        // compute search grid parameters
        n_cell_search = n_cell / search_to_base_refine_ratio;
        max_grid_size_search = max_grid_size / search_to_base_refine_ratio;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_cell_search * search_to_base_refine_ratio == n_cell, "Search space refinement doesn't divide nicely!");
    }
};

#endif