#ifndef SIMCONFIG_H_
#define SIMCONFIG_H_

#include <AMReX_ParmParse.H>
#include <AMReX_Array.H>
#include <string>

struct SimConfig
{
    // grid parameters
    int n_cell;
    int max_grid_size;
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
    amrex::Real source_tag_thresh;

    // FMM parameters
    int n_chebyshev;
    int n_lookup;

    // IO parameters
    bool write_plot = true;
    bool write_chk = true;
    bool write_log = true;
    bool start_from_chk = true;
    bool write_kedata = true;

    int plot_int = 100;
    int chk_int = 20;
    int log_int = 1;
    int kedata_int = 1;

    std::string plot_dir = "./Results";
    std::string chk_dir = "./Checkpoints/main";
    std::string altchk_dir = "./Checkpoints/alt";
    std::string log_dir = "./Logs";

    std::string plot_prefix = "/plt";
    std::string chk_prefix = "/chk";
    std::string log_prefix = "/log.txt";
    std::string kedata_prefix = "/kinetic_energy.dat";

    void readInputs()
    {
        amrex::ParmParse pp;

        pp.get("n_cell",n_cell);
        pp.get("max_grid_size",max_grid_size);
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
        pp.get("tagging_threshold", source_tag_thresh);

        pp.get("n_chebyshev", n_chebyshev);
        pp.get("n_lookup", n_lookup);

        pp.query("write_plot", write_plot);
        pp.query("plot_int", plot_int);
        pp.query("plot_dir", plot_dir);
        pp.query("plot_prefix", plot_prefix);

        pp.query("write_checkpoint", write_chk);
        pp.query("start_from_chk", start_from_chk);
        pp.query("chk_int", chk_int);
        pp.query("chk_dir", chk_dir);
        pp.query("altchk_dir", altchk_dir);
        pp.query("chk_prefix", chk_prefix);

        // PENDING: Log file backend working
        pp.query("write_log", write_log);
        pp.query("log_int", log_int);
        pp.query("log_dir", log_dir);
        pp.query("log_prefix", log_prefix);

        pp.query("write_kedata", write_kedata);
        pp.query("kedata_int", kedata_int);
        pp.query("kedata_prefix", kedata_prefix);
        
    }
};

#endif