#ifndef IOCONFIG_H_
#define IOCONFIG_H_

#include <AMReX_ParmParse.H>
#include <AMReX_Array.H>
#include <string>

struct IOConfig
{
    // IO parameters
    bool write_plot = true;
    bool plot_only_support = false;
    bool plot_pressure = false;      // lean plot: add the pressure multiplier
    bool plot_dense = false;         // dense plot every plot_dense_int steps
    bool plot_regrid = false;        // pre/post pair on every step the grid changes
    bool write_chk = true;
    bool write_log = true;
    bool start_from_chk = false;

    int plot_int = 100;
    int plot_dense_int = 1000;
    int chk_int = 20;
    int log_int = 1;

    std::string plot_dir = "./Results";
    std::string chk_dir = "./Checkpoints/main";
    std::string altchk_dir = "./Checkpoints/alt";
    std::string log_dir = "./Logs";

    std::string plot_prefix = "/plt";
    std::string chk_prefix = "/chk";
    std::string log_prefix = "/log.txt";

    void readInputs()
    {
        amrex::ParmParse pp;

        pp.query("write_plot", write_plot);
        pp.query("plot_pressure", plot_pressure);
        pp.query("plot_dense", plot_dense);
        pp.query("plot_regrid", plot_regrid);
        pp.query("plot_only_support", plot_only_support);
        pp.query("plot_int", plot_int);
        pp.query("plot_dense_int", plot_dense_int);
        pp.query("plot_dir", plot_dir);
        pp.query("plot_prefix", plot_prefix);

        pp.query("write_chk", write_chk);
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
    }
};

#endif