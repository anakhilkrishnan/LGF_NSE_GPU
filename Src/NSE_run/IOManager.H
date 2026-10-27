#ifndef IOMANAGER_H_
#define IOMANAGER_H_

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_BLProfiler.H>

#include <vector>
#include <utility>
#include <iomanip>
#include <array>
#include <string>
#include <fstream>
#include <cstdio>

#include <IOConfig.H>
#include <SolverConfig.H>
#include <FreestreamVelocity.H>
#include <FlowField.H>
#include <ProjectionWorkspace.H>
#include <InterfaceTracker.H>


// free function for averaging cell to node
void averageCellToNode (amrex::MultiFab& nd, int dcomp, const amrex::MultiFab& cc, int scomp);

// struct for reading status.dat files
struct ChkStatus
{
    int step = -1;
    amrex::Real time = -1.0;
    int is_safe = -1;
};

// abstraction to read and store ChkStatus structs
inline ChkStatus readCheckpointStatus(const std::string& chk_dir) 
{
    ChkStatus status;
    std::string status_file = chk_dir + "/status.dat";
    if (amrex::FileSystem::Exists(status_file)) 
    {
        std::ifstream ifs(status_file);
        if (ifs.is_open())
        {
            ifs >> status.step >> status.time >> status.is_safe;
        }
    }
    return status;
}

// struct for reading and using chk metadata from grid.dat files
struct ChkMetadata
{
    std::string curr_dir;
    int version = 0;
    int step = 0;
    amrex::Real time = 0.0;
    amrex::BoxArray ba, supp_ba;
    amrex::IntVect n_cell_base, n_cell;
    int refine_ratio = 1;
    amrex::RealVect prob_lo, prob_hi;
    amrex::Real dt = 0.0;
    amrex::Geometry geom;
    int n_interface = 0;            // 0: the checkpoint carries no psi
    amrex::Real psi_far = 0.0;
};

// free function to check chk metadata
inline int restartTransferRatio (const ChkMetadata& chk, const SolverConfig& sol_cfg)
{
    amrex::Real L = 0.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) 
    {
        L = std::max(L, sol_cfg.dom_hi[d] - sol_cfg.dom_lo[d]);
    }
    auto same = [L](const amrex::RealVect& a, const amrex::RealVect& b) 
    {
        for (int d = 0; d < AMREX_SPACEDIM; ++d) 
        {
            if (std::abs(a[d] - b[d]) > 1e-12 * L)
            {
                return false;
            }                    
        }
        return true;
    };

    // factor 1: the checkpoint is self-consistent
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(chk.n_cell == chk.n_cell_base * chk.refine_ratio,
        "restart: grid.dat inconsistent (n_cell != n_cell_base * refine_ratio)");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(chk.geom.Domain().contains(chk.ba.minimalBox()),
        "restart: checkpoint BoxArray lies outside its own domain");

    // factor 2: it belongs to this setup
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(chk.n_cell_base == sol_cfg.n_cell_base,
        "restart: n_cell_base differs from the checkpoint's");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(same(chk.prob_lo, sol_cfg.dom_lo) && same(chk.prob_hi, sol_cfg.dom_hi),
        "restart: physical domain differs from the checkpoint's");

    // factor 3: the requested resolution is reachable (divisibility also rejects coarsening)
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sol_cfg.refine_ratio % chk.refine_ratio == 0,
        "restart: refine_ratio must be an integer multiple of the checkpoint's");
    const int r = sol_cfg.refine_ratio / chk.refine_ratio;

    if (chk.dt > 0 && std::abs(chk.dt / r - sol_cfg.set_dt) > 1e-12 * sol_cfg.set_dt) 
    {
        amrex::Print() << "WARNING: set_dt is not the checkpoint's dt / " << r << "\n";
    }
    amrex::Print() << "Restart: checkpoint at refine_ratio " << chk.refine_ratio << " (n_cell " << chk.n_cell
                   << "), inputs at " << sol_cfg.refine_ratio << " (n_cell " << sol_cfg.n_cell
                   << ") -> transfer ratio " << r << "\n";
    return r;
}

// which plot to write: see IOManager::writePlot
enum class PlotKind { Lean, Dense, RegridPre, RegridPost };

class IOManager
{
    private:

        // simulation configuration object, passed by reference
        const IOConfig& io_cfg;
        const SolverConfig& sol_cfg;
        
        // correct chkfile to use
        std::string better_dir;
        
        void whichChkDirBetter();

    public:

        IOManager(const IOConfig& io_config, const SolverConfig& sol_config);
        void writeMyChkFile(bool writeMainChk, int step, amrex::Real time, const FlowField& state, const amrex::BoxArray& supp_ba,
                            const InterfaceTracker* tracker);       // nullptr: no psi written
        ChkMetadata readChkMetadata();
        void readChkData(const ChkMetadata& chk, FlowField& chk_state);
        void readChkInterface(const ChkMetadata& chk, InterfaceTracker& tracker);
        void writePlot(PlotKind kind, int step, amrex::Real time,
                       const FlowField& state, const DomainManager& dom_mgr,
                       const InterfaceTracker* tracker);            // nullptr: no psi plotted
        void writeRingDiagnostics(int step, amrex::Real time, amrex::Real dt, amrex::Real nu,
                                const FlowField& state, const DomainManager& dom_mgr);
};

#endif