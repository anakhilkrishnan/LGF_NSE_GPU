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

// struct for reading and using chk metadata from grid.dat or inputs.used files
struct ChkMetadata
{
    std::string curr_dir;
    int version = 0;
    int step = 0;
    amrex::Real time = 0.0;
    amrex::BoxArray ba, supp_ba;
    amrex::IntVect n_cell_base, n_cell;
    int refine_ratio = 1;
    amrex::ReaVect prob_lo, prob_hi;
    amrex::Real dt = 0.0;
    amrex::Geometry geom;
}

// which plot to write: see IOManager::writePlot
enum class PlotKind { Lean, Dense, RegridPre, RegridPost };

class IOManager
{
    private:

        // simulation configuration object, passed by reference
        const IOConfig& io_cfg;
        const SovlerConfig& sol_cfg;
        
        // correct chkfile to use
        std::string better_dir;
        
        void whichChkDirBetter();

    public:

        IOManager(const IOConfig& io_config, const SolverConfig& sol_config);
        void writeMyChkFile(bool writeMainChk, int step, amrex::Real time, const FlowField& state, const amrex::BoxArray& supp_ba);
        ChkMetadata readChkMetadata();
        void initializeFlowFieldFromChk(FlowField& init_state, DomainManager& dmgr, ProjectionWorkspace& workspace);
        void writePlot(PlotKind kind, int step, amrex::Real time,
                       const FlowField& state, const DomainManager& dom_mgr);
        void writeRingDiagnostics(int step, amrex::Real time, amrex::Real dt, amrex::Real nu,
                                const FlowField& state, const DomainManager& dom_mgr);
};

#endif