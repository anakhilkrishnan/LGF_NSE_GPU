#ifndef IOMANAGER_H_
#define IOMANAGER_H_

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_BLProfiler.H>
#include <string>
#include <fstream>
#include <cstdio>

#include <IOConfig.H>
#include <FlowField.H>
#include <ProjectionWorkspace.H>

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

class IOManager
{
    private:

        // simulation configuration object, passed by reference
        const IOConfig& cfg;
        
        // correct chkfile to use
        std::string better_dir;

        void whichChkDirBetter();

    public:

        IOManager(const IOConfig& config);
        void writeMyChkFile(bool writeMainChk, int step, amrex::Real time, const FlowField& state, const amrex::BoxArray& supp_ba);
        void initializeBoxArrFromChk(int& step, amrex::Real& time, amrex::BoxArray& chk_ba, amrex::BoxArray& chk_supp_ba);
        void initializeFlowFieldFromChk(FlowField& init_state);
        void writeMyPlotFile(int step, amrex::Real time, const FlowField& state, 
                                const MultiFab& divU_star,
                                const MultiFab& divN, 
                                const MultiFab& psi, 
                                const amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& vel_ref_err,
                                const amrex::Geometry& geom, 
                                const amrex::BoxArray& ba, 
                                const amrex::BoxArray& supp_ba,
                                const amrex::DistributionMapping& dm);
        void writeMyDiagnosticPlotFile(int diag_num, int step, amrex::Real time,
                                        const FlowField& state,
                                        const DomainManager& dom_mgr);
        void writeKEData(int step, amrex::Real time, const ProjectionWorkspace& workspace);
        void initializeWriteKEData(int step, amrex::Real time, const ProjectionWorkspace& workspace);
        

};

#endif