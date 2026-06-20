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

#include <SimConfig.H>
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
        const SimConfig& cfg;
        
        // correct chkfile to use
        std::string better_dir;

        void whichChkDirBetter();

    public:

        IOManager(const SimConfig& config);
        void writeMyChkFile(bool writeMainChk, int step, amrex::Real time, const FlowField& state);
        void initializeBAFromChk(int& step, amrex::Real& time, amrex::BoxArray& ba);
        void initializeFlowFieldFromChk(FlowField& init_state);
        void writeMyPlotFile(int step, amrex::Real time, const FlowField& state, const amrex::BoxArray ba, const amrex::DistributionMapping dm, const amrex::Geometry& geom);
        void writeKEData(int step, amrex::Real time, const ProjectionWorkspace& workspace);
        void initializeWriteKEData(int step, amrex::Real time, const ProjectionWorkspace& workspace);
        

};

#endif