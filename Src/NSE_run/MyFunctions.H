#ifndef MYFUNCTIONS_H_
#define MYFUNCTIONS_H_

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_BCRec.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_BLProfiler.H>
#include <string>

#include <SimConfig.H>
#include <IOManager.H>
#include <FlowField.H>
#include <ProjectionWorkspace.H>


using namespace amrex;

void extendedMain();
void initializeVelField(FlowField& init_state);

#endif