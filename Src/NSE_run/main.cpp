#include <MyFunctions.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    amrex::Print() << "Launching LGF-NSE solver..." << "\n";
    extendedMain();

    amrex::Finalize();
    return 0;
}

void extendedMain()
{
    BL_PROFILE("extendedMain()");

    auto overall_start_time = amrex::second();

    // creating IO config struct and reading inputs using ParmParse
    IOConfig io_cfg;
    io_cfg.readInputs();

    // creating IO object for plotting/data writing/checkpoint handling etc
    IOManager io(io_cfg);
    
    // creating Solver config struct and reading inputs using ParmParse
    SolverConfig sol_cfg;
    sol_cfg.readInputs();

    // creating domain handler object for BoxArray management, tagging
    // and updating of field variables
    DomainManager dmgr(sol_cfg);
    
    // creating timestepping variables beforehand
    amrex::Real time = 0.0;
    int step = 0;
    amrex::Real dt = 0.0;

    // PENDING: enable checkpoint restarts again for adaptive domain setup
    
    // create a coarse vorticity MultiFab with initial conditions, perform tagging,
    // extract the tagged boxes, refine and generate geom, ba, dm.
    dmgr.initializeBoxArray();

    // create flow field object on updated BoxArray that is fine and restricted
    FlowField state_n(dmgr.getGeom(), dmgr.getBoxArr(), dmgr.getDistMap(), sol_cfg);

    // create solver object on updated BoxArray that is fine and restricted
    ProjectionWorkspace workspace(dmgr.getGeom(), dmgr.getBoxArr(), dmgr.getDistMap(), sol_cfg);

    // initialize velocity data in state_n
    initializeVelField(state_n);
    state_n.setBoundary();

    // initialize pressure based on div.(NSE) at initial conditions
    workspace.initializePresField(state_n);

    // initialize kinetic energy field
    workspace.computeKEFromState(state_n);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        amrex::MultiFab::Copy(state_n.getKEComp(idim), workspace.kecomp_dir[idim], 0, 0, state_n.getKEComp(idim).nComp(), 0);
    }

    // fill ghost cells and apply BCs
    state_n.setBoundary();
    
    // plotting initial conditions
    if (io_cfg.write_plot && step == 0)
    {
        BL_PROFILE("<IO> Initial Plot()");
        io.writeMyPlotFile(step, time, state_n, state_n.getGeom(), state_n.getBoxArr(), state_n.getDistMap());

    }

    // logging initial kinetic energy data
    if (!cfg.start_from_chk && cfg.write_kedata)
    {
        // initialize kinetic_energy.dat
        io.initializeWriteKEData(step, time, workspace);
    }

    // switch for main and alt chk files
    bool writeMainChk = true;

    // tracking solver initialization time, from the moment 
    auto init_stop_time = amrex::second();
    auto init_duration = init_stop_time - overall_start_time;
    amrex::Print() << "Step: " << step << " | Time: " << time << " | dt: " << dt 
                    << " | WallTime: " << (init_duration) << "s | divU_star_max: " << workspace.divU_max_norm 
                    << " | divU_max: " << workspace.divU_at_end_max_norm << "\n";

    // timestepping logic begins
    while(time < cfg.t_stop && step < cfg.max_steps)
    {
        auto step_start_time = amrex::second();
        
        dt = workspace.computeDt(state_n, cfg.cfl, cfg.Re);

        // perform KEP check and write data
        if (step % cfg.kedata_int == 0 && cfg.write_kedata)
        {
            workspace.compareKE(state_n);
            io.writeKEData(step, time, workspace);
        }

        // advance time using RK for time, KEP Morinishi for space and LGF for
        // pressure poisson
        workspace.advanceTimeStep(state_n, dt, cfg.Re, cfg.rk_order, cfg.source_tag_thresh);

        // update counters
        time += dt;
        step++;

        //  plot in specified intervals
        if (step % cfg.plot_int == 0 && cfg.write_plot)
        {
            BL_PROFILE("<IO> Interval Plot()");
            io.writeMyPlotFile(step, time, state_n, ba, dm, geom);
        }

        // write checkpoints in specified intervals, write fallback 'alt' checkpoints
        // 5 steps after specified interval
        if ((step % cfg.chk_int == 0 || (step - 5) % cfg.chk_int == 0) && cfg.write_chk)
        {
            BL_PROFILE("<IO> Interval Checkpoint()");
            io.writeMyChkFile(writeMainChk, step, time, state_n);
            writeMainChk = !writeMainChk;
        }

        // track duration of timestep
        auto step_stop_time = amrex::second();
        auto step_duration = step_stop_time - step_start_time;

        // print to terminal each timestep
        amrex::Print() << "Step: " << step << " | Time: " << time << " | dt: " << dt 
                        << " | WallTime: " << (step_duration) << "s | divU_star_max: " << workspace.divU_max_norm 
                        << " | divU_max: " << workspace.divU_at_end_max_norm << "\n";
    }

    // perform KEP check and write data for the last time
    workspace.compareKE(state_n);
    io.writeKEData(step, time, workspace);

    // overall code walltime tracking
    auto overall_end_time = amrex::second();
    auto elapsed_time = overall_end_time - overall_start_time;

    // making copies to track slowest and fastest processor
    amrex::Real max_time = elapsed_time;
    amrex::Real min_time = elapsed_time;

    // performing a reduction over all the processors to track the slowest and
    // fastest MPI rank
    const int IOProc = amrex::ParallelDescriptor::IOProcessorNumber();
    amrex::ParallelDescriptor::ReduceRealMax(max_time, IOProc);
    amrex::ParallelDescriptor::ReduceRealMin(min_time, IOProc);

    amrex::Print() << "Max compute time (Slowest Rank): " << max_time << " s\n"
                   << "Min compute time (Fastest Rank): " << min_time << " s\n"
                   << "Time spread (Load Imbalance)   : " << (max_time - min_time) << " s\n";
}   

