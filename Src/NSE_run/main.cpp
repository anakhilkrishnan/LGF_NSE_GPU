#include <MyFunctions.H>

#include <memory>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    amrex::Print() << "Launching LGF-NSE solver..." << "\n";
#ifdef NSE_CASE_NAME
    amrex::Print() << "Case: " << NSE_CASE_NAME << "\n";
#endif
    extendedMain();

    amrex::Finalize();
    return 0;
}

// ends the per-step status line: appends the CFL number of the psi transport when
// tracking, and warns when it nears the stability limit of the upwind scheme with
// this three-stage tableau (about 1.4 for the linear fifth-order stencil)
void printInterfaceCFL(const InterfaceTracker* tracker, const FlowField& state, amrex::Real time, amrex::Real dt)
{
    if (tracker == nullptr) { amrex::Print() << "\n"; return; }

    const amrex::Real psi_cfl = tracker->computeCFL(state.getVelArr(), freestreamVelocity(time), dt);
    amrex::Print() << " | psi_cfl: " << psi_cfl << "\n";
    if (psi_cfl > 1.0)
    {
        amrex::Print() << "WARNING: psi_cfl = " << psi_cfl << " > 1, near the stability limit of the"
                          " interface transport; reduce set_dt_base\n";
    }
}

void extendedMain()
{
    BL_PROFILE("extendedMain()");

    amrex::Gpu::streamSynchronize();
    auto overall_start_time = amrex::second();

    // creating solver configuration object and reading inputs
    SolverConfig sol_cfg;
    sol_cfg.readInputs();

    DomainManager dmgr(sol_cfg);
    
    // creating io config object and reading inputs
    IOConfig io_cfg;
    io_cfg.readInputs(sol_cfg.refine_ratio); 

    // force VisMF to use only n_files_per_MF number of files for each plotdir
    amrex::VisMF::SetNOutFiles(io_cfg.n_files_per_MF);

    // initializing io manager object
    IOManager io(io_cfg, sol_cfg);

    // creating timestepping variables beforehand
    amrex::Real time_keeper = 0.0;
    amrex::Real dt_master = sol_cfg.set_dt; // master, not to be confused with workspace.dt
    int step_keeper = 0;

    // checkpoint metadata
    ChkMetadata chk;
    int transfer_ratio = 1; // ratio of cfg.refine_ratio to chk.refine_ratio
    
    // check if BoxArray comes from domain search and initialization, or checkpoint
    if (io_cfg.start_from_chk)
    {
        chk = io.readChkMetadata();
        transfer_ratio = restartTransferRatio(chk, sol_cfg);

        // psi cannot be refined yet: fail before any expensive restart work
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!sol_cfg.track_interface || transfer_ratio == 1,
            "restart: interface tracking cannot restart at a finer resolution yet (psi refinement is not"
            " implemented); restart at the checkpoint's refine_ratio");

        step_keeper = chk.step * transfer_ratio;   // steps of the current dt since t_start
        time_keeper = chk.time;

        // transfer ratio determins if refinement happens
        dmgr.restartSnugDomain(chk.ba, chk.supp_ba, transfer_ratio);
    }
    else
    {
        // extract correct region and update geom, boxarrs and distmap
        dmgr.initializeSnugDomain(sol_cfg.t_start);
    }

    // create flow field object
    FlowField state_n(dmgr.getGeom(), dmgr.getBoxArr(), dmgr.getDistMap(), sol_cfg.n_comp, sol_cfg.n_ghost);
    // create solver object
    ProjectionWorkspace workspace(dmgr.getGeom(), dmgr.getBoxArr(), dmgr.getDistMap(), sol_cfg);
    // create interface tracker (optional): psi lives on the same snug domain as the flow
    std::unique_ptr<InterfaceTracker> tracker;
    if (sol_cfg.track_interface)
    {
        tracker = std::make_unique<InterfaceTracker>(dmgr.getGeom(), dmgr.getBoxArr(), dmgr.getDistMap(), sol_cfg);
    }

    if (io_cfg.start_from_chk)
    {
        if (transfer_ratio == 1)
        {
            // no refinement needed, directly read chkdata and store in state_n
            io.readChkData(chk, state_n);
        }
        else
        {
            FlowField chk_state(chk.geom, chk.ba, amrex::DistributionMapping(chk.ba), sol_cfg.n_comp, sol_cfg.n_ghost);
            io.readChkData(chk, chk_state);
            chk_state.setBoundary();
            dmgr.refineVelField(state_n, chk_state, transfer_ratio, workspace.lgf_poisson_solver);
        }

        // initialize pressure fields
        workspace.initializePresField(state_n, dmgr.getSuppBoxArr(), time_keeper);
        state_n.setBoundary();

        workspace.divU_max_norm = 0.0;
        workspace.divU_at_end_max_norm = computeDivUMaxNorm(state_n);
        const amrex::BoxArray restart_supp_ba = dmgr.getSuppBoxArr(); // initialized to send as pointer to computeDivUMaxNorm
        workspace.divU_at_end_max_norm_support = computeDivUMaxNorm(state_n, &restart_supp_ba);

        // psi from the checkpoint (transfer ratio 1 guaranteed above)
        if (tracker) { io.readChkInterface(chk, *tracker); }
    }
    else
    {
        time_keeper = sol_cfg.t_start;
        step_keeper = 0;

        // populate initial conditions
        dmgr.initializeVelField(state_n, workspace.lgf_poisson_solver);
        
        // populating pressure based on divergence of Navier-Stokes at initial conditions
        workspace.initializePresField(state_n, dmgr.getSuppBoxArr(), time_keeper);
        state_n.setBoundary();

        // populate psi from the active case in InitialSDF.H
        if (tracker) { tracker->initializeSDF(); }
    }
    
    io.writeRingDiagnostics(step_keeper, time_keeper, dt_master, sol_cfg.invRe, state_n, dmgr);

    // plotting initial conditions
    if (step_keeper == 0)
    {
        BL_PROFILE("<IO> Initial Plot()");
        if (io_cfg.write_plot) { io.writePlot(PlotKind::Lean,  step_keeper, time_keeper, state_n, dmgr, tracker.get()); }
        if (io_cfg.plot_dense) { io.writePlot(PlotKind::Dense, step_keeper, time_keeper, state_n, dmgr, tracker.get()); }
    }

    // switch for main and alt chk files
    bool writeMainChk = true;

    // tracking solver initialization time, from the moment 
    amrex::Gpu::streamSynchronize();
    auto init_stop_time = amrex::second();
    auto init_duration = init_stop_time - overall_start_time;
    amrex::Print() << "Step: " << step_keeper << " | Time: " << time_keeper << " | dt: " << dt_master 
                    << " | WallTime: " << (init_duration) << "s | divU_star_max: " << workspace.divU_max_norm 
                    << " | divU_max: " << workspace.divU_at_end_max_norm
                    << " | divU_max_support: " << workspace.divU_at_end_max_norm_support;
    printInterfaceCFL(tracker.get(), state_n, time_keeper, dt_master);

    // check if max_steps is to be used or not
    const bool use_max_steps = (sol_cfg.max_steps >= 0);
    const bool use_t_stop    = (sol_cfg.t_stop > sol_cfg.t_start);  

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(use_max_steps || use_t_stop,
        "No stopping criterion: set max_steps >= 0 or t_stop > t_start");

    // timestepping logic begins
    while ((!use_max_steps || step_keeper < sol_cfg.max_steps) &&
            (!use_t_stop    || time_keeper < sol_cfg.t_stop))
    {
        amrex::Gpu::streamSynchronize();
        auto step_start_time = amrex::second();

        // IF-HERK integrator
        workspace.advanceTimeStep(state_n, 0.0, dmgr.getSuppBoxArr(), time_keeper, tracker.get());   // time at start of step

        // update counters
        time_keeper += dt_master;
        step_keeper++;

        // snug domain management after timestep
        amrex::Gpu::streamSynchronize();
        auto regrid_start_time = amrex::second();

        dmgr.checkAndUpdateSnugDomain(state_n, time_keeper);      // time of the new state

        // regrid pair: only on steps where the grid changes (the flag is known only from here on)
        const bool regrid_plots = io_cfg.plot_regrid && dmgr.did_snug_domain_change;
        if (regrid_plots) { io.writePlot(PlotKind::RegridPre, step_keeper, time_keeper, state_n, dmgr, tracker.get()); }

        if (dmgr.did_snug_domain_change)
        {
            workspace.regridOnto(dmgr.getGeom(), dmgr.getBoxArr(), dmgr.getDistMap());
            if (tracker) { tracker->regrid(dmgr.getGeom(), dmgr.getBoxArr(), dmgr.getDistMap()); }
        }

        dmgr.checkAndRefreshVelocity(state_n, workspace.lgf_poisson_solver, step_keeper);

        if (regrid_plots) { io.writePlot(PlotKind::RegridPost, step_keeper, time_keeper, state_n, dmgr, tracker.get()); }

        // solver time and output
        amrex::Gpu::streamSynchronize();
        auto regrid_stop_time = amrex::second();
        auto regrid_duration = regrid_stop_time - regrid_start_time;

        amrex::Print() << "Snug domain management at end of step " << step_keeper
                        << " complete | WallTime: " << (regrid_duration) << "s" << "\n";

        //  plot in specified intervals
        if (io_cfg.write_plot && step_keeper % io_cfg.plot_int == 0)
        {
            io.writePlot(PlotKind::Lean, step_keeper, time_keeper, state_n, dmgr, tracker.get());
        }
        if (io_cfg.plot_dense && step_keeper % io_cfg.plot_dense_int == 0)
        {
            io.writePlot(PlotKind::Dense, step_keeper, time_keeper, state_n, dmgr, tracker.get());
        }

        // refresh flag after all dependent processes are complete
        dmgr.did_snug_domain_change = false;

        // write out diagnostics for ring simulations
        io.writeRingDiagnostics(step_keeper, time_keeper, dt_master, sol_cfg.invRe, state_n, dmgr);

        // write checkpoints in specified intervals, write fallback 'alt' checkpoints
        // 5 steps after specified interval
        if ((step_keeper %io_cfg.chk_int == 0 || (step_keeper - 5) %io_cfg.chk_int == 0) && io_cfg.write_chk)
        {
            BL_PROFILE("<IO> Interval Checkpoint()");
            io.writeMyChkFile(writeMainChk, step_keeper, time_keeper, state_n, dmgr.getSuppBoxArr(), tracker.get());
            writeMainChk = !writeMainChk;
        }

        // track duration of timestep
        amrex::Gpu::streamSynchronize();
        auto step_stop_time = amrex::second();
        auto step_duration = step_stop_time - step_start_time;

        // print to terminal each timestep
        amrex::Print() << "Step: " << step_keeper << " | Time: " << time_keeper << " | dt: " << dt_master 
                        << " | WallTime: " << (step_duration) << "s | divU_star_max: " << workspace.divU_max_norm 
                        << " | divU_max: " << workspace.divU_at_end_max_norm
                        << " | divU_max_support: " << workspace.divU_at_end_max_norm_support;
        printInterfaceCFL(tracker.get(), state_n, time_keeper, dt_master);
    }

    // overall code walltime tracking
    amrex::Gpu::streamSynchronize();
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