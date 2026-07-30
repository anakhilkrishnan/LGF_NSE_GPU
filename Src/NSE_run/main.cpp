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
    
    // creating io config object and reading inputs
    IOConfig io_cfg;
    io_cfg.readInputs(); 

    // initializing io manager object
    IOManager io(io_cfg);

    // creating solver configuration object and reading inputs
    SolverConfig sol_cfg;
    sol_cfg.readInputs();

    // creating timestepping variables beforehand
    amrex::Real time = 0.0;
    amrex::Real dt_master = 0.0; // master, not to be confused with workspace.dt
    int step = 0;

    DomainManager dmgr(sol_cfg);
    
    // check if BoxArray comes from domain search and initialization, or checkpoint
    if (io_cfg.start_from_chk)
    {
        amrex::BoxArray chk_ba;
        amrex::BoxArray chk_supp_ba;
        io.initializeBoxArrFromChk(step, time, chk_ba, chk_supp_ba);
        dmgr.restartSnugDomain(chk_ba, chk_supp_ba);
    }
    else
    {
        // extract correct region and update geom, boxarr and distmap
        dmgr.initializeSnugDomain();
    }
    

    // create flow field object
    FlowField state_n(dmgr.getGeom(), dmgr.getBoxArr(), dmgr.getDistMap(), sol_cfg.n_comp, sol_cfg.n_ghost);
    // create solver object
    ProjectionWorkspace workspace(dmgr.getGeom(), dmgr.getBoxArr(), dmgr.getDistMap(), sol_cfg);

    if (io_cfg.start_from_chk)
    {
        io.initializeFlowFieldFromChk(state_n);
        state_n.setBoundary();

        const amrex::BoxArray restart_supp_ba = dmgr.getSuppBoxArr();
        workspace.divU_max_norm = 0.0;
        workspace.divU_at_end_max_norm = workspace.computeDivUMaxNorm(state_n);
        workspace.divU_at_end_max_norm_support = workspace.computeDivUMaxNorm(state_n, &restart_supp_ba);
    }
    else
    {
        time = sol_cfg.t_start;
        step = 0;

        // starting from initial conditions
        initializeVelField(state_n);

        // fill ghost cells and apply physical BCs
        state_n.setBoundary();

        // tag again on the fine grid to prep for the solver, don't shed outer layer here
        dmgr.computeSuppBoxArr(state_n, false);
        
        // populating pressure based on divergence of Navier-Stokes at initial conditions
        workspace.initializePresField(state_n, dmgr.getSuppBoxArr());
    }

    // populating KE comp arrays
    workspace.computeKEFromState(state_n);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        amrex::MultiFab::Copy(state_n.getKEComp(idim), workspace.kecomp_dir[idim], 0, 0, state_n.getKEComp(idim).nComp(), 0);
    }

    // fill ghost cells and apply physical BCs
    state_n.setBoundary();
    
    // plotting initial conditions
    if (io_cfg.write_plot && step == 0)
    {
        BL_PROFILE("<IO> Initial Plot()");
        io.writeMyPlotFile(0, io_cfg.plot_only_support, step, time, state_n, dmgr);

    }

    // logging initial kinetic energy data
    if (!io_cfg.start_from_chk && io_cfg.write_kedata)
    {
        // initialize kinetic_energy.dat
        io.initializeWriteKEData(step, time, workspace);
    }

    // switch for main and alt chk files
    bool writeMainChk = true;

    // tracking solver initialization time, from the moment 
    auto init_stop_time = amrex::second();
    auto init_duration = init_stop_time - overall_start_time;
    amrex::Print() << "Step: " << step << " | Time: " << time << " | dt: " << dt_master 
                    << " | WallTime: " << (init_duration) << "s | divU_star_max: " << workspace.divU_max_norm 
                    << " | divU_max: " << workspace.divU_at_end_max_norm
                    << " | divU_max_support: " << workspace.divU_at_end_max_norm_support << "\n";

    // timestepping logic begins
    while(time < sol_cfg.t_stop && step < sol_cfg.max_steps)
    {
        auto step_start_time = amrex::second();

        // perform KEP check and write data
        if (step %io_cfg.kedata_int == 0 &&io_cfg.write_kedata)
        {
            workspace.compareKE(state_n);
            io.writeKEData(step, time, workspace);
        }

        // always call computeDt() right before advanceTimeStep()
        dt_master = workspace.computeDt(state_n);

        // advance time using RK for time, KEP Morinishi for space and LGF for
        // pressure poisson
        workspace.advanceTimeStep(state_n, dt_master, dmgr.getSuppBoxArr());

        // update counters
        time += dt_master;
        step++;

        //  plot in specified intervals
        if (step % io_cfg.plot_int == 0 && io_cfg.write_plot)
        {
            BL_PROFILE("<IO> Interval Plot()");
            io.writeMyPlotFile(0, io_cfg.plot_only_support, step, time, state_n, dmgr);
        }

        // update domain based on results from timestep
        if (step % dmgr.computeRegridInterval(state_n) == 0)
        {
            auto regrid_start_time = amrex::second();
            dmgr.updateSnugDomain(state_n);
            dmgr.regridFlowFieldOntoNewSnugDomain(state_n, workspace.lgf_poisson_solver);
            workspace.regridOnto(dmgr.getGeom(), dmgr.getBoxArr(), dmgr.getDistMap());
            if (io_cfg.plot_post_regrid && step % io_cfg.plot_post_regrid_int == 0)
            {
                io.writeMyPlotFile(1, false, step, time, state_n, dmgr);
            }
            dmgr.computeSuppBoxArr(state_n);
            auto regrid_stop_time = amrex::second();
            auto regrid_duration = regrid_stop_time - regrid_start_time;

            amrex::Print() << "Regridding done at end of step " << step
                            << " | WallTime: " << (regrid_duration) << "s"
                            AMREX_D_TERM(<< " | u-refresh err: " << dmgr.refresh_err_max_norm[0],
                                        << " | v-refresh err: " << dmgr.refresh_err_max_norm[1],
                                        << " | w-refresh err: " << dmgr.refresh_err_max_norm[2])
                            << "\n";
        }

        // write checkpoints in specified intervals, write fallback 'alt' checkpoints
        // 5 steps after specified interval
        if ((step %io_cfg.chk_int == 0 || (step - 5) %io_cfg.chk_int == 0) && io_cfg.write_chk)
        {
            BL_PROFILE("<IO> Interval Checkpoint()");
            io.writeMyChkFile(writeMainChk, step, time, state_n, dmgr.getSuppBoxArr());
            writeMainChk = !writeMainChk;
        }

        // track duration of timestep
        auto step_stop_time = amrex::second();
        auto step_duration = step_stop_time - step_start_time;

        // print to terminal each timestep
        amrex::Print() << "Step: " << step << " | Time: " << time << " | dt: " << dt_master 
                        << " | WallTime: " << (step_duration) << "s | divU_star_max: " << workspace.divU_max_norm 
                        << " | divU_max: " << workspace.divU_at_end_max_norm
                        << " | divU_max_support: " << workspace.divU_at_end_max_norm_support << "\n";
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

