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

    // creating simulation configuration object and reading inputs
    SimConfig cfg;
    cfg.readInputs();
    
    // creating input/output object 
    IOManager io(cfg);

    // creating timestepping variables beforehand
    amrex::Real time;
    int step = 0;
    amrex::Real dt;

    // creating domain data objects
    amrex::IntVect dom_lo_iv(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect dom_hi_iv(AMREX_D_DECL(cfg.n_cell-1, cfg.n_cell-1, cfg.n_cell-1));
    amrex::Box domain(dom_lo_iv, dom_hi_iv);

    amrex::BoxArray ba;
    // boxarray taken from ChkPoints if needed
    if (cfg.start_from_chk)
    {
        io.initializeBAFromChk(step, time, ba);
    }
    else
    {
        ba.define(domain);
        ba.maxSize(cfg.max_grid_size);
    }
    
    amrex::DistributionMapping dm(ba);

    amrex::RealBox real_box(cfg.dom_lo, cfg.dom_hi);
    amrex::Vector<int> is_periodic(AMREX_SPACEDIM, 0); // infinite domain using zero-grad BC
    amrex::Geometry geom(domain, &real_box, amrex::CoordSys::cartesian, is_periodic.data());

    // create flow field object
    FlowField state_n(geom, ba, dm, cfg.n_comp, cfg.n_ghost);
    // create solver object
    ProjectionWorkspace workspace(geom, ba, dm, cfg.n_comp, cfg.n_ghost);

    if (cfg.start_from_chk)
    {
        io.initializeFlowFieldFromChk(state_n);
    }
    else 
    {
        // starting from initial conditions
        initializeVelField(state_n);

        // populating pressure based on divergence of Navier-Stokes at initial conditions
        workspace.initializePresField(state_n, cfg.Re, cfg.source_tag_thresh, cfg.n_chebyshev, cfg.n_lookup);

        // populating KE comp arrays
        auto init_kecomp_dir = computeKEFromState(state_n);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            amrex::MultiFab::Copy(state_n.getKEComp(idim), init_kecomp_dir[idim], 0, 0, state_n.getKEComp(idim).nComp(), 0);
        }

        time = cfg.t_start;
        step = 0;
    }
    
    // fill ghost cells and apply physical BCs
    state_n.setBoundary();
    
    // plotting initial conditions
    if (cfg.write_plot && step == 0)
    {
        BL_PROFILE("<IO> Initial Plot()");
        io.writeMyPlotFile(step, time, state_n, ba, dm, geom);

    }

    // logging initial kinetic energy data
    if (!cfg.start_from_chk && cfg.write_kedata)
    {
        // initialize kinetic_energy.dat
        io.initializeWriteKEData(step, time, workspace);
    }

    // switch for main and alt chk files
    bool writeMainChk = true;

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
        workspace.advanceTimeStep(state_n, dt, cfg.Re, cfg.rk_order, cfg.source_tag_thresh, cfg.n_chebyshev, cfg.n_lookup);

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
                       << " | WallTime: " << (step_duration) << "s | divU_max: " << workspace.divU_max_norm << "\n";
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

