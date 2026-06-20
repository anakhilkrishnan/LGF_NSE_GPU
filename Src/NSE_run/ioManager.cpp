#include <IOManager.H>

IOManager::IOManager(const SimConfig& config) : cfg(config)
{
    better_dir = "";
}

// PENDING: Write Chk and Plt data for KE as well!!

void IOManager::writeMyChkFile(bool writeMainChk, int step, amrex::Real time, const FlowField& state)
{
    // PENDING: Modify to write checkpoints for AMR data

    // writing to either main or alt checkpoint
    std::string cur_chkdir = writeMainChk ? cfg.chk_dir : cfg.altchk_dir;

    // variables for filenames to be used
    std::string status_file = cur_chkdir + "/status.dat";
    std::string final_write_dir = cur_chkdir + cfg.chk_prefix;
    std::string temp_write_dir= final_write_dir + "_temp";

    amrex::Print() << "Writing checkpoint data to: " << cur_chkdir << "\n";

    // creating status.dat file with format: step \t time \t status
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        amrex::UtilCreateDirectory(cur_chkdir, 0755); 
        std::ofstream ofs(status_file, std::ios::out | std::ios::trunc);
        ofs << step << "\t" << time << "\t0\n"; // setting status to 0 while writing
        ofs.close();
    }

    // building temp dir and header
    amrex::PreBuildDirectorHierarchy(temp_write_dir, "Level_", 1, true);

    // populating plaintext header file
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        std::string HeaderFileName(temp_write_dir + "/Header");
        std::ofstream HeaderFile(HeaderFileName, std::ofstream::out | std::ofstream::trunc);

        HeaderFile.precision(17);
        HeaderFile << "LGF_NSE_Checkpoint\n" << step << "\n" << time << "\n";
        state.getPres().boxArray().writeOn(HeaderFile);
        HeaderFile << "\n";
        HeaderFile.close();
    }

    // copying MultiFabs of importance (those that need previous time step data)
    amrex::VisMF::Write(state.getVel(0), amrex::MultiFabFileFullPrefix(0, temp_write_dir, "Level_", "vel_x"));
#if AMREX_SPACEDIM >= 2
    amrex::VisMF::Write(state.getVel(1), amrex::MultiFabFileFullPrefix(0, temp_write_dir, "Level_", "vel_y"));
#endif
#if AMREX_SPACEDIM == 3
    amrex::VisMF::Write(state.getVel(2), amrex::MultiFabFileFullPrefix(0, temp_write_dir, "Level_", "vel_z"));
#endif
    amrex::VisMF::Write(state.getPres(), amrex::MultiFabFileFullPrefix(0, temp_write_dir, "Level_", "pres"));

    // renaming and deleting older variants
    amrex::ParallelDescriptor::Barrier();

    if (amrex::ParallelDescriptor::IOProcessor())
    {

        if (amrex::FileSystem::Exists(final_write_dir))
        {
            amrex::FileSystem::RemoveAll(final_write_dir);
        }

        std::rename(temp_write_dir.c_str(), final_write_dir.c_str());
        
        std::ofstream ofs(status_file, std::ios::out | std::ios::trunc);
        ofs << step << "\t" << time << "\t1\n"; // Flag as Safe
        ofs.close();
    }

    amrex::Print() << "Checkpoint data safely written to: " << cur_chkdir << "\n";
}

// directory to compare main and alt checkpoints
void IOManager::whichChkDirBetter()
{
    ChkStatus main_status = readCheckpointStatus(cfg.chk_dir);
    ChkStatus alt_status = readCheckpointStatus(cfg.altchk_dir);

    // checking for most recent checkpoint, provided both are safe
    if (main_status.is_safe == 1 && alt_status.is_safe == 1) 
    {
        better_dir = (main_status.step > alt_status.step) ? cfg.chk_dir : cfg.altchk_dir;
    }
    else if (main_status.is_safe == 1)
    {
        better_dir = cfg.chk_dir;
    }
    else if (alt_status.is_safe == 1)
    {
        better_dir = cfg.altchk_dir;
    }
    else
    {
        amrex::Abort("Restart Failed: No safe checkpoint found!");
    }
}

// reading BA data from checkpoint
void IOManager::initializeBAFromChk(int& step, amrex::Real& time, amrex::BoxArray& ba)
{
    whichChkDirBetter();
    std::string restart_dir = better_dir + cfg.chk_prefix;
    amrex::Print() << "Restarting from: " << restart_dir << "\n";

    std::string HeaderFileName(restart_dir + "/Header");
    amrex::Vector<char> fileCharPtr;
    amrex::ParallelDescriptor::ReadAndBcastFile(HeaderFileName, fileCharPtr);
    std::string fileCharPtrString(fileCharPtr.dataPtr());
    std::istringstream is(fileCharPtrString, std::istringstream::in);

    std::string line;
    std::getline(is, line); // skipping heading "LGF_NSE_Checkpoint"
    is >> step;
    is >> time;

    // extracting BoxArray from chk file
    ba.readFrom(is);
}

void IOManager::initializeFlowFieldFromChk(FlowField& init_state)
{
    std::string restart_dir = better_dir + cfg.chk_prefix;

    // Read the native MultiFabs into the newly sized FlowField
    amrex::VisMF::Read(init_state.getVel(0), amrex::MultiFabFileFullPrefix(0, restart_dir, "Level_", "vel_x"));
#if AMREX_SPACEDIM >= 2
    amrex::VisMF::Read(init_state.getVel(1), amrex::MultiFabFileFullPrefix(0, restart_dir, "Level_", "vel_y"));
#endif
#if AMREX_SPACEDIM == 3
    amrex::VisMF::Read(init_state.getVel(2), amrex::MultiFabFileFullPrefix(0, restart_dir, "Level_", "vel_z"));
#endif
    amrex::VisMF::Read(init_state.getPres(), amrex::MultiFabFileFullPrefix(0, restart_dir, "Level_", "pres"));

    amrex::Print() << "Restarted from: " << restart_dir << "\n";
}

void IOManager::writeMyPlotFile(int step, amrex::Real time, const FlowField& state, const amrex::BoxArray ba, const amrex::DistributionMapping dm, const amrex::Geometry& geom)
{
    // checking total components for plotfile
    int ncomp_vort = (AMREX_SPACEDIM == 2) ? 1 : 3;
    int ncomp_plot = AMREX_SPACEDIM + ncomp_vort + 4 ;

    // building a multiFab with n dim + 2 components for plotting
   amrex::MultiFab plotFab(ba, dm, ncomp_plot, 0);

   // converting face-centered data to cell-centered data
    #if AMREX_SPACEDIM == 1
        amrex::average_face_to_cellcenter(plotFab, 0, amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>{&state.getVel(0)});
    #elif AMREX_SPACEDIM == 2
        amrex::average_face_to_cellcenter(plotFab, 0, amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>{&state.getVel(0), &state.getVel(1)});
    #elif AMREX_SPACEDIM == 3
        amrex::average_face_to_cellcenter(plotFab, 0, amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>{&state.getVel(0), &state.getVel(1), &state.getVel(2)});
    #endif
    
    amrex::MultiFab::Copy(plotFab, state.getPres(), 0, AMREX_SPACEDIM, 1, 0); 
    amrex::MultiFab::Copy(plotFab, state.getTagRegion(), 0, AMREX_SPACEDIM + 1, 1, 0);
    amrex::MultiFab::Copy(plotFab, state.getDivU(), 0, AMREX_SPACEDIM + 2, 1, 0);
    amrex::MultiFab::Copy(plotFab, state.getDivUAtEnd(), 0, AMREX_SPACEDIM + 3, 1, 0);
    amrex::MultiFab::Copy(plotFab, computeCellCenteredVorticity(state), 0, AMREX_SPACEDIM + 4, ncomp_vort, 0);


    // exporting the names of the MultiFabs
    amrex::Vector<std::string> varnames = {AMREX_D_DECL("x_velocity", "y_velocity", "z_velocity"), "pressure", "active_box_tag", "divU", "divUAtEnd"};
    #if AMREX_SPACEDIM == 2
        varnames.push_back("z_vorticity");
    #elif AMREX_SPACEDIM == 3
        varnames.push_back("x_vorticity");
        varnames.push_back("y_vorticity");
        varnames.push_back("z_vorticity");
    #endif

    // writing a simple plotfile
    const std::string& plotfile_name = amrex::Concatenate((cfg.plot_dir + cfg.plot_prefix), step, 5);
    amrex::Print() << "Writing plotfile to: " << plotfile_name << "\n";
    WriteSingleLevelPlotfile(plotfile_name, plotFab, varnames, geom, time, step);
    amrex::Print() << "Plotfile written to: " << plotfile_name << "\n";
}

void IOManager::writeKEData(int step, amrex::Real time, const ProjectionWorkspace& workspace)
{   
    std::string write_dir = cfg.plot_dir + cfg.kedata_prefix;

    if (amrex::ParallelDescriptor::IOProcessor())
    {
        std::ofstream ofs(write_dir, std::ios::out | std::ios::app);
        ofs.precision(17);
 
        ofs << step << "\t" << time;
 
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            ofs << "\t" << workspace.global_kecomp_dir[idim]
                << "\t" << workspace.global_kecomp[idim]
                << "\t" << workspace.global_kecomp_err[idim];
        }
 
        ofs << "\n";
        ofs.close();
    }


}

void IOManager::initializeWriteKEData(int step, amrex::Real time, const ProjectionWorkspace& workspace)
{
    std::string write_dir = cfg.plot_dir + "/" + cfg.kedata_prefix;
 
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        amrex::UtilCreateDirectory(cfg.plot_dir, 0755);
 
        std::ofstream ofs(write_dir, std::ios::out | std::ios::trunc);
 
        ofs << "Step\tTime";
 
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            ofs << "\ttotalKE_dir_comp" << idim
                << "\ttotalKE_evol_comp" << idim
                << "\ttotalKE_err_comp" << idim;
        }
 
        ofs << "\n";
        ofs.close();
    }
 
    amrex::ParallelDescriptor::Barrier();
}