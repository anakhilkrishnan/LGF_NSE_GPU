#include <IOManager.H>

IOManager::IOManager(const IOConfig& config) : cfg(config)
{
    better_dir = "";
}

// PENDING: Write Chk and Plt data for KE as well!!

void IOManager::writeMyChkFile(bool writeMainChk, int step, amrex::Real time, const FlowField& state, const amrex::BoxArray& supp_ba)
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
        supp_ba.writeOn(HeaderFile);
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
void IOManager::initializeBoxArrFromChk(int& step, amrex::Real& time, amrex::BoxArray& chk_ba, amrex::BoxArray& chk_supp_ba)
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
    chk_ba.readFrom(is);

    // extracting supp_ba from chk file
    chk_supp_ba.readFrom(is);
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

void IOManager::writeMyPlotFile(int diag_num, bool restrictToSupport, int step, amrex::Real time,
                                const FlowField& state, const DomainManager& dom_mgr)
{
    // Single consolidated plot writer. Two orthogonal switches:
    //   restrictToSupport : if true, the written grid is intersect(ba, supp_ba)
    //                       (buffer excluded); if false, the full ba is written.
    //   diag_num          : filename disambiguator so the SAME step can be
    //                       plotted at different points in the solver flow
    //                       without file collision. diag_num == 0 is the
    //                       mainstream plot (no suffix); diag_num >= 1 appends
    //                       an "xdiagNN" suffix.
    // All grid and field data is pulled from dom_mgr, so the plotted quantities
    // are whatever the domain manager currently holds (freshly valid after the
    // last computeSuppBoxArr / vor2vel).
 
    // extracting grid from the domain manager
    amrex::BoxArray ba = dom_mgr.getBoxArr();
    amrex::DistributionMapping dm = dom_mgr.getDistMap();
    amrex::Geometry geom = dom_mgr.getGeom();
    const amrex::BoxArray& supp_ba = dom_mgr.getSuppBoxArr();
 
    // construct MultiFab for plotting support region (all-ones on a restricted
    // grid; genuinely informative only when the full buffer is written)
    amrex::MultiFab tagRegion(ba, dm, 1, 0);
    for (MFIter mfi(tagRegion); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
        tagRegion[mfi].setVal<RunOn::Device>(supp_ba.intersects(bx) ? 1.0 : 0.0);
    }
 
    // checking total components for plotfile
    int ncomp_vort = (AMREX_SPACEDIM == 2) ? 1 : 3;
    const int ncomp_cc = (2 * AMREX_SPACEDIM) + 5;
    const int ncomp_nd = (2 * ncomp_vort);
 
    // building full-grid assembly fabs (support + buffer). face->cc averaging
    // requires the destination cc fab to share the source (full ba) layout, so
    // assembly always happens on the full grid regardless of restrictToSupport.
    amrex::MultiFab plotFab_cc_full(ba, dm, ncomp_cc, 0);
    amrex::MultiFab plotFab_nd_full(amrex::convert(ba, amrex::IntVect::TheNodeVector()), dm, ncomp_nd, 0);
    plotFab_cc_full.setVal(0.0);
    plotFab_nd_full.setVal(0.0);
 
    // create an array of pointers to the face-centered MultiFabs
    amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM> face_vels;
    amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM> face_errs;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        face_vels[d] = &state.getVel(d);
        face_errs[d] = &dom_mgr.getVelRefreshErrComp(d);
    }
 
    // averages all dimensions simultaneously into plotFab starting at component 0
    amrex::average_face_to_cellcenter(plotFab_cc_full, 0, face_vels);
    amrex::average_face_to_cellcenter(plotFab_cc_full, AMREX_SPACEDIM, face_errs);
 
    plotFab_cc_full.ParallelCopy(state.getPres(), 0, (2 * AMREX_SPACEDIM), 1, 0, 0);
    plotFab_cc_full.ParallelCopy(tagRegion, 0, (2 * AMREX_SPACEDIM) + 1, 1, 0, 0);
    plotFab_cc_full.ParallelCopy(computePlotDivU(state), 0, (2 * AMREX_SPACEDIM) + 2, 1, 0, 0);
    plotFab_cc_full.ParallelCopy(dom_mgr.getDivN(), 0, (2 * AMREX_SPACEDIM) + 3, 1, 0, 0);
    plotFab_cc_full.ParallelCopy(dom_mgr.getVort(), 0, (2 * AMREX_SPACEDIM) + 4, 1, 0, 0);
 
    plotFab_nd_full.ParallelCopy(dom_mgr.getPsi(), 0, 0, ncomp_vort, 0, 0);
    plotFab_nd_full.ParallelCopy(computeNodalVorticity(state), 0, ncomp_vort, ncomp_vort, 0, 0);
 
    // choose the grid actually written: restricted support grid, or full ba.
    // Bind references so the write section below is agnostic to which was chosen.
    amrex::BoxArray plot_ba;
    amrex::DistributionMapping plot_dm;
    amrex::MultiFab plotFab_cc_restricted;
    amrex::MultiFab plotFab_nd_restricted;
 
    if (restrictToSupport)
    {
        plot_ba = amrex::intersect(ba, supp_ba);   // cell-centered support grid
        plot_dm = amrex::DistributionMapping(plot_ba);
 
        plotFab_cc_restricted.define(plot_ba, plot_dm, ncomp_cc, 0);
        plotFab_nd_restricted.define(amrex::convert(plot_ba, amrex::IntVect::TheNodeVector()), plot_dm, ncomp_nd, 0);
        plotFab_cc_restricted.setVal(0.0);
        plotFab_nd_restricted.setVal(0.0);
 
        plotFab_cc_restricted.ParallelCopy(plotFab_cc_full, 0, 0, ncomp_cc, 0, 0);
        plotFab_nd_restricted.ParallelCopy(plotFab_nd_full, 0, 0, ncomp_nd, 0, 0);
    }
 
    amrex::MultiFab& plotFab_cc = restrictToSupport ? plotFab_cc_restricted : plotFab_cc_full;
    amrex::MultiFab& plotFab_nd = restrictToSupport ? plotFab_nd_restricted : plotFab_nd_full;
 
    // exporting the names of the MultiFabs
    amrex::Vector<std::string> varnames_cc = {AMREX_D_DECL("x_velocity", "y_velocity", "z_velocity"),
                                                AMREX_D_DECL("x_vel_refr_corr", "y_vel_refr_corr", "z_vel_refr_corr"),
                                                "pressure", "active_box_tag", "divUAtEnd", "tag_divN", "tag_vort"};
    amrex::Vector<std::string> varnames_nd;
 
    #if AMREX_SPACEDIM == 2
        varnames_nd.push_back("z_psi");
        varnames_nd.push_back("z_vorticity");
    #elif AMREX_SPACEDIM == 3
        varnames_nd.push_back("x_psi");
        varnames_nd.push_back("y_psi");
        varnames_nd.push_back("z_psi");
        varnames_nd.push_back("x_vorticity");
        varnames_nd.push_back("y_vorticity");
        varnames_nd.push_back("z_vorticity");
    #endif
 
    // filename: diag_num == 0 is the mainstream plot (no suffix); diag_num >= 1
    // appends an xdiagNN suffix so the same step can be replotted without clash.
    const std::string diag_suffix = (diag_num == 0) ? "" : amrex::Concatenate("_xdiag", diag_num, 2);
    const std::string plotfile_name = (amrex::Concatenate((cfg.plot_dir + cfg.plot_prefix), step, 5)) + diag_suffix;
    
    amrex::Print() << "Writing plotfiles to: " << plotfile_name << "\n";
    WriteSingleLevelPlotfile((plotfile_name + "_cc"), plotFab_cc, varnames_cc, geom, time, step);
    WriteSingleLevelPlotfile((plotfile_name + "_nd"), plotFab_nd, varnames_nd, geom, time, step);
    amrex::Print() << "Plotfiles written to: " << plotfile_name << "\n";
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
    std::string write_dir = cfg.plot_dir + cfg.kedata_prefix;
 
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