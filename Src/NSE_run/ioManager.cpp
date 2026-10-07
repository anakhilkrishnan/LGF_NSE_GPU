#include <IOManager.H>
#include <FreestreamVelocity.H>

#include <limits>

void averageCellToNode (amrex::MultiFab& nd, int dcomp, const amrex::MultiFab& cc, int scomp)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(cc.nGrowVect().allGE(1), "averageCellToNode: cc needs >= 1 ghost");
    constexpr amrex::Real w = amrex::Real(1.0) / amrex::Real(1 << AMREX_SPACEDIM);

    for (amrex::MFIter mfi(nd, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& out = nd.array(mfi);
        auto const& in  = cc.const_array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
#if AMREX_SPACEDIM == 3
            out(i,j,k,dcomp) = w * ( in(i-1,j-1,k-1,scomp) + in(i,j-1,k-1,scomp) + in(i-1,j,k-1,scomp) + in(i,j,k-1,scomp)
                                   + in(i-1,j-1,k  ,scomp) + in(i,j-1,k  ,scomp) + in(i-1,j,k  ,scomp) + in(i,j,k  ,scomp) );
#else
            out(i,j,k,dcomp) = w * ( in(i-1,j-1,k,scomp) + in(i,j-1,k,scomp) + in(i-1,j,k,scomp) + in(i,j,k,scomp) );
#endif
        });
    }
}

IOManager::IOManager(const IOConfig& config) : cfg(config)
{
    better_dir = "";
}

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

// switches AMReX's FAB output format for the lifetime of this object, then restores it
struct FabFormatGuard
{
    amrex::FABio::Format saved = amrex::FArrayBox::getFormat();
    explicit FabFormatGuard (amrex::FABio::Format fmt) { amrex::FArrayBox::setFormat(fmt); }
    ~FabFormatGuard () { amrex::FArrayBox::setFormat(saved); }
};

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

    // comment below for double precision plots
    FabFormatGuard single_precision(amrex::FABio::FAB_NATIVE_32);   // plots only

    // filename: diag_num == 0 is the mainstream plot (no suffix); diag_num >= 1
    // appends an xdiagNN suffix so the same step can be replotted without clash.
    const std::string diag_suffix = (diag_num == 0) ? "" : amrex::Concatenate("_xdiag", diag_num, 2);
    const std::string plotfile_name = (amrex::Concatenate((cfg.plot_dir + cfg.plot_prefix), step, 5)) + diag_suffix;
 
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
    
    // SECTION: FULL CC PLOTTER
    // checking total components for plotfile
    const int ncomp_cc = (2 * AMREX_SPACEDIM) + 5;
 
    // building full-grid assembly fabs (support + buffer). face->cc averaging
    // requires the destination cc fab to share the source (full ba) layout, so
    // assembly always happens on the full grid regardless of restrictToSupport.
    amrex::MultiFab plotFab_cc_full(ba, dm, ncomp_cc, 0);
    plotFab_cc_full.setVal(0.0);
 
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
    plotFab_cc_full.ParallelCopy(dom_mgr.getTagDivN(), 0, (2 * AMREX_SPACEDIM) + 3, 1, 0, 0);
    plotFab_cc_full.ParallelCopy(dom_mgr.getTagVort(), 0, (2 * AMREX_SPACEDIM) + 4, 1, 0, 0);
 
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
        plotFab_cc_restricted.setVal(0.0);
 
        plotFab_cc_restricted.ParallelCopy(plotFab_cc_full, 0, 0, ncomp_cc, 0, 0);
    }
 
    amrex::MultiFab& plotFab_cc = restrictToSupport ? plotFab_cc_restricted : plotFab_cc_full;
 
    // exporting the names of the MultiFabs
    amrex::Vector<std::string> varnames_cc = {AMREX_D_DECL("x_velocity", "y_velocity", "z_velocity"),
                                                AMREX_D_DECL("x_vel_refr_corr", "y_vel_refr_corr", "z_vel_refr_corr"),
                                                "pressure", "active_box_tag", "divUAtEnd", "tag_divN", "tag_vort"};

    // SECTION: OTHER INDEXTYPE PLOTTER
    const int ncomp_nd = 1;

    amrex::MultiFab plotFab_nd_full(amrex::convert(ba, amrex::IntVect::TheNodeVector()), dm, ncomp_nd, 0);
    plotFab_nd_full.setVal(0.0);

    amrex::MultiFab vmag_cc(ba, dm, 1, 1);
    vmag_cc.setVal(0.0);
    amrex::MultiFab::Copy(vmag_cc, computeTagVorticity(state), 0, 0, 1, 0);
    vmag_cc.FillBoundary(geom.periodicity());
    
    averageCellToNode(plotFab_nd_full, 0, vmag_cc, 0);

    if (restrictToSupport)
    {
        plotFab_nd_restricted.define(amrex::convert(plot_ba, amrex::IntVect::TheNodeVector()), plot_dm, ncomp_nd, 0);
        plotFab_nd_restricted.setVal(0.0);

        plotFab_nd_restricted.ParallelCopy(plotFab_nd_full, 0, 0, ncomp_nd, 0, 0);
    }

    amrex::MultiFab& plotFab_nd = restrictToSupport ? plotFab_nd_restricted : plotFab_nd_full;

    amrex::Vector<std::string> varnames_nd;
#if AMREX_SPACEDIM == 2
    varnames_nd.push_back("vort");
#elif AMREX_SPACEDIM == 3
    varnames_nd.push_back("vort_mag");
#endif
 
    // SECTION: FINAL OUTPUT WRITER
    amrex::Print() << "Writing plotfiles to: " << plotfile_name << "\n";
    WriteSingleLevelPlotfile((plotfile_name + "_cc"), plotFab_cc, varnames_cc, geom, time, step);
    WriteSingleLevelPlotfile((plotfile_name + "_nd"), plotFab_nd, varnames_nd, geom, time, step);
    amrex::Print() << "Plotfiles written to: " << plotfile_name << "\n";
}

// GpuTuple -> std::array (for the reduction results below)
template <typename T, std::size_t... I>
std::array<amrex::Real, sizeof...(I)> tupleToArray (T const& t, std::index_sequence<I...>)
{
    return { amrex::get<I>(t)... };
}

// Lower-half sums (Saffman centroid of the ring in z < z_split, i.e. its approach speed) are
// compiled in when the case header (InitialVorticity.H) defines RING_DIAG_LOWER_HALF.

// Per-step vortex-ring diagnostics (Liska & Colonius 2016, Eq. 55), appended to
// <log_dir>/ring_diagnostics.txt:
//   E = 1/2 int |w|^2,  K = int u.(x cross w),  J = int u.w,  I = 1/2 int x cross w,
//   X = 1/2 int ((x cross w).I / |I|^2) x   (Saffman centroid; no freestream term)
// u (faces) and w (edges) are averaged to cell centres and x is the cell centre.
// Sums run over the support boxes only: the integrands are compact, no point is
// counted twice, and the zero-ghost layer at the edge of D_xsoln is skipped.
// U = dX/dt and dK/dt are left to post-processing (central differences between steps).
void IOManager::writeRingDiagnostics (int step, amrex::Real time, amrex::Real dt, amrex::Real nu,
                                      const FlowField& state, const DomainManager& dom_mgr)
{
#if AMREX_SPACEDIM == 3
    BL_PROFILE("<IO> IOManager::writeRingDiagnostics()");

    const amrex::Geometry geom  = dom_mgr.getGeom();
    const amrex::BoxArray supp_ba = dom_mgr.getSuppBoxArr();
    const auto dx  = geom.CellSizeArray();
    const auto plo = geom.ProbLoArray();
    const amrex::Real dV = dx[0] * dx[1] * dx[2];
    // u (state) is u' = u - u_inf. Positions are measured in the fluid frame, x - s(t) with
    // s = int_0^t u_inf, so X and its time derivative U are fluid-frame quantities (Eq. 55).
    // E, K, J, I are unaffected (Galilean invariant for compact vorticity).
    const auto uinf = freestreamVelocity(time);
    const auto sinf = freestreamDisplacement(time);

    const amrex::Array<amrex::MultiFab, N_VORT> vort = computeVorticity(state);

    // 15 sums (E, K, J, I[3], M[3][3] with M_ab = 1/2 int (x cross w)_a x_b) and 2 maxima (|u|, |w|)
    using S = amrex::ReduceOpSum;
    using M = amrex::ReduceOpMax;
    using R = amrex::Real;
#ifdef RING_DIAG_LOWER_HALF
    // + 12 sums over z < z_split (I[3], M[3][3]), appended after the maxima
    constexpr R z_split = 0.0;
    amrex::ReduceOps<S,S,S, S,S,S, S,S,S,S,S,S,S,S,S, M,M, S,S,S, S,S,S,S,S,S,S,S,S> reduce_op;
    amrex::ReduceData<R,R,R, R,R,R, R,R,R,R,R,R,R,R,R, R,R, R,R,R, R,R,R,R,R,R,R,R,R> reduce_data(reduce_op);
    constexpr int NR = 29;
#else
    amrex::ReduceOps<S,S,S, S,S,S, S,S,S,S,S,S,S,S,S, M,M> reduce_op;
    amrex::ReduceData<R,R,R, R,R,R, R,R,R,R,R,R,R,R,R, R,R> reduce_data(reduce_op);
    constexpr int NR = 17;
#endif
    using ReduceTuple = typename decltype(reduce_data)::Type;

    std::vector<std::pair<int, amrex::Box>> isects;
    for (amrex::MFIter mfi(state.getPres()); mfi.isValid(); ++mfi)
    {
        supp_ba.intersections(mfi.validbox(), isects);
        if (isects.empty()) { continue; }

        auto const& u  = state.getVel(0).const_array(mfi);
        auto const& v  = state.getVel(1).const_array(mfi);
        auto const& w  = state.getVel(2).const_array(mfi);
        auto const& ex = vort[0].const_array(mfi);
        auto const& ey = vort[1].const_array(mfi);
        auto const& ez = vort[2].const_array(mfi);

        for (auto const& is : isects)
        {
            reduce_op.eval(is.second, reduce_data, [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                const R x[3]  = { plo[0] + (i + R(0.5)) * dx[0] - sinf[0],
                                  plo[1] + (j + R(0.5)) * dx[1] - sinf[1],
                                  plo[2] + (k + R(0.5)) * dx[2] - sinf[2] };
                // face -> cell centre
                const R uc[3] = { R(0.5) * (u(i,j,k) + u(i+1,j,k)),
                                  R(0.5) * (v(i,j,k) + v(i,j+1,k)),
                                  R(0.5) * (w(i,j,k) + w(i,j,k+1)) };
                // edge -> cell centre: the four edges of each direction bounding the cell
                const R wc[3] = { R(0.25) * (ex(i,j,k) + ex(i,j+1,k) + ex(i,j,k+1) + ex(i,j+1,k+1)),
                                  R(0.25) * (ey(i,j,k) + ey(i+1,j,k) + ey(i,j,k+1) + ey(i+1,j,k+1)),
                                  R(0.25) * (ez(i,j,k) + ez(i+1,j,k) + ez(i,j+1,k) + ez(i+1,j+1,k)) };
                const R xw[3] = { x[1]*wc[2] - x[2]*wc[1],
                                  x[2]*wc[0] - x[0]*wc[2],
                                  x[0]*wc[1] - x[1]*wc[0] };
                const R h = R(0.5) * dV;
#ifdef RING_DIAG_LOWER_HALF
                const R hl = (x[2] < z_split) ? h : R(0.0);
#endif
                return { h  * (wc[0]*wc[0] + wc[1]*wc[1] + wc[2]*wc[2]),           // E
                         dV * (uc[0]*xw[0] + uc[1]*xw[1] + uc[2]*xw[2]),           // K
                         dV * (uc[0]*wc[0] + uc[1]*wc[1] + uc[2]*wc[2]),           // J
                         h * xw[0], h * xw[1], h * xw[2],                          // I
                         h * xw[0] * x[0], h * xw[0] * x[1], h * xw[0] * x[2],     // M row 0
                         h * xw[1] * x[0], h * xw[1] * x[1], h * xw[1] * x[2],     // M row 1
                         h * xw[2] * x[0], h * xw[2] * x[1], h * xw[2] * x[2],     // M row 2
                         std::sqrt((uc[0]+uinf[0])*(uc[0]+uinf[0]) + (uc[1]+uinf[1])*(uc[1]+uinf[1])
                                 + (uc[2]+uinf[2])*(uc[2]+uinf[2])),                   // |u' + u_inf| (grid CFL)
                         std::sqrt(wc[0]*wc[0] + wc[1]*wc[1] + wc[2]*wc[2])        // |w|
#ifdef RING_DIAG_LOWER_HALF
                       , hl * xw[0], hl * xw[1], hl * xw[2],                       // I, z < z_split
                         hl * xw[0] * x[0], hl * xw[0] * x[1], hl * xw[0] * x[2],  // M row 0, z < z_split
                         hl * xw[1] * x[0], hl * xw[1] * x[1], hl * xw[1] * x[2],  // M row 1, z < z_split
                         hl * xw[2] * x[0], hl * xw[2] * x[1], hl * xw[2] * x[2]   // M row 2, z < z_split
#endif
                       };
            });
        }
    }

    const ReduceTuple hv = reduce_data.value(reduce_op);
    std::array<R, NR> r = tupleToArray(hv, std::make_index_sequence<NR>{});
    amrex::ParallelDescriptor::ReduceRealSum(r.data(), 15);
    amrex::ParallelDescriptor::ReduceRealMax(r[15]);
    amrex::ParallelDescriptor::ReduceRealMax(r[16]);

    // A centroid needs a nonzero impulse. With zero total impulse (e.g. two colliding rings)
    // I is round-off, ~1e-15 in Gamma0 R0^2 units, and M/I would be garbage: write nan instead.
    constexpr R I2_min = 1.0e-20;
    const R nan_R = std::numeric_limits<R>::quiet_NaN();
#ifdef RING_DIAG_LOWER_HALF
    amrex::ParallelDescriptor::ReduceRealSum(r.data() + 17, 12);
    const R Ilx = r[17], Ily = r[18], Ilz = r[19];
    const R Il2 = Ilx*Ilx + Ily*Ily + Ilz*Ilz;
    R Xl[3] = {nan_R, nan_R, nan_R};
    if (Il2 > I2_min) {
        for (int b = 0; b < 3; ++b) { Xl[b] = (Ilx * r[20 + b] + Ily * r[23 + b] + Ilz * r[26 + b]) / Il2; }
    }
#endif

    // Saffman centroid X_b = sum_a I_a M_ab / |I|^2
    const R Ix = r[3], Iy = r[4], Iz = r[5];
    const R I2 = Ix*Ix + Iy*Iy + Iz*Iz;
    R X[3] = {nan_R, nan_R, nan_R};
    if (I2 > I2_min) {
        for (int b = 0; b < 3; ++b) { X[b] = (Ix * r[6 + b] + Iy * r[9 + b] + Iz * r[12 + b]) / I2; }
    }
    const R cfl = r[15] * dt / amrex::min(dx[0], amrex::min(dx[1], dx[2]));

    if (amrex::ParallelDescriptor::IOProcessor())
    {
        const std::string fname = cfg.log_dir + "/ring_diagnostics.txt";
        const bool new_file = !amrex::FileSystem::Exists(fname);
        if (new_file) { amrex::UtilCreateDirectory(cfg.log_dir, 0755); }

        std::ofstream ofs(fname, std::ios::out | std::ios::app);
        ofs << std::scientific << std::setprecision(16);
        if (new_file)
        {
            ofs << "# Liska & Colonius (2016) Eq. 55 ring diagnostics; cell-centre sums over supp_ba\n"
                << "# X in the fluid frame (x - int_0^t u_inf); lab position = X + s(t). u_max = max|u' + u_inf|\n"
                << "# nu = " << nu << "  dx = " << dx[0] << "  dt = " << dt << "\n"
                << "# step time dt E K J Ix Iy Iz Xx Xy Xz u_max cfl omega_max n_active n_supp"
#ifdef RING_DIAG_LOWER_HALF
                << " Il_x Il_y Il_z Xl_x Xl_y Xl_z"
#endif
                << "\n";
        }
        ofs << step << ' ' << time << ' ' << dt << ' '
            << r[0] << ' ' << r[1] << ' ' << r[2] << ' '
            << Ix << ' ' << Iy << ' ' << Iz << ' '
            << X[0] << ' ' << X[1] << ' ' << X[2] << ' '
            << r[15] << ' ' << cfl << ' ' << r[16] << ' '
            << dom_mgr.getBoxArr().numPts() << ' ' << supp_ba.numPts()
#ifdef RING_DIAG_LOWER_HALF
            << ' ' << Ilx << ' ' << Ily << ' ' << Ilz << ' ' << Xl[0] << ' ' << Xl[1] << ' ' << Xl[2]
#endif
            << '\n';
    }
#else
    amrex::ignore_unused(step, time, dt, nu, state, dom_mgr);
#endif
}
