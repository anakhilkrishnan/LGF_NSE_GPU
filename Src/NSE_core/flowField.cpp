#include <FlowField.H>
#include <FreestreamVelocity.H>

FlowField::FlowField(const amrex::Geometry& geom, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, const int n_comp, const int n_ghost)
{
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // convert the box array to face centered
        amrex::BoxArray ba_face = amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim));

        // declare the specific velocity component
        vel[idim].define(ba_face, dm, n_comp, n_ghost);

        // initialize velocities upon creation
        vel[idim].setVal(0.0);
    }

    // initialize pressure upon creation
    pres.define(ba, dm, n_comp, n_ghost);
    pres.setVal(0.0);

    globalgeom = geom;
}

FlowField::FlowField(const FlowField& other)
{
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) 
    {
        vel[idim].define(other.vel[idim].boxArray(), 
                         other.vel[idim].DistributionMap(), 
                         other.vel[idim].nComp(), 
                         other.vel[idim].nGrow());

        amrex::MultiFab::Copy(vel[idim], other.vel[idim], 0, 0, vel[idim].nComp(), vel[idim].nGrow());
    }

    pres.define(other.pres.boxArray(), other. pres.DistributionMap(), other.pres.nComp(), other.pres.nGrow());
    amrex::MultiFab::Copy(pres, other.pres, 0, 0, pres.nComp(), pres.nGrow());

    globalgeom = other.globalgeom;
}

FlowField& FlowField::operator=(const FlowField& other) 
{
    if (this != &other) {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) 
        {
            amrex::MultiFab::Copy(vel[idim], other.vel[idim], 0, 0, vel[idim].nComp(), vel[idim].nGrow());
        }
        
        amrex::MultiFab::Copy(pres, other.pres, 0, 0, pres.nComp(), pres.nGrow());
        globalgeom = other.globalgeom;
    }
    return *this;
}

void FlowField::setBoundary()
{
    // update velocity fields
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // update ghost cells
        vel[idim].FillBoundary(globalgeom.periodicity());
    }

    // update pressure fields
    pres.FillBoundary(globalgeom.periodicity());
}

void FlowField::redefine(const amrex::Geometry& new_geom, const amrex::BoxArray& new_ba, const amrex::DistributionMapping& new_dm)
{
    BL_PROFILE("<MemMgmt> FlowField::redefine()");
    globalgeom = new_geom;

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        amrex::BoxArray new_ba_face = amrex::convert(new_ba, amrex::IntVect::TheDimensionVector(idim));
        vel[idim].define(new_ba_face, new_dm, vel[idim].nComp(), vel[idim].nGrow());

        vel[idim].setVal(0.0);
    }
    pres.define(new_ba, new_dm, pres.nComp(), pres.nGrow());
    pres.setVal(0.0);
}

// initialization procedure to ensure div(omega_0) = 0 is actually upheld discretely
void fluxAverageVorticity(amrex::Array<amrex::MultiFab, N_VORT>& w0, const amrex::Geometry& geom)
{
    const auto dx = geom.CellSizeArray();
    const auto lo = geom.ProbLoArray();

    // 4-point Gauss-Legendre on [-1/2, 1/2]; weights sum to 1, so the sum is the average
    const amrex::GpuArray<amrex::Real, 4> qx = {-0.4305681557970263, -0.1699905217924281,
                                                  0.1699905217924281,  0.4305681557970263};
    const amrex::GpuArray<amrex::Real, 4> qw = { 0.1739274225687269,  0.3260725774312731,
                                                  0.3260725774312731,  0.1739274225687269};

    for (int c = 0; c < N_VORT; ++c)
    {
        const int dir = (AMREX_SPACEDIM == 2) ? 2 : c;   // physical direction of this component
        const int d1  = (dir + 1) % 3;                   // directions spanning the dual face
        const int d2  = (dir + 2) % 3;                   // (2D: dir = 2, so d1 = 0, d2 = 1)

        for (amrex::MFIter mfi(w0[c], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box bx = mfi.growntilebox();    // omega_0 is known everywhere: fill ghosts too
            auto const& a = w0[c].array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // edge centre: nodal in d1, d2; cell-centred in dir (3D). 2D: a node, z = 0.
                const int ijk[3] = {i, j, k};
                amrex::GpuArray<amrex::Real, 3> xe = {0.0, 0.0, 0.0};
                for (int d = 0; d < AMREX_SPACEDIM; ++d) { xe[d] = lo[d] + ijk[d] * dx[d]; }
#if AMREX_SPACEDIM == 3
                xe[dir] += amrex::Real(0.5) * dx[dir];
#endif
                // average flux of omega_dir through the dual face
                amrex::Real sum = 0.0;
                for (int a1 = 0; a1 < 4; ++a1) {
                    for (int a2 = 0; a2 < 4; ++a2) {
                        amrex::GpuArray<amrex::Real, 3> xq = xe;
                        xq[d1] += qx[a1] * dx[d1];
                        xq[d2] += qx[a2] * dx[d2];
                        sum += qw[a1] * qw[a2] * initialVorticity(dir, xq[0], xq[1], xq[2]);
                    }
                }
                a(i,j,k) = sum;
            });
        }
    }
}

// Launch the F2E curl for one vorticity component. dim is a compile-time constant here.
template <int dim>
void fillVorticityComponent(amrex::MultiFab& vort,
                            const FlowField& state,
                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& invdx)
{
    static_assert(dim >= 0 && dim < 3, "fillVorticityComponent: dim is an edge direction (0, 1, 2)");
    static_assert(AMREX_SPACEDIM == 3 || dim == 2, "fillVorticityComponent: 2D only has omega_z (dim = 2)");

    for (amrex::MFIter mfi(vort, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& out = vort.array(mfi);
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel{
            AMREX_D_DECL(state.getVel(0).const_array(mfi),
                         state.getVel(1).const_array(mfi),
                         state.getVel(2).const_array(mfi))};

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            out(i,j,k) = discreteCurlF2E<dim>(i, j, k, invdx, vel);
        });
    }
}

template <int... comp>
void fillVorticity (amrex::Array<amrex::MultiFab, N_VORT>& vort,
                    const FlowField& state,
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& invdx,
                    std::integer_sequence<int, comp...>)
{
    (fillVorticityComponent<vort_dir<comp>>(vort[comp], state, invdx), ...);
}

amrex::Array<amrex::MultiFab, N_VORT> computeVorticity(const FlowField& state)
{
    BL_PROFILE("<Compute> computeVorticity()");

    const amrex::Geometry& geom = state.getGeom();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx = geom.InvCellSizeArray();

    const amrex::BoxArray& ba = state.getPres().boxArray();
    const amrex::DistributionMapping& dm = state.getPres().DistributionMap();
    const int n_ghost = state.getPres().nGrow();

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) 
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(state.getVel(idim).nGrow() >= 1,
            "computeVorticity: velocity needs >= 1 filled ghost cell");
    }

    amrex::Array<amrex::MultiFab, N_VORT> vort;
    for (int icomp = 0; icomp < N_VORT; ++icomp)
    {
        vort[icomp].define(amrex::convert(ba, TheVortEdgeVector(icomp)), dm, 1, n_ghost);
        vort[icomp].setVal(0.0);   // ghosts outside the domain stay zero
    }

    fillVorticity(vort, state, invdx, std::make_integer_sequence<int, N_VORT>{});

    for (int icomp = 0; icomp < N_VORT; ++icomp) 
    {
        vort[icomp].FillBoundary(geom.periodicity());
    }

    return vort;
}



// computes cell-averaged vorticity for tagging
amrex::MultiFab computeTagVorticity(const FlowField& state)
{
    BL_PROFILE("<Compute> computeTagVorticity()");

    const amrex::BoxArray& ba = state.getPres().boxArray();
    const amrex::DistributionMapping& dm = state.getPres().DistributionMap();

    amrex::MultiFab vort_cc(ba, dm, 1, 0);

    amrex::Array<amrex::MultiFab, N_VORT> vort_ed = computeVorticity(state);
#if AMREX_SPACEDIM == 2
    amrex::average_node_to_cellcenter(vort_cc, 0, vort_ed[0], 0, 1, 0);
#endif
#if AMREX_SPACEDIM == 3
    amrex::Vector<const amrex::MultiFab*> vort_ed_ptrs = {&vort_ed[0], &vort_ed[1], &vort_ed[2]};
 
    // average_edge_to_cellcenter writes 3 components (one per edge direction),
    // so it needs a 3-component destination; tag on the magnitude
    amrex::MultiFab vort_cc3(ba, dm, 3, 0);
    amrex::average_edge_to_cellcenter(vort_cc3, 0, vort_ed_ptrs, 0);
 
    auto const& out = vort_cc.arrays();
    auto const& in  = vort_cc3.const_arrays();
    amrex::ParallelFor(vort_cc, [=] AMREX_GPU_DEVICE (int b, int i, int j, int k) noexcept
    {
        const amrex::Real wx = in[b](i,j,k,0), wy = in[b](i,j,k,1), wz = in[b](i,j,k,2);
        out[b](i,j,k) = std::sqrt(wx*wx + wy*wy + wz*wz);
    });
    amrex::Gpu::streamSynchronize();
#endif 

    return vort_cc;
}

amrex::MultiFab computeDivNonLinearTerm(const FlowField& state, amrex::Real time)
{
    BL_PROFILE("computeDivNonLinearTerm()");

    const amrex::Geometry& geom = state.getGeom();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx = geom.InvCellSizeArray();
    const auto uinf = freestreamVelocity(time);   // N(u' + u_inf(t)), Eq. 43c

    const amrex::BoxArray& ba = state.getPres().boxArray();
    const amrex::DistributionMapping& dm = state.getPres().DistributionMap();

    // create base multifab to work on

    amrex::MultiFab divN(ba, dm, state.getPres().nComp(), 0);

    for (amrex::MFIter mfi(divN, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& divN_arr = divN.array(mfi);
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel{AMREX_D_DECL(state.getVel(0).const_array(mfi), state.getVel(1).const_array(mfi), state.getVel(2).const_array(mfi))};
        
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            divN_arr(i,j,k) = divNonLinearTerm(i,j,k, invdx, vel, uinf);
        });
    }
    
    return divN;
}

void computeDivU(amrex::MultiFab& output_divU, const FlowField& input_state)
{
    BL_PROFILE("<Compute> computeDivU()");

    // set divU to 0.0 and store fresh data
    output_divU.setVal(0.0);
    
    // extracting physical dx for computations
    const amrex::Geometry& geom = input_state.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx = geom.InvCellSizeArray();

    // compute divU and store in input_state
    for(amrex::MFIter mfi(output_divU, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& divU_arr = output_divU.array(mfi);
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel_arr;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) 
        {
            vel_arr[d] = input_state.getVel(d).const_array(mfi);
        }
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            divU_arr(i,j,k) = discreteDivergenceF2C(i, j, k, invdx, vel_arr);
        });
    }
}

// function to compute and RETURN divU for plotting divU_at_end
amrex::MultiFab computePlotDivU(const FlowField& state)
{
    // thin wrapper for plot convenience
    MultiFab out(state.getPres().boxArray(), state.getPres().DistributionMap(), 1, 0);
    computeDivU(out, state);

    return out;
}

amrex::Real computeDivUMaxNorm(const FlowField& input_state, const amrex::BoxArray* restrict_ba)
{
    BL_PROFILE("<Compute> computeDivUMaxNorm()");

    // function to reduce state directly into divUMaxNorm
    // optionally takes ba input and restricts norm computation to those boxes

    amrex::ReduceOps<amrex::ReduceOpMax> reduce_op;
    amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    // grid spacing for the stencil
    const auto invdx = input_state.getGeom().InvCellSizeArray();  // {1/dx, 1/dy, 1/dz}

    // You reduce over CELL-centered boxes (divergence is cell-centered),
    // so iterate something cell-centered — e.g. the pressure MultiFab —
    // to get the right box/loop domain, and read the face velocities into it.
    for (amrex::MFIter mfi(input_state.getPres()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        if (restrict_ba != nullptr && !restrict_ba->contains(bx)) { continue; }

        AMREX_D_TERM(auto const& u = input_state.getVel(0).const_array(mfi);,
                    auto const& v = input_state.getVel(1).const_array(mfi);,
                    auto const& w = input_state.getVel(2).const_array(mfi);)

        reduce_op.eval(bx, reduce_data, [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                // discrete divergence at cell (i,j,k) from surrounding faces
                amrex::Real div =
                    AMREX_D_TERM(  (u(i+1,j,k) - u(i,j,k)) * invdx[0],
                                + (v(i,j+1,k) - v(i,j,k)) * invdx[1],
                                + (w(i,j,k+1) - w(i,j,k)) * invdx[2] );
                return { amrex::Math::abs(div) };
            });
    }

    ReduceTuple hv = reduce_data.value(reduce_op);
    amrex::Real max_div = amrex::get<0>(hv);
    amrex::ParallelDescriptor::ReduceRealMax(max_div);
    return max_div;
}

    

