#include <FlowField.H>

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

// computes nodal vorticity for vor2vel(), via the discreteCurlF2E.
amrex::MultiFab computeNodalVorticity(const FlowField& state)
{
    // IMPORTANT: Limited to 2D at present
    BL_PROFILE("<Compute> computeNodalVorticity()")

    const amrex::Geometry& geom = state.getGeom();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx = geom.InvCellSizeArray();

    const amrex::BoxArray& ba = state.getPres().boxArray();
    const amrex::DistributionMapping& dm = state.getPres().DistributionMap();

    // 2D: omega_z lives at NODES. One nodal MultiFab.
    amrex::BoxArray ba_nd = amrex::convert(ba, amrex::IntVect::TheNodeVector());
    amrex::MultiFab vort_nd(ba_nd, dm, 1, 1);

    for (amrex::MFIter mfi(vort_nd, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& out = vort_nd.array(mfi);
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel{
            state.getVel(0).const_array(mfi),
            state.getVel(1).const_array(mfi)
        };
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            out(i,j,k) = discreteCurlF2E<2>(i, j, k, invdx, vel);  // omega_z at node
        });
    }

    // update ghost cells before returning
    vort_nd.FillBoundary(geom.periodicity());

    return vort_nd;
}

// computes cell-centered vorticity for plotting
amrex::MultiFab computePlotVorticity(const FlowField& state)
{
    BL_PROFILE("<Compute> computePlotVorticity()");

    amrex::MultiFab vort_nd = computeNodalVorticity(state);

    const amrex::BoxArray& ba = state.getPres().boxArray();
    const amrex::DistributionMapping& dm = state.getPres().DistributionMap();

    amrex::MultiFab vort_cc(ba, dm, 1, 0);
    amrex::average_node_to_cellcenter(vort_cc, 0, vort_nd, 0, 1, 0);

    return vort_cc;
}

amrex::MultiFab computeDivNonLinearTerm(const FlowField& state)
{
    BL_PROFILE("computeDivNonLinearTerm()");

    const amrex::Geometry& geom = state.getGeom();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> invdx = geom.InvCellSizeArray();

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
            divN_arr(i,j,k) = divNonLinearTerm(i,j,k, invdx, vel);
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

    

