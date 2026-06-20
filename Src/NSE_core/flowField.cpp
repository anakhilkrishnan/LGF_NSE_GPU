#include <FlowField.H>

struct DummyFillExtDir
{
    AMREX_GPU_DEVICE
    void operator() (const amrex::IntVect& /*iv*/, amrex::Array4<amrex::Real> const& /*dest*/,
                     const int /*dcomp*/, const int /*numcomp*/,
                     amrex::GeometryData const& /*geom*/, const amrex::Real /*time*/,
                     const amrex::BCRec* /*bcr*/, const int /*bcomp*/,
                     const int /*orig_comp*/) const {}
};

FlowField::FlowField(const amrex::Geometry& geom, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, const int n_comp, const int n_ghost)
{
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // convert the box array to face centered
        amrex::BoxArray ba_face = amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim));

        // declare the specific velocity component
        vel[idim].define(ba_face, dm, n_comp, n_ghost);
        kecomp[idim].define(ba_face, dm, n_comp, n_ghost);

        // initialize velocities upon creation
        vel[idim].setVal(0.0);
        kecomp[idim].setVal(0.0);
    }

    // initialize pressure upon creation
    pres.define(ba, dm, n_comp, n_ghost);
    pres.setVal(0.0);

    // initialize tagging regions
    tagRegion.define(ba, dm, n_comp, n_ghost);
    tagRegion.setVal(0.0);

    // initialize divU upon creation
    divU.define(ba, dm, n_comp, n_ghost);
    divU.setVal(0.0);

    divU_at_end.define(ba, dm, n_comp, n_ghost);
    divU_at_end.setVal(0.0);

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
        
        kecomp[idim].define(other.kecomp[idim].boxArray(), 
                         other.kecomp[idim].DistributionMap(), 
                         other.kecomp[idim].nComp(), 
                         other.kecomp[idim].nGrow());

        amrex::MultiFab::Copy(vel[idim], other.vel[idim], 0, 0, vel[idim].nComp(), vel[idim].nGrow());
        amrex::MultiFab::Copy(kecomp[idim], other.kecomp[idim], 0, 0, kecomp[idim].nComp(), kecomp[idim].nGrow());
    }

    pres.define(other.pres.boxArray(), other. pres.DistributionMap(), other.pres.nComp(), other.pres.nGrow());
    amrex::MultiFab::Copy(pres, other.pres, 0, 0, pres.nComp(), pres.nGrow());

    tagRegion.define(other.tagRegion.boxArray(), other.tagRegion.DistributionMap(), other.tagRegion.nComp(), other.tagRegion.nGrow());
    amrex::MultiFab::Copy(tagRegion, other.tagRegion, 0, 0, tagRegion.nComp(), tagRegion.nGrow());

    divU.define(other.divU.boxArray(), other.divU.DistributionMap(), other.divU.nComp(), other.divU.nGrow());
    amrex::MultiFab::Copy(divU, other.divU, 0, 0, divU.nComp(), divU.nGrow());
    
    divU_at_end.define(other.divU_at_end.boxArray(), other.divU_at_end.DistributionMap(), other.divU_at_end.nComp(), other.divU_at_end.nGrow());
    amrex::MultiFab::Copy(divU_at_end, other.divU_at_end, 0, 0, divU_at_end.nComp(), divU_at_end.nGrow());

    globalgeom = other.globalgeom;
}

FlowField& FlowField::operator=(const FlowField& other) 
{
    if (this != &other) {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) 
        {
            amrex::MultiFab::Copy(vel[idim], other.vel[idim], 0, 0, vel[idim].nComp(), vel[idim].nGrow());
            amrex::MultiFab::Copy(kecomp[idim], other.kecomp[idim], 0, 0, kecomp[idim].nComp(), kecomp[idim].nGrow());
        }
        
        amrex::MultiFab::Copy(pres, other.pres, 0, 0, pres.nComp(), pres.nGrow());
        amrex::MultiFab::Copy(tagRegion, other.tagRegion, 0, 0, tagRegion.nComp(), tagRegion.nGrow());
        amrex::MultiFab::Copy(divU, other.divU, 0, 0, divU.nComp(), divU.nGrow());
        amrex::MultiFab::Copy(divU_at_end, other.divU_at_end, 0, 0, divU_at_end.nComp(), divU_at_end.nGrow());
        globalgeom = other.globalgeom;
    }
    return *this;
}

amrex::MultiFab& FlowField::getVel(int idim) 
{
    return vel[idim]; 
}

const amrex::MultiFab& FlowField::getVel(int idim) const 
{
    return vel[idim]; 
}

amrex::MultiFab& FlowField::getKEComp(int idim) 
{
    return kecomp[idim]; 
}

const amrex::MultiFab& FlowField::getKEComp(int idim) const 
{
    return kecomp[idim]; 
}

amrex::MultiFab& FlowField::getPres() 
{
    return pres; 
}

const amrex::MultiFab& FlowField::getPres() const 
{
    return pres; 
}

amrex::MultiFab& FlowField::getTagRegion() 
{
    return tagRegion; 
}

const amrex::MultiFab& FlowField::getTagRegion() const 
{
    return tagRegion; 
}

amrex::MultiFab& FlowField::getDivU() 
{
    return divU; 
}

const amrex::MultiFab& FlowField::getDivU() const 
{
    return divU; 
}

amrex::MultiFab& FlowField::getDivUAtEnd() 
{
    return divU_at_end; 
}

const amrex::MultiFab& FlowField::getDivUAtEnd() const 
{
    return divU_at_end; 
}

const amrex::Geometry& FlowField::getGeom() const 
{
    return globalgeom; 
}

void FlowField::setBoundary()
{
    // initializing BCRec object 1 dimension at a time
    amrex::Vector<amrex::BCRec> bc(1);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) 
    {
        bc[0].setLo(idim, amrex::BCType::foextrap);
        bc[0].setHi(idim, amrex::BCType::foextrap);
    }
    
    amrex::GpuBndryFuncFab<DummyFillExtDir> bndry_func(DummyFillExtDir{});
    amrex::PhysBCFunct<decltype(bndry_func)> physbc(globalgeom, bc, bndry_func);

    // update velocity fields
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // update ghost cells
        vel[idim].FillBoundary(globalgeom.periodicity());
        kecomp[idim].FillBoundary(globalgeom.periodicity());
        
        // update physical domain BCs
        physbc(vel[idim], 0, 1, vel[idim].nGrowVect(), 0.0, 0);
        physbc(kecomp[idim], 0, 1, kecomp[idim].nGrowVect(), 0.0, 0);
    }

    // update pressure fields
    pres.FillBoundary(globalgeom.periodicity());
}

    

