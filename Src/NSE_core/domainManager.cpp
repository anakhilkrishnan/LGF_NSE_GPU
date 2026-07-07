#include <DomainManager.H>

DomainManager::DomainManager(const SolverConfig& config)
{
    n_cell = config.n_cell;
    max_grid_size = config.max_grid_size;
    dom_lo = config.dom_lo;
    dom_hi = config.dom_hi;
    periodicity = config.periodic;

    // creating coarse mesh geom, ba and dm
    amrex::IntVect dom_lo_iv(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect dom_hi_iv(AMREX_D_DECL(config.n_cell_init-1, config.n_cell_init-1, config.n_cell_init-1));
    amrex::Box domain(dom_lo_iv, dom_hi_iv);

    ba.define(domain);
    ba.maxSize(config.max_grid_size_init);
    
    dm(ba);

    amrex::RealBox real_box(dom_lo, dom_hi);
    geom(domain, &real_box, amrex::CoordSys::cartesian, periodicity.data());

    // allocate vorticity based on coarse mesh data
    vort_mag.define(ba, dm, config.n_comp, config.n_ghost);
    vort_mag.setVal(0.0);
}

const amrex::Geometry& DomainManager::getGeom() const
{
    return geom;
}

const amrex::BoxArray& DomainManager::getBoxArr() const
{
    return ba;
}

const amrex::DistributionMapping& DomainManager::getDistMap() const
{
    return dm;
}

void DomainManager::initializeCoarseVort()
{
    // extracting dx array and prob_lo to construct real x,y,z
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    // initializing coarse voriticy
    for(amrex::MFIter mfi(vort_mag, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& vort_mag_arr = vort_mag.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::Real x = prob_lo[0] + i * dx[0]; 
            amrex::Real y = prob_lo[1] + j * dx[1];
        #if AMREX_SPACEDIM == 3
            amrex::Real z = prob_lo[2] + k * dx[2];
        #endif

            vort_mag_arr(i,j,k) = vortMag2DAt(x,y,z);
        });
    }
}

void DomainManager::initializeBoxArray()
{
    
}