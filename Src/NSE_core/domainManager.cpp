#include <DomainManager.H>

DomainManager::DomainManager(const SolverConfig& config)
{
    n_cell = config.n_cell;
    max_grid_size = config.max_grid_size;
    dom_lo = config.dom_lo;
    dom_hi = config.dom_hi;
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

void DomainManager::initializeBoxArray(int n_cell_init, int max_grid_size_init)
{
    // creating domain data objects
    amrex::IntVect dom_lo_iv(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect dom_hi_iv(AMREX_D_DECL(n_cell_init-1, n_cell_init-1, n_cell_init-1));
    amrex::Box domain(dom_lo_iv, dom_hi_iv);

    amrex::BoxArray ba;
    ba.define(domain);
    ba.maxSize(max_grid_size_init);
    
    amrex::DistributionMapping dm(ba);

    amrex::RealBox real_box(cfg.dom_lo, cfg.dom_hi);
    amrex::Vector<int> is_periodic(AMREX_SPACEDIM, 0); // infinite domain using zero-grad BC
    amrex::Geometry geom(domain, &real_box, amrex::CoordSys::cartesian, is_periodic.data());

}