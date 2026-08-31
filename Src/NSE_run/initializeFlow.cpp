// #include <DomainManager.H>

// using namespace amrex;

// void initializeVelField(FlowField& init_state)
// {
//     BL_PROFILE("<Setup> initializeFlowField()");

//     const amrex::Geometry& geom = init_state.getGeom();
//     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx      = geom.CellSizeArray();
//     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
//     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = geom.ProbHiArray();

//     // ------------------------------------------------------------------
//     // Radially symmetric decaying vortex
//     //   (Shukla & Giri, JCP 276 (2014), Sec. 5.3.1)
//     //
//     //   omega(r, 0) = omega_0 (1 - r^2/r0^2) exp(-r^2/r0^2),   r = ||x - xc||
//     //
//     // The streamfunction satisfying grad^2 psi = -omega is, in 2D,
//     //
//     //   psi(r) = (omega_0 r0^2 / 4) exp(-r^2/r0^2)
//     //
//     // Check: for psi = A exp(-r^2/r0^2),
//     //   grad^2 psi = (1/r) d/dr (r dpsi/dr)
//     //              = -(4A/r0^2) (1 - r^2/r0^2) exp(-r^2/r0^2)
//     // so A = omega_0 r0^2 / 4 gives grad^2 psi = -omega exactly.
//     //
//     // Net circulation is identically zero:
//     //   int_0^inf omega 2 pi r dr = omega_0 pi r0^2 int_0^inf (1-s) e^-s ds = 0
//     // so the far field decays as a Gaussian, which suits the free-space
//     // LGF solve and keeps the snug support compact at t = 0.
//     //
//     // Reference solution for validation (paper Eq. 61), with
//     // sigma^2(t) = r0^2 + 4 nu t:
//     //   omega(r,t) = omega_0 (r0^2/sigma^2)^2 (1 - r^2/sigma^2) exp(-r^2/sigma^2)
//     //   psi(r,t)   = (omega_0 r0^4 / (4 sigma^2)) exp(-r^2/sigma^2)
//     // Paper parameters: nu = 1e-3, r0 = 1e-1, domain [-2,2]^2, sampled at t = 7.5.
//     // ------------------------------------------------------------------

//     const amrex::Real r0      = 0.1;
//     const amrex::Real omega_0 = 2.0 * std::sqrt(2.0 * std::exp(1.0)) / r0;
//     const amrex::Real psi_0   = 0.25 * omega_0 * r0 * r0;
//     const amrex::Real inv_r0sq = 1.0 / (r0 * r0);

//     // vortex sits at the centre of the physical domain
//     const amrex::Real xc = 0.5 * (prob_lo[0] + prob_hi[0]);
//     const amrex::Real yc = 0.5 * (prob_lo[1] + prob_hi[1]);

//     // GPU-safe lambda for the streamfunction at any (x,y) node
//     auto get_psi = [=] AMREX_GPU_DEVICE (amrex::Real x, amrex::Real y) -> amrex::Real
//     {
//         const amrex::Real r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc);
//         return psi_0 * std::exp(-r2 * inv_r0sq);
//     };

//     // Initialize x-velocity (u) on the x-faces
//     for (amrex::MFIter mfi(init_state.getVel(0), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
//     {
//         const amrex::Box& bx = mfi.tilebox();
//         auto const& u_arr = init_state.getVel(0).array(mfi);

//         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
//         {
//             // For the x-face at (i, j+0.5), evaluate psi at the North and South nodes
//             amrex::Real x   = prob_lo[0] + i * dx[0];
//             amrex::Real y_N = prob_lo[1] + (j + 1) * dx[1];
//             amrex::Real y_S = prob_lo[1] + j * dx[1];

//             // u = d(psi)/dy
//             u_arr(i,j,k) = (get_psi(x, y_N) - get_psi(x, y_S)) / dx[1];
//         });
//     }

//     // Initialize y-velocity (v) on the y-faces
//     for (amrex::MFIter mfi(init_state.getVel(1), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
//     {
//         const amrex::Box& bx = mfi.tilebox();
//         auto const& v_arr = init_state.getVel(1).array(mfi);

//         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
//         {
//             // For the y-face at (i+0.5, j), evaluate psi at the East and West nodes
//             amrex::Real x_E = prob_lo[0] + (i + 1) * dx[0];
//             amrex::Real x_W = prob_lo[0] + i * dx[0];
//             amrex::Real y   = prob_lo[1] + j * dx[1];

//             // v = -d(psi)/dx
//             v_arr(i,j,k) = -(get_psi(x_E, y) - get_psi(x_W, y)) / dx[0];
//         });
//     }
// }

#include <DomainManager.H>

using namespace amrex;

void initializeVelField(FlowField& init_state)
{
    BL_PROFILE("<Setup> initializeFlowField()");

    const amrex::Geometry& geom = init_state.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    amrex::Real r0 = 0.05;
    amrex::Real omega_0 = 2.0 * std::sqrt(2.0 * exp(1.0)) / r0;
    amrex::Real dTheta = M_PI / 32.0;
    amrex::Vector<amrex::Real> thetas = {0.0, 2.0*M_PI/3.0, 4.0*M_PI/3.0};

    amrex::GpuArray<amrex::Real, 6> cx;
    amrex::GpuArray<amrex::Real, 6> cy;
    
    int idx = 0;
    for (int t = 0; t < 3; ++t) 
    {
        cx[idx] = 0.5 * std::cos(thetas[t] - dTheta);
        cy[idx] = 0.5 * std::sin(thetas[t] - dTheta);
        idx++;
        cx[idx] = 0.5 * std::cos(thetas[t] + dTheta);
        cy[idx] = 0.5 * std::sin(thetas[t] + dTheta);
        idx++;
    }

    // GPU-safe lambda to compute the streamfunction at any (x,y) node
    auto get_psi = [=] AMREX_GPU_DEVICE (amrex::Real x, amrex::Real y) -> amrex::Real {
        amrex::Real psi_val = 0.0;
        for (int v = 0; v < 6; ++v) {
            amrex::Real r2 = (x - cx[v])*(x - cx[v]) + (y - cy[v])*(y - cy[v]);
            // The exact integral of the continuous velocity equations
            psi_val += 0.25 * omega_0 * r0 * r0 * std::exp(-r2 / (r0 * r0));
        }
        return psi_val;
    };

    // Initialize x-velocity (u) on the x-faces
    for (amrex::MFIter mfi(init_state.getVel(0), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& u_arr = init_state.getVel(0).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // For the x-face at (i, j+0.5), evaluate psi at the North and South nodes
            amrex::Real x   = prob_lo[0] + i * dx[0]; 
            amrex::Real y_N = prob_lo[1] + (j + 1) * dx[1]; 
            amrex::Real y_S = prob_lo[1] + j * dx[1]; 

            // u = d(psi)/dy
            u_arr(i,j,k) = (get_psi(x, y_N) - get_psi(x, y_S)) / dx[1];
        });
    }

    // Initialize y-velocity (v) on the y-faces
    for (amrex::MFIter mfi(init_state.getVel(1), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& v_arr = init_state.getVel(1).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // For the y-face at (i+0.5, j), evaluate psi at the East and West nodes
            amrex::Real x_E = prob_lo[0] + (i + 1) * dx[0]; 
            amrex::Real x_W = prob_lo[0] + i * dx[0]; 
            amrex::Real y   = prob_lo[1] + j * dx[1]; 

            // v = -d(psi)/dx
            v_arr(i,j,k) = -(get_psi(x_E, y) - get_psi(x_W, y)) / dx[0];
        });
    }
}