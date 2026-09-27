#include <DomainManager.H>
#include <InitialVorticity.H>

#if AMREX_SPACEDIM == 2
using namespace amrex;

void initializeVelField(FlowField& init_state)
{
    BL_PROFILE("<Setup> initializeFlowField()");

    const amrex::Geometry& geom = init_state.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx      = geom.CellSizeArray();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = geom.ProbHiArray();

    // ------------------------------------------------------------------
    // Radially symmetric decaying vortex
    //   (Shukla & Giri, JCP 276 (2014), Sec. 5.3.1)
    //
    //   omega(r, 0) = omega_0 (1 - r^2/r0^2) exp(-r^2/r0^2),   r = ||x - xc||
    //
    // The streamfunction satisfying grad^2 psi = -omega is, in 2D,
    //
    //   psi(r) = (omega_0 r0^2 / 4) exp(-r^2/r0^2)
    //
    // Check: for psi = A exp(-r^2/r0^2),
    //   grad^2 psi = (1/r) d/dr (r dpsi/dr)
    //              = -(4A/r0^2) (1 - r^2/r0^2) exp(-r^2/r0^2)
    // so A = omega_0 r0^2 / 4 gives grad^2 psi = -omega exactly.
    //
    // Net circulation is identically zero:
    //   int_0^inf omega 2 pi r dr = omega_0 pi r0^2 int_0^inf (1-s) e^-s ds = 0
    // so the far field decays as a Gaussian, which suits the free-space
    // LGF solve and keeps the snug support compact at t = 0.
    //
    // Reference solution for validation (paper Eq. 61), with
    // sigma^2(t) = r0^2 + 4 nu t:
    //   omega(r,t) = omega_0 (r0^2/sigma^2)^2 (1 - r^2/sigma^2) exp(-r^2/sigma^2)
    //   psi(r,t)   = (omega_0 r0^4 / (4 sigma^2)) exp(-r^2/sigma^2)
    // Paper parameters: nu = 1e-3, r0 = 1e-1, domain [-2,2]^2, sampled at t = 7.5.
    // ------------------------------------------------------------------

    const amrex::Real r0      = 0.1;
    const amrex::Real omega_0 = 2.0 * std::sqrt(2.0 * std::exp(1.0)) / r0;
    const amrex::Real psi_0   = 0.25 * omega_0 * r0 * r0;
    const amrex::Real inv_r0sq = 1.0 / (r0 * r0);

    // vortex sits at the centre of the physical domain
    const amrex::Real xc = 0.5 * (prob_lo[0] + prob_hi[0]);
    const amrex::Real yc = 0.5 * (prob_lo[1] + prob_hi[1]);

    // GPU-safe lambda for the streamfunction at any (x,y) node
    auto get_psi = [=] AMREX_GPU_DEVICE (amrex::Real x, amrex::Real y) -> amrex::Real
    {
        const amrex::Real r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc);
        return psi_0 * std::exp(-r2 * inv_r0sq);
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

// #include <DomainManager.H>

// using namespace amrex;

// void initializeVelField(FlowField& init_state)
// {
//     BL_PROFILE("<Setup> initializeFlowField()");

//     const amrex::Geometry& geom = init_state.getGeom();
//     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
//     amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

//     amrex::Real r0 = 0.05;
//     amrex::Real omega_0 = 2.0 * std::sqrt(2.0 * exp(1.0)) / r0;
//     amrex::Real dTheta = M_PI / 32.0;
//     amrex::Vector<amrex::Real> thetas = {0.0, 2.0*M_PI/3.0, 4.0*M_PI/3.0};

//     amrex::GpuArray<amrex::Real, 6> cx;
//     amrex::GpuArray<amrex::Real, 6> cy;
    
//     int idx = 0;
//     for (int t = 0; t < 3; ++t) 
//     {
//         cx[idx] = 0.5 * std::cos(thetas[t] - dTheta);
//         cy[idx] = 0.5 * std::sin(thetas[t] - dTheta);
//         idx++;
//         cx[idx] = 0.5 * std::cos(thetas[t] + dTheta);
//         cy[idx] = 0.5 * std::sin(thetas[t] + dTheta);
//         idx++;
//     }

//     // GPU-safe lambda to compute the streamfunction at any (x,y) node
//     auto get_psi = [=] AMREX_GPU_DEVICE (amrex::Real x, amrex::Real y) -> amrex::Real {
//         amrex::Real psi_val = 0.0;
//         for (int v = 0; v < 6; ++v) {
//             amrex::Real r2 = (x - cx[v])*(x - cx[v]) + (y - cy[v])*(y - cy[v]);
//             // The exact integral of the continuous velocity equations
//             psi_val += 0.25 * omega_0 * r0 * r0 * std::exp(-r2 / (r0 * r0));
//         }
//         return psi_val;
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

#endif // AMREX_SPACEDIM

#if (AMREX_SPACEDIM == 3)

namespace vortex_ring_detail
{
    constexpr int NR_FAR   = 16;  // Gauss-Legendre, radial, core-centred polar coords
    constexpr int NR_NEAR  = 48;  // Gauss-Legendre, radial, target-centred polar coords
    constexpr int NPHI     = 64;  // trapezoid (periodic), angular, near branch
    constexpr int PHI_SKIP = 2;   // far branch uses every 2nd angle (32 points)

    constexpr int OFF_XF = 0;
    constexpr int OFF_WF = OFF_XF + NR_FAR;
    constexpr int OFF_XN = OFF_WF + NR_FAR;
    constexpr int OFF_WN = OFF_XN + NR_NEAR;
    constexpr int OFF_C  = OFF_WN + NR_NEAR;
    constexpr int OFF_S  = OFF_C  + NPHI;
    constexpr int NBUF   = OFF_S  + NPHI;

    // theta-component of A at (r,z) from a unit-circulation circular filament through (rp,zp)
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real filamentAtheta (amrex::Real r,  amrex::Real z,
                                amrex::Real rp, amrex::Real zp) noexcept
    {
        if (r <= amrex::Real(0.0) || rp <= amrex::Real(0.0)) { return amrex::Real(0.0); }
        const amrex::Real pi  = amrex::Real(3.141592653589793238462643383279502884);
        const amrex::Real dz  = z - zp;
        const amrex::Real Dp2 = (r + rp) * (r + rp) + dz * dz;
        const amrex::Real Dm2 = (r - rp) * (r - rp) + dz * dz;
        const amrex::Real k2  = amrex::Real(4.0) * r * rp / Dp2;

        amrex::Real bracket;  // (1 - k^2/2) K(k) - E(k)
        if (k2 < amrex::Real(1.0e-3)) {
            bracket = pi / amrex::Real(32.0) * k2 * k2
                    * (amrex::Real(1.0) + amrex::Real(0.75) * k2 + amrex::Real(75.0/128.0) * k2 * k2);
        } else {
            amrex::Real a = amrex::Real(1.0), b = std::sqrt(Dm2 / Dp2);
            amrex::Real sum = amrex::Real(0.5) * k2, p2 = amrex::Real(0.5);
            const amrex::Real tol = amrex::Real(10.0) * std::numeric_limits<amrex::Real>::epsilon();
            for (int it = 0; it < 40; ++it) {
                const amrex::Real c  = amrex::Real(0.5) * (a - b);
                const amrex::Real an = amrex::Real(0.5) * (a + b);
                b = std::sqrt(a * b); a = an; p2 *= amrex::Real(2.0); sum += p2 * c * c;
                if (std::abs(c) <= tol * a) { break; }
            }
            const amrex::Real K = pi / (amrex::Real(2.0) * a);
            const amrex::Real E = K * (amrex::Real(1.0) - sum);
            bracket = (amrex::Real(1.0) - amrex::Real(0.5) * k2) * K - E;
        }
        return bracket * std::sqrt(Dp2) / (amrex::Real(2.0) * pi * r);
    }

    inline void gaussLegendre01 (int n, double* x, double* w)
    {
        const double pi = 3.141592653589793238462643383279502884;
        for (int i = 0; i < n; ++i) {
            double z = std::cos(pi * (i + 0.75) / (n + 0.5)), dp = 1.0;
            for (int it = 0; it < 100; ++it) {
                double p0 = 1.0, p1 = z;
                for (int j = 2; j <= n; ++j) {
                    const double p2 = ((2.0 * j - 1.0) * z * p1 - (j - 1.0) * p0) / j;
                    p0 = p1; p1 = p2;
                }
                dp = n * (z * p1 - p0) / (z * z - 1.0);
                const double dz = p1 / dp; z -= dz;
                if (std::abs(dz) < 1.0e-15) { break; }
            }
            x[i] = 0.5 * (1.0 + z);
            w[i] = 1.0 / ((1.0 - z * z) * dp * dp);
        }
    }

    struct GaussianRingPotential
    {
        amrex::Real R = 1.0, delta = 0.2, Gamma = 1.0;
        const amrex::Real* q = nullptr;

        AMREX_GPU_HOST_DEVICE
        amrex::Real operator() (amrex::Real r, amrex::Real z) const noexcept
        {
            const amrex::Real pi     = amrex::Real(3.141592653589793238462643383279502884);
            const amrex::Real rs     = amrex::Real(5.0) * delta;
            const amrex::Real inv_d2 = amrex::Real(1.0) / (delta * delta);
            const amrex::Real om0    = Gamma / (pi * delta * delta);
            const amrex::Real dphi   = amrex::Real(2.0) * pi / NPHI;
            const amrex::Real d      = std::sqrt((r - R) * (r - R) + z * z);
            amrex::Real A = amrex::Real(0.0);

            if (d >= amrex::Real(6.0) * delta) {
                // far: polar coords centred on the core
                for (int i = 0; i < NR_FAR; ++i) {
                    const amrex::Real rho = rs * q[OFF_XF + i];
                    const amrex::Real wr  = om0 * std::exp(-rho * rho * inv_d2)
                                          * rho * rs * q[OFF_WF + i] * (dphi * PHI_SKIP);
                    for (int j = 0; j < NPHI; j += PHI_SKIP) {
                        A += wr * filamentAtheta(r, z, R + rho * q[OFF_C + j], rho * q[OFF_S + j]);
                    }
                }
            } else {
                // near/inside core: polar coords centred on the target, rho = L u^2
                const amrex::Real L = d + rs;
                for (int i = 0; i < NR_NEAR; ++i) {
                    const amrex::Real u   = q[OFF_XN + i];
                    const amrex::Real rho = L * u * u;
                    const amrex::Real wr  = amrex::Real(2.0) * L * L * u * u * u * q[OFF_WN + i] * dphi;
                    for (int j = 0; j < NPHI; ++j) {
                        const amrex::Real rp = r + rho * q[OFF_C + j];
                        const amrex::Real zp = z + rho * q[OFF_S + j];
                        if (rp <= amrex::Real(0.0)) { continue; }
                        const amrex::Real s2 = (rp - R) * (rp - R) + zp * zp;
                        if (s2 > rs * rs) { continue; }
                        A += wr * om0 * std::exp(-s2 * inv_d2) * filamentAtheta(r, z, rp, zp);
                    }
                }
            }
            return A;
        }
    };

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void cubicWeights (amrex::Real t, amrex::Real* w) noexcept
    {
        const amrex::Real tm1 = t - 1, tm2 = t - 2, tp1 = t + 1;
        w[0] = -t   * tm1 * tm2 / amrex::Real(6.0);
        w[1] =  tp1 * tm1 * tm2 / amrex::Real(2.0);
        w[2] = -tp1 * t   * tm2 / amrex::Real(2.0);
        w[3] =  tp1 * t   * tm1 / amrex::Real(6.0);
    }
} // namespace vortex_ring_detail

void initializeVelField(FlowField& init_state)
{
    BL_PROFILE("<Setup> initializeFlowField()");
    using namespace vortex_ring_detail;

    const amrex::Geometry& geom = init_state.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx      = geom.CellSizeArray();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = geom.ProbHiArray();

    amrex::Real R0 = 1.0, Gamma0 = 1.0;
    int table_refine = 2;
    amrex::Vector<amrex::Real> center = { 0.5 * (prob_lo[0] + prob_hi[0]),
                                          0.5 * (prob_lo[1] + prob_hi[1]),
                                          0.5 * (prob_lo[2] + prob_hi[2]) };
    amrex::ParmParse pp("vortex_ring");
    pp.query("R0", R0);
    pp.query("Gamma0", Gamma0);
    pp.query("table_refine", table_refine);
    pp.queryarr("center", center);
    amrex::Real delta0 = 0.2 * R0;
    pp.query("delta0", delta0);
    AMREX_ALWAYS_ASSERT(center.size() == 3 && R0 > 0 && delta0 > 0 && table_refine >= 1);
    if (delta0 / R0 > 0.2) {
        amrex::Print() << "WARNING: vortex ring delta0/R0 = " << delta0 / R0
                       << " > 0.2; omega on the ring axis is no longer negligible.\n";
    }
    const amrex::Real xc = center[0], yc = center[1], zc = center[2];

    // quadrature nodes -> device
    std::vector<double> hbuf_d(NBUF);
    gaussLegendre01(NR_FAR,  &hbuf_d[OFF_XF], &hbuf_d[OFF_WF]);
    gaussLegendre01(NR_NEAR, &hbuf_d[OFF_XN], &hbuf_d[OFF_WN]);
    for (int j = 0; j < NPHI; ++j) {
        const double phi = 2.0 * 3.141592653589793238462643383279502884 * (j + 0.5) / NPHI;
        hbuf_d[OFF_C + j] = std::cos(phi);
        hbuf_d[OFF_S + j] = std::sin(phi);
    }
    std::vector<amrex::Real> hbuf(hbuf_d.begin(), hbuf_d.end());
    amrex::Gpu::DeviceVector<amrex::Real> qbuf(NBUF);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, hbuf.begin(), hbuf.end(), qbuf.begin());

    GaussianRingPotential ring;
    ring.R = R0; ring.delta = delta0; ring.Gamma = Gamma0; ring.q = qbuf.data();

    // (r,z) extent of the edges touched by this rank
    amrex::Real xa = std::numeric_limits<amrex::Real>::max(), xb = -xa;
    amrex::Real ya = xa, yb = -xa, za = xa, zb = -xa;
    bool have_boxes = false;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        for (amrex::MFIter mfi(init_state.getVel(d)); mfi.isValid(); ++mfi) {
            const amrex::Box& vb = mfi.validbox();
            const amrex::IntVect lo = vb.smallEnd(), hi = vb.bigEnd();
            xa = amrex::min(xa, prob_lo[0] + (lo[0] - 1) * dx[0]);
            xb = amrex::max(xb, prob_lo[0] + (hi[0] + 2) * dx[0]);
            ya = amrex::min(ya, prob_lo[1] + (lo[1] - 1) * dx[1]);
            yb = amrex::max(yb, prob_lo[1] + (hi[1] + 2) * dx[1]);
            za = amrex::min(za, prob_lo[2] + (lo[2] - 1) * dx[2]);
            zb = amrex::max(zb, prob_lo[2] + (hi[2] + 2) * dx[2]);
            have_boxes = true;
        }
    }
    if (!have_boxes) { return; }

    const amrex::Real ex = amrex::max(amrex::Real(0.0), amrex::max(xa - xc, xc - xb));
    const amrex::Real ey = amrex::max(amrex::Real(0.0), amrex::max(ya - yc, yc - yb));
    const amrex::Real r_min = std::sqrt(ex * ex + ey * ey);
    const amrex::Real fx = amrex::max(std::abs(xa - xc), std::abs(xb - xc));
    const amrex::Real fy = amrex::max(std::abs(ya - yc), std::abs(yb - yc));
    const amrex::Real r_max = std::sqrt(fx * fx + fy * fy);

    const amrex::Real h    = amrex::min(dx[0], amrex::min(dx[1], dx[2])) / table_refine;
    const amrex::Real invh = amrex::Real(1.0) / h;
    const amrex::Real r0   = r_min - 2 * h;
    const amrex::Real z0   = (za - zc) - 2 * h;
    const int nr = static_cast<int>(std::ceil((r_max - r_min) * invh)) + 5;
    const int nz = static_cast<int>(std::ceil((zb - za) * invh)) + 5;
    const int ntab = nr * nz;

    // tabulate A_theta(r,z); odd extension across the axis
    amrex::Gpu::DeviceVector<amrex::Real> table(ntab);
    amrex::Real* tab = table.data();
    auto fill = [=] AMREX_GPU_HOST_DEVICE (int n) noexcept {
        const int i = n % nr, j = n / nr;
        const amrex::Real r = r0 + i * h, z = z0 + j * h;
        tab[n] = (r >= amrex::Real(0.0)) ? ring(r, z) : -ring(-r, z);
    };
#ifdef AMREX_USE_GPU
    amrex::ParallelFor(ntab, fill);
    amrex::Gpu::streamSynchronize();
#else
#ifdef AMREX_USE_OMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (int n = 0; n < ntab; ++n) { fill(n); }
#endif

    auto get_Atheta = [=] AMREX_GPU_DEVICE (amrex::Real r, amrex::Real zrel) -> amrex::Real {
        const amrex::Real fr = (r - r0) * invh, fz = (zrel - z0) * invh;
        int i = static_cast<int>(std::floor(fr)), j = static_cast<int>(std::floor(fz));
        i = amrex::max(1, amrex::min(i, nr - 3));
        j = amrex::max(1, amrex::min(j, nz - 3));
        amrex::Real wr[4], wz[4];
        cubicWeights(fr - i, wr);
        cubicWeights(fz - j, wz);
        amrex::Real s = 0;
        for (int b = 0; b < 4; ++b) {
            const amrex::Real* row = tab + (j - 1 + b) * nr + (i - 1);
            s += wz[b] * (wr[0] * row[0] + wr[1] * row[1] + wr[2] * row[2] + wr[3] * row[3]);
        }
        return s;
    };

    const amrex::Real r_tiny = amrex::Real(1.0e-12) * dx[0];
    auto get_Ax = [=] AMREX_GPU_DEVICE (amrex::Real x, amrex::Real y, amrex::Real z) -> amrex::Real {
        const amrex::Real X = x - xc, Y = y - yc, r = std::sqrt(X * X + Y * Y);
        return (r < r_tiny) ? amrex::Real(0.0) : -get_Atheta(r, z - zc) * Y / r;
    };
    auto get_Ay = [=] AMREX_GPU_DEVICE (amrex::Real x, amrex::Real y, amrex::Real z) -> amrex::Real {
        const amrex::Real X = x - xc, Y = y - yc, r = std::sqrt(X * X + Y * Y);
        return (r < r_tiny) ? amrex::Real(0.0) :  get_Atheta(r, z - zc) * X / r;
    };

    // x-faces: u = -dAy/dz
    for (amrex::MFIter mfi(init_state.getVel(0), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& u_arr = init_state.getVel(0).array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            const amrex::Real x = prob_lo[0] + i * dx[0];
            const amrex::Real y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
            const amrex::Real z_B = prob_lo[2] + k * dx[2], z_T = prob_lo[2] + (k + 1) * dx[2];
            u_arr(i,j,k) = -(get_Ay(x, y, z_T) - get_Ay(x, y, z_B)) / dx[2];
        });
    }
    // y-faces: v = dAx/dz
    for (amrex::MFIter mfi(init_state.getVel(1), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& v_arr = init_state.getVel(1).array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            const amrex::Real x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
            const amrex::Real y = prob_lo[1] + j * dx[1];
            const amrex::Real z_B = prob_lo[2] + k * dx[2], z_T = prob_lo[2] + (k + 1) * dx[2];
            v_arr(i,j,k) = (get_Ax(x, y, z_T) - get_Ax(x, y, z_B)) / dx[2];
        });
    }
    // z-faces: w = dAy/dx - dAx/dy
    for (amrex::MFIter mfi(init_state.getVel(2), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& w_arr = init_state.getVel(2).array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            const amrex::Real z   = prob_lo[2] + k * dx[2];
            const amrex::Real x_W = prob_lo[0] + i * dx[0], x_E = prob_lo[0] + (i + 1) * dx[0];
            const amrex::Real y_m = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
            const amrex::Real x_m = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
            const amrex::Real y_S = prob_lo[1] + j * dx[1], y_N = prob_lo[1] + (j + 1) * dx[1];
            w_arr(i,j,k) = (get_Ay(x_E, y_m, z) - get_Ay(x_W, y_m, z)) / dx[0]
                         - (get_Ax(x_m, y_N, z) - get_Ax(x_m, y_S, z)) / dx[1];
        });
    }
    amrex::Gpu::streamSynchronize();
}

#endif // AMREX_SPACEDIM
