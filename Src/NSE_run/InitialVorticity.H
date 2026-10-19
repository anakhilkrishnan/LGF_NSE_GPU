#ifndef INITIALVORTICITY_H_
#define INITIALVORTICITY_H_

#include <AMReX_REAL.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>

#include <cmath>

// uncomment below for running colliding rings test
// #define RING_DIAG_LOWER_HALF

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real archerHashUniform (unsigned int x) noexcept          // lowbias32 integer hash -> [0, 1)
{
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return static_cast<amrex::Real>(x) * amrex::Real(2.3283064365386963e-10);
}

// ---------------------------------------------------------------------------
// General Gaussian-core vortex ring: any centre, any direction of travel.
//   omega = W(s) e_theta  (+ optional solenoidal centreline perturbation),
//   W(s)  = Gamma / (pi delta^2) exp(-s^2 / delta^2),  s = distance from the core centreline.
// The ring lies in the plane through `center` normal to `dir`; Gamma > 0 travels along +dir
// (Gamma < 0 is the same as reversing dir). The local frame (e1, e2, dir) is the identity for
// dir = +z, so the azimuthal angle -- and therefore the perturbation phases -- is measured from +x.
// 3D only (the 2D build only asks for comp = 2).
// ---------------------------------------------------------------------------
struct GaussianRing
{
    amrex::Real R         = 1.0;                // ring radius
    amrex::Real delta     = 0.2;                // core radius, exp(-s^2/delta^2) convention (sigma convention: delta = sqrt(2) sigma)
    amrex::Real Gamma     = 1.0;                // circulation; > 0 travels along +dir
    amrex::Real center[3] = {0.0, 0.0, 0.0};    // ring centre
    amrex::Real dir[3]    = {0.0, 0.0, 1.0};    // direction of travel (any nonzero length; normalised internally)
    amrex::Real cutoff    = 6.0;                // omega = 0 beyond cutoff*delta from the core (exp(-36) ~ 2e-16); <= 0: no cutoff

    // Optional centreline displacement  varsigma R sum_{n=1}^{nmodes} [A_n sin(n theta) + B_n cos(n theta)],
    // (A_n, B_n) = (cos phi_n, sin phi_n), phi_n random from `seed`, added as curl(a f W dir) so omega stays
    // divergence-free (see archerRingVorticity for the amplitude calibration). varsigma = 0 disables it.
    amrex::Real  varsigma = 0.0;
    int          nmodes   = 32;
    unsigned int seed     = 2008u;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ringVorticity (int comp, amrex::Real x, amrex::Real y, amrex::Real z,
                           const GaussianRing& ring) noexcept
{
    // unit direction of travel
    amrex::Real d0 = ring.dir[0], d1 = ring.dir[1], d2 = ring.dir[2];
    const amrex::Real dn = std::sqrt(d0*d0 + d1*d1 + d2*d2);
    d0 /= dn; d1 /= dn; d2 /= dn;

    // Right-handed orthonormal frame (e1, e2, d), stable for every d: on d2 >= 0 it is the smallest
    // rotation taking +z to d (identity for d = +z); on d2 < 0 it is built about -z (d = -z gives
    // e1 = +x, e2 = -y, i.e. the half-turn about x).
    const amrex::Real sg = (d2 >= 0.0) ? amrex::Real(1.0) : amrex::Real(-1.0);
    const amrex::Real a  = amrex::Real(-1.0) / (sg + d2);
    const amrex::Real b  = d0 * d1 * a;
    const amrex::Real e1[3] = {amrex::Real(1.0) + sg * d0 * d0 * a, sg * b, -sg * d0};
    const amrex::Real e2[3] = {b, sg + d1 * d1 * a, -d1};

    // local coordinates: (X, Y) in the plane of the ring, Z along the direction of travel
    const amrex::Real px = x - ring.center[0], py = y - ring.center[1], pz = z - ring.center[2];
    const amrex::Real X = px*e1[0] + py*e1[1] + pz*e1[2];
    const amrex::Real Y = px*e2[0] + py*e2[1] + pz*e2[2];
    const amrex::Real Z = px*d0    + py*d1    + pz*d2;

    const amrex::Real r = std::sqrt(X*X + Y*Y);
    if (r == 0.0) { return 0.0; }                               // axis: omega ~ exp(-R^2/delta^2)
    const amrex::Real s2 = (r - ring.R)*(r - ring.R) + Z*Z;
    if (ring.cutoff > 0.0 && s2 > ring.cutoff * ring.cutoff * ring.delta * ring.delta) { return 0.0; }

    const amrex::Real W  = ring.Gamma / (amrex::Math::pi<amrex::Real>() * ring.delta * ring.delta)
                         * std::exp(-s2 / (ring.delta * ring.delta));
    const amrex::Real c1 = X / r, s1 = Y / r;                   // cos(theta), sin(theta) in the ring frame

    // local components; the local "3" component is zero (both parts lie in planes normal to dir)
    amrex::Real w1 = -W * s1;
    amrex::Real w2 =  W * c1;

    if (ring.varsigma != 0.0)
    {
        constexpr amrex::Real twopi = 6.283185307179586;
        const amrex::Real g = -2.0 * (r - ring.R) / (ring.delta * ring.delta);    // (dW/dr) / W
        amrex::Real cn = 1.0, sn = 0.0, f = 0.0, fp = 0.0;
        for (int n = 1; n <= ring.nmodes; ++n)
        {
            const amrex::Real cnew = cn*c1 - sn*s1;             // cos(n theta), sin(n theta) by recurrence
            sn = sn*c1 + cn*s1;
            cn = cnew;
            const amrex::Real phi = twopi * archerHashUniform(ring.seed * 2654435761u + static_cast<unsigned int>(n));
            const amrex::Real An = std::cos(phi), Bn = std::sin(phi);
            f  += An * sn + Bn * cn;                            // A_n sin(n theta) + B_n cos(n theta)
            fp += n * (An * cn - Bn * sn);                      // d f / d theta
        }
        // Near-axis taper: psi = amp f W T(r). Untapered, the fp/r term is singular on the ring's own axis
        // (theta is undefined there but W ~ exp(-R^2/delta^2) is not exactly zero), which blows up at grid
        // points that sit on an off-centre or tilted axis up to round-off. T = 3q^2 - 2q^3 (q = r/rt) ~ r^2
        // removes the singularity; T = 1, T' = 0 for r >= rt, so the core is exactly the untapered form and
        // omega stays exactly divergence-free (psi is C^1).
        const amrex::Real rt = 0.1 * ring.R;
        amrex::Real T = 1.0, Tp = 0.0;
        if (r < rt) { const amrex::Real q = r / rt; T = q*q*(3.0 - 2.0*q); Tp = 6.0*q*(1.0 - q) / rt; }
        const amrex::Real amp = ring.varsigma * ring.R;
        // curl(psi e3) in the ring frame = (d psi/dY, -d psi/dX, 0)
        w1 += amp * W * (fp * X * T / (r*r) + f * (g*T + Tp) * s1);
        w2 += amp * W * (fp * Y * T / (r*r) - f * (g*T + Tp) * c1);
    }

    return w1 * e1[comp] + w2 * e2[comp];                      // back to global (x, y, z) components
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

// Thin viscous vortex ring, Gaussian core (Liska & Colonius 2016, Eq. 53), at the origin, travelling in +z.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real vortexRingVorticity (int comp, amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
    GaussianRing ring;                                          // R = 1, delta = 0.2, Gamma = 1
    return ringVorticity(comp, x, y, z, ring);
}

// Archer et al. (2008) case B3: the ring above with a small random centreline perturbation.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real archerRingVorticity (int comp, amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
    GaussianRing ring;                                          // R = 1, delta = 0.2, Gamma = 1
    // Archer et al. use varsigma = 2e-4 in their (non-solenoidal, then projected) form. The solenoidal form also
    // carries the full tilt of the displaced vorticity, which puts 3.4-4.2x more velocity amplitude into the
    // unstable band n = 9-11 for the same displacement; 5.3e-5 = 2e-4 / 3.8 matches their amplitude at n = 10
    // (within 12% for n = 9-11), so their timeline (still linear at t = 47.5) carries over.
    ring.varsigma = 5.3e-5;
    ring.nmodes   = 32;                                         // modes n = 1..32 (thin core)
    ring.seed     = 2008u;                                      // realisation of the random phases
    return ringVorticity(comp, x, y, z, ring);
}

// Head-on collision of two coaxial Gaussian-core rings, axis along z.
// Lower ring (z = -z0) travels in +z; upper ring (z = +z0) travels in -z.
// The field is antisymmetric about z = 0: total impulse I = 0, and u_z = 0 on the mid-plane.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real collidingRingsVorticity (int comp, amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
    constexpr amrex::Real z0 = 1.25;                            // half the initial centre separation

    GaussianRing ring;
    ring.R     = 1.0;
    ring.delta = 0.14142135623730950;                           // sqrt(2) * 0.1, i.e. sigma = 0.1
    ring.Gamma = 1.0;

    ring.center[2] = -z0; ring.dir[2] =  1.0;
    amrex::Real w = ringVorticity(comp, x, y, z, ring);
    ring.center[2] =  z0; ring.dir[2] = -1.0;
    return w + ringVorticity(comp, x, y, z, ring);
}

// Collision with shear-free walls R0 behind each ring, represented by first-order image rings.
// The image series is truncated, so the walls are only approximately shear-free
// (u_z ~ 0.0075 on the axis at each wall at t = 0, ~2.5% of a ring's self-speed).
namespace collision_params
{
    constexpr amrex::Real R = 1.0, delta = 0.1, Gamma = 1.0;
    constexpr amrex::Real z0 = 1.25;                            // rings at +-z0
    constexpr amrex::Real zw = z0 + 1.0;                        // stress-free walls R0 behind each ring
    constexpr amrex::Real zi = 2.0*zw - z0;                     // image rings
}
// #define RING_DIAG_ZLO (-collision_params::zw)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real collidingRingsWithWallImageVorticity (int comp, amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
    using namespace collision_params;
    const amrex::Real zc[4]   = {-z0,  z0, -zi,  zi};           // physical rings, then their images
    const amrex::Real dirz[4] = {1.0, -1.0, -1.0, 1.0};         // direction of travel along z

    GaussianRing ring;
    ring.R = R; ring.delta = delta; ring.Gamma = Gamma;
    amrex::Real w = 0.0;
    for (int k = 0; k < 4; ++k)
    {
        ring.center[2] = zc[k]; ring.dir[2] = dirz[k];
        w += ringVorticity(comp, x, y, z, ring);
    }
    return w;
}

// Three identical Gaussian-core rings centred on the vertices of an equilateral triangle.
// The triangle lies in the plane z = z0 with its centroid on the z axis and one vertex on +y, so the
// configuration is symmetric under 120-degree rotations about z. With tilt = 0 all three rings travel
// in +z; tilt > 0 inclines each ring's direction of travel toward the centroid (tilt < 0: away from it).
// The gap between facing cores (centre to centre, in the plane of the triangle) is side - 2R.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real threeRingTriangleVorticity (int comp, amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
    constexpr amrex::Real R        = 1.0;   // ring radius
    constexpr amrex::Real delta    = 0.2;   // core radius
    constexpr amrex::Real Gamma    = 1.0;   // circulation of each ring
    constexpr amrex::Real side     = 2.5;   // triangle side = centre-to-centre distance  ** PLACEHOLDER **
    constexpr amrex::Real z0       = 0.0;   // plane of the triangle
    constexpr amrex::Real tilt     = 0.0;   // inclination of each ring's travel toward the centroid [rad]
    constexpr amrex::Real varsigma = 0.0;   // per-ring centreline perturbation (0 = none); seeds differ per ring

    constexpr amrex::Real rho = side * 0.57735026918962576;     // circumradius, side / sqrt(3)
    constexpr amrex::Real h   = 0.86602540378443865;            // sqrt(3) / 2
    const amrex::Real ux[3] = {0.0, -h,   h  };                 // unit vectors centroid -> vertex k
    const amrex::Real uy[3] = {1.0, -0.5, -0.5};                // (90, 210, 330 degrees)
    const amrex::Real st = std::sin(tilt), ct = std::cos(tilt);

    amrex::Real w = 0.0;
    for (int k = 0; k < 3; ++k)
    {
        GaussianRing ring;
        ring.R = R; ring.delta = delta; ring.Gamma = Gamma;
        ring.center[0] = rho * ux[k]; ring.center[1] = rho * uy[k]; ring.center[2] = z0;
        ring.dir[0]    = -st * ux[k]; ring.dir[1]    = -st * uy[k]; ring.dir[2]    = ct;
        ring.varsigma  = varsigma;
        ring.seed      = 2008u + static_cast<unsigned int>(k);
        w += ringVorticity(comp, x, y, z, ring);
    }
    return w;
}

#if AMREX_SPACEDIM == 2
// Radially symmetric decaying vortex (Shukla & Giri 2014, Sec. 5.3.1), omega = omega_z(r).
// Analytic decay: omega(r,t) = omega_0 (r0^2/s^2)^2 (1 - r^2/s^2) exp(-r^2/s^2), s^2 = r0^2 + 4 nu t.
// 2D only: in 3D this is an infinite vortex column, not a compact field.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real decayingVortexVorticity (int comp, amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
    constexpr amrex::Real r0      = 0.1;
    constexpr amrex::Real omega_0 = 4.663287963194248 / r0; // 2 sqrt(2e) / r0
    constexpr amrex::Real xc = 0.0, yc = 0.0;               // vortex centre

    amrex::ignore_unused(z);
    if (comp != 2) { return 0.0; }

    const amrex::Real s = ((x - xc)*(x - xc) + (y - yc)*(y - yc)) / (r0*r0);
    return omega_0 * (1.0 - s) * std::exp(-s);
}
#endif

// ---------------------------------------------------------------------------
// Initial vorticity omega_0(x, y, z). Every case follows the same contract:
//   comp    : physical direction of the component (0 = x, 1 = y, 2 = z)
//   x, y, z : physical coordinates (the 2D build passes z = 0 and only asks for comp = 2)
//   returns : component comp of omega_0 at (x, y, z)
// omega_0 must be divergence-free and effectively compactly supported.
// Ring cases are built from ringVorticity(); parameters live inside each case, edit them there.
// ---------------------------------------------------------------------------

// The active case. The pipeline only ever calls this function.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real initialVorticity (int comp, amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
    // return vortexRingVorticity(comp, x, y, z);
    // return archerRingVorticity(comp, x, y, z);
    // return collidingRingsVorticity(comp, x, y, z);
    // return collidingRingsWithWallImageVorticity(comp, x, y, z);
    // return decayingVortexVorticity(comp, x, y, z);
    return threeRingTriangleVorticity(comp, x, y, z);
}

#endif