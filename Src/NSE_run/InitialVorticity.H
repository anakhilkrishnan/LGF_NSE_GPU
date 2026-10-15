#ifndef INITIALVORTICITY_H_
#define INITIALVORTICITY_H_

#include <AMReX_REAL.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>

#include <cmath>

// uncomment below for running colliding rings test
// #define RING_DIAG_LOWER_HALF

// ---------------------------------------------------------------------------
// Initial vorticity omega_0(x, y, z). Every case follows the same contract:
//   comp    : physical direction of the component (0 = x, 1 = y, 2 = z)
//   x, y, z : physical coordinates (the 2D build passes z = 0 and only asks for comp = 2)
//   returns : component comp of omega_0 at (x, y, z)
// omega_0 must be divergence-free and effectively compactly supported.
// Parameters live inside each case; edit them there.
// ---------------------------------------------------------------------------

// Thin viscous vortex ring, Gaussian core (Liska & Colonius 2016, Eq. 53), axis along z.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real vortexRingVorticity (int comp, amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
    constexpr amrex::Real R     = 1.0;                      // ring radius
    constexpr amrex::Real delta = 0.2;                      // core radius (paper: delta/R < 0.2)
    constexpr amrex::Real Gamma = 1.0;                      // circulation
    constexpr amrex::Real xc = 0.0, yc = 0.0, zc = 0.0;     // ring centre

    if (comp == 2) { return 0.0; }                          // omega = omega_theta e_theta

    const amrex::Real X = x - xc, Y = y - yc, Z = z - zc;
    const amrex::Real r = std::sqrt(X*X + Y*Y);
    if (r == 0.0) { return 0.0; }                           // axis: omega ~ exp(-R^2/delta^2)

    const amrex::Real wt = Gamma / (amrex::Math::pi<amrex::Real>() * delta * delta)
                         * std::exp(-((r - R)*(r - R) + Z*Z) / (delta*delta));
    return (comp == 0) ? -wt * Y / r : wt * X / r;
}

// Gaussian-core ring with explicit parameters, for building multi-ring cases.
// Axis along z through (xc, yc); Gamma > 0 makes the ring translate in +z.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real gaussianRingVorticity (int comp, amrex::Real x, amrex::Real y, amrex::Real z,
                                   amrex::Real R, amrex::Real delta, amrex::Real Gamma,
                                   amrex::Real xc, amrex::Real yc, amrex::Real zc) noexcept
{
    if (comp == 2) { return 0.0; }                          // omega = omega_theta e_theta

    const amrex::Real X = x - xc, Y = y - yc, Z = z - zc;
    const amrex::Real r = std::sqrt(X*X + Y*Y);
    if (r == 0.0) { return 0.0; }                           // axis: omega ~ exp(-R^2/delta^2)

    const amrex::Real wt = Gamma / (amrex::Math::pi<amrex::Real>() * delta * delta)
                         * std::exp(-((r - R)*(r - R) + Z*Z) / (delta*delta));
    return (comp == 0) ? -wt * Y / r : wt * X / r;
}

// Head-on collision of two coaxial Gaussian-core rings, axis along z.
// Lower ring (z = -z0, +Gamma) moves up; upper ring (z = +z0, -Gamma) moves down.
// The field is antisymmetric about z = 0: total impulse I = 0, and u_z = 0 on the mid-plane.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real collidingRingsVorticity (int comp, amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
    constexpr amrex::Real R     = 1.0;                      // ring radius
    constexpr amrex::Real delta = 0.14142135623730950;      // core radius
    constexpr amrex::Real Gamma = 1.0;                      // circulation magnitude
    constexpr amrex::Real z0    = 1.25;                      // half the initial centre separation
    constexpr amrex::Real xc = 0.0, yc = 0.0;               // common axis

    return gaussianRingVorticity(comp, x, y, z, R, delta,  Gamma, xc, yc, -z0)
         + gaussianRingVorticity(comp, x, y, z, R, delta, -Gamma, xc, yc,  z0);
}

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

// below for collision simulation with shear-free wall behavior using image rings
namespace collision_params
{
    constexpr amrex::Real R = 1.0, delta = 0.1, Gamma = 1.0;
    constexpr amrex::Real z0 = 1.25;                     // rings at +-z0
    constexpr amrex::Real zw = z0 + 1.0;                 // stress-free walls R0 behind each ring
    constexpr amrex::Real zi = 2.0*zw - z0;              // image rings
}
// #define RING_DIAG_ZLO (-collision_params::zw)  
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real collidingRingsWithWallImageVorticity (int comp, amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
    using namespace collision_params;
    return gaussianRingVorticity(comp, x, y, z, R, delta,  Gamma, 0.0, 0.0, -z0)
         + gaussianRingVorticity(comp, x, y, z, R, delta, -Gamma, 0.0, 0.0,  z0)
         + gaussianRingVorticity(comp, x, y, z, R, delta, -Gamma, 0.0, 0.0, -zi)
         + gaussianRingVorticity(comp, x, y, z, R, delta,  Gamma, 0.0, 0.0,  zi);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real archerHashUniform (unsigned int x) noexcept          // lowbias32 integer hash -> [0, 1)
{
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return static_cast<amrex::Real>(x) * amrex::Real(2.3283064365386963e-10);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real archerRingVorticity (int comp, amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
    constexpr amrex::Real R0 = 1.0, delta = 0.2, Gamma = 1.0;
    // Archer et al. use varsigma = 2e-4 in their (non-solenoidal, then projected) form. The solenoidal form also
    // carries the full tilt of the displaced vorticity, which puts 3.4-4.2x more velocity amplitude into the
    // unstable band n = 9-11 for the same displacement; 5.3e-5 = 2e-4 / 3.8 matches their amplitude at n = 10
    // (within 12% for n = 9-11), so their timeline (still linear at t = 47.5) carries over.
    constexpr amrex::Real varsigma = 5.3e-5;
    constexpr int nmodes = 32;                                  // modes n = 1..32 (thin core)
    constexpr unsigned int seed = 2008u;                        // realisation of the random phases
    constexpr amrex::Real twopi = 6.283185307179586;
 
    if (comp == 2) { return 0.0; }                              // omega_z = 0 (both parts lie in z = const planes)
    const amrex::Real r = std::sqrt(x*x + y*y);
    if (r == 0.0) { return 0.0; }
    const amrex::Real s2 = (r - R0)*(r - R0) + z*z;
    if (s2 > 36.0 * delta * delta) { return 0.0; }              // beyond 6 delta: < 1e-15 of the peak
    const amrex::Real W = Gamma / (amrex::Math::pi<amrex::Real>() * delta * delta) * std::exp(-s2 / (delta * delta));
    const amrex::Real g = -2.0 * (r - R0) / (delta * delta);    // (dW/dr) / W
 
    const amrex::Real c1 = x / r, s1 = y / r;                   // cos(theta), sin(theta)
    amrex::Real cn = 1.0, sn = 0.0, f = 0.0, fp = 0.0;
    for (int n = 1; n <= nmodes; ++n)
    {
        const amrex::Real cnew = cn*c1 - sn*s1;                 // cos(n theta), sin(n theta) by recurrence
        sn = sn*c1 + cn*s1;
        cn = cnew;
        const amrex::Real phi = twopi * archerHashUniform(seed * 2654435761u + static_cast<unsigned int>(n));
        const amrex::Real An = std::cos(phi), Bn = std::sin(phi);
        f  += An * sn + Bn * cn;                                // A_n sin(n theta) + B_n cos(n theta)
        fp += n * (An * cn - Bn * sn);                          // d f / d theta
    }
    const amrex::Real a = varsigma * R0;
    // omega = W e_theta + curl(a f W z_hat):  curl(psi z_hat) = (d psi/dy, -d psi/dx, 0)
    if (comp == 0) { return -W * s1 + a * W * (fp * x / (r*r) + f * g * s1); }
    else           { return  W * c1 + a * W * (fp * y / (r*r) - f * g * c1); }
}

// The active case. The pipeline only ever calls this function.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real initialVorticity (int comp, amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
    // return vortexRingVorticity(comp, x, y, z);
    // return decayingVortexVorticity(comp, x, y, z);
    // return collidingRingsVorticity(comp, x, y, z);
    // return collidingRingsWithWallImageVorticity(comp, x, y, z);
    return archerRingVorticity(comp, x, y, z);
}

#endif