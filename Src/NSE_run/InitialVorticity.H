#ifndef INITIALVORTICITY_H_
#define INITIALVORTICITY_H_

#include <AMReX_REAL.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>

#include <cmath>

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
    constexpr amrex::Real delta = 0.2;                      // core radius
    constexpr amrex::Real Gamma = 1.0;                      // circulation magnitude
    constexpr amrex::Real z0    = 1.5;                      // half the initial centre separation
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

// The active case. The pipeline only ever calls this function.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real initialVorticity (int comp, amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
    // return vortexRingVorticity(comp, x, y, z);
    // return decayingVortexVorticity(comp, x, y, z);
    return collidingRingsVorticity(comp, x, y, z);
}

#endif