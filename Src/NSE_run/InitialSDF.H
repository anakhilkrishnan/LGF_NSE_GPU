#ifndef INITIALSDF_H_
#define INITIALSDF_H_

#include <AMReX.H>
#include <AMReX_REAL.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Algorithm.H>

#include <cmath>

#include <InitialVorticity.H>

// SIGN CONVENTION: psi > 0 inside an interface, psi < 0 outside, |psi| = distance to it.
// Interfaces are closed surfaces, so everything far away is "outside"; the tracker clips psi to
// +-psi_far and fills cells new to the snug domain with -psi_far.

// ---------------------------------------------------------------------------
// Primitives: exact signed distance to one shape
// ---------------------------------------------------------------------------

// Sphere (a circle in 2D, where z = 0).
struct SphereSDF
{
    amrex::Real radius    = 0.5;
    amrex::Real center[3] = {0.0, 0.0, 0.0};
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real sphereSDF (amrex::Real x, amrex::Real y, amrex::Real z, const SphereSDF& s) noexcept
{
    const amrex::Real px = x - s.center[0], py = y - s.center[1], pz = z - s.center[2];
    return s.radius - std::sqrt(px*px + py*py + pz*pz);
}

// Torus: a tube of radius r_tube around a circle of radius R, in the plane through `center`
// normal to `axis`. Same frame as GaussianRing (axis = the ring's direction of travel), so
// ringCoreTorus() below marks a vortex ring's core. In 2D (z = 0) an axis along z gives an
// annulus, and an in-plane axis gives the two discs of a ring's cross-section.
struct TorusSDF
{
    amrex::Real R         = 1.0;                // radius of the tube's centreline
    amrex::Real r_tube    = 0.4;                // tube radius
    amrex::Real center[3] = {0.0, 0.0, 0.0};
    amrex::Real axis[3]   = {0.0, 0.0, 1.0};    // symmetry axis (any nonzero length; normalised internally)
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real torusSDF (amrex::Real x, amrex::Real y, amrex::Real z, const TorusSDF& t) noexcept
{
    const amrex::Real an = std::sqrt(t.axis[0]*t.axis[0] + t.axis[1]*t.axis[1] + t.axis[2]*t.axis[2]);
    const amrex::Real px = x - t.center[0], py = y - t.center[1], pz = z - t.center[2];

    const amrex::Real h   = (px*t.axis[0] + py*t.axis[1] + pz*t.axis[2]) / an;          // offset along the axis
    const amrex::Real rho = std::sqrt(amrex::max(px*px + py*py + pz*pz - h*h, amrex::Real(0.0)));   // distance from it
    return t.r_tube - std::sqrt((rho - t.R)*(rho - t.R) + h*h);
}

// The torus that marks a Gaussian-core ring's core: same centre, axis and ring radius, with the
// tube reaching n_delta core radii (|omega| has fallen to exp(-n_delta^2) of its peak there).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
TorusSDF ringCoreTorus (const GaussianRing& ring, amrex::Real n_delta) noexcept
{
    TorusSDF t;
    t.R      = ring.R;
    t.r_tube = n_delta * ring.delta;
    for (int d = 0; d < 3; ++d) { t.center[d] = ring.center[d]; t.axis[d] = ring.dir[d]; }
    return t;
}

// ---------------------------------------------------------------------------
// Combining shapes (psi > 0 inside). The sign and the zero contour are always exact; the
// distance is exact on one side and a bound on the other (union: exact outside; intersection
// and difference: exact inside). Reinitialization will turn the result into a true distance.
// There is deliberately no stand-alone complement: it would make the far field "inside".
// ---------------------------------------------------------------------------

// A or B
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real sdfUnion (amrex::Real a, amrex::Real b) noexcept { return amrex::max(a, b); }

// A and B
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real sdfIntersection (amrex::Real a, amrex::Real b) noexcept { return amrex::min(a, b); }

// A with B cut out
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real sdfDifference (amrex::Real a, amrex::Real b) noexcept { return amrex::min(a, -b); }

// A or B with the crease filled by a fillet of size ~k (k = 0 is sdfUnion). Not a distance near
// the fillet; the zero contour is smooth there, which is the point.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real sdfSmoothUnion (amrex::Real a, amrex::Real b, amrex::Real k) noexcept
{
    if (k <= 0.0) { return amrex::max(a, b); }
    const amrex::Real h = amrex::Clamp(amrex::Real(0.5) + amrex::Real(0.5) * (a - b) / k,
                                       amrex::Real(0.0), amrex::Real(1.0));
    return b + h * (a - b) + k * h * (amrex::Real(1.0) - h);
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

// One sphere of radius 0.5 at the origin.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real singleSphereSDF (int n, amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
    amrex::ignore_unused(n);
    SphereSDF s;
    return sphereSDF(x, y, z, s);
}

// The core of the default ring in vortexRingVorticity (R = 1, delta = 0.2, at the origin, axis +z),
// marked out to two core radii.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real vortexRingCoreSDF (int n, amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
    amrex::ignore_unused(n);
    GaussianRing ring;
    return torusSDF(x, y, z, ringCoreTorus(ring, 2.0));
}

// Two interfaces: the default ring's core (n = 0), and a sphere of fluid sitting on its axis
// ahead of it (n = 1) that the ring will run into.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ringAndSphereSDF (int n, amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
    if (n == 0)
    {
        GaussianRing ring;
        return torusSDF(x, y, z, ringCoreTorus(ring, 2.0));
    }
    SphereSDF s;
    s.radius = 0.4; s.center[2] = 1.5;
    return sphereSDF(x, y, z, s);
}

// The cores of the three rings of threeRingTriangleVorticity, one interface per ring (n = ring k).
// The geometry comes from threeRingTriangleRing(), so the tori always sit on the actual cores. The
// tube reaches one core radius (|omega| = e^-1 of its peak there), which keeps the three tubes
// apart: facing cores are side - 2R = 0.5 apart, the tubes 0.1.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real threeRingTriangleCoreSDF (int n, amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
    constexpr amrex::Real n_delta = 1.0;    // tube radius in core radii
    return torusSDF(x, y, z, ringCoreTorus(threeRingTriangleRing(n), n_delta));
}

// ---------------------------------------------------------------------------
// Initial signed distance psi_0(x, y, z). Every case follows the same contract:
//   n       : interface index, 0 <= n < N_INTERFACE (one psi component each)
//   x, y, z : physical coordinates (the 2D build passes z = 0)
//   returns : signed distance to interface n, psi > 0 inside
// Each interface must be a closed surface that fits inside the snug domain with its band
// (|psi| < psi_far): the tracker checks this after initialization. Exact distances are not
// required, but psi should behave like one (|grad psi| close to 1) near the zero contour.
// ---------------------------------------------------------------------------

// Number of interfaces the active case defines. Must match the case below.
inline constexpr int N_INTERFACE = 3;

// The active case. The pipeline only ever calls this function.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real initialSDF (int n, amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
    // return singleSphereSDF(n, x, y, z);          // N_INTERFACE = 1
    // return vortexRingCoreSDF(n, x, y, z);        // N_INTERFACE = 1
    // return ringAndSphereSDF(n, x, y, z);         // N_INTERFACE = 2
    return threeRingTriangleCoreSDF(n, x, y, z);    // N_INTERFACE = 3
}

#endif