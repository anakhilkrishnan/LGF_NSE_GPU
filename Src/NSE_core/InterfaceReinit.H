#ifndef INTERFACEREINIT_H_
#define INTERFACEREINIT_H_

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>

// device kernels for the geometric (closest-point) reinitialization of psi. The zero level
// set is not moved by a pseudo-time PDE: it is rebuilt from psi's own values by local
// interpolation, and psi is reset to the distance to it.
//
// Every band cell x is handled on its own, with a fixed number of steps:
//   1. p, the tensor-product Lagrange interpolant of psi of degree REINIT degree (3: the 4^3
//      cells around the evaluation point; 5: 6^3), re-centred on whichever cell the current
//      iterate lies in;
//   2. a starting point on the interface by projection along the gradient,
//      y <- y - p(y) grad p(y) / |grad p(y)|^2, steps clamped to REINIT_STEP_PROJ cells;
//   3. Newton's method on the Lagrangian of  min |y - x|^2  subject to  p(y) = 0 (the
//      closest-point step of Saye's high-order method, as in algoim's hocp.hpp), steps
//      clamped to REINIT_STEP_NEWTON cells;
//   4. psi(x) = sign(psi_old(x)) |x - y*|, clipped to psi_far;
//   5. a zero-set correction (InterfaceTracker::reinitializeComponent, step 4): without it the
//      zero set of the new psi's interpolant sits about (dx/r)^3 dx (degree 3, radius of
//      curvature r) off the old one, always to the same side, and repeated reinitialization
//      walks the interface.
// Cells that do not converge, or converge on a point that is only locally closest (beside a
// medial surface, where two parts of the interface are about equally near), are retried
// from the closest points their 26 neighbours found; these passes also carry closest points
// outward to cells the first pass could not resolve (the outer band of a badly distorted psi). There are no seeds and no search tree:
// the work and the memory reads of a cell are bounded and local.
//
// Measured (reinitTest.cpp; a numpy prototype of the same sequence agreed with algoim's
// reinit, seeds + kd-tree, to 6e-3 dx within 6 cells of the interface on a seeded torus):
//   - error against the exact distance: 4th order for degree 3 (2e-4 dx at 8 cells per tube
//     radius, 3e-5 dx at 16) and 6th order for degree 5; distorted psi (|grad psi| 0.4 ... 1.6)
//     and non-distance psi are restored to |grad psi| = 1 at the same accuracy;
//   - the sign of psi never changes, so no cell crosses the interface;
//   - repeated reinitialization with the correction: mean zero-set drift 5e-7 dx per
//     application for degree 3 at 8 cells per tube radius (3e-5 dx without it); none
//     measurable for degree 5;
//   - transport by strain + rotation, reinitialized every step for 95 steps: zero-set error
//     6e-3 dx max, 3.5e-4 dx mean (no reinitialization: 1.2e-3 / 1.4e-4; every step without
//     the correction: 3.2e-2 / 4.3e-3).

#if (AMREX_SPACEDIM == 3)

inline constexpr int REINIT_N_PROJ = 4;                    // projection steps from the cell centre
inline constexpr int REINIT_N_NEWTON = 4;                  // constrained Newton steps
inline constexpr amrex::Real REINIT_STEP_PROJ = 3.0;       // longest projection step, in cells
inline constexpr amrex::Real REINIT_STEP_NEWTON = 1.0;     // longest Newton step, in cells
inline constexpr amrex::Real REINIT_TOL_SURFACE = 1.0e-4;  // |p(y)| / |grad p(y)|, in cells
inline constexpr amrex::Real REINIT_TOL_ANGLE = 1.0e-2;    // |y - x + lambda grad p| / max(|x - y|, dx)
inline constexpr amrex::Real REINIT_REACH_MARGIN = 2.0;    // cells beyond psi_far an iterate may go

// status of a cell's closest point (component 3 of the closest-point MultiFab)
inline constexpr amrex::Real REINIT_OUTSIDE = -1.0;        // not in the band: psi = +-psi_far
inline constexpr amrex::Real REINIT_FAILED = 0.0;          // no closest point and no estimate
inline constexpr amrex::Real REINIT_ESTIMATE = 0.5;        // a neighbour's closest point that Newton could not
                                                           // polish: a point on the interface, so its distance
                                                           // is an upper bound and close to the true one
inline constexpr amrex::Real REINIT_CONVERGED = 1.0;

// Lagrange basis on the P nodes first, ..., first + P - 1 (first = 1 - P/2: -1 ... 2 for
// P = 4, -2 ... 3 for P = 6), with its first and second derivatives, at t in [0, 1)
template <int P>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void lagrangeWeights(amrex::Real t, amrex::Real w[P], amrex::Real dw[P], amrex::Real d2w[P])
{
    constexpr int first = 1 - P / 2;
    amrex::Real e[P];
    for (int m = 0; m < P; ++m) { e[m] = t - amrex::Real(first + m); }

    for (int k = 0; k < P; ++k)
    {
        amrex::Real denom = 1.0, v = 1.0, d1 = 0.0, d2 = 0.0;
        for (int m = 0; m < P; ++m)
        {
            if (m == k) { continue; }
            denom *= amrex::Real(k - m);
            v *= e[m];
            amrex::Real p1 = 1.0;
            for (int l = 0; l < P; ++l)
            {
                if (l == k || l == m) { continue; }
                p1 *= e[l];
                amrex::Real p2 = 1.0;
                for (int q = 0; q < P; ++q)
                {
                    if (q != k && q != m && q != l) { p2 *= e[q]; }
                }
                d2 += p2;   // ordered pairs (m, l): the second derivative of the product
            }
            d1 += p1;
        }
        w[k] = v / denom;
        dw[k] = d1 / denom;
        d2w[k] = d2 / denom;
    }
}

// value, gradient and Hessian (xx, yy, zz, xy, xz, yz) of the interpolant at a point
struct InterpValue
{
    amrex::Real v;
    amrex::Real g[3];
    amrex::Real h[6];
};

// the interpolant of psi at physical point y, from the P^3 cells around it. False when that
// stencil is not inside phi's box (valid + ghost cells), which the caller treats as failure
template <int P>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool interpolatePsi(amrex::Array4<amrex::Real const> const& phi, const amrex::Real y[3],
                    amrex::GpuArray<amrex::Real, 3> const& problo,
                    amrex::GpuArray<amrex::Real, 3> const& invdx, InterpValue& f)
{
    constexpr int first = 1 - P / 2;
    int base[3];
    amrex::Real w[3][P], dw[3][P], d2w[3][P];
    for (int d = 0; d < 3; ++d)
    {
        // continuous cell index: cell i is centred at problo + (i + 0.5) dx
        const amrex::Real s = (y[d] - problo[d]) * invdx[d] - 0.5;
        const amrex::Real c = std::floor(s);
        base[d] = static_cast<int>(c) + first;
        lagrangeWeights<P>(s - c, w[d], dw[d], d2w[d]);
    }
    if (!phi.contains(base[0], base[1], base[2]) ||
        !phi.contains(base[0] + P - 1, base[1] + P - 1, base[2] + P - 1)) { return false; }

    // contract z, then y, then x; everything stays in registers
    amrex::Real v = 0.0, gx = 0.0, gy = 0.0, gz = 0.0;
    amrex::Real hxx = 0.0, hyy = 0.0, hzz = 0.0, hxy = 0.0, hxz = 0.0, hyz = 0.0;
    for (int a = 0; a < P; ++a)
    {
        amrex::Real t00 = 0.0, t10 = 0.0, t01 = 0.0, t20 = 0.0, t11 = 0.0, t02 = 0.0;
        for (int b = 0; b < P; ++b)
        {
            amrex::Real s0 = 0.0, s1 = 0.0, s2 = 0.0;
            for (int c = 0; c < P; ++c)
            {
                const amrex::Real val = phi(base[0] + a, base[1] + b, base[2] + c);
                s0 += w[2][c] * val;
                s1 += dw[2][c] * val;
                s2 += d2w[2][c] * val;
            }
            t00 += w[1][b] * s0;
            t10 += dw[1][b] * s0;
            t01 += w[1][b] * s1;
            t20 += d2w[1][b] * s0;
            t11 += dw[1][b] * s1;
            t02 += w[1][b] * s2;
        }
        v   += w[0][a] * t00;
        gx  += dw[0][a] * t00;
        gy  += w[0][a] * t10;
        gz  += w[0][a] * t01;
        hxx += d2w[0][a] * t00;
        hyy += w[0][a] * t20;
        hzz += w[0][a] * t02;
        hxy += dw[0][a] * t10;
        hxz += dw[0][a] * t01;
        hyz += w[0][a] * t11;
    }

    f.v = v;
    f.g[0] = gx * invdx[0];
    f.g[1] = gy * invdx[1];
    f.g[2] = gz * invdx[2];
    f.h[0] = hxx * invdx[0] * invdx[0];
    f.h[1] = hyy * invdx[1] * invdx[1];
    f.h[2] = hzz * invdx[2] * invdx[2];
    f.h[3] = hxy * invdx[0] * invdx[1];
    f.h[4] = hxz * invdx[0] * invdx[2];
    f.h[5] = hyz * invdx[1] * invdx[2];
    return true;
}

// solve A z = b in place (b becomes z) by Gaussian elimination with partial pivoting; false
// when A is numerically singular (the iterate sits at a centre of curvature)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool solve4(amrex::Real A[4][4], amrex::Real b[4])
{
    for (int c = 0; c < 4; ++c)
    {
        int piv = c;
        amrex::Real best = amrex::Math::abs(A[c][c]);
        for (int r = c + 1; r < 4; ++r)
        {
            if (amrex::Math::abs(A[r][c]) > best) { best = amrex::Math::abs(A[r][c]); piv = r; }
        }
        if (!(best > 1.0e-14)) { return false; }
        if (piv != c)
        {
            for (int q = 0; q < 4; ++q) { const amrex::Real tmp = A[c][q]; A[c][q] = A[piv][q]; A[piv][q] = tmp; }
            const amrex::Real tmp = b[c]; b[c] = b[piv]; b[piv] = tmp;
        }
        for (int r = c + 1; r < 4; ++r)
        {
            const amrex::Real fct = A[r][c] / A[c][c];
            for (int q = c; q < 4; ++q) { A[r][q] -= fct * A[c][q]; }
            b[r] -= fct * b[c];
        }
    }
    for (int r = 3; r >= 0; --r)
    {
        amrex::Real s = b[r];
        for (int q = r + 1; q < 4; ++q) { s -= A[r][q] * b[q]; }
        b[r] = s / A[r][r];
    }
    return true;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real distance2(const amrex::Real a[3], const amrex::Real b[3])
{
    return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]);
}

// closest point y on the zero set of the interpolant to the cell centre x. from_x: start at
// x and project along the gradient first; otherwise y holds a starting point already on (or
// next to) the interface, such as a neighbour's closest point. h is the cell size and reach
// the farthest an iterate may stray from x. True when converged: on the surface to
// REINIT_TOL_SURFACE cells, with x - y along grad p to REINIT_TOL_ANGLE (a small tangential
// offset changes the distance only at second order, and the piecewise interpolant's gradient
// jumps slightly at cell faces, which stops Newton at about that level)
template <int P>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool closestPoint(amrex::Array4<amrex::Real const> const& phi, const amrex::Real x[3], amrex::Real y[3],
                  bool from_x, amrex::GpuArray<amrex::Real, 3> const& problo,
                  amrex::GpuArray<amrex::Real, 3> const& invdx, amrex::Real h, amrex::Real reach)
{
    constexpr amrex::Real tiny = 1.0e-24;   // |grad p|^2 below this: flat, no direction to the interface
    const amrex::Real reach2 = reach * reach;
    InterpValue f;

    if (from_x)
    {
        for (int d = 0; d < 3; ++d) { y[d] = x[d]; }
        for (int it = 0; it < REINIT_N_PROJ; ++it)
        {
            if (!interpolatePsi<P>(phi, y, problo, invdx, f)) { return false; }
            const amrex::Real gg = f.g[0] * f.g[0] + f.g[1] * f.g[1] + f.g[2] * f.g[2];
            if (!(gg > tiny)) { return false; }
            amrex::Real st[3], ns = 0.0;
            for (int d = 0; d < 3; ++d) { st[d] = -f.v * f.g[d] / gg; ns += st[d] * st[d]; }
            ns = std::sqrt(ns);
            const amrex::Real sc = (ns > REINIT_STEP_PROJ * h) ? REINIT_STEP_PROJ * h / ns : 1.0;
            for (int d = 0; d < 3; ++d) { y[d] += sc * st[d]; }
            if (distance2(x, y) > reach2) { return false; }
        }
    }

    if (!interpolatePsi<P>(phi, y, problo, invdx, f)) { return false; }
    amrex::Real gg = f.g[0] * f.g[0] + f.g[1] * f.g[1] + f.g[2] * f.g[2];
    if (!(gg > tiny)) { return false; }
    amrex::Real lam = ((x[0] - y[0]) * f.g[0] + (x[1] - y[1]) * f.g[1] + (x[2] - y[2]) * f.g[2]) / gg;

    for (int it = 0; it < REINIT_N_NEWTON; ++it)
    {
        if (it > 0 && !interpolatePsi<P>(phi, y, problo, invdx, f)) { return false; }

        // [I + lam H, grad p; grad p^T, 0] [dy; dlam] = -[y - x + lam grad p; p]
        amrex::Real A[4][4] = {{1.0 + lam * f.h[0], lam * f.h[3],       lam * f.h[4],       f.g[0]},
                               {lam * f.h[3],       1.0 + lam * f.h[1], lam * f.h[5],       f.g[1]},
                               {lam * f.h[4],       lam * f.h[5],       1.0 + lam * f.h[2], f.g[2]},
                               {f.g[0],             f.g[1],             f.g[2],             0.0}};
        amrex::Real b[4] = {-(y[0] - x[0] + lam * f.g[0]), -(y[1] - x[1] + lam * f.g[1]),
                            -(y[2] - x[2] + lam * f.g[2]), -f.v};
        if (!solve4(A, b)) { return false; }

        const amrex::Real ns = std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
        const amrex::Real sc = (ns > REINIT_STEP_NEWTON * h) ? REINIT_STEP_NEWTON * h / ns : 1.0;
        for (int d = 0; d < 3; ++d) { y[d] += sc * b[d]; }
        lam += sc * b[3];
        if (distance2(x, y) > reach2) { return false; }
    }

    if (!interpolatePsi<P>(phi, y, problo, invdx, f)) { return false; }
    gg = f.g[0] * f.g[0] + f.g[1] * f.g[1] + f.g[2] * f.g[2];
    if (!(gg > tiny)) { return false; }

    const amrex::Real res_surface = amrex::Math::abs(f.v) / std::sqrt(gg);
    amrex::Real res_angle = 0.0;
    for (int d = 0; d < 3; ++d) { res_angle += (y[d] - x[d] + lam * f.g[d]) * (y[d] - x[d] + lam * f.g[d]); }
    res_angle = std::sqrt(res_angle);
    const amrex::Real dist = std::sqrt(distance2(x, y));

    return res_surface <= REINIT_TOL_SURFACE * h && res_angle <= REINIT_TOL_ANGLE * amrex::max(dist, h);
}

#endif

#endif
