#ifndef INTERFACEDISCRETIZATION_H_
#define INTERFACEDISCRETIZATION_H_

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <utility>

// device kernels for transporting the interface's signed distance function psi
// with the face velocities of the staggered grid, in advective (Hamilton-Jacobi)
// form:
//
//     d(psi)/dt = - sum_dim u_dim * D_dim(psi),   u_dim = average of the cell's two dim-faces + u_inf
//
// D_dim is the upwind one-sided derivative (D- for u >= 0, D+ otherwise) from a
// five-point WENO5 or HOUC5 reconstruction of face gradients (Jiang & Peng).
// The same one-sided derivatives are what a reinitialization's Godunov
// Hamiltonian needs.
//
// WHY NOT THE FLUX FORM: with Du = 0 the flux form -div(u psi) is the same PDE,
// but its O(dx^2) error is proportional to the strain du_dim/dx_dim; averaging
// the velocity instead leaves an O(dx^2) error proportional to d2u/dx2 only. On a
// pure strain field the flux form measured ~100x larger error at 128^3 (second
// order versus fifth). Conserving the integral of psi buys nothing for an SDF.

// ghost cells psi needs: D- reads cells i-3 ... i+2, D+ reads i-2 ... i+3
inline constexpr int LS_N_GHOST = 3;

// regularization of the WENO5 weights; the reconstructed values are gradients of
// a signed distance function, which are O(1)
inline constexpr amrex::Real WENO5_EPS = 1.0e-6;

enum class AdvectionScheme : int { WENO5 = 0, HOUC5 = 1 };

// value of a at the cell m steps away from (i,j,k) along direction dim
template <int dim>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real valueAlong(amrex::Array4<amrex::Real const> const& a, int i, int j, int k, int n, int m)
{
    if constexpr (dim == 0)
    {
        return a(i+m, j, k, n);
    }
    else if constexpr (dim == 1)
    {
        return a(i, j+m, k, n);
    }
    else
    {
        return a(i, j, k+m, n);
    }
}

// the three third-order candidate reconstructions shared by WENO5 and HOUC5.
// v0 ... v4 are five consecutive values ordered from far upwind (v0) to
// downwind (v4); each candidate reconstructs the value between v2 and v3
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, 3> upwindCandidates(amrex::Real v0, amrex::Real v1, amrex::Real v2,
                                                 amrex::Real v3, amrex::Real v4)
{
    return {( 2.0*v0 - 7.0*v1 + 11.0*v2) / 6.0,
            (-v1 + 5.0*v2 + 2.0*v3) / 6.0,
            ( 2.0*v2 + 5.0*v3 - v4) / 6.0};
}

// upwind-biased five-point reconstruction of the value between v2 and v3.
// HOUC5 is the fixed optimal combination (1/10, 6/10, 3/10) of the candidates,
// i.e. the linear fifth-order upwind stencil; WENO5 (Jiang & Shu) replaces the
// fixed weights with smoothness-dependent ones, recovering HOUC5 where psi is
// smooth and dropping candidates that straddle a kink
template <AdvectionScheme scheme>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real reconstructUpwind(amrex::Real v0, amrex::Real v1, amrex::Real v2, amrex::Real v3,
                              amrex::Real v4, amrex::Real eps)
{
    const amrex::GpuArray<amrex::Real, 3> q = upwindCandidates(v0, v1, v2, v3, v4);

    if constexpr (scheme == AdvectionScheme::HOUC5)
    {
        amrex::ignore_unused(eps);
        return 0.1*q[0] + 0.6*q[1] + 0.3*q[2];
    }
    else
    {
        // smoothness indicators: curvature (c) and slope (s) of each candidate
        const amrex::Real c0 = v0 - 2.0*v1 + v2,  s0 = v0 - 4.0*v1 + 3.0*v2;
        const amrex::Real c1 = v1 - 2.0*v2 + v3,  s1 = v1 - v3;
        const amrex::Real c2 = v2 - 2.0*v3 + v4,  s2 = 3.0*v2 - 4.0*v3 + v4;

        const amrex::Real b0 = (13.0/12.0)*c0*c0 + 0.25*s0*s0;
        const amrex::Real b1 = (13.0/12.0)*c1*c1 + 0.25*s1*s1;
        const amrex::Real b2 = (13.0/12.0)*c2*c2 + 0.25*s2*s2;

        const amrex::Real a0 = 0.1 / ((eps + b0)*(eps + b0));
        const amrex::Real a1 = 0.6 / ((eps + b1)*(eps + b1));
        const amrex::Real a2 = 0.3 / ((eps + b2)*(eps + b2));

        return (a0*q[0] + a1*q[1] + a2*q[2]) / (a0 + a1 + a2);
    }
}

// gradient of psi along dim at the low face of the cell m steps from (i,j,k)
template <int dim>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real faceGradient(amrex::Array4<amrex::Real const> const& psi, int i, int j, int k, int n, int m,
                         amrex::Real invdx)
{
    return (valueAlong<dim>(psi, i, j, k, n, m) - valueAlong<dim>(psi, i, j, k, n, m - 1)) * invdx;
}

// one-sided derivative of psi along dim at cell (i,j,k): side < 0 gives D-
// (biased to the low side), side > 0 gives D+. The five face gradients are
// passed from far upwind to downwind, so D+ is D- mirrored about the cell
template <AdvectionScheme scheme, int dim>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real oneSidedDerivative(int i, int j, int k, int n, int side,
                               amrex::Array4<amrex::Real const> const& psi, amrex::Real invdx)
{
    if (side < 0)
    {
        return reconstructUpwind<scheme>(faceGradient<dim>(psi, i, j, k, n, -2, invdx),
                                         faceGradient<dim>(psi, i, j, k, n, -1, invdx),
                                         faceGradient<dim>(psi, i, j, k, n,  0, invdx),
                                         faceGradient<dim>(psi, i, j, k, n,  1, invdx),
                                         faceGradient<dim>(psi, i, j, k, n,  2, invdx), WENO5_EPS);
    }
    return reconstructUpwind<scheme>(faceGradient<dim>(psi, i, j, k, n,  3, invdx),
                                     faceGradient<dim>(psi, i, j, k, n,  2, invdx),
                                     faceGradient<dim>(psi, i, j, k, n,  1, invdx),
                                     faceGradient<dim>(psi, i, j, k, n,  0, invdx),
                                     faceGradient<dim>(psi, i, j, k, n, -1, invdx), WENO5_EPS);
}

// u_dim * D_dim(psi) at cell (i,j,k), with u_dim the average of the cell's two
// dim-faces plus the freestream and D_dim taken from the upwind side
template <AdvectionScheme scheme, int dim>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real advectionTerm(int i, int j, int k, int n,
                          amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& invdx,
                          amrex::Array4<amrex::Real const> const& psi,
                          amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> const& vel,
                          amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& uinf)
{
    auto const& u_face = vel[dim];
    const amrex::Real u = uinf[dim] + 0.5 * (u_face(i, j, k) + u_face(i + (dim == 0 ? 1 : 0), j + (dim == 1 ? 1 : 0), k + (dim == 2 ? 1 : 0)));

    return u * oneSidedDerivative<scheme, dim>(i, j, k, n, (u >= 0.0) ? -1 : 1, psi, invdx[dim]);
}

// right-hand side of the transport equation for cell (i,j,k), component n
template <AdvectionScheme scheme, int... dim>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real advectionRHS(int i, int j, int k, int n,
                         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& invdx,
                         amrex::Array4<amrex::Real const> const& psi,
                         amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> const& vel,
                         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& uinf,
                         std::integer_sequence<int, dim...>)
{
    return -(advectionTerm<scheme, dim>(i, j, k, n, invdx, psi, vel, uinf) + ...);
}

#endif
