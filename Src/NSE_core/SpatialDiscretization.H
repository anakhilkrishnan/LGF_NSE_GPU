#ifndef SPATIALDISCRETIZATION_H_
#define SPATIALDISCRETIZATION_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BCRec.H>
#include <AMReX_Math.H>

#include <utility>

using namespace amrex;

// HELPERS for unified handling of 2D and 3D vorticity

// number of vorticity components 
inline constexpr int N_VORT = (AMREX_SPACEDIM == 2) ? 1 : 3;

// edge IndexType
inline amrex::IntVect TheVortEdgeVector(int dim)
{
#if AMREX_SPACEDIM == 2
    amrex::ignore_unused(dim);
    return amrex::IntVect::TheNodeVector();
#elif AMREX_SPACEDIM == 3
    return (amrex::IntVect::TheNodeVector() - amrex::IntVect::TheDimensionVector(dim));
#endif
}

// direction index (component) for other templated functions that require dimension
template <int comp>
inline constexpr int vort_dir = (AMREX_SPACEDIM == 2) ? 2 : comp;



// set of discrete operators used in S.Liska and T.Colonius' paper, "A fast
// lattice Green's function method for solving viscous incompressible flows on
// unbounded domains"

// gradient computation for a scalar field
// returns the scalar for template <dim> dimension
// written for cell-centered input, face-centered output
template <int dim>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real discreteGradientC2F(int i, int j, int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& invdx, amrex::Array4<amrex::Real const> const& phi)
{
    if constexpr (dim == 0)
    {
        return ((phi(i,j,k) - phi(i-1,j,k)) * invdx[0]);
    }
    else if constexpr (dim == 1)
    {
        return ((phi(i,j,k) - phi(i,j-1,k)) * invdx[1]);
    }
    else if constexpr (dim == 2)
    {
        return ((phi(i,j,k) - phi(i,j,k-1)) * invdx[2]);
    }
    else
    {
        // this should never be reached
        return 0.0;
    }
}

// divergence computations for a vector field with 2/3 components
// returns a scalar
// written for face-centered input, cell-centered output
AMREX_GPU_DEVICE AMREX_FORCE_INLINE 
amrex::Real discreteDivergenceF2C(int i, int j, int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& invdx, amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vec)
{

    // extract components from vec
    auto const& u = vec[0];
    amrex::Real div = (u(i+1,j,k) - u(i,j,k)) * invdx[0];

    auto const& v = vec[1];
    div += (v(i,j+1,k) - v(i,j,k)) * invdx[1];

#if AMREX_SPACEDIM == 3
    auto const& w = vec[2];
    div += (w(i,j,k+1) - w(i,j,k)) * invdx[2];
#endif

    return div;
}

// curl computations for a vector field with 2/3 components
// returns the scalar corresponding to template <dim> dimension
// written for face-centered input, edge-centered output
template <int dim>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real discreteCurlF2E(int i, int j, int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& invdx, amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vec)
{
    // extract components from vec
    auto const& u = vec[0];
    auto const& v = vec[1];
#if AMREX_SPACEDIM == 3
    auto const& w = vec[2];
#endif

    // return component-wise
    if constexpr (dim == 2)
    {
        return (((v(i,j,k) - v(i-1,j,k)) * invdx[0]) - ((u(i,j,k) - u(i,j-1,k)) * invdx[1]));
    }

#if AMREX_SPACEDIM == 3
    else if constexpr (dim == 1)
    {
        return (((u(i,j,k) - u(i,j,k-1)) * invdx[2]) - ((w(i,j,k) - w(i-1,j,k)) * invdx[0]));
    }
    else if constexpr (dim == 0)
    {
        return (((w(i,j,k) - w(i,j-1,k)) * invdx[1]) - ((v(i,j,k) - v(i,j,k-1)) * invdx[2]));
    }
#endif

    else
    {
        // this should never be reached
        return 0.0;
    }
}

// discrete curl C^T: edge-centered input, face-centered output
// 3D: vec = {A_x on x-edges, A_y on y-edges, A_z on z-edges}
// 2D: vec[0] = nodal streamfunction psi (= A_z); vec[1] is unused
template <int dim>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real discreteCurlE2F(int i, int j, int k,
                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& invdx,
                            amrex::GpuArray<amrex::Array4<amrex::Real const>, N_VORT> vec)
{
#if AMREX_SPACEDIM == 2
    auto const& psi = vec[0];

    if constexpr (dim == 0) 
    {   // u =  dpsi/dy
        return  (psi(i,j+1,k) - psi(i,j,k)) * invdx[1];
    }
    else
    {   // v = -dpsi/dx
        return -(psi(i+1,j,k) - psi(i,j,k)) * invdx[0];
    }
#elif AMREX_SPACEDIM == 3
    auto const& ax = vec[0];
    auto const& ay = vec[1];
    auto const& az = vec[2];

    if constexpr (dim == 0)
    {
        // u on x-faces
        return ((az(i,j+1,k) - az(i,j,k)) * invdx[1]) - ((ay(i,j,k+1) - ay(i,j,k)) * invdx[2]);
    }
    else if constexpr (dim == 1)
    {   // v on y-faces
        return ((ax(i,j,k+1) - ax(i,j,k)) * invdx[2]) - ((az(i+1,j,k) - az(i,j,k)) * invdx[0]);
    }
    else
    {   // w on z-faces
        return ((ay(i+1,j,k) - ay(i,j,k)) * invdx[0]) - ((ax(i,j+1,k) - ax(i,j,k)) * invdx[1]);
    }    
#endif
}

// laplacian computations for a scalar field
// returns scalar value
// written for in-place return
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real discreteLaplacian(int i, int j, int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& invdx2, amrex::Array4<amrex::Real const> const& phi)
{
    // extract components from vec
    amrex::Real laplacian = (phi(i+1,j,k) - (2 * phi(i,j,k)) + phi(i-1,j,k)) * invdx2[0];

    laplacian += (phi(i,j+1,k) - (2 * phi(i,j,k)) + phi(i,j-1,k)) * invdx2[1];

#if AMREX_SPACEDIM == 3
    laplacian += (phi(i,j,k+1) - (2 * phi(i,j,k)) + phi(i,j,k-1)) * invdx2[2];
#endif

    return laplacian;
}

// convective term (vort cross vel, without 1/2|u|^2) computations for vector
// velocity. vel_arr holds u' = u - u_inf; the uniform freestream uinf is added
// to every face velocity, so the product is omega x (u' + u_inf). omega is
// unchanged (a uniform field has no curl).
// returns scalar corresponding to template <dim> dimension 
// written for in-place return
template <int dim>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real nonLinearTerm(int i, int j, int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& invdx,
                            amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel_arr,
                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& uinf)
{
    // extract velocities from vel_arr
    auto const& u = vel_arr[0];
    auto const& v = vel_arr[1];
#if AMREX_SPACEDIM == 3
    auto const& w = vel_arr[2];
#endif
    // notation: w1, w2, w3 are components of vorticity, not to be confused with w (z-velocity)
    // notation: u1, u2, u3 are components of velocity, not to be confused with u (x-velocity)
    // above only used for variable naming; formula use u,v,w and the discreteCurl() function
    if constexpr (dim == 0)
    {
        amrex::Real w3u2 = 0.25 * ( (discreteCurlF2E<2>(i, j+1, k, invdx, vel_arr) * (v(i, j+1, k) + v(i-1, j+1, k) + 2.0*uinf[1])) 
                                    + (discreteCurlF2E<2>(i, j, k, invdx, vel_arr) * (v(i, j, k) + v(i-1, j, k) + 2.0*uinf[1])) );
#if AMREX_SPACEDIM == 3
        amrex::Real w2u3 = 0.25 * ( (discreteCurlF2E<1>(i, j, k+1, invdx, vel_arr) * (w(i, j, k+1) + w(i-1, j, k+1) + 2.0*uinf[2])) 
                                    + (discreteCurlF2E<1>(i, j, k, invdx, vel_arr) * (w(i, j, k) + w(i-1, j, k) + 2.0*uinf[2])) );
        return (w2u3 - w3u2);
#else 
        return (-w3u2);
#endif
    }
    else if constexpr (dim == 1)
    {
        amrex::Real w3u1 = 0.25 * ( (discreteCurlF2E<2>(i+1, j, k, invdx, vel_arr) * (u(i+1, j, k) + u(i+1, j-1, k) + 2.0*uinf[0]))
                                    + (discreteCurlF2E<2>(i, j, k, invdx, vel_arr) * (u(i, j, k) + u(i, j-1, k) + 2.0*uinf[0])) );
#if AMREX_SPACEDIM == 3
        amrex::Real w1u3 = 0.25 * ( (discreteCurlF2E<0>(i, j, k+1, invdx, vel_arr) * (w(i, j, k+1) + w(i, j-1, k+1) + 2.0*uinf[2]))
                                    + (discreteCurlF2E<0>(i, j, k, invdx, vel_arr) * (w(i, j, k) + w(i, j-1, k) + 2.0*uinf[2])) );


        return (w3u1 - w1u3);
#else 
        return (w3u1);
#endif
    }
#if AMREX_SPACEDIM == 3
    else if constexpr (dim == 2)
    {
        amrex::Real w1u2 = 0.25 * ( (discreteCurlF2E<0>(i, j+1, k, invdx, vel_arr) * (v(i, j+1, k) + v(i, j+1, k-1) + 2.0*uinf[1]))
                                    + (discreteCurlF2E<0>(i, j, k, invdx, vel_arr) * (v(i, j, k) + v(i, j, k-1) + 2.0*uinf[1])) );
        amrex::Real w2u1 = 0.25 * ( (discreteCurlF2E<1>(i+1, j, k, invdx, vel_arr) * (u(i+1 ,j, k) + u(i+1, j, k-1) + 2.0*uinf[0]))
                                    + (discreteCurlF2E<1>(i, j, k, invdx, vel_arr) * (u(i, j,k) + u(i, j,k-1) + 2.0*uinf[0])) );
        return (w1u2 - w2u1);

    }
#endif
    else
    {
        // this should never be reached
        return 0.0;
    }
}

// single call function for divergence of nonLinearTerm
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real divNonLinearTerm(int i, int j, int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& invdx,
                                    amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel_arr,
                                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& uinf)
{
    amrex::Real div = ((nonLinearTerm<0>(i+1,j,k, invdx, vel_arr, uinf) - nonLinearTerm<0>(i,j,k, invdx, vel_arr, uinf)) * invdx[0]
                        + (nonLinearTerm<1>(i,j+1,k, invdx, vel_arr, uinf) - nonLinearTerm<1>(i,j,k, invdx, vel_arr, uinf)) * invdx[1]);
#if AMREX_SPACEDIM == 3
    div += (nonLinearTerm<2>(i,j,k+1, invdx, vel_arr, uinf) - nonLinearTerm<2>(i,j,k, invdx, vel_arr, uinf)) * invdx[2];
#endif 

    return div;
}

#endif