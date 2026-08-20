#ifndef SPATIALDISCRETIZATION_H_
#define SPATIALDISCRETIZATION_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BCRec.H>
#include <AMReX_Math.H>

using namespace amrex;

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
// velocity 
// returns scalar corresponding to template <dim> dimension 
// written for in-place return
template <int dim>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real nonLinearTerm(int i, int j, int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& invdx,
                            amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel_arr)
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
        amrex::Real w3u2 = 0.25 * ( (discreteCurlF2E<2>(i, j+1, k, invdx, vel_arr) * (v(i, j+1, k) + v(i-1, j+1, k))) 
                                    + (discreteCurlF2E<2>(i, j, k, invdx, vel_arr) * (v(i, j, k) + v(i-1, j, k))) );
#if AMREX_SPACEDIM == 3
        amrex::Real w2u3 = 0.25 * ( (discreteCurlF2E<1>(i, j, k+1, invdx, vel_arr) * (w(i, j, k+1) + w(i-1, j, k+1))) 
                                    + (discreteCurlF2E<1>(i, j, k, invdx, vel_arr) * (w(i, j, k) + w(i-1, j, k))) );
        return (w2u3 - w3u2);
#else 
        return (-w3u2);
#endif
    }
    else if constexpr (dim == 1)
    {
        amrex::Real w3u1 = 0.25 * ( (discreteCurlF2E<2>(i+1, j, k, invdx, vel_arr) * (u(i+1, j, k) + u(i+1, j-1, k)))
                                    + (discreteCurlF2E<2>(i, j, k, invdx, vel_arr) * (u(i, j, k) + u(i, j-1, k))) );
#if AMREX_SPACEDIM == 3
        amrex::Real w1u3 = 0.25 * ( (discreteCurlF2E<0>(i, j, k+1, invdx, vel_arr) * (w(i, j, k+1) + w(i, j-1, k+1)))
                                    + (discreteCurlF2E<0>(i, j, k, invdx, vel_arr) * (w(i, j, k) + w(i, j-1, k))) );


        return (w3u1 - w1u3);
#else 
        return (w3u1);
#endif
    }
#if AMREX_SPACEDIM == 3
    else if constexpr (dim == 2)
    {
        amrex::Real w1u2 = 0.25 * ( (discreteCurlF2E<0>(i, j+1, k, invdx, vel_arr) * (v(i, j+1, k) + v(i, j+1, k-1)))
                                    + (discreteCurlF2E<0>(i, j, k, invdx, vel_arr) * (v(i, j, k) + v(i, j, k-1))) );
        amrex::Real w2u1 = 0.25 * ( (discreteCurlF2E<1>(i+1, j, k, invdx, vel_arr) * (u(i+1 ,j, k) + u(i+1, j, k-1)))
                                    + (discreteCurlF2E<1>(i, j, k, invdx, vel_arr) * (u(i, j,k) + u(i, j,k-1))) );
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
                                    amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel_arr)
{
    amrex::Real div = ((nonLinearTerm<0>(i+1,j,k, invdx, vel_arr) - nonLinearTerm<0>(i,j,k, invdx, vel_arr)) * invdx[0]
                        + (nonLinearTerm<1>(i,j+1,k, invdx, vel_arr) - nonLinearTerm<1>(i,j,k, invdx, vel_arr)) * invdx[1]);
#if AMREX_SPACEDIM == 3
    div += (nonLinearTerm<2>(i,j,k+1, invdx, vel_arr) - nonLinearTerm<2>(i,j,k, invdx, vel_arr)) * invdx[2];
#endif 

    return div;
}

// Morinishi convective term computations for vector velocity and scalar pressure fields
// returns scalar corresponding to template <dim> dimension
// written with face-centered veclocity/ cell-centered pressure and face-centered return in mind

template <int dim>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real morinishiConvective(int i, int j, int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& invdx, amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel_arr)
{
    // extract velocities from vel_arr
    auto const& u = vel_arr[0];
    auto const& v = vel_arr[1];
#if AMREX_SPACEDIM == 3
    auto const& w = vel_arr[2];
#endif

    amrex::Real convterm = 0.0;
    // computing the convective term with the following references
    // considering a box around the point of stored data
    // F/B -> +- z-direction
    // N/S -> +- y-direction
    // E/W -> +- x-direction

    if constexpr (dim == 0)
    {
        amrex::Real u_E = 0.5 * (u(i+1,j,k) + u(i,j,k));
        amrex::Real u_W = 0.5 * (u(i,j,k) + u(i-1,j,k));
        
        amrex::Real conv_x = (u_E * u_E - u_W * u_W) * invdx[0];
        convterm = conv_x;

        amrex::Real u_N = 0.5 * (u(i,j+1,k) + u(i,j,k));
        amrex::Real u_S = 0.5 * (u(i,j,k) + u(i,j-1,k));

        amrex::Real v_N = 0.5 * (v(i,j+1,k) + v(i-1,j+1,k));
        amrex::Real v_S = 0.5 * (v(i,j,k) + v(i-1,j,k));

        amrex::Real conv_y = (u_N * v_N - u_S * v_S) * invdx[1];
        convterm += conv_y;

        #if AMREX_SPACEDIM == 3
            amrex::Real u_F = 0.5 * (u(i,j,k+1) + u(i,j,k));
            amrex::Real u_B = 0.5 * (u(i,j,k) + u(i,j,k-1));

            amrex::Real w_F = 0.5 * (w(i,j,k+1) + w(i-1,j,k+1));
            amrex::Real w_B = 0.5 * (w(i,j,k) + w(i-1,j,k));

            amrex::Real conv_z = (u_F * w_F - u_B * w_B) * invdx[2];
            convterm += conv_z;
        #endif
        
        return convterm;
    }
    else if constexpr (dim == 1)
    {
        amrex::Real u_E = 0.5 * (u(i+1,j,k) + u(i+1,j-1,k));
        amrex::Real u_W = 0.5 * (u(i,j,k) + u(i,j-1,k));
        
        amrex::Real v_E = 0.5 * (v(i+1,j,k) + v(i,j,k));
        amrex::Real v_W = 0.5 * (v(i,j,k) + v(i-1,j,k));

        amrex::Real conv_x = (u_E * v_E - u_W * v_W) * invdx[0];
        convterm = conv_x;

        amrex::Real v_N = 0.5 * (v(i,j+1,k) + v(i,j,k));
        amrex::Real v_S = 0.5 * (v(i,j,k) + v(i,j-1,k));

        amrex::Real conv_y = (v_N * v_N - v_S * v_S) * invdx[1];
        convterm += conv_y;
        
        #if AMREX_SPACEDIM == 3
            amrex::Real w_F = 0.5 * (w(i,j,k+1) + w(i,j-1,k+1));
            amrex::Real w_B = 0.5 * (w(i,j,k) + w(i,j-1,k));

            amrex::Real v_F = 0.5 * (v(i,j,k+1) + v(i,j,k));
            amrex::Real v_B = 0.5 * (v(i,j,k) + v(i,j,k-1));

            amrex::Real conv_z = (v_F * w_F - v_B * w_B) * invdx[2];
            convterm += conv_z;
        #endif

        return convterm;
    }
#if AMREX_SPACEDIM == 3  
    else if constexpr (dim == 2)
    {
        amrex::Real w_E = 0.5 * (w(i+1,j,k) + w(i,j,k));
        amrex::Real w_W = 0.5 * (w(i,j,k) + w(i-1,j,k));

        amrex::Real u_E = 0.5 * (u(i+1,j,k) + u(i+1,j,k-1));
        amrex::Real u_W = 0.5 * (u(i,j,k) + u(i,j,k-1));

        amrex::Real conv_x = (w_E * u_E - w_W * u_W) * invdx[0];
        convterm = conv_x;

        amrex::Real w_N = 0.5 * (w(i,j+1,k) + w(i,j,k));
        amrex::Real w_S = 0.5 * (w(i,j,k) + w(i,j-1,k));

        amrex::Real v_N = 0.5 * (v(i,j+1,k) + v(i,j+1,k-1));
        amrex::Real v_S = 0.5 * (v(i,j,k) + v(i,j,k-1));

        amrex::Real conv_y = (w_N * v_N - w_S * v_S) * invdx[1];
        convterm += conv_y;
        
        amrex::Real w_F = 0.5 * (w(i,j,k+1) + w(i,j,k));
        amrex::Real w_B = 0.5 * (w(i,j,k) + w(i,j,k-1));

        amrex::Real conv_z = (w_F * w_F - w_B * w_B) * invdx[2];
        convterm += conv_z;

        return convterm;
    }
#endif
    else 
    {
        // this should never be reached
        return 0.0;
    }
}

// single call function for divergence of morinishiConvective
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real divMorinishiConvective(int i, int j, int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& invdx,
                                    amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel_arr)
{
    amrex::Real div = ((morinishiConvective<0>(i+1,j,k, invdx, vel_arr) - morinishiConvective<0>(i,j,k, invdx, vel_arr)) * invdx[0]
                        + (morinishiConvective<1>(i,j+1,k, invdx, vel_arr) - morinishiConvective<1>(i,j,k, invdx, vel_arr)) * invdx[1]);
#if AMREX_SPACEDIM == 3
    div += (morinishiConvective<2>(i,j,k+1, invdx, vel_arr) - morinishiConvective<2>(i,j,k, invdx, vel_arr)) * invdx[2];
#endif 

    return div;
}

// single call function for RHS of momentum update in RK-stages
template <int dim>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real morinishiFlux(int i, int j, int k, amrex::Real invRe,
                          amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& invdx, 
                          amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& invdx2,
                          amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel_arr, 
                          amrex::Array4<amrex::Real const> const& p)
{
    amrex::Real presterm = discreteGradientC2F<dim>(i, j, k, invdx, p);
    amrex::Real viscterm = invRe * discreteLaplacian(i, j, k, invdx2, vel_arr[dim]);
    // amrex::Real convterm = morinishiConvective<dim>(i, j, k, invdx, vel_arr);
    amrex::Real convterm = nonLinearTerm<dim>(i, j, k, invdx, vel_arr);

    return (viscterm - presterm - convterm); 
}

#endif