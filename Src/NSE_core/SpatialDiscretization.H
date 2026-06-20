#ifndef SPATIALDISCRETIZATION_H_
#define SPATIALDISCRETIZATION_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BCRec.H>
#include <AMReX_Math.H>

using namespace amrex;

// Morinishi KEP discretization for RHS in RK-stage
template <int dim>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real morinishiFlux(int i, int j, int k,
                            amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel_arr,
                            amrex::Array4<amrex::Real const> const& p,
                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
                            amrex::Real Re)
{

    // extract velocities from vel_arr
    auto const& u = vel_arr[0];
    #if AMREX_SPACEDIM >= 2
        auto const& v = vel_arr[1];
    #endif
    #if AMREX_SPACEDIM == 3
        auto const& w = vel_arr[2];
    #endif

    // precompute important divisions
    amrex::Real Re_inv = 1/Re;

    amrex::Real idx  = 1.0 / dx[0];
    amrex::Real idx2 = idx * idx;
    
    #if AMREX_SPACEDIM >= 2
        amrex::Real idy  = 1.0 / dx[1];
        amrex::Real idy2 = idy * idy;
    #endif

    #if AMREX_SPACEDIM == 3
        amrex::Real idz  = 1.0 / dx[2];
        amrex::Real idz2 = idz * idz;
    #endif

    amrex::Real presterm = 0.0;
    amrex::Real viscterm = 0.0;
    amrex::Real convterm = 0.0;

    if constexpr (dim == 0)
    {
        // computing the pressure term
        presterm = idx * (p(i,j,k) - p(i-1,j,k));

        // computing the viscous term dimension by dimension
        amrex::Real d2udx2 = (u(i+1,j,k) - 2.0*u(i,j,k) + u(i-1,j,k)) * idx2;
        viscterm = d2udx2;

        #if AMREX_SPACEDIM >= 2
            amrex::Real d2udy2 = (u(i,j+1,k) - 2.0*u(i,j,k) + u(i,j-1,k)) * idy2;
            viscterm += d2udy2;
        #endif

        #if AMREX_SPACEDIM == 3
            amrex::Real d2udz2 = (u(i,j,k+1) - 2.0*u(i,j,k) + u(i,j,k-1)) * idz2;
            viscterm += d2udz2;
        #endif

        viscterm *= Re_inv;

        // computing the convective term with the following references
        // considering a box around the point of stored data
        // F/B -> +- z-direction
        // N/S -> +- y-direction
        // E/W -> +- x-direction

        amrex::Real u_E = 0.5 * (u(i+1,j,k) + u(i,j,k));
        amrex::Real u_W = 0.5 * (u(i,j,k) + u(i-1,j,k));
        
        amrex::Real conv_x = (u_E * u_E - u_W * u_W) * idx;
        convterm = conv_x;

        #if AMREX_SPACEDIM >= 2
            amrex::Real u_N = 0.5 * (u(i,j+1,k) + u(i,j,k));
            amrex::Real u_S = 0.5 * (u(i,j,k) + u(i,j-1,k));

            amrex::Real v_N = 0.5 * (v(i,j+1,k) + v(i-1,j+1,k));
            amrex::Real v_S = 0.5 * (v(i,j,k) + v(i-1,j,k));

            amrex::Real conv_y = (u_N * v_N - u_S * v_S) * idy;
            convterm += conv_y;
        #endif
        
        #if AMREX_SPACEDIM == 3
            amrex::Real u_F = 0.5 * (u(i,j,k+1) + u(i,j,k));
            amrex::Real u_B = 0.5 * (u(i,j,k) + u(i,j,k-1));

            amrex::Real w_F = 0.5 * (w(i,j,k+1) + w(i-1,j,k+1));
            amrex::Real w_B = 0.5 * (w(i,j,k) + w(i-1,j,k));

            amrex::Real conv_z = (u_F * w_F - u_B * w_B) * idz;
            convterm += conv_z;
        #endif
    }
    #if AMREX_SPACEDIM >= 2
    else if constexpr (dim == 1)
    {
        // computing the pressure term
        presterm = idy * (p(i,j,k) - p(i,j-1,k));

        // computing the viscous term dimension by dimension
        amrex::Real d2vdx2 = (v(i+1,j,k) - 2.0*v(i,j,k) + v(i-1,j,k)) * idx2;
        viscterm = d2vdx2;

        amrex::Real d2vdy2 = (v(i,j+1,k) - 2.0*v(i,j,k) + v(i,j-1,k)) * idy2;
        viscterm += d2vdy2;
        

        #if AMREX_SPACEDIM == 3
            amrex::Real d2vdz2 = (v(i,j,k+1) - 2.0*v(i,j,k) + v(i,j,k-1)) * idz2;
            viscterm += d2vdz2;
        #endif

        viscterm *= Re_inv;

        // computing the convective term with the following references
        // considering a box around the point of stored data
        // F/B -> +- z-direction
        // N/S -> +- y-direction
        // E/W -> +- x-direction

        amrex::Real u_E = 0.5 * (u(i+1,j,k) + u(i+1,j-1,k));
        amrex::Real u_W = 0.5 * (u(i,j,k) + u(i,j-1,k));
        
        amrex::Real v_E = 0.5 * (v(i+1,j,k) + v(i,j,k));
        amrex::Real v_W = 0.5 * (v(i,j,k) + v(i-1,j,k));

        amrex::Real conv_x = (u_E * v_E - u_W * v_W) * idx;
        convterm = conv_x;

        amrex::Real v_N = 0.5 * (v(i,j+1,k) + v(i,j,k));
        amrex::Real v_S = 0.5 * (v(i,j,k) + v(i,j-1,k));

        amrex::Real conv_y = (v_N * v_N - v_S * v_S) * idy;
        convterm += conv_y;
        
        #if AMREX_SPACEDIM == 3
            amrex::Real w_F = 0.5 * (w(i,j,k+1) + w(i,j-1,k+1));
            amrex::Real w_B = 0.5 * (w(i,j,k) + w(i,j-1,k));

            amrex::Real v_F = 0.5 * (v(i,j,k+1) + v(i,j,k));
            amrex::Real v_B = 0.5 * (v(i,j,k) + v(i,j,k-1));

            amrex::Real conv_z = (v_F * w_F - v_B * w_B) * idz;
            convterm += conv_z;
        #endif
    }
    #endif

    #if AMREX_SPACEDIM == 3
    else if constexpr (dim == 2)
    {
        // computing the pressure term
        presterm = idz * (p(i,j,k) - p(i,j,k-1));

        // computing the viscous term dimension by dimension
        amrex::Real d2wdx2 = (w(i+1,j,k) - 2.0*w(i,j,k) + w(i-1,j,k)) * idx2;
        viscterm = d2wdx2;

        amrex::Real d2wdy2 = (w(i,j+1,k) - 2.0*w(i,j,k) + w(i,j-1,k)) * idy2;
        viscterm += d2wdy2;

        amrex::Real d2wdz2 = (w(i,j,k+1) - 2.0*w(i,j,k) + w(i,j,k-1)) * idz2;
        viscterm += d2wdz2;

        viscterm *= Re_inv;

        // computing the convective term with the following references
        // considering a box around the point of stored data
        // F/B -> +- z-direction
        // N/S -> +- y-direction
        // E/W -> +- x-direction

        amrex::Real w_E = 0.5 * (w(i+1,j,k) + w(i,j,k));
        amrex::Real w_W = 0.5 * (w(i,j,k) + w(i-1,j,k));

        amrex::Real u_E = 0.5 * (u(i+1,j,k) + u(i+1,j,k-1));
        amrex::Real u_W = 0.5 * (u(i,j,k) + u(i,j,k-1));

        amrex::Real conv_x = (w_E * u_E - w_W * u_W) * idx;
        convterm = conv_x;

        amrex::Real w_N = 0.5 * (w(i,j+1,k) + w(i,j,k));
        amrex::Real w_S = 0.5 * (w(i,j,k) + w(i,j-1,k));

        amrex::Real v_N = 0.5 * (v(i,j+1,k) + v(i,j+1,k-1));
        amrex::Real v_S = 0.5 * (v(i,j,k) + v(i,j,k-1));

        amrex::Real conv_y = (w_N * v_N - w_S * v_S) * idy;
        convterm += conv_y;
        
        amrex::Real w_F = 0.5 * (w(i,j,k+1) + w(i,j,k));
        amrex::Real w_B = 0.5 * (w(i,j,k) + w(i,j,k-1));

        amrex::Real conv_z = (w_F * w_F - w_B * w_B) * idz;
        convterm += conv_z;
    }
    #endif
    
    return (viscterm - presterm - convterm);
}

// Morinishi KEP divergence computation for RHS in LGF step
AMREX_GPU_DEVICE AMREX_FORCE_INLINE 
amrex::Real discreteDivergence(int i, int j, int k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
                                 amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel_arr)
{

    // extract velocities from vel_arr
    auto const& u = vel_arr[0];
    amrex::Real div = (u(i+1,j,k) - u(i,j,k)) / dx[0];

    #if AMREX_SPACEDIM >= 2
    auto const& v = vel_arr[1];
    div += (v(i,j+1,k) - v(i,j,k)) / dx[1];
    #endif
    #if AMREX_SPACEDIM == 3
    auto const& w = vel_arr[2];
    div += (w(i,j,k+1) - w(i,j,k)) / dx[2];
    #endif

    return div;
}

template <int dim>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real discreteGradient(int i, int j, int k, amrex::Array4<amrex::Real const> const& p,
                                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx)
{
    if constexpr (dim == 0)
    {
        return ((p(i,j,k) - p(i-1,j,k))/dx[0]);
    }
    else if constexpr (dim == 1)
    {
        return ((p(i,j,k) - p(i,j-1,k))/dx[1]);
    }
    else if constexpr (dim == 2)
    {
        return ((p(i,j,k) - p(i,j,k-1))/dx[2]);
    }
}

// Morinishi KEP discretization for component-wise Kinetic Energy equation
template <int dim>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real morinishiKECompFlux(int i, int j, int k,
                            amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel,
                            amrex::Array4<amrex::Real const> const& p,
                            amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> kecomp,
                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
                            amrex::Real Re)
{
    // extract velocities and kinetic energy components from vel and kecomp
    auto const& u = vel[0];
    auto const& ke_u = kecomp[0];
    #if AMREX_SPACEDIM >= 2
        auto const& v = vel[1];
        auto const& ke_v = kecomp[1];
    #endif
    #if AMREX_SPACEDIM == 3
        auto const& w = vel[2];
        auto const& ke_w = kecomp[2];
    #endif

    // precompute important divisions
    amrex::Real Re_inv = 1/Re;

    amrex::Real idx  = 1.0 / dx[0];
    amrex::Real idx2 = idx * idx;
    
    #if AMREX_SPACEDIM >= 2
        amrex::Real idy  = 1.0 / dx[1];
        amrex::Real idy2 = idy * idy;
    #endif

    #if AMREX_SPACEDIM == 3
        amrex::Real idz  = 1.0 / dx[2];
        amrex::Real idz2 = idz * idz;
    #endif

    amrex::Real presterm = 0.0;
    amrex::Real viscterm = 0.0;
    amrex::Real convterm = 0.0;

    if constexpr (dim == 0)
    {
        // computing the pressure term
        presterm = u(i,j,k) * idx * (p(i,j,k) - p(i-1,j,k));

        // computing the viscous term dimension by dimension
        amrex::Real d2udx2 = (u(i+1,j,k) - 2.0*u(i,j,k) + u(i-1,j,k)) * idx2;
        viscterm = d2udx2;

    #if AMREX_SPACEDIM >= 2
        amrex::Real d2udy2 = (u(i,j+1,k) - 2.0*u(i,j,k) + u(i,j-1,k)) * idy2;
        viscterm += d2udy2;
    #endif

    #if AMREX_SPACEDIM == 3
        amrex::Real d2udz2 = (u(i,j,k+1) - 2.0*u(i,j,k) + u(i,j,k-1)) * idz2;
        viscterm += d2udz2;
    #endif

        viscterm *= u(i,j,k) * Re_inv;

        // computing the convective term with the following references
        // considering a box around the point of stored data
        // F/B -> +- z-direction
        // N/S -> +- y-direction
        // E/W -> +- x-direction

        // expressions correspond to the "divergence" form of the ke_i
        // convective term; which corresponds to the skew form of the 
        // momentum convection term in Morinishi et. al.

        // evaluating the advecting velocity
        amrex::Real u_E = 0.5 * (u(i+1,j,k) + u(i,j,k));
        amrex::Real u_W = 0.5 * (u(i,j,k) + u(i-1,j,k));

        // evaluating KE
        amrex::Real ke_u_E = 0.5 * (u(i+1,j,k) * u(i,j,k));
        amrex::Real ke_u_W = 0.5 * (u(i,j,k) * u(i-1,j,k));

        amrex::Real conv_x = (u_E * ke_u_E - u_W * ke_u_W) * idx;
        convterm = conv_x;

    #if AMREX_SPACEDIM >= 2

        amrex::Real v_N = 0.5 * (v(i,j+1,k) + v(i-1,j+1,k));
        amrex::Real v_S = 0.5 * (v(i,j,k) + v(i-1,j,k));

        amrex::Real ke_u_N = 0.5 * (u(i,j+1,k) * u(i,j,k));
        amrex::Real ke_u_S = 0.5 * (u(i,j,k) * u(i,j-1,k));

        amrex::Real conv_y = (v_N * ke_u_N - v_S * ke_u_S) * idy;
        convterm += conv_y;
    #endif
    
    #if AMREX_SPACEDIM == 3

        amrex::Real w_F = 0.5 * (w(i,j,k+1) + w(i-1,j,k+1));
        amrex::Real w_B = 0.5 * (w(i,j,k) + w(i-1,j,k));

        amrex::Real ke_u_F = 0.5 * (u(i,j,k+1) * u(i,j,k));
        amrex::Real ke_u_B = 0.5 * (u(i,j,k) * u(i,j,k-1));

        amrex::Real conv_z = (w_F * ke_u_F - w_B * ke_u_B) * idz;
        convterm += conv_z;
    #endif
    }
#if AMREX_SPACEDIM >= 2
    else if constexpr (dim == 1)
    {
        // computing the pressure term
        presterm = v(i,j,k) * idy * (p(i,j,k) - p(i,j-1,k));

        // computing the viscous term dimension by dimension
        amrex::Real d2vdx2 = (v(i+1,j,k) - 2.0*v(i,j,k) + v(i-1,j,k)) * idx2;
        viscterm = d2vdx2;

        amrex::Real d2vdy2 = (v(i,j+1,k) - 2.0*v(i,j,k) + v(i,j-1,k)) * idy2;
        viscterm += d2vdy2;
        
    #if AMREX_SPACEDIM == 3
        amrex::Real d2vdz2 = (v(i,j,k+1) - 2.0*v(i,j,k) + v(i,j,k-1)) * idz2;
        viscterm += d2vdz2;
    #endif

        viscterm *= v(i,j,k) * Re_inv;

        // computing the convective term with the following references
        // considering a box around the point of stored data
        // F/B -> +- z-direction
        // N/S -> +- y-direction
        // E/W -> +- x-direction

        amrex::Real u_E = 0.5 * (u(i+1,j,k) + u(i+1,j-1,k));
        amrex::Real u_W = 0.5 * (u(i,j,k) + u(i,j-1,k));
        
        amrex::Real ke_v_E = 0.5 * (v(i+1,j,k) * v(i,j,k));
        amrex::Real ke_v_W = 0.5 * (v(i,j,k) * v(i-1,j,k));

        amrex::Real conv_x = (u_E * ke_v_E - u_W * ke_v_W) * idx;
        convterm = conv_x;

        amrex::Real v_N = 0.5 * (v(i,j+1,k) + v(i,j,k));
        amrex::Real v_S = 0.5 * (v(i,j,k) + v(i,j-1,k));

        amrex::Real ke_v_N = 0.5 * (v(i,j+1,k) * v(i,j,k));
        amrex::Real ke_v_S = 0.5 * (v(i,j,k) * v(i,j-1,k));

        amrex::Real conv_y = (v_N * ke_v_N - v_S * ke_v_S) * idy;
        convterm += conv_y;
        
    #if AMREX_SPACEDIM == 3
        amrex::Real w_F = 0.5 * (w(i,j,k+1) + w(i,j-1,k+1));
        amrex::Real w_B = 0.5 * (w(i,j,k) + w(i,j-1,k));

        amrex::Real ke_v_F = 0.5 * (v(i,j,k+1) * v(i,j,k));
        amrex::Real ke_v_B = 0.5 * (v(i,j,k) * v(i,j,k-1));

        amrex::Real conv_z = (w_F * ke_v_F - w_B * ke_v_B) * idz;
        convterm += conv_z;
    #endif
    }
#endif
#if AMREX_SPACEDIM == 3
    else if constexpr (dim == 2)
    {
        // computing the pressure term
        presterm = w(i,j,k) * idz * (p(i,j,k) - p(i,j,k-1));

        // computing the viscous term dimension by dimension
        amrex::Real d2wdx2 = (w(i+1,j,k) - 2.0*w(i,j,k) + w(i-1,j,k)) * idx2;
        viscterm = d2wdx2;

        amrex::Real d2wdy2 = (w(i,j+1,k) - 2.0*w(i,j,k) + w(i,j-1,k)) * idy2;
        viscterm += d2wdy2;

        amrex::Real d2wdz2 = (w(i,j,k+1) - 2.0*w(i,j,k) + w(i,j,k-1)) * idz2;
        viscterm += d2wdz2;

        viscterm *= w(i,j,k) * Re_inv;

        // computing the convective term with the following references
        // considering a box around the point of stored data
        // F/B -> +- z-direction
        // N/S -> +- y-direction
        // E/W -> +- x-direction

        amrex::Real u_E = 0.5 * (u(i+1,j,k) + u(i+1,j,k-1));
        amrex::Real u_W = 0.5 * (u(i,j,k) + u(i,j,k-1));

        amrex::Real ke_w_E = 0.5 * (w(i+1,j,k) * w(i,j,k));
        amrex::Real ke_w_W = 0.5 * (w(i,j,k) * w(i-1,j,k));

        amrex::Real conv_x = (u_E * ke_w_E - u_W * ke_w_W) * idx;
        convterm = conv_x;

        amrex::Real v_N = 0.5 * (v(i,j+1,k) + v(i,j+1,k-1));
        amrex::Real v_S = 0.5 * (v(i,j,k) + v(i,j,k-1));

        amrex::Real ke_w_N = 0.5 * (w(i,j+1,k) * w(i,j,k));
        amrex::Real ke_w_S = 0.5 * (w(i,j,k) * w(i,j-1,k));

        amrex::Real conv_y = (v_N * ke_w_N - v_S * ke_w_S) * idy;
        convterm += conv_y;
        
        amrex::Real w_F = 0.5 * (w(i,j,k+1) + w(i,j,k));
        amrex::Real w_B = 0.5 * (w(i,j,k) + w(i,j,k-1));

        amrex::Real ke_w_F = 0.5 * (w(i,j,k+1) * w(i,j,k));
        amrex::Real ke_w_B = 0.5 * (w(i,j,k) * w(i,j,k-1));

        amrex::Real conv_z = (w_F * ke_w_F - w_B * ke_w_B) * idz;
        convterm += conv_z;
    }
#endif

    return (viscterm - presterm - convterm);
}

#endif
