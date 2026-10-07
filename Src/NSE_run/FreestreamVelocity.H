#ifndef FREESTREAMVELOCITY_H_
#define FREESTREAMVELOCITY_H_

#include <AMReX_REAL.H>
#include <AMReX_Array.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>

// prescribe freestream velocity behavior through inlined functions
// intended to be computed host-side once, and passed to any device kernel

namespace freestream_params
{
    // general case, no freestream velocity
    // constexpr amrex::Real U[3] = {0.0, 0.0, 0.0};

    // Liska & Colonius (2016), Sec. 6.4: constant u_inf = (0, 0, -0.18686) Gamma0/R0
    constexpr amrex::Real Ux = 0.0, Uy = 0.0, Uz = -0.18686;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> freestreamVelocity (amrex::Real t) noexcept
{
    amrex::ignore_unused(t);

    using namespace freestream_params;
    return {AMREX_D_DECL(Ux, Uy, Uz)};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> freestreamDisplacement (amrex::Real t) noexcept
{
    using namespace freestream_params;
    
    // correct for constant freestream
    return {AMREX_D_DECL(Ux * t, Uy * t, Uz * t)};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> freestreamAcceleration (amrex::Real t) noexcept
{
    amrex::ignore_unused(t);

    // correct for constant freestream
    return {AMREX_D_DECL(0.0, 0.0, 0.0)};
}

#endif
