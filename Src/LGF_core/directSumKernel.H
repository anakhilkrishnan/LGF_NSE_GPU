#ifndef DIRECTSUMKERNEL_H_
#define DIRECTSUMKERNEL_H_

#include <AMReX.H>
#include <AMReX_Math.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <LGFKernels.H>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE 
amrex::Real computeLGF(int n_lookup,
                    AMREX_D_DECL(amrex::Real x_tar, amrex::Real y_tar, amrex::Real z_tar),
                    AMREX_D_DECL(amrex::Real x_src, amrex::Real y_src, amrex::Real z_src),
                    AMREX_D_DECL(amrex::Real dx_val, amrex::Real dy_val, amrex::Real dz_val))
{
    // compute physical distance
    amrex:: Real AMREX_D_DECL(dx_dist = x_tar - x_src,
                            dy_dist = y_tar - y_src,
                            dz_dist = z_tar - z_src);
    
    // compute non-dimensional distances without roundoff
    amrex::Real AMREX_D_DECL(deltax = dx_dist / dx_val,
                            deltay = dy_dist / dy_val,
                            deltaz = dz_dist / dz_val);
    
    // convert physical distance to integer grid offsets
    int AMREX_D_DECL(nx = amrex::Math::round(amrex::Math::abs(dx_dist / dx_val)),
                    ny = amrex::Math::round(amrex::Math::abs(dy_dist / dy_val)),
                    nz = amrex::Math::round(amrex::Math::abs(dz_dist / dz_val)));

    // 2D Kernel computations
#if AMREX_SPACEDIM == 2
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_lookup <= 32, "n_lookup larger than lookup table size!");
    if ((nx * nx + ny * ny <= ((n_lookup - 1) * (n_lookup - 1))))
    {
        return nearField2D(nx, ny);
    }

    return farField2D(deltax, deltay);
#endif

#if AMREX_SPACEDIM == 3
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_lookup <= 16, "n_lookup larger than lookup table size!");
    if ((nx * nx + ny * ny + nz * nz <= ((n_lookup - 1) * (n_lookup - 1))))
    {
        return nearField3D(nx, ny, nz);
    }

    return farField3D(deltax, deltay, deltaz);
#endif
}

#endif