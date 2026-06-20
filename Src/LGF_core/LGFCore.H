#ifndef LGFCORE_H_
#define LGFCORE_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BCRec.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_BLProfiler.H>


// defining structs used within LGFcore
struct FabMetaData
{
    /**
     * @brief This is a basic struct that stores box-wise meta data: box number offset, box lo, box hi
     * 
     */
    int offset;
    amrex::IntVect lo;
    amrex::IntVect hi;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx;
};

struct ConsolidatedData 
{
    amrex::Vector<amrex::Real> data;
    amrex::Vector<FabMetaData> metadata;
};

// defining key functions within LGFcore
ConsolidatedData consolidateMultiFab(const amrex::MultiFab& phifab, const amrex::Geometry& geom, amrex::Vector<int> box_tag_arr);
void addEverySourceBox(const amrex::MultiFab& source, amrex::MultiFab& target, const amrex::Geometry& geom, amrex::Vector<int> box_tag_arr);
amrex::Vector<int> tagSource(const amrex::MultiFab& phifab, const amrex::Real source_threshold = 1.0e-8);
// defining function for BBFMM
void solveFMM(const amrex::MultiFab& source, amrex::MultiFab& target, const amrex::Geometry& geom, int n_chebyshev, int n_lookup); 

// function to compute discrete LGF kernel for 2D
AMREX_GPU_DEVICE AMREX_FORCE_INLINE 
amrex::Real computeLGF(AMREX_D_DECL(amrex::Real x_tar, amrex::Real y_tar, amrex::Real z_tar),
                                        AMREX_D_DECL(amrex::Real x_src, amrex::Real y_src, amrex::Real z_src), amrex::Real dx_val)
{
    // compute physical distance
    amrex::Real dx_dist = x_tar - x_src;
    amrex::Real dy_dist = y_tar - y_src;
    
    // 1. Convert physical distance to integer grid offsets
    int n = amrex::Math::abs(amrex::Math::round(dx_dist / dx_val));
    int m = amrex::Math::abs(amrex::Math::round(dy_dist / dx_val));

    // 2. Check lookup table for near-field core
    if (n*n + m*m <= 49) 
    {
        static constexpr amrex::Real exact_core[8][8] = {
            {-0.000000000000, 0.250000000000, 0.363380227632, 0.430281365795, 0.476993647394, 0.512902329079, 0.542115430423, 0.566760291019},
            {0.250000000000, 0.318309886184, 0.386619772368, 0.440375794076, 0.482395447352, 0.516250119249, 0.544399550798, 0.568421922546},
            {0.363380227632, 0.386619772368, 0.424413181578, 0.462206590789, 0.495962228688, 0.525303149768, 0.550810730973, 0.573181291838},
            {0.430281365795, 0.440375794076, 0.462206590789, 0.488075158815, 0.513943726841, 0.538189520163, 0.560358931488, 0.580479669282},
            {0.476993647394, 0.482395447352, 0.495962228688, 0.513943726841, 0.533547999699, 0.553152272556, 0.571955805532, 0.589632760981},
            {0.512902329079, 0.516250119249, 0.525303149768, 0.538189520163, 0.553152272556, 0.568915764830, 0.584679257104, 0.599992430844},
            {0.542115430423, 0.544399550798, 0.550810730973, 0.560358931488, 0.571955805532, 0.584679257104, 0.597853027210, 0.611026797317},
            {0.566760291019, 0.568421922546, 0.573181291838, 0.580479669282, 0.589632760981, 0.599992430844, 0.611026797317, 0.622338403071},
        };
        return exact_core[n][m];
    }
    
    // 3. Far-Field: Asymptotic Expansion with Multipole Correction
    // Non-dimensionalize r^2 to the grid
    amrex::Real rsquare = (dx_dist*dx_dist + dy_dist*dy_dist) / (dx_val*dx_val);
    
    amrex::Real r2 = rsquare;
    amrex::Real r4 = r2 * r2;
    amrex::Real r6 = r4 * r2;
    
    // Base Continuous LGF + Constant Asymptotic Shift for 2D
    amrex::Real C = 0.2573420803; 
    amrex::Real G_cont = (1.0 / (4.0 * amrex::Math::pi<amrex::Real>())) * std::log(r2) + C;
    
    // Anisotropy (Multipole) Correction for the 2D discrete grid
    amrex::Real n2 = static_cast<amrex::Real>(n * n);
    amrex::Real m2 = static_cast<amrex::Real>(m * m);
    amrex::Real multipole = (n2*n2 + m2*m2 - 6.0*n2*m2) / (24.0 * amrex::Math::pi<amrex::Real>() * r6);
    
    // Subtract the error correction from the continuous approximation
    return G_cont - multipole;
}
#endif