#ifndef RKCOEFFICIENTS_H_
#define RKCOEFFICIENTS_H_

#include <AMReX_Vector.H>
#include <AMReX_REAL.H>

struct RKCoeffs
{
    amrex::Real alp;
    amrex::Real bet;
    amrex::Real gam;
};

// Return a Vector instead of an Array!
inline amrex::Vector<RKCoeffs> getRKCoeffs(int rk_stages)
{
    amrex::Vector<RKCoeffs> coeffs;

    if (rk_stages == 1) 
    {
        // Forward Euler (1 Stage)
        coeffs.push_back({0.0, 1.0, 1.0});
    }
    else if (rk_stages == 2) 
    {
        // SSP-RK2 (2 Stages)
        coeffs.push_back({0.0, 1.0, 1.0});
        coeffs.push_back({0.5, 0.5, 2.0});
    }
    else if (rk_stages == 3) 
    {
        // SSP-RK3 (3 Stages)
        coeffs.push_back({0.0, 1.0, 1.0});
        coeffs.push_back({0.75, 0.25, 4.0});
        coeffs.push_back({1.0/3.0, 2.0/3.0, 1.5});
    }
    else 
    {
        amrex::Abort("Error: Unsupported number of RK stages! Choose 1, 2, or 3.");
    }

    return coeffs;
}

#endif