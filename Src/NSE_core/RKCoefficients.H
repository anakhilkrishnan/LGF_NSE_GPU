#ifndef RKCOEFFICIENTS_H_
#define RKCOEFFICIENTS_H_

#include <AMReX_REAL.H>
#include <cmath>

// struct for defining the Butcher RK tableau for a desired scheme
// the indexing is 0-based
// the struct is used by ProjectionWorkspace:
// -> validation
// -> precomputing IFs based on RK scheme
// -> shifting indices to "shifted RK tableau", with 1-based indexing

struct RKButcher
{
    // size for declaring arrays beforehand
    static constexpr int MAX_STAGES = 3;   

    int         n_stages;
    amrex::Real A[MAX_STAGES][MAX_STAGES];
    amrex::Real b[MAX_STAGES];
    amrex::Real c[MAX_STAGES];
};

// set values for the desired HERK scheme below
inline RKButcher getRKButcher()
{
    const amrex::Real r3 = std::sqrt(3.0);

    RKButcher t{}; // zero-init: only nonzero entries below
    t.n_stages = 3;

    t.c[0] = 0.0;
    t.c[1] = 0.5;
    t.c[2] = 1.0;

    t.b[0] = (3.0+r3)/6.0;
    t.b[1] = -r3/3.0;
    t.b[2] = (3.0+r3)/6.0;

    t.A[1][0] = 0.5;
    t.A[2][0] = r3/3.0;
    t.A[2][1] = (3.0-r3)/3.0;

    return t;
}

#endif