#ifndef LGFNSEOMP_H_
#define LGFNSEOMP_H_

#include <AMReX_Gpu.H> // to enforce include order, as AMReX doesn't pass the flag AMReX_OMP directly and includes it in a config header

#ifdef AMREX_USE_OMP
#define LGF_NSE_OMP_MFITER _Pragma("omp parallel if (amrex::Gpu::notInLaunchRegion())")
#else
#define LGF_NSE_OMP_MFITER
#endif

#if defined(AMREX_USE_OMP) && !defined(AMREX_USE_GPU)
#define LGF_NSE_OMP_HOST 1
#else
#define LGF_NSE_OMP_HOST 0
#endif

#endif
