import mpmath as mp
import math

import numpy as np
import scipy

def integrand_finite_c0(lbd, n,m,alpha,theta):
    a = alpha*np.cos(theta)*(0-2) + lbd
    K = (a + np.sqrt(np.square(a) - 4))/2
    res = ( np.cos(theta*n) *((1/K)**m ) - 1) / (K - 1/K)
    return res

def eval_lgf_GK_finite_c0(n,m, alpha, eps = 1e-12):
    # Evaluating the LGF using Gauss-Kronrod quadrature but with the 1D finite integral
    lbd = 2 + 2*alpha
    res1 = scipy.integrate.quad(lambda x: integrand_finite_c0(lbd, n,m,alpha,x), -np.pi, 0, epsrel=-1, epsabs=eps, limit = 100000)
    res2 = scipy.integrate.quad(lambda x: integrand_finite_c0(lbd, n,m,alpha,x), 0, np.pi, epsrel=-1, epsabs=eps, limit = 100000)
    return (res1[0]+res2[0])/2/np.pi

size = 32
print(f"static constexpr amrex::Real exact_core[{size}][{size}] = {{")
for n in range(size):
    row = []
    for m in range(size):
        # alpha = 1.0 assumes dx = dy
        val = -1.0 * eval_lgf_GK_finite_c0(n, m, 1.0) 
        row.append(f"{val:.12f}")
    print("    {" + ", ".join(row) + "},")
print("};")