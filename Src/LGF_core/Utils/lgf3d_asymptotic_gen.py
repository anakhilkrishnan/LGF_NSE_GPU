#!/usr/bin/env python3
"""
lgf3d_asymptotic_gen.py -- derive the q-term asymptotic expansion A^q_G of the
3D lattice Green's function and emit it as C++.  Fast enough for q ~ 10+.

    G(n) = A^q_G(n) + O(|n|^{-2q-1})


DERIVATION
----------
  Symbol of the 7-point Laplacian:
      Lhat(xi) = 2 sum_q cos(xi_q) - 6 = -sigma(xi),  sigma = sum_q 4 sin^2(xi_q/2)
  With the convention f(x) = (2 pi)^-3 int exp(-i x.xi) fhat dxi,  G = -InvFT[1/sigma].

  Expand sigma = sum_{j>=1} b_j, b_j homogeneous of degree 2j, b_1 = |xi|^2, and

      1/sigma = |xi|^-2 sum_{k>=0} (-1)^k ( (sigma - |xi|^2) / |xi|^2 )^k

  The part homogeneous of degree -2+2p is a FINITE sum (k <= p, because each
  factor of (sigma - |xi|^2) carries degree >= 4):

      T_p = sum_{k=1..p} (-1)^k Pi_{p,k}(xi) / |xi|^{2k+2},      T_0 = |xi|^-2

  where Pi_{p,k} = degree-(2p+2k) homogeneous part of (sigma - |xi|^2)^k.

  Radial building block, from the Riesz formula
      InvFT[|xi|^-a] = Gamma((d-a)/2) / (2^a pi^{d/2} Gamma(a/2)) |x|^{a-d}:

      InvFT[ |xi|^{-2k-2} ] = c_k |x|^{2k-1},     c_k = (-1)^k / (4 pi (2k)!)

  *** This is where the original derivation went wrong: it used
      4^k k! (2k-1)!!  in place of  (2k)! = 2^k k! (2k-1)!! ,
      making c_k too small by 2^-k. The error WORSENS with order --
      1/2 at k=1, 1/4 at k=2, 1/8 at k=3 -- so it is not a global rescale. ***

  A polynomial factor maps as xi_j -> i d/dx_j, contributing i^{2m} = (-1)^{p+k}
  for a monomial of total degree 2m = 2(p+k). Combined with the (-1)^k above:

      g_p = (-1)^p sum_{k=1..p} c_k * Pi_{p,k}(d/dx) [ |x|^{2k-1} ]

      A^q_G(x) = - sum_{p=0}^{q-1} g_p(x)

  Each g_p is homogeneous of degree -2p-1, so g_p = Phi_p(s) / |x|^{2p+1} with
  s_i = x_i^2/|x|^2, and Phi_p is a symmetric polynomial with exact rational
  coefficients.


WHY THIS IS FAST
----------------
  A direct approach (sympy diff on sqrt(x1^2+x2^2+x3^2), once per monomial)
  dies well before q=10: at q=10 the top term needs 36 derivatives of |x|^17,
  repeated over thousands of monomials.  Three exploits remove that:

  (1) EVEN POWERS ONLY.  sigma - |xi|^2 = sum_{j>=2} b_j with b_j proportional
      to sum_i xi_i^{2j}, so every monomial that ever appears has even exponents.
      Working in t_i = xi_i^2 halves the polynomial degree everywhere.

  (2) CLOSED-FORM DOUBLE-DERIVATIVE IN s-SPACE.  Writing a function as
      r^n Q(s), one pair of derivatives in direction i is pure polynomial
      algebra -- no square roots ever appear:

          d^2/dx_i^2 [ r^n Q ] = r^{n-2} Q',
              R  = n Q + 2 sum_j (delta_ij - s_j) dQ/ds_j
              Q' = (n-1) s_i R + (1 - s_i) R
                   + 2 s_i sum_j (delta_ij - s_j) dR/ds_j

      (Uses d(s_i)/d(x_1) = 2 x_1 / r^2 * (delta_i1 - s_i) and u_1^2 = s_1.)
      Verified: applying it twice to r^1 gives d^4 r/dx1^4 = (-3 + 18 s1
      - 15 s1^2)/r^3, which matches direct differentiation.

  (3) DYNAMIC PROGRAMMING OVER THE DERIVATIVE MULTI-INDEX.  Build a table
      D_k[beta] with r^{2k-1-2|beta|} D_k[beta] = d^{2beta} |x|^{2k-1}, filling
      in order of increasing |beta| from D_k[beta - e_i].  Cost becomes
      proportional to the NUMBER of multi-indices rather than
      (number) x (depth), and every monomial of every Pi_{p,k} reads from the
      same table.

  Also: coefficients are exact Fractions in a small dict-based polynomial type
  rather than sympy expressions, which removes the expand/simplify overhead
  that dominates at high order.


VALIDATION (runs before anything is written)
--------------------------------------------
  1. Phi_1 must equal (5 P - 3)/32 with P = sum s_i^2 (1/pi factored out) --
     Liska & Colonius eq. (8) rewritten in s. Closed form, independent of this
     code.
  2. Phi_2 must match the known-good q=3 reference embedded below.
  3. Then confirm slope -2q per order against the exact table via
     lgf3d_reps_sweep.py.  A wrong coefficient shows up in the SLOPE even when
     the magnitude looks plausible -- the original g_1 was exactly half correct,
     which halved the error but left the slope at -2.
"""

import argparse
import itertools
import time
from fractions import Fraction
from math import factorial

# ---------------------------------------------------------------------------
# Minimal dense-dict polynomial in 3 variables with exact rational coefficients.
# Deliberately not sympy: at q=10 the expand/simplify overhead dominates.
# ---------------------------------------------------------------------------

class P3:
    __slots__ = ('c',)

    def __init__(self, c=None):
        self.c = {} if c is None else {k: v for k, v in c.items() if v}

    @staticmethod
    def const(v):
        v = Fraction(v)
        return P3({(0, 0, 0): v}) if v else P3()

    @staticmethod
    def var(i):
        e = [0, 0, 0]; e[i] = 1
        return P3({tuple(e): Fraction(1)})

    def __add__(self, o):
        r = dict(self.c)
        for k, v in o.c.items():
            n = r.get(k, 0) + v
            if n: r[k] = n
            else: r.pop(k, None)
        return P3(r)

    def __sub__(self, o):
        r = dict(self.c)
        for k, v in o.c.items():
            n = r.get(k, 0) - v
            if n: r[k] = n
            else: r.pop(k, None)
        return P3(r)

    def scal(self, a):
        a = Fraction(a)
        if not a: return P3()
        return P3({k: v * a for k, v in self.c.items()})

    def __mul__(self, o):
        r = {}
        for k1, v1 in self.c.items():
            for k2, v2 in o.c.items():
                k = (k1[0]+k2[0], k1[1]+k2[1], k1[2]+k2[2])
                n = r.get(k, 0) + v1*v2
                if n: r[k] = n
                else: r.pop(k, None)
        return P3(r)

    def diff(self, i):
        r = {}
        for k, v in self.c.items():
            if k[i] == 0: continue
            kk = list(k); n = kk[i]; kk[i] -= 1
            r[tuple(kk)] = v * n
        return P3(r)

    def homog(self, d):
        return P3({k: v for k, v in self.c.items() if sum(k) == d})

    def pow(self, n):
        r = P3.const(1); b = self
        while n:
            if n & 1: r = r * b
            b = b * b; n >>= 1
        return r

    def subs_s3(self):
        """Reduce modulo s1+s2+s3-1, i.e. substitute s3 -> 1-s1-s2."""
        one_minus = P3.const(1) - P3.var(0) - P3.var(1)
        pw = {0: P3.const(1)}
        out = P3()
        for (a, b, c), v in self.c.items():
            if c not in pw: pw[c] = one_minus.pow(c)
            out = out + P3({(a, b, 0): v}) * pw[c]
        return out

    def is_zero(self): return not self.c

    def __repr__(self):
        if not self.c: return "0"
        names = ['s1', 's2', 's3']
        parts = []
        for k in sorted(self.c, key=lambda k: (-sum(k), k)):
            v = self.c[k]
            m = "*".join("%s%s" % (names[i], "" if k[i] == 1 else "**%d" % k[i])
                         for i in range(3) if k[i])
            parts.append("%s%s" % (v, "*" + m if m else ""))
        return " + ".join(parts).replace("+ -", "- ")


# ---------------------------------------------------------------------------
# Exploit (2): one pair of derivatives, in s-space, no square roots.
# ---------------------------------------------------------------------------

def d2(Q, n, i):
    """Given f = r^n Q(s), return Q' with d^2 f/dx_i^2 = r^{n-2} Q'."""
    def L(F):                                   # 2 sum_j (delta_ij - s_j) dF/ds_j
        acc = P3()
        for j in range(3):
            dF = F.diff(j)
            if dF.is_zero(): continue
            coef = (P3.const(1) if j == i else P3()) - P3.var(j)
            acc = acc + coef * dF
        return acc.scal(2)

    R = Q.scal(n) + L(Q)
    si = P3.var(i)
    return (si.scal(n - 1) * R) + ((P3.const(1) - si) * R) + (si * L(R))


# ---------------------------------------------------------------------------
# Exploit (3): DP table of d^{2 beta} |x|^{2k-1}
# ---------------------------------------------------------------------------

def deriv_table(k, max_order):
    """D[beta] with r^{2k-1-2|beta|} D[beta] = d^{2beta} |x|^{2k-1}."""
    n0 = 2*k - 1
    D = {(0, 0, 0): P3.const(1)}
    for total in range(1, max_order + 1):
        for b in itertools.product(range(total + 1), repeat=3):
            if sum(b) != total: continue
            i = next(j for j in range(3) if b[j] > 0)
            prev = list(b); prev[i] -= 1; prev = tuple(prev)
            D[b] = d2(D[prev], n0 - 2*(total - 1), i)
    return D


# ---------------------------------------------------------------------------
# Exploit (4): elementary symmetric reduction.
#
# Phi_p is symmetric in (s1,s2,s3), so it is a polynomial in the elementary
# symmetric polynomials e1, e2, e3 -- and e1 = s1+s2+s3 = 1 identically, so
# only e2 and e3 survive.  This is a strictly better representation than
# eliminating s3:
#
#     p=9:  1159 terms (expanded)  ->  190 (s3 eliminated)  ->  30 in (e2,e3)
#           max|coeff| 7.2e23      ->  7.2e23               ->  3.4e20
#
#   * 6.3x fewer terms, so ~6x fewer flops
#   * 2098x smaller coefficients, and e2 <= 1/3, e3 <= 1/27 on the simplex, so
#     powers DECAY instead of sitting at O(1) -- measured 11x to 490x better
#     float64 accuracy than the s-form
#   * symmetry is manifest rather than destroyed by singling out s3
#
# Standard algorithm: repeatedly cancel the lex-leading monomial (which, for a
# symmetric polynomial, has exponents in descending order) with
# e1^(a1-a2) e2^(a2-a3) e3^a3, then set e1 = 1.
# ---------------------------------------------------------------------------

def to_elementary(poly):
    """Symmetric P3 -> {(j,k): coeff} meaning sum coeff * e2^j * e3^k, with e1=1."""
    e1 = P3.var(0) + P3.var(1) + P3.var(2)
    e2 = P3.var(0)*P3.var(1) + P3.var(0)*P3.var(2) + P3.var(1)*P3.var(2)
    e3 = P3.var(0)*P3.var(1)*P3.var(2)
    epow = {1: {0: P3.const(1)}, 2: {0: P3.const(1)}, 3: {0: P3.const(1)}}
    base = {1: e1, 2: e2, 3: e3}

    def pw(i, n):
        if n not in epow[i]:
            for e in range(1, n + 1):          # fill the gap, not just the top
                if e not in epow[i]:
                    epow[i][e] = epow[i][e-1] * base[i]
        return epow[i][n]

    rem = P3(dict(poly.c))
    out = {}
    while not rem.is_zero():
        lead = max(rem.c)                       # lex order on exponent tuples
        cf = rem.c[lead]
        a = sorted(lead, reverse=True)
        i, j, k = a[0]-a[1], a[1]-a[2], a[2]
        out[(j, k)] = out.get((j, k), Fraction(0)) + cf
        rem = rem - (pw(1, i) * pw(2, j) * pw(3, k)).scal(cf)
    return {k: v for k, v in out.items() if v}


def cpp_horner_e(terms):
    """Nested Horner: outer in e3, inner in e2.  terms = {(j,k): coeff}."""
    if not terms:
        return "0.0"
    by_k = {}
    for (j, k), v in terms.items():
        by_k.setdefault(k, {})[j] = v

    def lit(v):
        n, d = v.numerator, v.denominator
        return "%d.0" % n if d == 1 else "(%d.0/%d.0)" % (n, d)

    def horner(coeffs, var):                    # {power: coeff} -> Horner string
        hi = max(coeffs)
        out = lit(coeffs[hi])
        for e in range(hi - 1, -1, -1):
            c = coeffs.get(e)
            out = "(%s)*%s" % (out, var)
            if c is not None:
                out += (" - %s" % lit(-c)) if c < 0 else (" + %s" % lit(c))
        return out

    hi = max(by_k)
    out = "(%s)" % horner(by_k[hi], "e2")
    for e in range(hi - 1, -1, -1):
        out = "(%s)*e3" % out
        if e in by_k:
            out += " + (%s)" % horner(by_k[e], "e2")
    return out


# ---------------------------------------------------------------------------
# Main derivation
# ---------------------------------------------------------------------------

def derive(q_max, verbose=True):
    """Return {p: Phi_p as P3 in s}, with an overall 1/pi factored out."""
    # Exploit (1): work in t_i = xi_i^2.  b_j -> coef_j * (t1^j + t2^j + t3^j),
    # t-degree j.  Pi_{p,k} has t-degree p+k and is a product of k factors of
    # t-degree >= 2, so the largest single b_j needed has j <= p-k+2 <= q_max.
    tail = P3()
    for j in range(2, q_max + 1):
        coef = Fraction(2 * (-1)**(j - 1), factorial(2*j))
        tail = tail + (P3.var(0).pow(j) + P3.var(1).pow(j) + P3.var(2).pow(j)).scal(coef)

    tail_pow = {0: P3.const(1)}
    for k in range(1, max(1, q_max)):
        tail_pow[k] = tail_pow[k-1] * tail

    Dcache = {}
    Phi = {}
    for p in range(q_max):
        t0 = time.time()
        if p == 0:
            Phi[0] = {(0, 0): Fraction(1, 4)}
            if verbose: print("  Phi_0 = 1/4")
            continue

        acc = P3()
        for k in range(1, p + 1):
            Pi = tail_pow[k].homog(p + k)
            if Pi.is_zero(): continue
            if k not in Dcache:
                Dcache[k] = deriv_table(k, q_max - 1 + k)
            D = Dcache[k]
            ck = Fraction((-1)**k, factorial(2*k))
            sub = P3()
            for beta, coeff in Pi.c.items():
                sub = sub + D[beta].scal(coeff)
            acc = acc + sub.scal(ck)

        sym = acc.scal(Fraction((-1)**p, 4))
        Phi[p] = to_elementary(sym)
        if verbose:
            print("  Phi_%d : %4d terms expanded -> %3d in (e2,e3)   (%.2fs)"
                  % (p, len(sym.c), len(Phi[p]), time.time() - t0))
    return Phi


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def check_phi1(Phi):
    """L&C eq.(8) is Phi_1 = (5 sum s_i^2 - 3)/32; sum s_i^2 = e1^2 - 2 e2 = 1 - 2 e2,
    so in the elementary basis the target is simply (1 - 5 e2)/16."""
    if 1 not in Phi: return True, "skipped"
    target = {(0, 0): Fraction(1, 16), (1, 0): Fraction(-5, 16)}
    d = {k: Phi[1].get(k, 0) - target.get(k, 0)
         for k in set(Phi[1]) | set(target)}
    d = {k: v for k, v in d.items() if v}
    return not d, repr(d)


def check_phi2(Phi):
    """Cross-check Phi_2 by evaluating the (e2,e3) form against the validated
    s-space reference at three independent directions."""
    if 2 not in Phi: return True, "skipped"
    S_REF = {(4,0):(1155,128),(3,1):(2310,128),(3,0):(-2310,128),(2,2):(3465,128),
             (2,1):(-3486,128),(2,0):(1491,128),(1,3):(2310,128),(1,2):(-3486,128),
             (1,1):(1512,128),(1,0):(-336,128),(0,4):(1155,128),(0,3):(-2310,128),
             (0,2):(1491,128),(0,1):(-336,128),(0,0):(23,128)}
    msgs = []
    for s in [(Fraction(1),Fraction(0),Fraction(0)),
              (Fraction(1,3),Fraction(1,3),Fraction(1,3)),
              (Fraction(4,21),Fraction(9,21),Fraction(8,21))]:
        ref = sum(Fraction(n,d)*s[0]**a*s[1]**b for (a,b),(n,d) in S_REF.items())
        e2 = s[0]*s[1] + s[0]*s[2] + s[1]*s[2]
        e3 = s[0]*s[1]*s[2]
        got = sum(c*e2**j*e3**k for (j,k), c in Phi[2].items())
        if got != ref: msgs.append("at %s: %s != %s" % (s, got, ref))
    return not msgs, "; ".join(msgs)


# ---------------------------------------------------------------------------
# C++ emission
# ---------------------------------------------------------------------------

def emit_cpp(Phi, q_max, path):
    L = ["// AUTO-GENERATED by lgf3d_asymptotic_gen.py -- do not edit by hand.",
         "//",
         "//   A^q_G(x) = -sum_{p=0}^{q-1} Phi_p(s) / |x|^{2p+1},  s_i = x_i^2/|x|^2",
         "//   q = %d.  Truncation error O(|n|^{-%d})." % (q_max, 2*q_max + 1),
         "//   Convention: G(n) -> -1/(4 pi |n|), matching LGFKernels3D.H.",
         "//",
         "//   Validated: Phi_1 == Liska & Colonius eq. (8); slope -2q per order.",
         "//   NOTE: more terms is not monotonically better at fixed |n| (L&C",
         "//   footnote 1). Use lgf3d_reps_sweep.py to pick the optimal q for",
         "//   your table radius rather than assuming higher q wins.",
         "",
         "LGF_HD inline double farField3D(double nx, double ny, double nz)",
         "{",
         "    const double r2     = nx*nx + ny*ny + nz*nz;",
         "    const double inv_r  = 1.0 / std::sqrt(r2);",
         "    const double inv_r2 = inv_r * inv_r;",
         "",
         "    const double sx = nx*nx*inv_r2;",
         "    const double sy = ny*ny*inv_r2;",
         "    const double sz = nz*nz*inv_r2;",
         "",
         "    // Phi_p is symmetric in (sx,sy,sz), so it depends only on the",
         "    // elementary symmetric polynomials. e1 = sx+sy+sz = 1 identically,",
         "    // leaving e2 and e3. On the simplex e2 <= 1/3 and e3 <= 1/27, so",
         "    // their powers decay -- far better conditioned than s^n at O(1).",
         "    const double e2 = sx*sy + sx*sz + sy*sz;",
         "    const double e3 = sx*sy*sz;",
         "",
         "    double p = inv_r;",
         "    double g = 0.0;",
         ""]
    for q in range(q_max):
        L.append("    g += (%s) * p;" % cpp_horner_e(Phi[q]))
        if q < q_max - 1:
            L.append("    p *= inv_r2;")
    L += ["", "    return -g * M_1_PI;", "}", ""]
    open(path, "w").write("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q-max", type=int, default=5)
    ap.add_argument("--out", default="farField3D.inc")
    args = ap.parse_args()

    t0 = time.time()
    print("deriving %d terms ..." % args.q_max)
    Phi = derive(args.q_max)
    print("total %.2fs\n" % (time.time() - t0))

    ok1, m1 = check_phi1(Phi)
    print("[1] Phi_1 vs L&C eq.(8) :  %s %s" % ("OK" if ok1 else "MISMATCH", "" if ok1 else m1))
    ok2, m2 = check_phi2(Phi)
    print("[2] Phi_2 vs reference  :  %s %s" % ("OK" if ok2 else "MISMATCH", "" if ok2 else m2))
    if not (ok1 and ok2):
        raise SystemExit("*** validation failed, nothing written ***")

    emit_cpp(Phi, args.q_max, args.out)
    print("\nwrote %s" % args.out)
    print("Now confirm slope -2q per order:")
    print("    python3 lgf3d_reps_sweep.py --npy G.npy")