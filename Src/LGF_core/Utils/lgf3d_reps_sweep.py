#!/usr/bin/env python3
"""
lgf3d_reps_sweep.py -- measure r_eps, the radius at which the asymptotic
expansion A^q_G meets a target relative tolerance, reading the expansion
directly from the generated C++ so the sweep and the solver can never disagree
about what A^q_G is.

    python3 lgf3d_table_gen.py      --n-lookup 16 --out lgf3d_table.inc --npy G_R15.npy
    python3 lgf3d_asymptotic_gen.py --q-max 10    --out farField3D.inc
    python3 lgf3d_reps_sweep.py     --inc farField3D.inc --npy G_R15.npy


WHY THIS NUMBER MATTERS

  r_eps simultaneously fixes three things you would otherwise be guessing at:

    1. n_lookup.  The near-field table must cover |n| <= r_eps, or the solver
       silently uses the expansion where it is not yet converged.

    2. How many expansion terms to compile.  More is NOT always better.

    3. Whether the table stays `static constexpr` or moves to a device pointer.
       n_lookup <= 20 is 64 KB and fits the constant bank; 21 is 74 KB and does
       not.

  If you later bolt on an FMM, r_eps is ALSO the minimum separation at which
  cluster approximations (M2L) may be used, because M2L evaluates the kernel at
  off-lattice interpolation points where the table cannot be consulted. No
  amount of raising the multipole order fixes a violation of that.


THE SERIES IS ASYMPTOTIC, NOT CONVERGENT

  Liska & Colonius footnote 1: it is not always possible to improve accuracy at
  fixed |n| by adding terms. This is not a caveat, it is the dominant effect at
  small radius -- at r=4, A^10 is WORSE than A^1 by more than an order of
  magnitude. So this script reports the OPTIMAL q per radius rather than
  assuming the highest available q wins. Compile farField3D at the q matched to
  your n_lookup, not at the highest q you can generate.


WHAT IT MEASURES

  For each integer shell r:

      max over  r-0.5 < |n| <= r+0.5  of  | G(n) - A^q_G(n) | / | G(n) |

  The max over the SHELL, not one direction: the leading correction is a
  harmonic quartic, maximal along the axes and vanishing elsewhere, so sampling
  only (r,0,0) or only (r,r,r) is misleading either way.

  It also reports a round-off floor per order. The high-order Phi_p carry
  enormous coefficients that cancel -- at p=9, terms of size 4e21 cancel down
  to 4e9, losing 12 of the 16 digits double precision has. That floor currently
  sits below the truncation error, but it will bite first if you push q higher,
  and it is invisible unless you look for it.
"""

import argparse
import re

import numpy as np


# ---------------------------------------------------------------------------
# Parse the generated C++.
#
# Format emitted by lgf3d_asymptotic_gen.py:
#     g += (<poly in s1,s2,s3>) * p;
#     p *= inv_r2;
#     ...
#     return -g * M_1_PI;
#
# The polynomial body is already valid Python arithmetic -- rational literals
# are written (num.0/den.0) and monomials as repeated multiplication -- so it
# compiles directly with no translation.
#
# Because the emitted terms are cumulative, ONE file gives every order
# q = 1..q_max, so the whole sweep comes from a single generation run.
# ---------------------------------------------------------------------------

TERM_RE = re.compile(r"g\s*\+=\s*\((.*)\)\s*\*\s*p\s*;")
LIT_RE = re.compile(r"\(?(-?\d+\.0)(?:/(\d+\.0))?\)?")


def load_inc(path):
    """Return (compiled Phi_p list, max|coefficient| per p)."""
    terms, maxc = [], []
    with open(path) as f:
        for line in f:
            m = TERM_RE.search(line)
            if not m:
                continue
            body = m.group(1)
            terms.append(compile(body, "<Phi_%d>" % len(terms), "eval"))
            mx = 0.0
            for num, den in LIT_RE.findall(body):
                mx = max(mx, abs(float(num) / (float(den) if den else 1.0)))
            maxc.append(mx)
    if not terms:
        raise SystemExit("no 'g += (...) * p;' lines found in %s" % path)
    return terms, maxc


def evaluate(terms, q, s1, s2, s3, r):
    """A^q_G, vectorised. Mirrors the emitted C++ exactly, including -g/pi."""
    # supply both spellings so the sweep works with either emitted form:
    # the older expanded s1,s2 polynomials and the current (e2,e3) Horner form.
    e2 = s1*s2 + s1*s3 + s2*s3
    e3 = s1*s2*s3
    env = {"s1": s1, "s2": s2, "s3": s3,
           "sx": s1, "sy": s2, "sz": s3, "e2": e2, "e3": e3}
    g = np.zeros_like(r)
    p = 1.0 / r
    inv_r2 = p * p
    for k in range(q):
        g = g + eval(terms[k], {"__builtins__": {}}, env) * p
        p = p * inv_r2
    return -g / np.pi


# ---------------------------------------------------------------------------
# Independent hardcoded references -- these verify the PARSER, not the physics
# ---------------------------------------------------------------------------

def A1_ref(x, y, z):
    return -1.0 / (4.0 * np.pi * np.sqrt(x*x + y*y + z*z))


def A2_ref(x, y, z):
    """Liska & Colonius (2014) eq. (8), verbatim."""
    x2, y2, z2 = x*x, y*y, z*z
    r = np.sqrt(x2 + y2 + z2)
    quartic = x2*x2 + y2*y2 + z2*z2 - 3.0*(x2*y2 + x2*z2 + y2*z2)
    return -1.0/(4.0*np.pi*r) - quartic/(16.0*np.pi*r**7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inc", default="farField3D.inc")
    ap.add_argument("--npy", default="G_R15.npy")
    ap.add_argument("--tolerances", type=float, nargs="+",
                    default=[1e-5, 1e-6, 1e-8, 1e-10, 1e-12])
    args = ap.parse_args()

    terms, maxc = load_inc(args.inc)
    q_max = len(terms)
    G = np.load(args.npy)
    R = G.shape[0] - 1
    print("expansion : %s  (q up to %d)" % (args.inc, q_max))
    print("table     : %s  (exact for |n| <= %d)\n" % (args.npy, R))

    # flatten the 1/48 wedge once
    idx = np.array([(a, b, c) for a in range(R+1)
                              for b in range(a, R+1)
                              for c in range(b, R+1)], dtype=float)
    a, b, c = idx[:, 0], idx[:, 1], idx[:, 2]
    rad = np.sqrt(a*a + b*b + c*c)
    keep = rad > 0
    a, b, c, rad = a[keep], b[keep], c[keep], rad[keep]
    gex = G[a.astype(int), b.astype(int), c.astype(int)]
    inv_r2 = 1.0 / (rad * rad)
    s1, s2, s3 = a*a*inv_r2, b*b*inv_r2, c*c*inv_r2

    # parser self-check
    for name, ref, q in (("A^1", A1_ref, 1), ("A^2", A2_ref, 2)):
        if q > q_max:
            continue
        d = np.max(np.abs(evaluate(terms, q, s1, s2, s3, rad) - ref(a, b, c)))
        print("parser check %s vs hardcoded reference : %.3e  %s"
              % (name, d, "OK" if d < 1e-13 else "*** MISMATCH ***"))
    print()

    err = {q: np.abs(gex - evaluate(terms, q, s1, s2, s3, rad)) / np.abs(gex)
           for q in range(1, q_max + 1)}

    shells = list(range(2, R + 1))
    show = sorted(set([1, 2, 3, 5, q_max]) & set(range(1, q_max + 1)))

    print("relative error, max over shell:  |G - A^q| / |G|")
    print("   r  " + "".join("    A^%-2d  " % q for q in show))
    print("  " + "-" * (5 + 10*len(show)))
    best = {}
    for r in shells:
        m = (rad > r - 0.5) & (rad <= r + 0.5)
        if not m.any():
            continue
        print("  %3d  " % r + "".join("  %.2e" % err[q][m].max() for q in show))
        best[r] = min((err[q][m].max(), q) for q in range(1, q_max + 1))

    print("\noptimal q per radius (asymptotic series -- more terms is not better)")
    print("     r   best q   best error    A^%d is" % q_max)
    print("  " + "-" * 46)
    for r in shells:
        if r not in best:
            continue
        e, q = best[r]
        m = (rad > r - 0.5) & (rad <= r + 0.5)
        print("   %3d     %2d     %.3e   %8.1fx worse"
              % (r, q, e, err[q_max][m].max() / e))

    print("\nr_eps: smallest radius meeting each tolerance, and the q that does it")
    print("     tol       r_eps   q   n_lookup   table size   storage")
    print("  " + "-" * 60)
    for tol in args.tolerances:
        hit = next(((r, best[r][1]) for r in shells
                    if r in best and best[r][0] <= tol), None)
        if hit is None:
            print("   %-9.0e   > %d   (not reached inside the exact table;"
                  " regenerate at larger --n-lookup)" % (tol, R))
            continue
        r, q = hit
        n, = (r + 1,)
        kb = (n**3 * 8) / 1024.0
        print("   %-9.0e    %3d  %2d      %3d     %8.1f KB   %s"
              % (tol, r, q, n, kb, "constexpr" if kb <= 64 else "device ptr"))

    print("\nround-off floor from coefficient cancellation (eps = 2.2e-16)")
    print("     p    max|coeff|      r=8        r=15       r=30")
    print("  " + "-" * 52)
    for p in range(q_max):
        if maxc[p] == 0:
            continue
        f = [4.0 * maxc[p] * 2.2e-16 / r**(2*p) for r in (8.0, 15.0, 30.0)]
        print("    %2d    %.3e   %.2e   %.2e   %.2e" % (p, maxc[p], f[0], f[1], f[2]))
    print("\n  If a floor exceeds your target tolerance at your working radius,")
    print("  that order cannot be evaluated usefully in double precision --")
    print("  reduce q rather than enlarging the table.")


if __name__ == "__main__":
    main()