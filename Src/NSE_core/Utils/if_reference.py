#!/usr/bin/env python3
"""
Reference values and self-checks for the IF-HERK integrating factor.

    g_alpha(n) = exp(-2 alpha) I_n(2 alpha),     alpha = gap * dt / (dx^2 Re)

Two things are checked, and they catch different bugs:

  TABLE   the Bessel values themselves, computed three independent ways.
          A wrong argument (I_n(alpha) instead of I_n(2 alpha)) will NOT be
          caught by the composition test below -- the addition theorem holds
          for any consistent scaling -- so this has to be checked separately.

  SWEEP   E(a)E(a) == E(2a), and E applied to a delta reproduces the kernel.
          Catches transposed indices, wrong sweep direction, ping-pong errors.

Prints a paste-ready C++ assert block at the end.

    python3 if_reference.py --dt 0.01 --dx 0.05 --Re 1000 --n-IF 14
"""

import argparse, math
import numpy as np

# --------------------------------------------------------------------------
# three independent routes to g_alpha(n)
# --------------------------------------------------------------------------

def g_series(n, a):
    """Ascending series -- mirrors the C++ in precomputeIFs()."""
    z = 2.0 * a
    term = 1.0
    for p in range(1, n + 1):
        term *= (0.5 * z) / p
    s = term
    for k in range(200):
        term *= (0.25 * z * z) / ((k + 1) * (n + k + 1))
        s += term
        if term < 1e-300:
            break
    return math.exp(-z) * s

def g_quad(n, a, M=4096):
    """Integral representation, trapezoid on a periodic integrand.
    g(n) = (1/2pi) int_{-pi}^{pi} exp(2a(cos t - 1)) cos(n t) dt
    Spectrally accurate and shares no code with the series."""
    t = np.linspace(-np.pi, np.pi, M, endpoint=False)
    return float(np.mean(np.exp(2.0 * a * (np.cos(t) - 1.0)) * np.cos(n * t)))

def g_scipy(n, a):
    from scipy.special import ive
    return float(ive(n, 2.0 * a))

# --------------------------------------------------------------------------
# sweep, structured exactly like applyIF()
# --------------------------------------------------------------------------

def apply_if(fld, tab, ndim):
    for sd in range(ndim):
        out = tab[0] * fld
        for m in range(1, len(tab)):
            out += tab[m] * (np.roll(fld, -m, axis=sd) + np.roll(fld, m, axis=sd))
        fld = out
    return fld

# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dt",   type=float, default=0.002)
    p.add_argument("--dx",   type=float, default=0.004)
    p.add_argument("--Re",   type=float, default=10000.0)
    p.add_argument("--n-IF", type=int,   dest="nIF", default=14)
    p.add_argument("--ndim", type=int,   default=2)
    p.add_argument("--N",    type=int,   default=48, help="test grid size")
    a = p.parse_args()

    r3   = math.sqrt(3.0)
    gaps = [0.5, 0.5, 0.0]                      # Scheme A shifted widths
    uniq = sorted({g for g in gaps if g > 1e-14})
    beta = a.dt / (a.dx * a.dx * a.Re)

    print(f"dt={a.dt}  dx={a.dx}  Re={a.Re}  ->  beta = dt/(dx^2 Re) = {beta:.6g}")
    print(f"Scheme A gaps {gaps} -> {len(uniq)} distinct: {uniq}")
    print(f"n_IF = {a.nIF}, {a.ndim}D\n")

    try:
        g_scipy(0, 0.1); have_scipy = True
    except Exception:
        have_scipy = False

    tables = {}
    for gap in uniq:
        alpha = gap * beta
        tab_s = np.array([g_series(n, alpha) for n in range(a.nIF + 1)])
        tab_q = np.array([g_quad  (n, alpha) for n in range(a.nIF + 1)])
        tables[gap] = tab_s

        print(f"--- gap = {gap}   alpha = {alpha:.6g} " + "-" * 34)
        d_sq = np.abs(tab_s - tab_q).max()
        print(f"  series vs quadrature      : {d_sq:.3e}")
        if have_scipy:
            tab_c = np.array([g_scipy(n, alpha) for n in range(a.nIF + 1)])
            print(f"  series vs scipy           : {np.abs(tab_s - tab_c).max():.3e}")

        mass = tab_s[0] + 2.0 * tab_s[1:].sum()
        print(f"  retained mass             : {mass:.16f}  (deficit {1-mass:.2e})")
        if 1 - mass > 1e-8:
            print(f"  *** n_IF={a.nIF} is TOO SMALL for alpha={alpha:.4g}")

        need = {}
        for tol in (1e-8, 1e-10, 1e-12):
            n = 0
            while 1 - (tab_q[0] + 2*sum(g_quad(k, alpha) for k in range(1, n+1))) >= tol:
                n += 1
                if n > 200: break
            need[tol] = n
        print(f"  n_IF needed for 1e-8/1e-10/1e-12: "
              f"{need[1e-8]} / {need[1e-10]} / {need[1e-12]}")

        # wrong-argument sanity: what the table looks like if you forget the 2
        wrong = np.array([g_series(n, 0.5 * alpha) for n in range(a.nIF + 1)])
        print(f"  (a table built with I_n(alpha) would differ by "
              f"{np.abs(tab_s - wrong).max():.3e} -- composition alone won't catch it)")
        print()

    # ---- sweep checks ----------------------------------------------------
    gap   = uniq[0]
    alpha = gap * beta
    tabA  = tables[gap]
    tab2A = np.array([g_series(n, 2.0 * alpha) for n in range(a.nIF + 1)])

    shape = (a.N,) * a.ndim
    rng   = np.random.default_rng(0)
    f     = rng.normal(size=shape)

    lhs = apply_if(apply_if(f, tabA, a.ndim), tabA, a.ndim)
    rhs = apply_if(f, tab2A, a.ndim)
    scale = np.abs(rhs).max()
    print("--- sweep checks " + "-" * 46)
    print(f"  |E(a)E(a) - E(2a)|_inf    : {np.abs(lhs-rhs).max():.3e}  (field scale {scale:.3g})")

    ones = np.ones(shape)
    print(f"  |E(a)[1] - 1|_inf         : {np.abs(apply_if(ones, tabA, a.ndim)-1).max():.3e}"
          f"   <- nonzero means truncation bias")

    delta = np.zeros(shape); delta[(0,) * a.ndim] = 1.0
    out   = apply_if(delta, tabA, a.ndim)
    if a.ndim == 2:
        ref = np.outer(np.concatenate([tabA[::-1], tabA[1:]]),
                       np.concatenate([tabA[::-1], tabA[1:]]))
        got = np.roll(np.roll(out, a.nIF, 0), a.nIF, 1)[:2*a.nIF+1, :2*a.nIF+1]
        print(f"  |E(a)[delta] - kernel|_inf: {np.abs(got-ref).max():.3e}"
              f"   <- nonzero means transposed index")

    # ---- paste-ready C++ -------------------------------------------------
    print("\n--- paste into a C++ unit check " + "-" * 31)
    print("//  expects dt, dx, Re as above")
    for gap in uniq:
        alpha = gap * beta
        print(f"//  gap {gap}, alpha {alpha:.10g}")
        for n in (0, 1, 2, a.nIF // 2, a.nIF):
            print(f"AMREX_ALWAYS_ASSERT(std::abs(h[{n}] - {tables[gap][n]:.17e}) < 1.0e-14);")
        m = tables[gap][0] + 2.0 * tables[gap][1:].sum()
        print(f"AMREX_ALWAYS_ASSERT(std::abs(mass - {m:.17e}) < 1.0e-14);")

if __name__ == "__main__":
    main()
