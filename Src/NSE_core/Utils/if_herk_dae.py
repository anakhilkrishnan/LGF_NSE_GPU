#!/usr/bin/env python3
"""
IF-HERK on a linear index-2 DAE  --  a stripped-down analogue of the
Liska & Colonius (JCP 316, 2016) time integrator.

The point of this script is to run the *exact* algorithm of their Eqs.
(25)-(31) and (35) -- shifted tableau, q/w aging recursion, per-stage
projection, (a_ss*dt)^-1 rescaling -- on a problem small enough that
both the solution variable y and the constraint variable z have
closed-form truth, so both orders of accuracy can be read off directly.

    PROBLEM
    -------
        dy/dt = -N~(y) + A_d y - G z,      G^T y = 0
        N~(y) = -A_skew y

    so the full RHS is  A y - G z  with  A = A_skew + A_d.  The pieces
    map onto the fluid problem as:

        y        <->  u        (velocity, face-centred)
        z        <->  d        (total pressure, cell-centred)
        A_skew   <->  -N~      (convection; imaginary spectrum)
        A_d      <->  L_F/Re   (viscosity; negative real spectrum)
        G        <->  G        (gradient)
        G^T      <->  -D       (divergence)
        G^T G    <->  -L_C     (cell Laplacian)

    Differentiating the constraint gives the index-2 reduction

        z    = (G^T G)^-1 G^T A y
        dy/dt = P A y,    P = I - G (G^T G)^-1 G^T

    so y(t) = expm(P A t) y0 and z(t) = (G^T G)^-1 G^T A y(t) are exact.

    A_d is diagonal and G lies in one of its eigenspaces, which
    reproduces the commutation  G^T H_F = H_C G^T  that collapses the
    stage saddle-point system to a single Poisson solve (their Eq. 35).
    This is asserted numerically at startup, and the constraint residual
    is checked at every stage of every step.

Run:  python3 if_herk_dae.py
"""

import numpy as np
from scipy.linalg import expm

np.set_printoptions(precision=6, suppress=True)

TOL = 1e-12


# ----------------------------------------------------------------------
# Butcher tableaux and the shifted form of Eq. (23)
# ----------------------------------------------------------------------

class Tableau:
    """Explicit RK tableau plus the shifted tableau used by IF-HERK."""

    def __init__(self, name, A, b, c, note=""):
        self.name = name
        self.note = note
        self.A = np.asarray(A, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.c = np.asarray(c, dtype=float)
        self.s = len(self.b)

        # Eq. (23):  a~_{i,j} = a_{i+1,j} for i<s,  a~_{s,j} = b_j
        #            c~_i     = c_{i+1}   for i<s,  c~_s     = 1
        self.At = np.vstack([self.A[1:, :], self.b])
        self.ct = np.concatenate([self.c[1:], [1.0]])

        # sub-step widths  c~_i - c~_{i-1},  with c~_0 := 0.
        # gaps[i-1] is the argument of H^i in Eq. (27).
        self.gaps = np.diff(np.concatenate([[0.0], self.ct]))

        self._validate()

    def _validate(self):
        s = self.s
        assert np.allclose(self.A, np.tril(self.A, -1)), \
            f"{self.name}: A must be strictly lower triangular (explicit)"
        assert np.allclose(self.A.sum(axis=1), self.c), \
            f"{self.name}: row sums of A must equal c"
        assert abs(self.b.sum() - 1.0) < TOL, f"{self.name}: sum(b) != 1"
        # Eq. (30) divides by a~_{i,i}
        for i in range(s):
            assert abs(self.At[i, i]) > TOL, \
                f"{self.name}: a~_{{{i+1},{i+1}}} = 0, cannot form w^{{i,i}}"

    # -- classical order of the underlying ERK (for the y variable) -----
    @property
    def classical_order(self):
        A, b, c = self.A, self.b, self.c
        ok = lambda lhs, rhs: abs(lhs - rhs) < 1e-12
        if not ok(b.sum(), 1.0):
            return 0
        if not ok(b @ c, 0.5):
            return 1
        if not (ok(b @ c**2, 1/3) and ok(b @ (A @ c), 1/6)):
            return 2
        if not (ok(b @ c**3, 1/4) and ok(b @ (c * (A @ c)), 1/8)
                and ok(b @ (A @ c**2), 1/12) and ok(b @ (A @ (A @ c)), 1/24)):
            return 3
        return 4

    @property
    def linear_order(self):
        """Order of R(z) as an approximation to exp(z).

        On a LINEAR problem an explicit RK collapses to
        y_{n+1} = R(dt*M) y_n, so only R(z) matters -- two tableaux with
        the same stability polynomial produce bit-identical iterates
        regardless of their classical order.  All four schemes here share
        R(z) = 1 + z + z^2/2 + z^3/6, which is why the no-IF y-columns
        below are identical.  The classical order is recovered only by a
        nonlinear problem, or by switching the IF on (which breaks the
        polynomial structure)."""
        coef = [self.b.sum(), self.b @ self.c, self.b @ (self.A @ self.c),
                self.b @ (self.A @ (self.A @ self.c))]
        exact = [1.0, 1/2, 1/6, 1/24]
        p = 0
        for k, (got, want) in enumerate(zip(coef, exact), start=1):
            if abs(got - want) > 1e-12:
                return p
            p = k
        return p

    @property
    def if_compatible(self):
        """IF needs every sub-step to advance forward in time: E(-a) is
        anti-diffusion and amplifies the grid-scale mode."""
        return bool(np.all(self.gaps > -TOL))

    @property
    def n_distinct_alpha(self):
        nz = self.gaps[self.gaps > TOL]
        return len(np.unique(np.round(nz, 12)))

    def stability_polynomial(self, z):
        """R(z) = 1 + z b^T (I - zA)^-1 e,  Eq. (C.5)."""
        e = np.ones(self.s)
        return 1.0 + z * (self.b @ np.linalg.solve(np.eye(self.s) - z * self.A, e))

    def imag_axis_limit(self):
        """mu* = sup{ mu : |R(i mu)| < 1 }, the quantity in Eq. (C.6)."""
        # |R(i mu)| -> 1 as mu -> 0 to machine precision, so compare
        # against 1 + eps rather than strict <.
        mu = np.linspace(0, 4, 400001)[1:]
        good = np.abs([self.stability_polynomial(1j * m) for m in mu]) <= 1.0 + 1e-9
        if not good[0]:
            return 0.0
        if good.all():
            return mu[-1]
        return mu[int(np.argmin(good)) - 1]


r3 = np.sqrt(3.0)

TABLEAUX = {
    # Shu-Osher SSP-RK3 converted to Butcher form.  Note c = [0, 1, 1/2]:
    # the nodes are NOT monotone, so c~ = [1, 1/2, 1] goes backwards.
    "SSP-RK3": Tableau(
        "SSP-RK3",
        A=[[0, 0, 0], [1, 0, 0], [1/4, 1/4, 0]],
        b=[1/6, 1/6, 2/3],
        c=[0, 1, 1/2],
        note="Shu-Osher SSP, standard in explicit projection solvers",
    ),
    # Eq. (33) of the paper.
    "Scheme A": Tableau(
        "Scheme A",
        A=[[0, 0, 0], [1/2, 0, 0], [r3/3, (3 - r3)/3, 0]],
        b=[(3 + r3)/6, -r3/3, (3 + r3)/6],
        c=[0, 1/2, 1],
        note="designed for IF-HERK: equispaced nodes -> one integrating factor",
    ),
    "Scheme B": Tableau(
        "Scheme B",
        A=[[0, 0, 0], [1/3, 0, 0], [-1, 2, 0]],
        b=[0, 3/4, 1/4],
        c=[0, 1/3, 1],
        note="from Brasey & Hairer",
    ),
    "Scheme C": Tableau(
        "Scheme C",
        A=[[0, 0, 0], [8/15, 0, 0], [1/4, 5/12, 0]],
        b=[1/4, 0, 3/4],
        c=[0, 8/15, 2/3],
        note="RK coefficients of the Le & Moin fractional-step method",
    ),
}

# Table 1 of the paper (specialised order conditions, constant f_z and g_y).
PAPER_TABLE1 = {"Scheme A": (2, 2), "Scheme B": (3, 2), "Scheme C": (3, 1)}


# ----------------------------------------------------------------------
# The DAE
# ----------------------------------------------------------------------

class LinearIndex2DAE:
    def __init__(self, omega, d_diag, G):
        self.n = 3
        w1, w2, w3 = omega
        # A_skew v = omega x v  -- purely imaginary spectrum, the "advection"
        self.A_skew = np.array([[0, -w3, w2],
                                [w3, 0, -w1],
                                [-w2, w1, 0]], dtype=float)
        self.d_diag = np.asarray(d_diag, dtype=float)   # the "viscosity"
        self.A_d = np.diag(self.d_diag)
        self.A = self.A_skew + self.A_d

        G = np.asarray(G, dtype=float).reshape(self.n, -1)
        self.G = G / np.linalg.norm(G, axis=0)
        self.GtG = self.G.T @ self.G
        self.GtG_inv = np.linalg.inv(self.GtG)
        self.P = np.eye(self.n) - self.G @ self.GtG_inv @ self.G.T

        self._check_commutation()

        y0 = self.P @ np.array([0.9, -0.4, 1.3])
        self.y0 = y0 / np.linalg.norm(y0)

    def _check_commutation(self):
        """G^T H(a) must equal H_C(a) G^T for the Eq. (35) collapse to hold.
        With A_d diagonal this needs G to sit in one of its eigenspaces."""
        for a in (0.13, 0.71, 1.9):
            H = np.exp(self.d_diag * a)
            lhs = self.G.T * H              # G^T H
            lam = (self.G.T @ (self.d_diag[:, None] * self.G)) @ self.GtG_inv
            rhs = np.exp(lam[0, 0] * a) * self.G.T
            assert np.allclose(lhs, rhs, atol=1e-13), (
                "G is not an eigenvector of A_d; the commutation "
                "G^T H_F = H_C G^T fails and Eq. (35) does not apply."
            )

    # -- splitting -----------------------------------------------------
    def N_tilde(self, y, use_if):
        """The explicitly-integrated term.  With IF on, viscosity is
        pulled out into H; with IF off, it stays in the explicit RHS."""
        M = self.A_skew if use_if else self.A
        return -(M @ y)

    def H(self, alpha, use_if):
        """Integrating factor exp(A_d * alpha), returned as a diagonal
        vector.  In the real solver this is the LGF convolution with
        G_E(alpha) = prod_q exp(-2a) I_{n_q}(2a)."""
        if not use_if:
            return np.ones(self.n)
        return np.exp(self.d_diag * alpha)

    def multiplier(self, r):
        """Eq. (35):  d^ = -L_C^-1 G^T r  =  (G^T G)^-1 G^T r."""
        return self.GtG_inv @ (self.G.T @ r)

    # -- closed-form truth ---------------------------------------------
    def y_exact(self, t):
        return expm(self.P @ self.A * t) @ self.y0

    def z_exact(self, t):
        return self.GtG_inv @ (self.G.T @ (self.A @ self.y_exact(t)))


# ----------------------------------------------------------------------
# IF-HERK, Eqs. (25)-(31) with the stage solve replaced by Eq. (35)
# ----------------------------------------------------------------------

def if_herk_step(dae, y_k, t_k, dt, tab, use_if, constraint_log=None):
    s, At, ct, gaps = tab.s, tab.At, tab.ct, tab.gaps

    y_prev = y_k.copy()          # u^0_k = u_k                      Eq. (25)
    q = y_k.copy()               # q^1_k = u^0_k                    Eq. (29)
    w = np.zeros((s, dae.n))     # w[j-1] holds w^{i,j} at the current i
    d_hat = None

    for i in range(1, s + 1):
        ii = i - 1

        # --- age everything carried over by one sub-step, H^{i-1} -----
        # This is the whole trick: each stored field is advanced to the
        # current stage's time level, so all IF arguments stay positive.
        if i > 1:
            H_prev = dae.H(gaps[i - 2] * dt, use_if)
            q = H_prev * q                                        # Eq. (29)
            for j in range(1, i):
                w[j - 1] = H_prev * w[j - 1]                      # Eq. (30)

        # --- fresh nonlinear term, NOT aged --------------------------
        g = -At[ii, ii] * dt * dae.N_tilde(y_prev, use_if)        # Eq. (28)

        # --- assemble ------------------------------------------------
        acc = np.zeros(dae.n)
        for j in range(1, i):
            acc += At[ii, j - 1] * w[j - 1]
        r = q + dt * acc + g                                      # Eq. (27)

        # --- project, then diffuse -----------------------------------
        d_hat = dae.multiplier(r)                                 # Eq. (35)
        H_i = dae.H(gaps[ii] * dt, use_if)
        y_new = H_i * (r - dae.G @ d_hat)                         # Eq. (35)

        # --- the new slope, at this stage's level --------------------
        w[ii] = (g - dae.G @ d_hat) / (At[ii, ii] * dt)           # Eq. (30)

        if constraint_log is not None:
            constraint_log.append(float(np.abs(dae.G.T @ y_new).max()))

        y_prev = y_new

    # Eq. (31).  The (a~_ss dt)^-1 rescaling is what makes d_hat^s a
    # consistent approximation to the multiplier rather than to a
    # stage-local quantity.  Getting this wrong costs an order in z only.
    return y_new, t_k + dt, d_hat / (At[s - 1, s - 1] * dt)


def integrate(dae, tab, use_if, T, n_steps, constraint_log=None):
    dt = T / n_steps
    y, t, d = dae.y0.copy(), 0.0, None
    for _ in range(n_steps):
        y, t, d = if_herk_step(dae, y, t, dt, tab, use_if, constraint_log)
        if not np.all(np.isfinite(y)) or np.linalg.norm(y) > 1e6:
            return None, None
    return y, d


# ----------------------------------------------------------------------
# Convergence study
# ----------------------------------------------------------------------

def convergence(dae, tab, use_if, T, refinements):
    y_ex, z_ex = dae.y_exact(T), dae.z_exact(T)
    ny, nz = np.linalg.norm(y_ex, np.inf), np.linalg.norm(z_ex, np.inf)
    rows = []
    for N in refinements:
        y, d = integrate(dae, tab, use_if, T, N)
        if y is None:
            rows.append((N, np.nan, np.nan))
            continue
        rows.append((N,
                     np.linalg.norm(y - y_ex, np.inf) / ny,
                     np.linalg.norm(d - z_ex, np.inf) / nz))
    return rows


def observed_order(rows, col, window=3):
    """Least-squares slope over the finest `window` refinements."""
    pts = [(N, e) for N, *errs in rows
           for e in [errs[col]]
           if np.isfinite(e) and e > 1e-13]
    if len(pts) < 2:
        return float("nan")
    pts = pts[-(window + 1):]
    x = np.log2([p[0] for p in pts])
    ylog = np.log2([p[1] for p in pts])
    return -np.polyfit(x, ylog, 1)[0]


def print_convergence(label, rows):
    print(f"\n  {label}")
    print(f"    {'N':>6}  {'err_y':>11} {'p_y':>6}   {'err_z':>11} {'p_z':>6}")
    prev = None
    for N, ey, ez in rows:
        if not np.isfinite(ey):
            print(f"    {N:>6}  {'DIVERGED':>11}")
            prev = None
            continue
        if prev is None:
            print(f"    {N:>6}  {ey:11.3e} {'--':>6}   {ez:11.3e} {'--':>6}")
        else:
            py = np.log2(prev[0] / ey) if ey > 0 else np.nan
            pz = np.log2(prev[1] / ez) if ez > 0 else np.nan
            print(f"    {N:>6}  {ey:11.3e} {py:6.2f}   {ez:11.3e} {pz:6.2f}")
        prev = (ey, ez)


# ----------------------------------------------------------------------

def main():
    line = "=" * 74

    # ---------------- tableau report ----------------------------------
    print(line)
    print("TABLEAU INSPECTION")
    print(line)
    print(f"\n  {'scheme':<10} {'ERK':>4} {'lin':>4} {'c~':<22} {'gaps':<22} "
          f"{'IF?':>4} {'#a':>3}")
    print("  " + "-" * 72)
    for name, tab in TABLEAUX.items():
        ct = "[" + ", ".join(f"{v:.3f}" for v in tab.ct) + "]"
        gp = "[" + ", ".join(f"{v:+.3f}" for v in tab.gaps) + "]"
        flag = "yes" if tab.if_compatible else "NO"
        print(f"  {name:<10} {tab.classical_order:>4} {tab.linear_order:>4} "
              f"{ct:<22} {gp:<22} {flag:>4} {tab.n_distinct_alpha:>3}")

    print("\n  'ERK' = classical (nonlinear) order, 'lin' = order of R(z) vs exp(z).")
    print("  CAVEAT: this test problem is linear, so an explicit RK collapses")
    print("  to y_{n+1} = R(dt M) y_n.  All four tableaux share")
    print("  R(z) = 1 + z + z^2/2 + z^3/6, so their no-IF y-iterates are")
    print("  bit-identical and p_y measures 'lin', not 'ERK'.  Switching the")
    print("  IF on breaks that structure and Scheme A drops to its true 2.")
    print("  The z-column has no such degeneracy -- the multiplier depends on")
    print("  the tableau directly -- so p_z is discriminating throughout.")

    print("\n  '#a' = number of distinct nonzero integrating-factor arguments.")
    print("  Scheme A needs one, and its last gap is zero so H^s = I.")
    bad = [n for n, t in TABLEAUX.items() if not t.if_compatible]
    for n in bad:
        g = TABLEAUX[n].gaps
        neg = [f"{v:+.3f}" for v in g if v < -TOL]
        print(f"\n  {n} is IF-INCOMPATIBLE: sub-step width(s) {', '.join(neg)}.")
        print("    A negative argument means E(-a), which amplifies the")
        print("    grid-scale mode by exp(4d|a|).  Not a tuning problem --")
        print("    the tableau has to change.")

    print("\n  Imaginary-axis stability limit mu* (Eq. C.6):")
    for name, tab in TABLEAUX.items():
        mu = tab.imag_axis_limit()
        print(f"    {name:<10} mu* = {mu:.4f}"
              + ("   (= sqrt(3))" if abs(mu - r3) < 2e-3 else ""))
    print("  All three-stage third-order-stable schemes share R(z);")
    print("  mu*/sqrt(3) = 1 recovers the paper's CFL_max = 1 in 3D.")

    # ---------------- problem ------------------------------------------
    dae = LinearIndex2DAE(omega=(0.35, -0.70, 1.10),
                          d_diag=(-2.0, -2.0, -5.0),
                          G=[0.6, 0.8, 0.0])

    print("\n" + line)
    print("PROBLEM")
    print(line)
    print(f"\n  A_skew spectrum : {np.sort_complex(np.linalg.eigvals(dae.A_skew))}")
    print(f"  A_d  (diagonal) : {dae.d_diag}")
    print(f"  G^T             : {dae.G.T[0]}")
    print(f"  G^T y0          : {(dae.G.T @ dae.y0)[0]:.3e}   (constraint at t=0)")
    print("\n  Commutation G^T H(a) = H_C(a) G^T verified at three arguments.")
    print("  Note G^T A_d y = 0 on the constraint manifold, so viscosity")
    print("  contributes nothing to the multiplier -- the discrete analogue")
    print("  of G^T q^i_k = 0 in the paper.")

    # ---------------- convergence --------------------------------------
    T = 2.0
    refin = [16, 32, 64, 128, 256, 512, 1024]

    print("\n" + line)
    print(f"CONVERGENCE   (T = {T}, error vs. expm truth, inf-norm relative)")
    print(line)

    summary = []
    for name, tab in TABLEAUX.items():
        modes = [("no IF ", False)]
        if tab.if_compatible:
            modes.append(("IF    ", True))
        for mlabel, use_if in modes:
            rows = convergence(dae, tab, use_if, T, refin)
            print_convergence(f"{name}   [{mlabel.strip() or 'no IF'}]", rows)
            summary.append((name, mlabel.strip(),
                            observed_order(rows, 0),
                            observed_order(rows, 1)))

    # ---------------- summary ------------------------------------------
    print("\n" + line)
    print("OBSERVED ORDERS   (fit over the four finest refinements)")
    print(line)
    print(f"\n  {'scheme':<10} {'mode':<7} {'p_y':>6} {'p_z':>6}   {'Table 1 (y,z)':>14}")
    print("  " + "-" * 52)
    for name, mode, py, pz in summary:
        exp = PAPER_TABLE1.get(name)
        expstr = f"{exp[0]}, {exp[1]}" if exp else "not in paper"
        print(f"  {name:<10} {mode:<7} {py:6.2f} {pz:6.2f}   {expstr:>14}")

    print("\n  Table 1 lists the specialised orders (constant f_z, g_y),")
    print("  which is the column this problem exercises.")

    # ---------------- constraint ---------------------------------------
    print("\n" + line)
    print("CONSTRAINT RESIDUAL")
    print(line)
    log = []
    integrate(dae, TABLEAUX["Scheme A"], True, T, 128, constraint_log=log)
    print(f"\n  max |G^T y^i| over all stages of all steps : {max(log):.3e}")
    print("  Zero to roundoff at every stage, not just at step ends --")
    print("  the multiplier enforces the constraint, it does not correct it.")

    # ---------------- stability ----------------------------------------
    print("\n" + line)
    print("STIFFNESS")
    print(line)
    stiff = LinearIndex2DAE(omega=(0.35, -0.70, 1.10),
                            d_diag=(-2.0, -2.0, -60.0),
                            G=[0.6, 0.8, 0.0])
    tab = TABLEAUX["Scheme A"]
    print(f"\n  Stiffest viscous eigenvalue: {stiff.d_diag[-1]:.1f}")
    print(f"  Explicit real-axis limit for this R(z): |z| < 2.512")
    print(f"\n  {'N':>6} {'dt':>8} {'|lam|dt':>9}  {'no IF':>12}  {'IF':>12}")
    print("  " + "-" * 54)
    for N in (16, 32, 64, 128, 256):
        dt = T / N
        y_p, _ = integrate(stiff, tab, False, T, N)
        y_i, _ = integrate(stiff, tab, True, T, N)
        ex = stiff.y_exact(T)
        ne = np.linalg.norm(ex, np.inf)
        f = lambda y: "diverged" if y is None else \
            f"{np.linalg.norm(y - ex, np.inf) / ne:.3e}"
        print(f"  {N:>6} {dt:8.4f} {abs(stiff.d_diag[-1]) * dt:9.3f}  "
              f"{f(y_p):>12}  {f(y_i):>12}")

    # pure diffusion: IF should be exact at any dt
    print("\n  Pure diffusion (A_skew = 0), IF has nothing left to discretise:")
    pure = LinearIndex2DAE(omega=(0.0, 0.0, 0.0),
                           d_diag=(-2.0, -2.0, -5.0),
                           G=[0.6, 0.8, 0.0])
    ex = pure.y_exact(T)
    for N in (2, 8, 64):
        y_i, _ = integrate(pure, tab, True, T, N)
        y_p, _ = integrate(pure, tab, False, T, N)
        e = lambda y: np.linalg.norm(y - ex, np.inf) / np.linalg.norm(ex, np.inf)
        print(f"    N={N:>4}   no IF: {e(y_p):.3e}    IF: {e(y_i):.3e}")
    print()


if __name__ == "__main__":
    main()
