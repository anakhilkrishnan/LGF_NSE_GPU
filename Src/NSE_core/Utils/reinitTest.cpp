// standalone verification of InterfaceTracker::reinitialize() (closest point, InterfaceReinit.H),
// no flow solver. 3D only; needs AMReX Base, InitialSDF.H / InitialVorticity.H (for N_INTERFACE;
// component 0 carries the test field, the others are empty) and SolverConfig.H (LgfNseOmp.H).
//   [1] run_static=1 (default): static fields with a known zero level set (a torus): the exact
//       distance, a distorted one (|grad psi| 0.4 ... 1.6) and the squared form (not a distance
//       at all), at 8 and 16 cells per tube radius, degree 3 and 5, on a box layout with holes.
//       Error against the exact distance, sign preservation, a second reinitialization's change,
//       |grad psi| drift before and after, cost per band cell;
//   [2] run_transport=1 (default): the torus carried by a steady strain + rotation (exact flow
//       map known) with WENO5 + SSP-RK3, with no reinitialization, every step (degree 3 and 5),
//       every 10 steps, and the drift trigger (mode=N runs only mode N; correct=0 turns the
//       zero-set correction off). The zero level set's error is measured at its crossings of
//       grid edges (cubic root along the edge) against the exact interface;
//   [3] run_repeat=1: the exact torus reinitialized 1 ... 20 times, with and without the
//       zero-set correction: the zero level set must not walk.
// About 2 minutes per part on one CPU core.

#include <InterfaceTracker.H>

#include <cmath>
#include <string>
#include <vector>

using namespace amrex;

namespace {

// ---------------------------------------------------------------- geometry and layouts
struct Case { Geometry geom; BoxArray ba; DistributionMapping dm; };

// the domain [lo, hi] at cell size dx, chunked into boxes of mgs cells; boxes whose centre is
// farther than r_keep from the origin, or nearer than r_hole to the z axis, are removed
// (a ragged outer edge and an interior hole, like the snug domain)
Case makeCase(RealVect lo, RealVect hi, Real dx, int mgs, Real r_keep, Real r_hole)
{
    Case cs;
    IntVect n;
    for (int d = 0; d < 3; ++d) { n[d] = static_cast<int>(std::lround((hi[d] - lo[d]) / dx)); }
    Box domain(IntVect(0), n - 1);
    RealBox rb({lo[0], lo[1], lo[2]}, {lo[0] + n[0] * dx, lo[1] + n[1] * dx, lo[2] + n[2] * dx});
    cs.geom.define(domain, rb, CoordSys::cartesian, Array<int, 3>{0, 0, 0});
    BoxArray full(domain);
    full.maxSize(mgs);
    BoxList bl;
    for (int b = 0; b < full.size(); ++b)
    {
        const Box& bx = full[b];
        Real xc[3];
        for (int d = 0; d < 3; ++d) { xc[d] = cs.geom.ProbLo(d) + 0.5 * (bx.smallEnd(d) + bx.bigEnd(d) + 1) * dx; }
        if (std::sqrt(xc[0]*xc[0] + xc[1]*xc[1] + xc[2]*xc[2]) < r_keep && std::hypot(xc[0], xc[1]) >= r_hole)
        {
            bl.push_back(bx);
        }
    }
    cs.ba = BoxArray(std::move(bl));
    cs.dm = DistributionMapping(cs.ba);
    return cs;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real torusSDF(Real x, Real y, Real z, Real R, Real a)
{
    return a - std::sqrt((std::hypot(x, y) - R) * (std::hypot(x, y) - R) + z * z);
}

enum class Field : int { Exact = 0, Distorted = 1, Squared = 2 };
const char* fieldName[3] = {"exact    ", "distorted", "squared  "};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real testField(Field f, Real x, Real y, Real z, Real R, Real a)
{
    const Real d = torusSDF(x, y, z, R, a);
    if (f == Field::Distorted) { return d * (1.0 + 0.6 * std::sin(3.0 * x + 0.3) * std::sin(2.5 * y) * std::cos(4.0 * z)); }
    if (f == Field::Squared)   { const Real s = a - d; return a * a - s * s; }
    return d;
}

// component 0: the field, clipped like initializeSDF; the other components: empty
void setField(InterfaceTracker& tr, const Geometry& geom, Field f, Real R, Real a)
{
    const Real far = tr.getPsiFar();
    const auto plo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();
    MultiFab& psi = tr.getPsi();
    for (MFIter mfi(psi); mfi.isValid(); ++mfi)
    {
        auto const& p = psi.array(mfi);
        ParallelFor(mfi.fabbox(), psi.nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
        {
            p(i, j, k, n) = (n > 0) ? -far
                          : amrex::Clamp(testField(f, plo[0] + (i + 0.5) * dx[0], plo[1] + (j + 0.5) * dx[1],
                                                   plo[2] + (k + 0.5) * dx[2], R, a), -far, far);
        });
    }
}

// error of component 0 against the exact torus distance: max and mean over cells within
// band of the interface, and max over cells within one cell of it
struct Err { Real max_band = 0.0, mean_band = 0.0, max_near = 0.0; };
Err sdfError(const MultiFab& psi, const Geometry& geom, Real R, Real a, Real band)
{
    const auto plo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();
    ReduceOps<ReduceOpMax, ReduceOpSum, ReduceOpSum, ReduceOpMax> op;
    ReduceData<Real, Real, Long, Real> data(op);
    for (MFIter mfi(psi); mfi.isValid(); ++mfi)
    {
        auto const& p = psi.const_array(mfi);
        op.eval(mfi.validbox(), data, [=] AMREX_GPU_DEVICE (int i, int j, int k) -> GpuTuple<Real, Real, Long, Real>
        {
            const Real d = torusSDF(plo[0] + (i + 0.5) * dx[0], plo[1] + (j + 0.5) * dx[1], plo[2] + (k + 0.5) * dx[2], R, a);
            if (std::abs(d) > band) { return {0.0, 0.0, 0, 0.0}; }
            const Real e = std::abs(p(i, j, k) - d);
            return {e, e, 1, std::abs(d) <= dx[0] ? e : 0.0};
        });
    }
    const auto r = data.value(op);
    Err e;
    e.max_band = get<0>(r);
    e.mean_band = get<1>(r) / std::max(Long(1), get<2>(r));
    e.max_near = get<3>(r);
    return e;
}

// max |a - b| over all valid cells and over cells with |b| < near, and the number of cells whose
// sign differs (component 0)
struct Cmp { Real all = 0.0, near = 0.0; Long sign = 0; };
Cmp compare(const MultiFab& a, const MultiFab& b, Real near)
{
    ReduceOps<ReduceOpMax, ReduceOpMax, ReduceOpSum> op;
    ReduceData<Real, Real, Long> data(op);
    for (MFIter mfi(a); mfi.isValid(); ++mfi)
    {
        auto const& pa = a.const_array(mfi);
        auto const& pb = b.const_array(mfi);
        op.eval(mfi.validbox(), data, [=] AMREX_GPU_DEVICE (int i, int j, int k) -> GpuTuple<Real, Real, Long>
        {
            const Real e = std::abs(pa(i, j, k) - pb(i, j, k));
            return {e, std::abs(pb(i, j, k)) < near ? e : 0.0, ((pa(i, j, k) >= 0.0) != (pb(i, j, k) >= 0.0)) ? Long(1) : Long(0)};
        });
    }
    const auto r = data.value(op);
    return {get<0>(r), get<1>(r), get<2>(r)};
}

// ---------------------------------------------------------------- transport
// u = S x - Om y, v = Om x - S y, w = 0 on the faces: divergence free, linear, so the exact flow
// map is x(t) = exp(A t) x0 with A = [[S, -Om], [Om, -S]] in the x-y plane (A^2 = mu^2 I)
void setVelocity(Array<MultiFab, 3>& vel, const Geometry& geom, Real S, Real Om)
{
    const auto plo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();
    for (int d = 0; d < 3; ++d)
    {
        for (MFIter mfi(vel[d]); mfi.isValid(); ++mfi)
        {
            auto const& u = vel[d].array(mfi);
            ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                const Real x = plo[0] + (i + (d == 0 ? 0.0 : 0.5)) * dx[0];
                const Real y = plo[1] + (j + (d == 1 ? 0.0 : 0.5)) * dx[1];
                ignore_unused(k);
                u(i, j, k) = (d == 0) ? S * x - Om * y : (d == 1) ? Om * x - S * y : 0.0;
            });
        }
    }
}

// back-map M = exp(-A t) of the x-y block
struct BackMap { Real m00, m01, m10, m11; };
BackMap backMap(Real S, Real Om, Real t)
{
    const Real mu = std::sqrt(S * S - Om * Om);   // S > Om here
    const Real c = std::cosh(mu * t), s = std::sinh(mu * t) / mu;
    return {c - s * S, s * Om, -s * Om, c + s * S};
}

// position error of the zero level set of component 0: at every crossing of a grid edge
// (located by the root of the cubic through the four values along the edge), the distance
// of the crossing to the exact interface, |psi_e| / |grad psi_e| with psi_e = psi0(M x)
struct PosErr { Real max = 0.0, mean = 0.0; Long n = 0; };
PosErr zeroSetError(const MultiFab& psi, const Geometry& geom, Real R, Real a, BackMap M)
{
    const auto plo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();
    ReduceOps<ReduceOpMax, ReduceOpSum, ReduceOpSum> op;
    ReduceData<Real, Real, Long> data(op);
    for (MFIter mfi(psi); mfi.isValid(); ++mfi)
    {
        auto const& p = psi.const_array(mfi);
        op.eval(mfi.validbox(), data, [=] AMREX_GPU_DEVICE (int i, int j, int k) -> GpuTuple<Real, Real, Long>
        {
            Real emax = 0.0, esum = 0.0; Long cnt = 0;
            for (int d = 0; d < 3; ++d)
            {
                const int e0 = (d == 0), e1 = (d == 1), e2 = (d == 2);
                const Real f0 = p(i - e0, j - e1, k - e2), f1 = p(i, j, k), f2 = p(i + e0, j + e1, k + e2),
                           f3 = p(i + 2*e0, j + 2*e1, k + 2*e2);
                if ((f1 >= 0.0) == (f2 >= 0.0)) { continue; }
                // cubic through nodes -1, 0, 1, 2; root in [0, 1] by safeguarded Newton
                auto cub = [=] (Real t, Real& dv) {
                    const Real w0 = -t*(t-1)*(t-2)/6, w1 = (t+1)*(t-1)*(t-2)/2, w2 = -(t+1)*t*(t-2)/2, w3 = (t+1)*t*(t-1)/6;
                    const Real d0 = (-3*t*t + 6*t - 2)/6, d1 = (3*t*t - 4*t - 1)/2, d2 = -(3*t*t - 2*t - 2)/2, d3 = (3*t*t - 1)/6;
                    dv = d0*f0 + d1*f1 + d2*f2 + d3*f3;
                    return w0*f0 + w1*f1 + w2*f2 + w3*f3;
                };
                Real lo = 0.0, hi = 1.0, t = f1 / (f1 - f2);
                for (int it = 0; it < 30; ++it)
                {
                    Real dv; const Real v = cub(t, dv);
                    if ((v >= 0.0) == (f1 >= 0.0)) { lo = t; } else { hi = t; }
                    Real tn = t - v / dv;
                    if (!(tn > lo && tn < hi)) { tn = 0.5 * (lo + hi); }
                    if (std::abs(tn - t) < 1e-14) { t = tn; break; }
                    t = tn;
                }
                Real x = plo[0] + (i + 0.5 + t * e0) * dx[0];
                Real y = plo[1] + (j + 0.5 + t * e1) * dx[1];
                Real z = plo[2] + (k + 0.5 + t * e2) * dx[2];
                // exact level set psi0(M x) and its gradient M^T grad psi0
                const Real X = M.m00 * x + M.m01 * y, Y = M.m10 * x + M.m11 * y;
                const Real rho = std::hypot(X, Y), s = std::sqrt((rho - R) * (rho - R) + z * z);
                const Real val = a - s;
                const Real gX = -(rho - R) / s * X / rho, gY = -(rho - R) / s * Y / rho, gZ = -z / s;
                const Real gx = M.m00 * gX + M.m10 * gY, gy = M.m01 * gX + M.m11 * gY;
                const Real err = std::abs(val) / std::sqrt(gx * gx + gy * gy + gZ * gZ);
                emax = std::max(emax, err); esum += err; ++cnt;
            }
            return {emax, esum, cnt};
        });
    }
    const auto r = data.value(op);
    PosErr e;
    e.max = get<0>(r);
    e.n = get<2>(r);
    e.mean = get<1>(r) / std::max(Long(1), e.n);
    return e;
}

void sspRK3Step(InterfaceTracker& tr, Array<MultiFab, 3>& vel, Real dt)
{
    const GpuArray<Real, 3> uinf{0.0, 0.0, 0.0};
    MultiFab& psi = tr.getPsi();
    MultiFab psi0, stage, rhs;
    tr.defineLikePsi(psi0, 0);
    tr.defineLikePsi(stage, LS_N_GHOST);
    tr.defineLikePsi(rhs, 0);
    const int nc = psi.nComp();
    MultiFab::Copy(psi0, psi, 0, 0, nc, 0);
    tr.computeAdvectionRHS(psi, rhs, vel, uinf);
    MultiFab::LinComb(stage, 1.0, psi0, 0, dt, rhs, 0, 0, nc, 0);
    tr.computeAdvectionRHS(stage, rhs, vel, uinf);
    MultiFab::LinComb(stage, 0.25, stage, 0, 0.25 * dt, rhs, 0, 0, nc, 0);
    MultiFab::Saxpy(stage, 0.75, psi0, 0, 0, nc, 0);
    tr.computeAdvectionRHS(stage, rhs, vel, uinf);
    MultiFab::LinComb(psi, 2.0 / 3.0, stage, 0, 2.0 / 3.0 * dt, rhs, 0, 0, nc, 0);
    MultiFab::Saxpy(psi, 1.0 / 3.0, psi0, 0, 0, nc, 0);
}

} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        ParmParse pp;
        int run1 = 1, run2 = 1;
        pp.query("run_static", run1);
        pp.query("run_transport", run2);
        Print() << "N_INTERFACE = " << N_INTERFACE << " (component 0 carries the test field)\n";

        // ---------------- [1] static fields ----------------
        if (run1)
        {
            const Real R = 1.0, a = 0.25;
            Print() << "\n[1] static torus fields, R = " << R << ", a = " << a << ", psi_far = 8 dx; errors in cells,"
                       " over cells within psi_far - 2 dx of the interface (band) and within 1 dx (near)\n";
            for (int cells : {8, 16})
            {
                const Real dx = a / cells;
                Case cs = makeCase(RealVect(-1.6, -1.6, -0.6), RealVect(1.6, 1.6, 0.6), dx, 32, 1.95, 0.3);
                for (int f = 0; f < 3; ++f)
                {
                    for (int degree : {3, 5})
                    {
                        InterfaceTracker tr(cs.geom, cs.ba, cs.dm, AdvectionScheme::WENO5, 8.0 * dx);
                        tr.setReinitialization(1, 0.0, degree);
                        setField(tr, cs.geom, static_cast<Field>(f), R, a);
                        const InterfaceTracker::GradientDrift before = tr.computeGradientDrift(0.01);

                        MultiFab old(cs.ba, cs.dm, 1, 0);
                        MultiFab::Copy(old, tr.getPsi(), 0, 0, 1, 0);
                        Gpu::streamSynchronize();
                        const Real t0 = second();
                        const InterfaceTracker::ReinitStats st = tr.reinitialize();
                        const Real wall = second() - t0;
                        const Err e = sdfError(tr.getPsi(), cs.geom, R, a, 6.0 * dx);
                        const Cmp signs = compare(tr.getPsi(), old, 0.0);

                        MultiFab once(cs.ba, cs.dm, 1, 0);
                        MultiFab::Copy(once, tr.getPsi(), 0, 0, 1, 0);
                        tr.reinitialize();
                        const Cmp again = compare(tr.getPsi(), once, 6.0 * dx);
                        const InterfaceTracker::GradientDrift after = tr.computeGradientDrift(0.01);

                        Print() << "  " << cells << " cells/a  " << fieldName[f] << " degree " << degree
                                << " | band cells " << st.n_band << ", first pass unresolved " << st.n_first_pass_failed
                                << ", estimates " << st.n_estimated << ", beyond " << st.n_beyond << ", unresolved " << st.n_unconverged
                                << ", correction " << st.max_correction
                                << " | error max " << e.max_band / dx << ", mean " << e.mean_band / dx << ", near max " << e.max_near / dx
                                << " | sign changes " << signs.sign << " | 2nd reinit changes psi by " << again.near / dx
                                << " within 6 dx, " << again.all / dx << " anywhere"
                                << " | drift max " << before.max_drift << " -> " << after.max_drift
                                << " | " << wall << " s (" << 1e6 * wall / std::max(Long(1), st.n_band) << " us/band cell)\n";
                    }
                }
            }
        }

        // ---------------- [3] repeated reinitialization of a static field ----------------
        int run3 = 0;
        pp.query("run_repeat", run3);
        if (run3)
        {
            const Real R = 1.0, a = 0.25, dx = a / 8.0;
            Case cs = makeCase(RealVect(-1.6, -1.6, -0.6), RealVect(1.6, 1.6, 0.6), dx, 32, 1.95, 0.3);
            Print() << "\n[3] the exact torus (8 cells per tube radius) reinitialized again and again; zero level set"
                       " error against the exact torus at edge crossings, in cells (max / mean)\n";
            for (int degree : {3, 5})
            {
                for (int corr : {0, 1})
                {
                    InterfaceTracker tr(cs.geom, cs.ba, cs.dm, AdvectionScheme::WENO5, 8.0 * dx);
                    tr.setReinitialization(1, 0.0, degree);
                    tr.setReinitCorrection(corr != 0);
                    setField(tr, cs.geom, Field::Exact, R, a);
                    tr.computeGradientDrift(0.1);   // fills ghost cells
                    const PosErr e0 = zeroSetError(tr.getPsi(), cs.geom, R, a, BackMap{1.0, 0.0, 0.0, 1.0});
                    Print() << "  degree " << degree << (corr ? ", corrected  " : ", uncorrected") << " | 0: "
                            << e0.max / dx << " / " << e0.mean / dx;
                    int done = 0;
                    for (int target : {1, 2, 5, 10, 20})
                    {
                        while (done < target) { tr.reinitialize(); ++done; }
                        const PosErr e = zeroSetError(tr.getPsi(), cs.geom, R, a, BackMap{1.0, 0.0, 0.0, 1.0});
                        Print() << " | " << target << ": " << e.max / dx << " / " << e.mean / dx;
                    }
                    Print() << "\n";
                }
            }
        }

        // ---------------- [4] the same transport driven by the IF-HERK stage loop ----------------
        // psi advanced by advanceStage(i, ...) / finishStep() with the solver's tableau (inviscid
        // IFHERK: no integrating factor), reinitialized between steps as main() does, and
        // regridded at mid-run onto a re-chunked layout (the stage registers are dropped there)
        int run4 = 0;
        pp.query("run_ifherk", run4);
        if (run4)
        {
            const Real R = 0.5, a = 0.25, S = 1.0, Om = 0.5, T = 0.4;
            const Real dx = 1.0 / 32.0;
            Case cs = makeCase(RealVect(-1.5, -1.0, -0.625), RealVect(1.5, 1.0, 0.625), dx, 16, 10.0, 0.0);
            Print() << "\n[4] as [2], but psi advanced through the IF-HERK stage loop (advanceStage / finishStep,"
                       " tableau from RKCoefficients.H) with a regrid onto 32^3 boxes at mid-run\n";
            struct Mode4 { const char* name; int interval; Real drift; };
            const std::vector<Mode4> modes = {{"no reinitialization    ", 0, 0.0},
                                              {"reinitialized each step", 1, 0.0},
                                              {"drift trigger 0.1      ", 0, 0.1}};
            for (const Mode4& md : modes)
            {
                BoxArray ba = cs.ba;
                DistributionMapping dmap = cs.dm;
                Array<MultiFab, 3> vel;
                auto defineVel = [&] () {
                    for (int d = 0; d < 3; ++d) { vel[d].define(convert(ba, IntVect::TheDimensionVector(d)), dmap, 1, 0); }
                    setVelocity(vel, cs.geom, S, Om);
                };
                defineVel();

                InterfaceTracker tr(cs.geom, ba, dmap, AdvectionScheme::WENO5, 8.0 * dx);
                tr.setReinitialization(md.interval, md.drift, 3);
                setField(tr, cs.geom, Field::Exact, R, a);

                const GpuArray<Real, 3> uinf{0.0, 0.0, 0.0};
                const Real dt = 0.5 / tr.computeCFL(vel, uinf, 1.0);
                const int nsteps = static_cast<int>(std::ceil(T / dt));
                const Real dts = T / nsteps;
                IFHERK integrator(getRKButcher(), cs.geom, 0.0, dts, 0, 1.0e-12);

                int n_reinit = 0;
                for (int step = 1; step <= nsteps; ++step)
                {
                    for (int i = 1; i <= integrator.nStages(); ++i) { tr.advanceStage(i, integrator, vel, uinf); }
                    tr.finishStep();
                    if (step == nsteps / 2)
                    {
                        BoxArray nba(cs.geom.Domain());
                        nba.maxSize(32);
                        ba = nba;
                        dmap = DistributionMapping(ba);
                        tr.regrid(cs.geom, ba, dmap);
                        defineVel();
                    }
                    if (md.interval == 1) { tr.reinitialize(); ++n_reinit; }
                    else if (md.drift > 0.0) { n_reinit += tr.reinitializeIfDue(step) ? 1 : 0; }
                }
                tr.computeGradientDrift(0.1);   // refreshes ghost cells for the edge crossings
                const PosErr pe = zeroSetError(tr.getPsi(), cs.geom, R, a, backMap(S, Om, T));
                const InterfaceTracker::GradientDrift gd = tr.computeGradientDrift(0.1);
                Print() << "  " << md.name << " | " << nsteps << " steps, " << n_reinit << " reinitializations, "
                        << tr.getPsi().boxArray().size() << " boxes at the end"
                        << " | zero level set error: max " << pe.max / dx << " dx, mean " << pe.mean / dx << " dx over "
                        << pe.n << " edge crossings | drift max " << gd.max_drift << ", share over 0.1: "
                        << 100.0 * gd.n_over / std::max(Long(1), gd.n_near) << "%\n";
            }
        }

        // ---------------- [2] transport by strain + rotation ----------------
        if (run2)
        {
            const Real R = 0.5, a = 0.25, S = 1.0, Om = 0.5, T = 0.4;
            const Real dx = 1.0 / 32.0;
            Case cs = makeCase(RealVect(-1.5, -1.0, -0.625), RealVect(1.5, 1.0, 0.625), dx, 16, 10.0, 0.0);
            Array<MultiFab, 3> vel;
            for (int d = 0; d < 3; ++d) { vel[d].define(convert(cs.ba, IntVect::TheDimensionVector(d)), cs.dm, 1, 0); }
            setVelocity(vel, cs.geom, S, Om);

            Print() << "\n[2] torus R = " << R << ", a = " << a << " (" << a / dx << " cells) carried by strain S = " << S
                    << " + rotation Om = " << Om << " to t = " << T << ", WENO5 + SSP-RK3, psi_far = 8 dx\n";
            struct Mode { const char* name; int interval; Real drift; int degree; };
            const std::vector<Mode> modes = {{"no reinitialization        ", 0, 0.0, 3},
                                             {"each step, degree 3        ", 1, 0.0, 3},
                                             {"each step, degree 5        ", 1, 0.0, 5},
                                             {"every 10 steps, degree 3   ", 10, 0.0, 3},
                                             {"drift trigger 0.1, degree 3", 0, 0.1, 3}};
            int only = -1;
            pp.query("mode", only);
            for (int mode = 0; mode < static_cast<int>(modes.size()); ++mode)
            {
                if (only >= 0 && mode != only) { continue; }
                const Mode& md = modes[mode];
                InterfaceTracker tr(cs.geom, cs.ba, cs.dm, AdvectionScheme::WENO5, 8.0 * dx);
                tr.setReinitialization(md.interval, md.drift, md.degree);
                int corr = 1;
                pp.query("correct", corr);
                tr.setReinitCorrection(corr != 0);
                setField(tr, cs.geom, Field::Exact, R, a);

                const Real dt = 0.5 / (tr.computeCFL(vel, GpuArray<Real, 3>{0.0, 0.0, 0.0}, 1.0));
                const int nsteps = static_cast<int>(std::ceil(T / dt));
                const Real dts = T / nsteps;
                int n_reinit = 0;
                Long unconverged = 0;
                Gpu::streamSynchronize();
                const Real t0 = second();
                for (int step = 1; step <= nsteps; ++step)
                {
                    sspRK3Step(tr, vel, dts);
                    if (md.interval == 1) { unconverged += tr.reinitialize().n_unconverged; ++n_reinit; }
                    else if (md.interval > 1 || md.drift > 0.0) { n_reinit += tr.reinitializeIfDue(step) ? 1 : 0; }
                }
                const Real wall = second() - t0;
                tr.computeGradientDrift(0.1);   // refreshes ghost cells for the edge crossings
                const PosErr pe = zeroSetError(tr.getPsi(), cs.geom, R, a, backMap(S, Om, T));
                const InterfaceTracker::GradientDrift gd = tr.computeGradientDrift(0.1);
                Print() << "  " << md.name << " | " << nsteps << " steps, " << n_reinit << " reinitializations"
                        << (md.interval == 1 ? " (" + std::to_string(unconverged) + " unresolved cells in all)" : std::string())
                        << " | zero level set error: max " << pe.max / dx << " dx, mean " << pe.mean / dx << " dx over "
                        << pe.n << " edge crossings | drift max " << gd.max_drift << ", share over 0.1: "
                        << 100.0 * gd.n_over / std::max(Long(1), gd.n_near) << "% | " << wall << " s\n";
            }
        }
    }
    amrex::Finalize();
}
