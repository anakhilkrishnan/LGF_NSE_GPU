// standalone verification of InterfaceTracker with prescribed face velocities
// (no flow solver): operator accuracy for translation/rotation/strain, ghost
// filling on a snug-like layout with holes, a short SSP-RK3 run with a regrid,
// and the solver path (initializeSDF with the active case, computeCFL). 3D only;
// needs AMReX Base and InitialVorticity.H (built and run against AMReX 25.09).
// Run with guard_test=edge, dropped or init to see the band check abort.
#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>

#include <InterfaceTracker.H>

#include <cmath>
#include <string>
#include <vector>

using namespace amrex;

namespace {

// a snug-like layout: the domain chunked into boxes of size mgs, with every
// box removed whose center lies outside a sphere of radius r_keep (holes and a
// ragged outer edge, like Dxsoln)
BoxArray snugLayout(const Geometry& geom, int mgs, Real r_keep)
{
    BoxArray full(geom.Domain());
    full.maxSize(mgs);
    BoxList bl;
    for (int b = 0; b < full.size(); ++b)
    {
        const Box& bx = full[b];
        Real r2 = 0.0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            const Real xc = geom.ProbLo(d) + 0.5 * (bx.smallEnd(d) + bx.bigEnd(d) + 1) * geom.CellSize(d);
            r2 += xc * xc;
        }
        if (std::sqrt(r2) < r_keep) { bl.push_back(bx); }
    }
    return BoxArray(std::move(bl));
}

// face velocities of a uniform translation plus a rigid rotation about z:
// u = U - Om*y, v = V + Om*x, w = W (discretely divergence free on faces)
void setVelocity(Array<MultiFab, AMREX_SPACEDIM>& vel, const Geometry& geom, GpuArray<Real, 3> U, Real Om, Real S = 0.0)
{
    const auto plo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        for (MFIter mfi(vel[d]); mfi.isValid(); ++mfi)
        {
            auto const& a = vel[d].array(mfi);
            const Box bx = mfi.fabbox();
            ParallelFor(bx, [=] (int i, int j, int k)
            {
                // face-centered coordinates in direction d, cell-centered otherwise
                const Real x = plo[0] + (i + (d == 0 ? 0.0 : 0.5)) * dx[0];
                const Real y = plo[1] + (j + (d == 1 ? 0.0 : 0.5)) * dx[1];
                ignore_unused(k);
                if (d == 0) { a(i,j,k) = U[0] - Om * y + S * x; }
                if (d == 1) { a(i,j,k) = U[1] + Om * x - S * y; }
                if (d == 2) { a(i,j,k) = U[2]; }
            });
        }
    }
}

// max |rhs - exact| over cells within `band` of the interface, for a sphere of
// radius R centered at c: exact rhs = -u . grad(psi) with grad(psi) = -(x-c)/|x-c|
Real rhsError(const MultiFab& rhs, const Geometry& geom, GpuArray<Real, 3> c, Real R,
              GpuArray<Real, 3> U, Real Om, Real band, Real S = 0.0)
{
    const auto plo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();
    ReduceOps<ReduceOpMax> op;
    ReduceData<Real> data(op);
    for (MFIter mfi(rhs); mfi.isValid(); ++mfi)
    {
        auto const& r = rhs.const_array(mfi);
        op.eval(mfi.validbox(), data, [=] (int i, int j, int k) -> GpuTuple<Real>
        {
            const Real x = plo[0] + (i + 0.5) * dx[0];
            const Real y = plo[1] + (j + 0.5) * dx[1];
            const Real z = plo[2] + (k + 0.5) * dx[2];
            const Real px = x - c[0], py = y - c[1], pz = z - c[2];
            const Real dist = std::sqrt(px*px + py*py + pz*pz);
            if (std::abs(R - dist) > band) { return {0.0}; }
            const Real u = U[0] - Om * y + S * x, v = U[1] + Om * x - S * y, w = U[2];
            const Real exact = (u * px + v * py + w * pz) / dist;   // -u . grad(psi)
            return {std::abs(r(i,j,k) - exact)};
        });
    }
    return amrex::get<0>(data.value());
}

// max |psi - exact SDF| near the interface of a sphere centered at c
Real psiError(const MultiFab& psi, const Geometry& geom, GpuArray<Real, 3> c, Real R, Real band)
{
    const auto plo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();
    ReduceOps<ReduceOpMax> op;
    ReduceData<Real> data(op);
    for (MFIter mfi(psi); mfi.isValid(); ++mfi)
    {
        auto const& p = psi.const_array(mfi);
        op.eval(mfi.validbox(), data, [=] (int i, int j, int k) -> GpuTuple<Real>
        {
            const Real px = plo[0] + (i + 0.5) * dx[0] - c[0];
            const Real py = plo[1] + (j + 0.5) * dx[1] - c[1];
            const Real pz = plo[2] + (k + 0.5) * dx[2] - c[2];
            const Real exact = R - std::sqrt(px*px + py*py + pz*pz);
            if (std::abs(exact) > band) { return {0.0}; }
            return {std::abs(p(i,j,k) - exact)};
        });
    }
    return amrex::get<0>(data.value());
}

Real volumeInside(const MultiFab& psi, const Geometry& geom)
{
    // sum of cells with psi > 0 corrected linearly would be nicer; a smoothed
    // Heaviside of width dx is enough to watch drift
    const Real eps = 0.5 * geom.CellSize(0);
    const Real dv = AMREX_D_TERM(geom.CellSize(0), * geom.CellSize(1), * geom.CellSize(2));
    ReduceOps<ReduceOpSum> op;
    ReduceData<Real> data(op);
    for (MFIter mfi(psi); mfi.isValid(); ++mfi)
    {
        auto const& p = psi.const_array(mfi);
        op.eval(mfi.validbox(), data, [=] (int i, int j, int k) -> GpuTuple<Real>
        {
            return {0.5 + 0.5 * std::tanh(p(i,j,k) / (2.0 * eps))};
        });
    }
    return amrex::get<0>(data.value()) * dv;
}

// the solver initializes psi from the active case in InitialSDF.H; the tests place their own
// sphere instead (clipped like initializeSDF, valid and ghost cells)
void setSphere(InterfaceTracker& tr, const Geometry& geom, GpuArray<Real, 3> c, Real R)
{
    SphereSDF s;
    s.radius = R;
    for (int d = 0; d < 3; ++d) { s.center[d] = c[d]; }
    const Real clip = tr.getPsiFar();
    const auto plo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();
    MultiFab& psi = tr.getPsi();
    for (MFIter mfi(psi); mfi.isValid(); ++mfi)
    {
        auto const& p = psi.array(mfi);
        ParallelFor(mfi.fabbox(), [=] (int i, int j, int k)
        {
            p(i,j,k) = amrex::Clamp(sphereSDF(plo[0] + (i + 0.5) * dx[0], plo[1] + (j + 0.5) * dx[1],
                                              plo[2] + (k + 0.5) * dx[2], s), -clip, clip);
        });
    }
}

struct Case
{
    Geometry geom;
    BoxArray ba;
    DistributionMapping dm;
};

Case makeCase(int n, int mgs, Real r_keep)
{
    Case cs;
    Box domain(IntVect(0), IntVect(n - 1));
    RealBox rb({AMREX_D_DECL(-1.0, -1.0, -1.0)}, {AMREX_D_DECL(1.0, 1.0, 1.0)});
    Array<int, AMREX_SPACEDIM> per{AMREX_D_DECL(0, 0, 0)};
    cs.geom.define(domain, rb, CoordSys::cartesian, per);
    cs.ba = snugLayout(cs.geom, mgs, r_keep);
    cs.dm = DistributionMapping(cs.ba);
    return cs;
}

void sspRK3Step(InterfaceTracker& tr, Array<MultiFab, AMREX_SPACEDIM>& vel,
                GpuArray<Real, AMREX_SPACEDIM> uinf, Real dt)
{
    MultiFab& psi = tr.getPsi();
    MultiFab psi0, stage, rhs;
    tr.defineLikePsi(psi0, 0);
    tr.defineLikePsi(stage, LS_N_GHOST);
    tr.defineLikePsi(rhs, 0);
    const int nc = psi.nComp();

    MultiFab::Copy(psi0, psi, 0, 0, nc, 0);
    tr.computeAdvectionRHS(psi, rhs, vel, uinf);
    MultiFab::LinComb(stage, 1.0, psi0, 0, dt, rhs, 0, 0, nc, 0);                 // u1
    tr.computeAdvectionRHS(stage, rhs, vel, uinf);
    MultiFab::LinComb(stage, 0.25, stage, 0, 0.25 * dt, rhs, 0, 0, nc, 0);
    MultiFab::Saxpy(stage, 0.75, psi0, 0, 0, nc, 0);                              // u2
    tr.computeAdvectionRHS(stage, rhs, vel, uinf);
    MultiFab::LinComb(psi, 2.0 / 3.0, stage, 0, 2.0 / 3.0 * dt, rhs, 0, 0, nc, 0);
    MultiFab::Saxpy(psi, 1.0 / 3.0, psi0, 0, 0, nc, 0);                           // u3
}

} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        const Real R = 0.35;
        const Real psi_far = 0.4;
        const GpuArray<Real, AMREX_SPACEDIM> uinf{AMREX_D_DECL(0.0, 0.0, 0.0)};

        // ---------------- 1. operator accuracy ----------------
        Print() << "\n[1] advection RHS error in the band |psi| < 0.1 (exact SDF data)\n";
        const char* vname[3] = {"translation", "rotation   ", "strain     "};
        for (int scheme_i = 0; scheme_i < 2; ++scheme_i)
        {
            const AdvectionScheme sch = scheme_i == 0 ? AdvectionScheme::WENO5 : AdvectionScheme::HOUC5;
            for (int vcase = 0; vcase < 3; ++vcase)
            {
                const GpuArray<Real, 3> U  = vcase == 0 ? GpuArray<Real, 3>{0.8, -0.5, 0.3} : GpuArray<Real, 3>{0.0, 0.0, 0.0};
                const Real Om = vcase == 1 ? 1.0 : 0.0;
                const Real S  = vcase == 2 ? 1.0 : 0.0;
                const GpuArray<Real, 3> c{0.1, -0.05, 0.02};
                Real e_prev = -1.0;
                for (int n : {32, 64, 128})
                {
                    Case cs = makeCase(n, 16, 10.0);   // full domain, several boxes
                    InterfaceTracker tr(cs.geom, cs.ba, cs.dm, sch, psi_far);
                    setSphere(tr, cs.geom, c, R);

                    Array<MultiFab, AMREX_SPACEDIM> vel;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        vel[d].define(convert(cs.ba, IntVect::TheDimensionVector(d)), cs.dm, 1, 0);
                    }
                    setVelocity(vel, cs.geom, U, Om, S);

                    MultiFab rhs;
                    tr.defineLikePsi(rhs, 0);
                    tr.computeAdvectionRHS(tr.getPsi(), rhs, vel, uinf);
                    const Real e = rhsError(rhs, cs.geom, c, R, U, Om, 0.1, S);

                    Print() << "  " << (scheme_i == 0 ? "WENO5" : "HOUC5") << "  " << vname[vcase] << "  n = " << n
                            << "  err = " << e;
                    if (e_prev > 0.0) { Print() << "  (order " << std::log2(e_prev / e) << ")"; }
                    Print() << "\n";
                    e_prev = e;
                }
            }
        }

        // ---------------- 2. ghost filling on a layout with holes ----------------
        Print() << "\n[2] ghost fill: linear field, snug layout with holes\n";
        {
            Case cs = makeCase(64, 16, 1.0);
            Print() << "  boxes kept: " << cs.ba.size() << " of 64\n";
            InterfaceTracker tr(cs.geom, cs.ba, cs.dm, AdvectionScheme::WENO5, psi_far);
            MultiFab& psi = tr.getPsi();
            const auto plo = cs.geom.ProbLoArray();
            const auto dx = cs.geom.CellSizeArray();
            auto lin = [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) {
                return 0.3 + 0.7 * (plo[0] + (i + 0.5) * dx[0]) - 0.2 * (plo[1] + (j + 0.5) * dx[1])
                           + 0.5 * (plo[2] + (k + 0.5) * dx[2]);
            };
            for (MFIter mfi(psi); mfi.isValid(); ++mfi)
            {
                auto const& p = psi.array(mfi);
                const Box vbx = mfi.validbox();
                ParallelFor(mfi.fabbox(), [=] (int i, int j, int k)
                {
                    p(i,j,k) = vbx.contains(IntVect(i,j,k)) ? lin(i,j,k) : 1.0e30;   // poison ghosts
                });
            }
            tr.fillPsiBoundary(psi);
            ReduceOps<ReduceOpMax> op;
            ReduceData<Real> data(op);
            for (MFIter mfi(psi); mfi.isValid(); ++mfi)
            {
                auto const& p = psi.const_array(mfi);
                op.eval(mfi.fabbox(), data, [=] (int i, int j, int k) -> GpuTuple<Real>
                {
                    return {std::abs(p(i,j,k) - lin(i,j,k))};
                });
            }
            Print() << "  max |ghost - linear| over all ghost cells: " << amrex::get<0>(data.value()) << "\n";
        }

        // ---------------- 3. SSP-RK3 run, regrid halfway ----------------
        Print() << "\n[3] translate + rotate a sphere with SSP-RK3, regrid halfway\n";
        const Real psi_far_run = 0.3;   // keeps the whole band inside the layouts below (> 6 dx at n = 64)
        const Real keep_before[3] = {10.0, 1.2, 1.0};
        const Real keep_after[3]  = {10.0, 10.0, 1.4};
        std::string guard_test;                      // "edge" or "dropped": run a case the guard must reject
        ParmParse pp;
        pp.query("guard_test", guard_test);
        const int lay_first = (guard_test == "edge") ? 2 : 0;
        const int lay_last  = (guard_test == "edge") ? 2 : 1;
        if (guard_test == "init")
        {
            // the active case's band does not fit: the ring core reaches |x| = 1.4, plus psi_far = 0.4
            Box domain(IntVect(0), IntVect(47));
            RealBox rb({AMREX_D_DECL(-1.5, -1.5, -1.5)}, {AMREX_D_DECL(1.5, 1.5, 1.5)});
            Array<int, AMREX_SPACEDIM> per{AMREX_D_DECL(0, 0, 0)};
            Geometry gi(domain, rb, CoordSys::cartesian, per);
            BoxArray bi(domain); bi.maxSize(16);
            InterfaceTracker tr(gi, bi, DistributionMapping(bi), AdvectionScheme::WENO5, 0.4);
            Print() << "  guard_test=init: initializeSDF on [-1.5, 1.5]^3\n";
            tr.initializeSDF();
            Print() << "  ERROR: the guard did not fire\n";
        }
        if (guard_test == "dropped")
        {
            // shrink the domain over the band right after initialization: must abort
            Case cs = makeCase(64, 16, 10.0);
            InterfaceTracker tr(cs.geom, cs.ba, cs.dm, AdvectionScheme::WENO5, psi_far);
            setSphere(tr, cs.geom, {0.2, 0.0, -0.1}, R);
            const BoxArray small_ba = snugLayout(cs.geom, 16, 0.6);
            Print() << "  guard_test=dropped: regrid " << cs.ba.size() << " -> " << small_ba.size() << " boxes\n";
            tr.regrid(cs.geom, small_ba, DistributionMapping(small_ba));
            Print() << "  ERROR: the guard did not fire\n";
        }
        for (int lay = lay_first; lay <= lay_last && guard_test != "dropped" && guard_test != "init"; ++lay)
        for (int scheme_i = 0; scheme_i < 2; ++scheme_i)
        {
            const AdvectionScheme sch = scheme_i == 0 ? AdvectionScheme::WENO5 : AdvectionScheme::HOUC5;
            const int n = 64;
            Case cs = makeCase(n, 16, keep_before[lay]);
            const GpuArray<Real, 3> U{0.4, 0.0, 0.2};
            const Real Om = 1.0;
            GpuArray<Real, 3> c0{0.2, 0.0, -0.1};
            InterfaceTracker tr(cs.geom, cs.ba, cs.dm, sch, psi_far_run);
            setSphere(tr, cs.geom, c0, R);
            const Real vol0 = volumeInside(tr.getPsi(), cs.geom);

            auto makeVel = [&] (const BoxArray& ba, const DistributionMapping& dm) {
                Array<MultiFab, AMREX_SPACEDIM> vel;
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    vel[d].define(convert(ba, IntVect::TheDimensionVector(d)), dm, 1, 0);
                }
                setVelocity(vel, cs.geom, U, Om);
                return vel;
            };
            auto vel = makeVel(cs.ba, cs.dm);

            const Real T = 0.5;
            const Real umax = 0.4 + 0.2 + 2.0 * Om;    // crude bound on |u|_1 over the domain
            const int nsteps = static_cast<int>(std::ceil(T / (0.4 * cs.geom.CellSize(0) / umax)));
            const Real dt = T / nsteps;
            for (int step = 0; step < nsteps; ++step)
            {
                if (step == nsteps / 2)
                {
                    // grow the snug domain: new boxes appear around the old ones
                    const BoxArray new_ba = snugLayout(cs.geom, 16, keep_after[lay]);
                    const DistributionMapping new_dm(new_ba);
                    tr.regrid(cs.geom, new_ba, new_dm);
                    vel = makeVel(new_ba, new_dm);
                    Print() << "  regrid: " << cs.ba.size() << " -> " << new_ba.size() << " boxes\n";
                }
                sspRK3Step(tr, vel, uinf, dt);
            }

            // exact: the center moves along the translation + rotation
            // dx/dt = U - Om*y, dy/dt = Om*x, i.e. the rotation of (x + U/Om...) about the fixed point
            const Real xs = -U[1] / Om, ys = U[0] / Om;    // fixed point of the planar velocity field
            const Real ang = Om * T;
            GpuArray<Real, 3> c1;
            c1[0] = xs + std::cos(ang) * (c0[0] - xs) - std::sin(ang) * (c0[1] - ys);
            c1[1] = ys + std::sin(ang) * (c0[0] - xs) + std::cos(ang) * (c0[1] - ys);
            c1[2] = c0[2] + U[2] * T;
            const Real e = psiError(tr.getPsi(), cs.geom, c1, R, 3.0 * cs.geom.CellSize(0));
            const Real vol1 = volumeInside(tr.getPsi(), cs.geom);
            Print() << "  " << (scheme_i == 0 ? "WENO5" : "HOUC5") << "  steps = " << nsteps
                    << "  max psi error near interface = " << e
                    << "  (dx = " << cs.geom.CellSize(0) << ")"
                    << "  rel. volume change = " << (vol1 - vol0) / vol0 << "\n";
        }

        // ---------------- 4. the solver path: active case of InitialSDF.H ----------------
        if (guard_test.empty())
        {
            Print() << "\n[4] initializeSDF() with the active case (N_INTERFACE = " << N_INTERFACE << ")\n";
            Box domain(IntVect(0), IntVect(63));
            RealBox rb({AMREX_D_DECL(-2.0, -2.0, -2.0)}, {AMREX_D_DECL(2.0, 2.0, 2.0)});
            Array<int, AMREX_SPACEDIM> per{AMREX_D_DECL(0, 0, 0)};
            Geometry g4(domain, rb, CoordSys::cartesian, per);
            BoxArray b4(domain); b4.maxSize(16);
            DistributionMapping d4(b4);
            InterfaceTracker tr(g4, b4, d4, AdvectionScheme::WENO5, 0.4);
            tr.initializeSDF();
            // torus core of the default ring: R = 1, tube 0.4 -> psi(R, 0, 0) = 0.4 (clipped), psi at the
            // tube surface (1.4, 0, 0) = 0, at the origin = 0.4 - 1 = -0.6 -> clipped to -0.4
            const Real vmax = tr.getPsi().max(0), vmin = tr.getPsi().min(0);
            Print() << "  psi range [" << vmin << ", " << vmax << "] (expect [-0.4, ~0.4]), band check passed\n";
            Array<MultiFab, AMREX_SPACEDIM> vel;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                vel[d].define(convert(b4, IntVect::TheDimensionVector(d)), d4, 1, 0);
            }
            setVelocity(vel, g4, {0.0, 0.0, 0.5}, 1.0);
            const Real dt = 0.01;
            Print() << "  computeCFL: " << tr.computeCFL(vel, uinf, dt)
                    << "  (expect (|u| + |v| + |w|) dt/dx at the corner cell = (1.96875 + 1.96875 + 0.5) * 0.16 = 0.71)\n";
        }
    }
    amrex::Finalize();
}
