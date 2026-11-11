#include <IFHERK.H>
#include <LgfNseOmp.H>   // LGF_NSE_OMP_MFITER

namespace {
    // largest half-width considered when n_IF is chosen automatically
    constexpr int NMAX_AUTO = 128;
}

IFHERK::IFHERK (const RKButcher& tableau_in, const amrex::Geometry& geom_in, amrex::Real invRe_in, amrex::Real dt_in, int n_IF_in, amrex::Real IF_eps_in)
    : IF_eps(IF_eps_in), n_IF(n_IF_in), geom(geom_in), dt(dt_in), invRe(invRe_in)   // declaration order
{
    assertIsotropic();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dt > 0.0, "dt must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(invRe >= 0.0, "invRe must be non-negative");

    setupRKCoeffs(tableau_in);

    if (invRe == 0.0)
    {
        // inviscid: alpha = 0 for every sub-step, so every H is exactly the identity. Skip the
        // tables and the sweeps instead of convolving with the kernel (1, 0, 0, ...)
        for (int i = 1; i <= rk_stages; ++i) { rk_if_idx(i) = -1; }
        rk_unique_gaps.clear();
        n_IF = 0;
    }
    else if (n_IF <= 0)
    {
        // automatic stencil half-width: smallest n meeting IF_eps for every sub-step
        n_IF = 1;
        for (auto g : rk_unique_gaps)
        {
            const auto h = besselKernel(alphaOf(g), NMAX_AUTO);
            int n = 1;
            while (n < NMAX_AUTO && kernelTail(h, n) > IF_eps) { ++n; }
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(kernelTail(h, n) <= IF_eps,
                "IFHERK: no n_IF <= 128 meets IF_eps; alpha is too large");
            n_IF = std::max(n_IF, n);
        }
    }

    precomputeIFs();
}

void IFHERK::setTimeStep (amrex::Real dt_new)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dt_new > 0.0, "dt must be positive");
    dt = dt_new;
    precomputeIFs();   // aborts if the fixed n_IF no longer meets IF_eps
}

void IFHERK::setGeometry (const amrex::Geometry& geom_new)
{
    // the stored geometry always follows the caller (applyIF takes its periodicity from it);
    // the tables only depend on dx
    const amrex::Real dx_old = geom.CellSize(0);
    geom = geom_new;
    assertIsotropic();
    if (std::abs(geom.CellSize(0) - dx_old) > 1.0e-12 * dx_old)
    {
        precomputeIFs();   // aborts if the fixed n_IF no longer meets IF_eps
    }
}

void IFHERK::assertIsotropic () const
{
    // IMPORTANT ASSUMPTION: dx == dy == dz
    for (int d = 1; d < AMREX_SPACEDIM; ++d) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(geom.CellSize(d) - geom.CellSize(0)) < 1.0e-12 * geom.CellSize(0),
            "IFHERK assumes dx == dy == dz");
    }
}

amrex::Real IFHERK::alphaOf (amrex::Real gap) const
{
    // the one place alpha is formed, always as gap * dt * invRe / dx^2 in this order.
    // Multiplying by (1/dx)^2 instead changes the last bits of alpha and with them every
    // table entry, which breaks a bit-for-bit comparison against earlier runs
    const amrex::Real dx = geom.CellSize(0);
    return gap * dt * invRe / (dx * dx);
}

// ---------------------------------------------------------------------------
// shifted tableau
// ---------------------------------------------------------------------------

void IFHERK::setupRKCoeffs (const RKButcher& rkbt)
{
    BL_PROFILE("<Setup> IFHERK::setupRKCoeffs()");

    rk_stages = rkbt.n_stages;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(rk_stages >= 3 && rk_stages <= RKButcher::MAX_STAGES,
        "incompatible rk_stages and MAX_STAGES!");

    // check if tableau is valid
    const amrex::Real eps = 1.0e-14;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(rkbt.c[0]) < eps,
        "c_1 must be 0 (explicit first stage)");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(rkbt.c[rk_stages-1] - 1.0) < eps,
        "c_s must be 1: required for second-order constraints, and what makes "
        "the final integrating factor the identity (footnote 16)");

    amrex::Real bsum = 0.0;
    for (int j = 0; j < rk_stages; ++j) { bsum += rkbt.b[j]; }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(bsum - 1.0) < eps, "sum(b) != 1");

    for (int i = 0; i < rk_stages; ++i)
    {
        amrex::Real rowsum = 0.0;
        for (int j = 0; j < rk_stages; ++j)
        {
            if (j >= i) { AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(rkbt.A[i][j]) < eps,
                "A must be strictly lower triangular (explicit scheme)"); }
            rowsum += rkbt.A[i][j];
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(rowsum - rkbt.c[i]) < eps,
            "row sums of A must equal c (consistency)");
    }

    for (int i = 1; i <= RKButcher::MAX_STAGES; ++i)
    {
        rk_ct(i) = 0.0;  rk_gap(i) = 0.0;  rk_if_idx(i) = -1;
        for (int j = 1; j <= RKButcher::MAX_STAGES; ++j) { rk_at(i,j) = 0.0; }
    }

    // shifted coefficients
    for (int i = 1; i < rk_stages; ++i)
    {
        rk_ct(i) = rkbt.c[i];
        for (int j = 1; j <= rk_stages; ++j) { rk_at(i,j) = rkbt.A[i][j-1]; }
    }
    rk_ct(rk_stages) = 1.0;
    for (int j = 1; j <= rk_stages; ++j) { rk_at(rk_stages,j) = rkbt.b[j-1]; }

    // c~_i - c~_{i-1}
    amrex::Real prev = 0.0;
    for (int i = 1; i <= rk_stages; ++i) { rk_gap(i) = rk_ct(i) - prev; prev = rk_ct(i); }

    // validate the shifted tableau
    for (int i = 1; i <= rk_stages; ++i)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(rk_at(i,i)) > eps,
            "a~_{i,i} == 0: Eq. (30) divides by it when forming w^{i,i}");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(rk_gap(i) > -eps,
            "negative sub-step width: H^i would be E(-alpha), which amplifies the "
            "grid-scale mode. Tableau is IF-incompatible -- this is what rules out "
            "SSP-RK3, whose c = [0, 1, 1/2]");
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(rk_gap(rk_stages) < eps,
        "final sub-step width should vanish when c_s == 1");

    // unique sub-steps, and the stage -> table map
    rk_unique_gaps.clear();
    for (int i = 1; i <= rk_stages; ++i)
    {
        if (rk_gap(i) < 1.0e-14) { rk_if_idx(i) = -1; continue; }   // H^i = I

        int found = -1;
        for (int m = 0; m < static_cast<int>(rk_unique_gaps.size()); ++m)
        {
            if (std::abs(rk_gap(i) - rk_unique_gaps[m]) < 1.0e-14) { found = m; break; }
        }
        if (found < 0)
        {
            rk_unique_gaps.push_back(rk_gap(i));
            found = static_cast<int>(rk_unique_gaps.size()) - 1;
        }
        rk_if_idx(i) = found;
    }
}

// ---------------------------------------------------------------------------
// 1D heat-kernel tables  h_alpha(n) = exp(-2 alpha) I_n(2 alpha)
// ---------------------------------------------------------------------------

amrex::Vector<amrex::Real> IFHERK::besselKernel (amrex::Real alpha, int nmax)
{
    // ascending series: I_n(z) = sum_k (z/2)^{n+2k} / (k! (n+k)!)
    amrex::Vector<amrex::Real> h(nmax + 1);
    const amrex::Real z = 2.0 * alpha;
    for (int n = 0; n <= nmax; ++n)
    {
        amrex::Real term = 1.0;
        for (int p = 1; p <= n; ++p) { term *= (0.5 * z) / amrex::Real(p); }
        amrex::Real sum = term;
        int k = 0;
        for (; k < 200; ++k)
        {
            term *= (0.25 * z * z) / (amrex::Real(k + 1) * amrex::Real(n + k + 1));
            sum  += term;
            if (term < 1.0e-300) { break; }
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(k < 200 || term < 1.0e-16 * sum,
            "IFHERK: Bessel series did not converge; alpha is too large");
        h[n] = std::exp(-z) * sum;
    }
    return h;
}

amrex::Real IFHERK::kernelTail (const amrex::Vector<amrex::Real>& h, int n)
{
    // 2 * sum_{m > n} h(m), summed directly: 1 - mass cancels below ~1e-16
    amrex::Real tail = 0.0;
    for (int m = static_cast<int>(h.size()) - 1; m > n; --m) { tail += 2.0 * h[m]; }
    return tail;
}

void IFHERK::precomputeIFs ()
{
    BL_PROFILE("<Setup> IFHERK::precomputeIFs()");

    if_table.resize(rk_unique_gaps.size());
    if_tail .resize(rk_unique_gaps.size());

    for (int m = 0; m < static_cast<int>(rk_unique_gaps.size()); ++m)
    {
        const amrex::Real alpha = alphaOf(rk_unique_gaps[m]);

        // evaluate a few extra terms so the truncation tail can be measured
        const auto h_ext = besselKernel(alpha, n_IF + 32);
        if_tail[m] = kernelTail(h_ext, n_IF);

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(if_tail[m] <= IF_eps,
            "IF kernel tail exceeds IF_eps: n_IF is too small for this dt. "
            "Construct IFHERK with the largest dt (or a larger n_IF).");

        // blocking copy: h_ext goes out of scope at the end of this iteration, and an
        // asynchronous copy is not guaranteed to have read it by then
        if_table[m].resize(n_IF + 1);
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                         h_ext.begin(), h_ext.begin() + (n_IF + 1),
                         if_table[m].begin());
    }
}

amrex::Real IFHERK::maxKernelTail () const
{
    amrex::Real t = 0.0;
    for (auto v : if_tail) { t = std::max(t, v); }
    return t;
}

void IFHERK::printSummary () const
{
    amrex::Print() << "IFHERK: " << rk_stages << " stages, dt = " << dt
                   << ", invRe = " << invRe << ", dx = " << geom.CellSize(0) << ", n_IF = " << n_IF << "\n";
    for (int i = 1; i <= rk_stages; ++i)
    {
        amrex::Print() << "  stage " << i << ": c~ = " << rk_ct(i)
                       << ", gap = " << rk_gap(i) << ", a~ =";
        for (int j = 1; j <= rk_stages; ++j) { amrex::Print() << " " << rk_at(i,j); }
        amrex::Print() << ", IF table = " << rk_if_idx(i) << "\n";
    }
    for (int m = 0; m < static_cast<int>(rk_unique_gaps.size()); ++m)
    {
        amrex::Print() << "  IF table " << m << ": gap = " << rk_unique_gaps[m]
                       << ", alpha = " << alphaOf(rk_unique_gaps[m])
                       << ", truncation tail = " << if_tail[m] << "\n";
    }
}

// ---------------------------------------------------------------------------
// applyIF: d separable 1D sweeps, ping-ponging through the caller's buffer
// ---------------------------------------------------------------------------

void IFHERK::applyIF (amrex::MultiFab& fld, amrex::MultiFab& buff, int if_idx) const
{
    BL_PROFILE("<Compute> IFHERK::applyIF()");

    const amrex::Periodicity period = geom.periodicity();

    // M^xsoln: cells outside the grid are zero. FillBoundary never reaches the ghost cells
    // that no neighbouring box covers, and fld and buff trade storage on every sweep, so
    // without setBndry those cells would carry whatever field last used the buffer
    if (if_idx < 0)                                   // H = I
    {
        fld.setBndry(0.0);
        fld.FillBoundary(period);
        return;
    }

    // always on: a violation reads or writes out of bounds silently in a release GPU build
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(if_idx < static_cast<int>(if_table.size()),
        "IFHERK::applyIF: no such IF table");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(fld.nGrowVect().allGE(n_IF) && buff.nGrowVect().allGE(n_IF),
        "IFHERK::applyIF: field and buffer need nGhostRequired() ghost cells");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(fld.ixType() == buff.ixType() && fld.size() == buff.size(),
        "IFHERK::applyIF: field and buffer live on different grids");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(fld.nComp() == 1 && buff.nComp() == 1,
        "IFHERK::applyIF: the sweeps only act on component 0");
    AMREX_ASSERT(fld.boxArray() == buff.boxArray());
    AMREX_ASSERT(fld.DistributionMap() == buff.DistributionMap());

    const amrex::Real* tab = if_table[if_idx].dataPtr();
    const int          n   = n_IF;

    for (int sd = 0; sd < AMREX_SPACEDIM; ++sd)
    {
        fld.setBndry(0.0);
        fld.FillBoundary(period);

        const int di = (sd == 0), dj = (sd == 1), dk = (sd == 2);

        LGF_NSE_OMP_MFITER
        for (amrex::MFIter mfi(fld, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox();
            auto const& src = fld.const_array(mfi);
            auto const& dst = buff.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real s = tab[0] * src(i,j,k);
                for (int m = 1; m <= n; ++m)
                {
                    s += tab[m] * ( src(i + m*di, j + m*dj, k + m*dk)
                                  + src(i - m*di, j - m*dj, k - m*dk) );
                }
                dst(i,j,k) = s;
            });
        }
        std::swap(fld, buff);                         // result back in fld
    }
    fld.setBndry(0.0);
    fld.FillBoundary(period);
}