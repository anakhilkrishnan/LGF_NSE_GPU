#!/usr/bin/env python3
"""
Error analysis for the radially symmetric decaying vortex.

Compares LGF-NSE plotfile output against the exact solution of
Shukla & Giri, JCP 276 (2014), Sec. 5.3.1 / Eq. (61).

    sigma^2(t) = r0^2 + 4 nu t
    omega(r,t) = omega_0 (r0^2/sigma^2)^2 (1 - r^2/sigma^2) exp(-r^2/sigma^2)
    psi(r,t)   = (omega_0 r0^4 / (4 sigma^2)) exp(-r^2/sigma^2)
    u          =  d(psi)/dy = -2 A (y-yc)/sigma^2 exp(-r^2/sigma^2)
    v          = -d(psi)/dx = +2 A (x-xc)/sigma^2 exp(-r^2/sigma^2)
    with A = omega_0 r0^4 / (4 sigma^2),  omega_0 = 2 sqrt(2e) / r0

Reads the *_cc (cell-centred) plotfiles only. See NOTE ON NODAL DATA below.

Usage
-----
    python vortex_error.py --dir ./Results
    python vortex_error.py --dir ./Results --support-only --csv err.csv
    python vortex_error.py --dir ./Results --nu 1e-3 --r0 0.1 --plot err.png

Requires: yt >= 4.0, numpy.  (matplotlib only for --plot)
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

try:
    import yt
except ImportError:
    sys.exit("yt not found.  pip install yt   (or: conda install -c conda-forge yt)")


# ----------------------------------------------------------------------
# exact solution
# ----------------------------------------------------------------------

def exact_fields(x, y, t, r0, nu, xc, yc):
    """Return (u, v, omega) of the decaying vortex at positions (x,y), time t."""
    omega_0 = 2.0 * np.sqrt(2.0 * np.e) / r0
    sigma2 = r0 * r0 + 4.0 * nu * t

    dx = x - xc
    dy = y - yc
    r2 = dx * dx + dy * dy
    g = np.exp(-r2 / sigma2)

    omega = omega_0 * (r0 * r0 / sigma2) ** 2 * (1.0 - r2 / sigma2) * g

    A = omega_0 * r0 ** 4 / (4.0 * sigma2)
    u = -2.0 * A * dy / sigma2 * g
    v = 2.0 * A * dx / sigma2 * g

    return u, v, omega


# ----------------------------------------------------------------------
# norms
# ----------------------------------------------------------------------

def norms(err, ref, dV):
    """Volume-weighted L1 / L2 / Linf, absolute and relative to ||ref||."""
    tot = dV.sum()
    l1 = (np.abs(err) * dV).sum() / tot
    l2 = np.sqrt((err ** 2 * dV).sum() / tot)
    li = np.abs(err).max() if err.size else np.nan

    r1 = (np.abs(ref) * dV).sum() / tot
    r2 = np.sqrt((ref ** 2 * dV).sum() / tot)
    ri = np.abs(ref).max() if ref.size else np.nan

    rel = lambda a, b: a / b if b > 0 else np.nan
    return dict(L1=l1, L2=l2, Linf=li,
                rL1=rel(l1, r1), rL2=rel(l2, r2), rLinf=rel(li, ri))


# ----------------------------------------------------------------------
# plotfile handling
# ----------------------------------------------------------------------

def find_field(ds, name):
    """Locate a field by short name regardless of which ftype yt assigned."""
    for ftype, fname in ds.field_list:
        if fname == name:
            return (ftype, fname)
    return None


def step_from_path(path):
    m = re.search(r"plt(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def analyse(path, args):
    ds = yt.load(path)
    t = float(ds.current_time)
    ad = ds.all_data()

    x = np.asarray(ad["index", "x"].to_ndarray(), dtype=np.float64)
    y = np.asarray(ad["index", "y"].to_ndarray(), dtype=np.float64)
    dV = np.asarray(ad["index", "cell_volume"].to_ndarray(), dtype=np.float64)

    def get(name):
        f = find_field(ds, name)
        if f is None:
            return None
        return np.asarray(ad[f].to_ndarray(), dtype=np.float64)

    u_num = get("x_velocity")
    v_num = get("y_velocity")
    w_num = get("tag_vort")          # cell-centred vorticity (one tagging pass stale)
    tag = get("active_box_tag")

    if u_num is None or v_num is None:
        raise RuntimeError(f"{path}: x_velocity/y_velocity not found. "
                           f"Fields present: {[f[1] for f in ds.field_list]}")

    mask = np.ones_like(x, dtype=bool)
    if args.support_only:
        if tag is None:
            print(f"  warning: active_box_tag absent in {path}, ignoring --support-only")
        else:
            mask = tag > 0.5
    if args.radius is not None:
        r = np.hypot(x - args.center[0], y - args.center[1])
        mask &= r <= args.radius

    x, y, dV = x[mask], y[mask], dV[mask]
    u_num, v_num = u_num[mask], v_num[mask]
    if w_num is not None:
        w_num = w_num[mask]

    u_ex, v_ex, w_ex = exact_fields(x, y, t, args.r0, args.nu,
                                    args.center[0], args.center[1])

    out = {"step": step_from_path(path), "time": t, "ncell": x.size}
    out.update({f"u_{k}": v for k, v in norms(u_num - u_ex, u_ex, dV).items()})
    out.update({f"v_{k}": v for k, v in norms(v_num - v_ex, v_ex, dV).items()})

    # combined velocity magnitude error
    speed_err = np.hypot(u_num - u_ex, v_num - v_ex)
    speed_ref = np.hypot(u_ex, v_ex)
    out.update({f"U_{k}": v for k, v in norms(speed_err, speed_ref, dV).items()})

    # ---- amplitude decay check, from velocity (never normalized) ----
    # |u|_exact peaks at r = sigma/sqrt(2) and decays as sigma^-3, so this is a
    # direct test of the diffusion coefficient with no free parameters.
    # Both maxima are taken on the same grid points, so they are comparable.
    out["Umax_num"] = np.hypot(u_num, v_num).max()
    out["Umax_ex"] = speed_ref.max()

    sigma2 = args.r0 ** 2 + 4.0 * args.nu * t
    out["sigma_over_r0"] = np.sqrt(sigma2) / args.r0

    # ---- vorticity ----
    out["wmax_num"] = np.abs(w_num).max() if w_num is not None else np.nan
    out["wmax_ex"] = np.abs(w_ex).max()

    if w_num is not None and not args.no_vorticity:
        wmax = out["wmax_num"]
        normalized = args.vort_normalized
        if normalized is None:  # auto-detect
            normalized = abs(wmax - 1.0) < 1e-6
        if normalized:
            # tag_vort is stored normalized by its own max, so the amplitude is
            # unrecoverable. Compare SHAPE only: normalize the exact field the
            # same way. This still tests radial spreading (sigma growth).
            ref = w_ex / out["wmax_ex"] if out["wmax_ex"] > 0 else w_ex
            num = w_num / wmax if wmax > 0 else w_num
            out["w_normalized"] = 1
        else:
            ref, num = w_ex, w_num
            out["w_normalized"] = 0
        out.update({f"w_{k}": v for k, v in norms(num - ref, ref, dV).items()})

    return out


# ----------------------------------------------------------------------

def load_scalar(path, fieldname, args):
    """Load one scalar plus cell-centre coordinates. Handles nodal plotfiles."""
    ds = yt.load(path)
    t = float(ds.current_time)
    ad = ds.all_data()

    f = find_field(ds, fieldname)
    if f is None:
        raise RuntimeError(f"{path}: field {fieldname!r} not present. "
                           f"Have: {[q[1] for q in ds.field_list]}")

    x = np.asarray(ad["index", "x"].to_ndarray(), dtype=np.float64)
    y = np.asarray(ad["index", "y"].to_ndarray(), dtype=np.float64)
    v = np.asarray(ad[f].to_ndarray(), dtype=np.float64)

    dx = float(ds.domain_width[0] / ds.domain_dimensions[0])
    dy = float(ds.domain_width[1] / ds.domain_dimensions[1])

    # A nodal MultiFab written through WriteSingleLevelPlotfile is described in
    # the header as if it were cell-centred, so an N+1-point nodal box is read
    # back as N+1 cells. yt therefore reports positions half a cell high, and
    # the domain one cell too wide. Detect and undo.
    shift = args.node_shift
    if shift is None:
        shift = ds.domain_dimensions[0] % 2 == 1  # 1025 vs 1024 etc.
    if shift:
        x = x - 0.5 * dx
        y = y - 0.5 * dy
        print(f"  [nd] nodal offset removed: coords shifted by -dx/2 "
              f"({0.5 * dx:.3e}); domain_dimensions={tuple(ds.domain_dimensions)}")

    return x, y, v, t, dx, dy


def to_image(x, y, v, dx, dy):
    """Scatter cell values onto a regular array by index arithmetic (no interp).

    Cells absent from the snug domain stay NaN so they render blank.
    """
    i = np.rint(x / dx - 0.5).astype(np.int64)
    j = np.rint(y / dy - 0.5).astype(np.int64)
    i0, j0 = i.min(), j.min()
    img = np.full((j.max() - j0 + 1, i.max() - i0 + 1), np.nan)
    img[j - j0, i - i0] = v
    extent = (i0 * dx, (i.max() + 1) * dx, j0 * dy, (j.max() + 1) * dy)
    return img, extent


def query_plot(path, args, outfile):
    """Shukla & Giri Fig. 13 style: spatial distribution of the error field."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y, w_num, t, dx, dy = load_scalar(path, args.query_field, args)
    _, _, w_ex = exact_fields(x, y, t, args.r0, args.nu,
                              args.center[0], args.center[1])

    if args.query_normalize:
        n_num = np.abs(w_num).max()
        n_ex = np.abs(w_ex).max()
        if n_num > 0:
            w_num = w_num / n_num
        if n_ex > 0:
            w_ex = w_ex / n_ex
        unit = " (normalized)"
    else:
        unit = ""

    err = w_num - w_ex
    r = np.hypot(x - args.center[0], y - args.center[1])

    # anisotropy: spread of the numerical field within thin radial shells.
    # a perfectly isotropic scheme collapses each shell onto a single value.
    nb = 80
    edges = np.linspace(0, r.max(), nb + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, nb - 1)
    aniso = np.zeros(nb)
    rmid = 0.5 * (edges[:-1] + edges[1:])
    for b in range(nb):
        m = idx == b
        if m.sum() > 1:
            aniso[b] = w_num[m].max() - w_num[m].min()
    aniso_max = aniso.max()

    im_num, ext = to_image(x, y, w_num, dx, dy)
    im_ex, _ = to_image(x, y, w_ex, dx, dy)
    im_err, _ = to_image(x, y, np.abs(err), dx, dy)

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 10))

    lim = np.nanmax(np.abs(im_num))
    for a, im, ttl in ((ax[0, 0], im_num, f"numerical {args.query_field}{unit}"),
                       (ax[0, 1], im_ex, f"exact{unit}")):
        h = a.imshow(im, origin="lower", extent=ext, cmap="RdBu_r",
                     vmin=-lim, vmax=lim)
        a.set_title(ttl); a.set_xlabel("x"); a.set_ylabel("y")
        a.set_aspect("equal")
        fig.colorbar(h, ax=a, fraction=0.046)

    h = ax[1, 0].imshow(im_err, origin="lower", extent=ext, cmap="viridis")
    ax[1, 0].set_title(f"absolute error   max={np.nanmax(im_err):.3e}")
    ax[1, 0].set_xlabel("x"); ax[1, 0].set_ylabel("y")
    ax[1, 0].set_aspect("equal")
    fig.colorbar(h, ax=ax[1, 0], fraction=0.046)

    # radial collapse: vertical spread at fixed r IS the grid-orientation error
    ax[1, 1].plot(r, w_num, ".", ms=1.2, alpha=0.25, label="numerical (all cells)")
    order = np.argsort(r)
    ax[1, 1].plot(r[order], w_ex[order], "k-", lw=1.4, label="exact")
    ax[1, 1].set_xlabel("r"); ax[1, 1].set_ylabel(f"{args.query_field}{unit}")
    ax[1, 1].set_title(f"radial collapse   max shell spread={aniso_max:.3e}")
    ax[1, 1].legend(markerscale=8); ax[1, 1].grid(alpha=.3)
    if args.query_rmax:
        ax[1, 1].set_xlim(0, args.query_rmax)

    fig.suptitle(f"{os.path.basename(path)}    t = {t:.5f}    "
                 f"sigma/r0 = {np.sqrt(args.r0**2 + 4*args.nu*t)/args.r0:.4f}")
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    print(f"\nwrote {outfile}")
    print(f"  cells={x.size}  t={t:.6f}  "
          f"max|err|={np.abs(err).max():.4e}  "
          f"rms err={np.sqrt((err**2).mean()):.4e}  "
          f"max shell spread={aniso_max:.4e}")


# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="./Results", help="directory holding plt*_cc files")
    p.add_argument("--pattern", default="plt*_cc", help="glob pattern (default: plt*_cc)")
    p.add_argument("--r0", type=float, default=0.1, help="vortex core radius")
    p.add_argument("--nu", type=float, default=1e-3, help="viscosity (== invRe in inputs)")
    p.add_argument("--center", type=float, nargs=2, default=[0.0, 0.0],
                   metavar=("XC", "YC"), help="vortex centre")
    p.add_argument("--support-only", action="store_true",
                   help="restrict norms to cells where active_box_tag > 0.5")
    p.add_argument("--radius", type=float, default=None,
                   help="further restrict to r <= RADIUS about the centre")
    p.add_argument("--no-vorticity", action="store_true",
                   help="skip the tag_vort comparison")
    p.add_argument("--vort-normalized", dest="vort_normalized",
                   action="store_const", const=True, default=None,
                   help="tag_vort is stored max-normalized; compare shape only "
                        "(auto-detected when peak |tag_vort| == 1)")
    p.add_argument("--vort-raw", dest="vort_normalized",
                   action="store_const", const=False,
                   help="force tag_vort to be treated as an unnormalized field")
    p.add_argument("--csv", default=None, help="write results to CSV")
    p.add_argument("--plot", default=None, help="write an error-vs-time PNG")

    q = p.add_argument_group("single-step error field (--query)")
    q.add_argument("--query", type=int, default=None, metavar="STEP",
                   help="plot the error field for this step and exit")
    q.add_argument("--query-source", choices=["cc", "nd"], default="cc",
                   help="cc = normalized tag_vort (default); nd = true vorticity")
    q.add_argument("--query-field", default=None,
                   help="override the field name (default: tag_vort / z_vorticity)")
    q.add_argument("--query-out", default=None, help="output PNG path")
    q.add_argument("--query-rmax", type=float, default=None,
                   help="x-limit on the radial-collapse panel")
    q.add_argument("--query-raw", dest="query_normalize", action="store_false",
                   default=True, help="do not max-normalize before comparing")
    q.add_argument("--node-shift", dest="node_shift", action="store_const",
                   const=True, default=None,
                   help="force removal of the nodal half-cell offset "
                        "(auto-detected from odd domain_dimensions)")
    q.add_argument("--no-node-shift", dest="node_shift", action="store_const",
                   const=False, help="force-disable the nodal offset correction")
    p.add_argument("--quiet-yt", action="store_true", default=True)
    args = p.parse_args()

    if args.quiet_yt:
        yt.set_log_level(50)

    if args.query is not None:
        suffix = "_cc" if args.query_source == "cc" else "_nd"
        if args.query_field is None:
            args.query_field = "tag_vort" if args.query_source == "cc" else "z_vorticity"
        path = os.path.join(args.dir, f"plt{args.query:05d}{suffix}")
        if not os.path.exists(path):
            cand = sorted(glob.glob(os.path.join(args.dir, f"plt*{suffix}")))
            sys.exit(f"{path} not found.\navailable: "
                     + ", ".join(os.path.basename(c) for c in cand[:20]))
        out = args.query_out or f"errfield_{args.query:05d}_{args.query_field}.png"
        query_plot(path, args, out)
        return

    files = sorted(glob.glob(os.path.join(args.dir, args.pattern)),
                   key=step_from_path)
    if not files:
        sys.exit(f"no plotfiles matching {args.pattern!r} under {args.dir!r}")

    print(f"# vortex: r0={args.r0}  nu={args.nu}  centre=({args.center[0]}, {args.center[1]})")
    print(f"# {len(files)} plotfiles from {args.dir}")
    if args.support_only:
        print("# norms restricted to active_box_tag > 0.5")
    print()
    hdr = (f"{'step':>6} {'time':>10} {'ncell':>9} "
           f"{'|U|_L2':>11} {'rel_L2':>10} {'|U|_Linf':>11} {'rel_Linf':>10} "
           f"{'w_relL2':>10} {'sig/r0':>7}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for f in files:
        try:
            r = analyse(f, args)
        except Exception as exc:
            print(f"  !! {os.path.basename(f)}: {exc}")
            continue
        rows.append(r)
        print(f"{r['step']:6d} {r['time']:10.5f} {r['ncell']:9d} "
              f"{r['U_L2']:11.4e} {r['U_rL2']:10.3e} "
              f"{r['U_Linf']:11.4e} {r['U_rLinf']:10.3e} "
              f"{r.get('w_rL2', float('nan')):10.3e} {r['sigma_over_r0']:7.4f}")

    if not rows:
        sys.exit("nothing analysed")

    if args.csv:
        import csv
        keys = list(rows[0].keys())
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nwrote {args.csv}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = [r["time"] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].semilogy(t, [r["U_rL2"] for r in rows], "o-", label="rel L2")
        ax[0].semilogy(t, [r["U_rLinf"] for r in rows], "s-", label="rel Linf")
        ax[0].set_xlabel("time"); ax[0].set_ylabel("velocity error")
        ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(t, [r["Umax_num"] for r in rows], "o-", label="numerical")
        ax[1].plot(t, [r["Umax_ex"] for r in rows], "k--", label="exact")
        ax[1].set_xlabel("time")
        ax[1].set_ylabel(r"peak $|\mathbf{u}|$   (decays as $\sigma^{-3}$)")
        ax[1].legend(); ax[1].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
