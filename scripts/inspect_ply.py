"""Measure a 3DGS scene. Nine probes, each answering a question that
affects a later design decision.

Run from the repo root:

    python scripts/probe_ply.py data/raw/garden.ply

Prints a report and writes out/probe_<name>.png. Nothing here modifies the
scene; it only measures it.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bozkir.ply import load_ply, SH_C0  # noqa: E402

AXES = ("x", "y", "z")


def core_mask(s, pct=90.0):
    """Splats inside the `pct` percentile of distance from the median centre.

    Uses the median rather than the mean because floaters drag the mean.
    """
    centre = np.median(s.xyz, axis=0)
    r = np.linalg.norm(s.xyz - centre, axis=1)
    return r <= np.percentile(r, pct), centre, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    s = load_ply(args.path)
    n = len(s)
    print(f"\n{args.path.name}: {n:,} splats, SH degree {s.sh_degree}")

    core, centre, radius = core_mask(s, 90.0)
    xyz_core = s.xyz[core]

    # ---- 1. Which way is up ----------------------------------------------
    # The dense part of a captured outdoor scene is dominated by ground.
    # PCA on it gives two long axes (the ground) and one short one (the
    # normal). The short axis is "up", whatever the file calls it.
    d = xyz_core - xyz_core.mean(axis=0)
    cov = (d.T @ d) / len(d)
    evals, evecs = np.linalg.eigh(cov)          # ascending
    normal = evecs[:, 0]
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal                        # fix arbitrary sign
    align = np.abs(normal)
    up_axis = int(np.argmax(align))
    planarity = float(evals[0] / evals[2])
    tilt = float(np.degrees(np.arccos(np.clip(align[up_axis], 0, 1))))

    # Refit using only the lowest band along that axis: Garden has a table
    # and a plant standing on the ground, and including them tilts the fit.
    h = xyz_core[:, up_axis]
    low = h <= np.percentile(h, 30)
    if low.sum() > 100:
        d2 = xyz_core[low] - xyz_core[low].mean(axis=0)
        ev2, evec2 = np.linalg.eigh((d2.T @ d2) / len(d2))
        n2 = evec2[:, 0]
        if n2[np.argmax(np.abs(n2))] < 0:
            n2 = -n2
        tilt2 = float(np.degrees(np.arccos(np.clip(np.abs(n2)[up_axis], 0, 1))))
        planarity2 = float(ev2[0] / ev2[2])
    else:
        n2, tilt2, planarity2 = normal, tilt, planarity

    print("\n1. Up axis")
    print(f"   plane normal      ({normal[0]:+.3f}, {normal[1]:+.3f}, {normal[2]:+.3f})")
    print(f"   closest to        {AXES[up_axis]}  (off by {tilt:.1f} deg)")
    print(f"   planarity ratio   {planarity:.4f}"
          f"   ({'plane-like' if planarity < 0.05 else 'not strongly planar'})")
    print(f"   ground only (lowest 30%):")
    print(f"     normal          ({n2[0]:+.3f}, {n2[1]:+.3f}, {n2[2]:+.3f})")
    print(f"     off {AXES[up_axis]} by       {tilt2:.1f} deg"
          f"   (planarity {planarity2:.4f})")

    # ---- 2. How much of the bounding box is real -------------------------
    bb_min, bb_max = s.xyz.min(axis=0), s.xyz.max(axis=0)
    p_lo = np.percentile(s.xyz, 1, axis=0)
    p_hi = np.percentile(s.xyz, 99, axis=0)
    vol_full = float(np.prod(bb_max - bb_min))
    vol_p98 = float(np.prod(p_hi - p_lo))

    print("\n2. Extent")
    print(f"   min/max        {np.round(bb_max - bb_min, 1)}   volume {vol_full:,.0f}")
    print(f"   1-99 pct       {np.round(p_hi - p_lo, 1)}   volume {vol_p98:,.0f}"
          f"   ({vol_p98 / vol_full:.1%} of the box)")
    print(f"   2% of splats occupy {1 - vol_p98 / vol_full:.1%} of the volume"
          f"  -> use percentiles, not min/max")

    # ---- 3. How flat are the splats --------------------------------------
    # Ratio of longest to shortest axis. StopThePop clamps 1/scale at 1e3;
    # anything with 1/scale above that is affected by the clamp.
    sc_sorted = np.sort(s.scale, axis=1)
    aniso = sc_sorted[:, 2] / np.maximum(sc_sorted[:, 0], 1e-20)
    inv_min = 1.0 / np.maximum(sc_sorted[:, 0], 1e-20)
    clamped = float((inv_min > 1e3).mean())

    print("\n3. Anisotropy (long axis / short axis)")
    for p in (50, 90, 99, 100):
        print(f"   p{p:<4d} {np.percentile(aniso, p):>12,.1f}")
    print(f"   {clamped:.1%} of splats exceed StopThePop's 1/scale clamp of 1e3")

    # ---- 4. Screen footprint ---------------------------------------------
    # A splat is a flattened ellipsoid. Its two LARGEST axes form the face
    # you see; the smallest is thickness. Measuring only the longest axis
    # overstates how much screen it covers.
    focal = 1000.0
    dist = float(np.percentile(radius[core], 50))
    px_long = focal * sc_sorted[:, 2] / max(dist, 1e-6)
    px_short = focal * sc_sorted[:, 1] / max(dist, 1e-6)
    area_px = np.pi * px_long * px_short

    print(f"\n4. Screen footprint (focal {focal:.0f}px, distance {dist:.2f})")
    print(f"   {'':10s}{'long':>8s}{'short':>8s}{'area':>10s}")
    for p_ in (50, 90, 99):
        print(f"   p{p_:<9d}{np.percentile(px_long, p_):>8.2f}"
              f"{np.percentile(px_short, p_):>8.2f}"
              f"{np.percentile(area_px, p_):>10.2f}")
    print(f"   {(area_px < 1.0).mean():.1%} cover less than one pixel of area")
    print(f"   {(px_short < 1.0).mean():.1%} are thinner than a pixel"
          f"   -> most splats are slivers, not discs")

    # ---- 5. Where the visible mass is ------------------------------------
    # "Ink" = how much a splat can contribute: screen area times opacity.
    # Sorting by it shows how many splats actually matter.
    ink = area_px * s.opacity
    order = np.argsort(ink)[::-1]
    cum = np.cumsum(ink[order])
    cum /= cum[-1]

    print("\n5. Contribution concentration")
    for frac in (0.50, 0.90, 0.99):
        k = int(np.searchsorted(cum, frac)) + 1
        print(f"   {frac:.0%} of visible mass in top {k:,} splats ({k / n:.1%})")
    dead = float(((s.opacity < 0.05) | (area_px < 0.1)).mean())
    print(f"   {dead:.1%} are near-transparent or far sub-pixel -> free to cull")

    # ---- 6. Spacing vs size ----------------------------------------------
    # If splats are smaller than the gaps between them the surface has holes;
    # if much larger they overlap heavily. This ratio predicts how much
    # blending order matters, and later, how tile boundaries behave.
    try:
        from scipy.spatial import cKDTree
        sample = rng.choice(np.flatnonzero(core), size=min(20_000, core.sum()),
                            replace=False)
        tree = cKDTree(s.xyz[core])
        nn, _ = tree.query(s.xyz[sample], k=2)
        nn = nn[:, 1]
        med_nn = float(np.median(nn))
        med_sz = float(np.median(sc_sorted[sample, 2]))
        print("\n6. Spacing vs size (core)")
        print(f"   median nearest neighbour  {med_nn:.4f}")
        print(f"   median splat long axis    {med_sz:.4f}")
        print(f"   size / spacing            {med_sz / med_nn:.2f}"
              f"   ({'overlapping' if med_sz > med_nn else 'sparse'})")
    except ImportError:
        med_nn = None
        print("\n6. Spacing vs size: skipped (scipy not installed)")

    # ---- 7. Depth complexity ---------------------------------------------
    # How many splats stack along a typical view ray. This is the number
    # that decides whether blend order matters at all.
    grid = 128
    lo, hi = p_lo.copy(), p_hi.copy()
    flat_axes = [i for i in range(3) if i != up_axis]
    inside = np.all((s.xyz >= lo) & (s.xyz <= hi), axis=1)
    ij = np.floor((s.xyz[inside][:, flat_axes] - lo[flat_axes])
                  / (hi[flat_axes] - lo[flat_axes] + 1e-9) * (grid - 1)).astype(int)
    counts = np.bincount(ij[:, 0] * grid + ij[:, 1], minlength=grid * grid)
    occ = counts[counts > 0]

    print(f"\n7. Depth complexity ({grid}x{grid} columns along {AXES[up_axis]})")
    print(f"   occupied columns   {len(occ):,} / {grid * grid:,}"
          f"  ({len(occ) / grid ** 2:.1%})")
    for p in (50, 90, 99):
        print(f"   p{p:<4d} {np.percentile(occ, p):>10,.0f} splats deep")

    # ---- 8. Cost of dropping SH degree -----------------------------------
    # Measured over several view directions, in 0-255 units so the numbers
    # mean something perceptually.
    sub = rng.choice(n, size=min(50_000, n), replace=False)
    small = Splats_subset(s, sub)
    dirs = _hemisphere_dirs(24, up_axis)
    err = {1: [], 0: []}
    for v in dirs:
        ref = small.rgb(v)
        for deg in (1, 0):
            err[deg].append(np.abs(small.truncate_sh(deg).rgb(v) - ref))
    print("\n8. Cost of truncating SH (error vs degree 3, 0-255 scale)")
    for deg in (1, 0):
        e = np.concatenate(err[deg]) * 255.0
        print(f"   degree {deg}:  mean {e.mean():5.2f}   "
              f"p99 {np.percentile(e, 99):6.2f}   max {e.max():6.2f}")
    for deg in (3, 1, 0):
        k = (deg + 1) ** 2
        print(f"   degree {deg} storage: {k * 3:2d} floats/splat"
              f"  = {n * k * 3 * 4 / 1e6:6.1f} MB")

    # ---- 9. Is the DC-only shortcut safe ---------------------------------
    raw = SH_C0 * s.sh_dc + 0.5
    out_lo = float((raw < 0).mean())
    out_hi = float((raw > 1).mean())
    print("\n9. DC-only colour")
    print(f"   {out_lo:.1%} below 0, {out_hi:.1%} above 1"
          f"   -> higher-order terms carry these back in range")

    # ---- figure ----------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.suptitle(f"{args.path.name} - probes")

    keep = rng.choice(n, size=min(60_000, n), replace=False)
    a, b = flat_axes
    ax[0, 0].scatter(s.xyz[keep, a], s.xyz[keep, b], s=0.4,
                     c=s.rgb(np.float32([0, 0, 1]))[keep], alpha=0.5, linewidths=0)
    ax[0, 0].set_xlim(p_lo[a], p_hi[a]); ax[0, 0].set_ylim(p_lo[b], p_hi[b])
    ax[0, 0].set_aspect("equal")
    ax[0, 0].set_title(f"ground plane ({AXES[a]}, {AXES[b]}), 1-99 pct")

    ax[0, 1].scatter(s.xyz[keep, a], s.xyz[keep, up_axis], s=0.4,
                     c=s.rgb(np.float32([0, 0, 1]))[keep], alpha=0.5, linewidths=0)
    ax[0, 1].set_xlim(p_lo[a], p_hi[a]); ax[0, 1].set_ylim(p_lo[up_axis], p_hi[up_axis])
    ax[0, 1].set_aspect("equal")
    ax[0, 1].set_title(f"elevation ({AXES[a]}, {AXES[up_axis]} = up)")

    ax[0, 2].plot(np.arange(1, n + 1) / n, cum, color="#444")
    ax[0, 2].set_xscale("log")
    ax[0, 2].axhline(0.9, ls="--", lw=0.8, color="crimson")
    ax[0, 2].set_title("cumulative visible mass")
    ax[0, 2].set_xlabel("fraction of splats, brightest first")

    ax[1, 0].hist(np.log10(aniso), bins=100, color="#444")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_title("log10 anisotropy")

    ax[1, 1].hist(np.log10(np.maximum(area_px, 1e-4)), bins=100, color="#444")
    ax[1, 1].axvline(0.0, ls="--", lw=0.8, color="crimson")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_title("log10 screen area (px2); red = 1px2")

    ax[1, 2].hist(np.log10(np.maximum(occ, 1)), bins=60, color="#444")
    ax[1, 2].set_title("log10 splats per column")

    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f"probe_{args.path.stem}.png"
    fig.tight_layout()
    fig.savefig(dest, dpi=110)
    print(f"\n   wrote {dest}\n")


def Splats_subset(s, idx):
    from bozkir.ply import Splats
    return Splats(xyz=s.xyz[idx], opacity=s.opacity[idx], scale=s.scale[idx],
                  rot=s.rot[idx], sh_dc=s.sh_dc[idx], sh_rest=s.sh_rest[idx],
                  sh_degree=s.sh_degree)


def _hemisphere_dirs(k, up_axis):
    """k roughly-even directions on the UPPER hemisphere.

    Cameras look at terrain from above, never from underneath. Averaging
    over the full sphere includes directions no camera will ever occupy.
    """
    i = np.arange(k) + 0.5
    phi = np.arccos(1 - i / k)          # 0..90 deg only
    theta = np.pi * (1 + 5 ** 0.5) * i
    d = np.stack([np.cos(theta) * np.sin(phi),
                  np.sin(theta) * np.sin(phi),
                  np.cos(phi)], axis=1).astype(np.float32)
    # rotate so the pole lands on up_axis
    return np.roll(d, up_axis - 2, axis=1)


if __name__ == "__main__":
    main()