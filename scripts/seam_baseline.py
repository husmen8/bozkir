"""Naive-merge baseline: measure the tile boundary artifact.

Cuts one patch out of a scene, places two copies edge to edge, and renders
the result two ways from a sweep of viewing angles:

  global  every splat sorted together          (the reference)
  tiled   each tile sorted alone, then composited (what GSWT does)

Both renders see identical geometry. The only difference is whether splats
from the two tiles are allowed to sort against each other. Whatever
separates the images is the boundary artifact and nothing else.

Using two copies of the SAME patch matters: identical content on both
sides means a visible seam cannot be a content mismatch.

    python scripts/seam_baseline.py data/aligned/garden.ply
    python scripts/seam_baseline.py data/aligned/garden.ply --size 1.5 --elev 3
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bozkir.ply import load_ply  # noqa: E402
from bozkir.select import remove_floaters, remove_large  # noqa: E402
from bozkir.tile import (extract_patch, grid, render_global,  # noqa: E402
                         render_tiled, seam_camera)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--size", type=float, default=1.5,
                    help="patch edge length in world units")
    ap.add_argument("--tiles", type=int, nargs=2, default=(2, 1),
                    metavar=("NX", "NY"),
                    help="grid of tile copies; more tiles means more "
                         "boundaries stacked along a grazing view ray")
    ap.add_argument("--centre", type=float, nargs=2, default=None,
                    help="patch centre in the ground plane (default: scene median)")
    ap.add_argument("--elev", type=float, default=5.0,
                    help="camera elevation; low is the grazing case")
    ap.add_argument("--azim", type=float, nargs="*", default=None,
                    help="angles from the seam direction; 0 looks along it")
    ap.add_argument("--dist", type=float, default=None)
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--height", type=int, default=300)
    ap.add_argument("--fov", type=float, default=50.0)
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--up-axis", type=int, default=2, choices=(0, 1, 2))
    ap.add_argument("--out", type=Path, default=Path("out/seam"))
    args = ap.parse_args()

    azims = args.azim if args.azim else [0, 15, 30, 45, 60, 75, 90]
    plane = [i for i in range(3) if i != args.up_axis]

    s = load_ply(args.path)
    print(f"{args.path.name}: {len(s):,} splats")

    centre = (args.centre if args.centre is not None
              else np.median(s.xyz[:, plane], axis=0))
    patch = extract_patch(s, centre, args.size, up_axis=args.up_axis)
    print(f"  patch {args.size} x {args.size} at "
          f"({centre[0]:.2f}, {centre[1]:.2f}): {len(patch):,} splats")

    if not args.no_clean:
        n0 = len(patch)
        patch, _ = remove_large(patch,
                                float(np.percentile(patch.scale.max(axis=1), 99)))
        patch, _ = remove_floaters(patch, std_ratio=2.0)
        print(f"  cleaned: {n0:,} -> {len(patch):,}")

    if len(patch) < 500:
        raise SystemExit("patch is nearly empty - try a different --centre "
                         "or a larger --size")

    # Two copies side by side. The seam is the plane between them, running
    # along the second ground axis through the origin.
    nx, ny = args.tiles
    tiles = grid(patch, nx, ny, args.size, up_axis=args.up_axis)
    seam = np.zeros(3, dtype=np.float32)
    seam[args.up_axis] = float(np.percentile(patch.xyz[:, args.up_axis], 60))
    dist = args.dist if args.dist is not None else args.size * 1.6

    print(f"  {nx}x{ny} tiles ({len(tiles)} total), seam at {np.round(seam, 2)}, "
          f"camera distance {dist:.2f}")
    print(f"\n  {'azim':>5} {'splats':>9} {'affected':>9} {'mean|d|':>9} "
          f"{'p99':>8} {'max':>8}")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for az in azims:
        cam = seam_camera(seam, dist, args.elev, az, up_axis=args.up_axis,
                          fov_deg=args.fov, width=args.width, height=args.height)
        t0 = time.time()
        g, pg = render_global(cam, tiles, sh_degree=0)
        t, pt = render_tiled(cam, tiles, sh_degree=0)
        dt = time.time() - t0

        d = np.abs(g - t).sum(axis=2) / 3.0
        lit = g.sum(axis=2) > 0.01
        # Averaging over the whole frame buries a strong local effect in a
        # large area of untouched pixels. Separate the two questions: how
        # much of the image is affected, and how strong it is where it is.
        hit = lit & (d > 1.0 / 255.0)          # one 8-bit level
        frac = float(hit.sum() / max(lit.sum(), 1))
        mean_hit = float(d[hit].mean()) if hit.any() else 0.0
        p99 = float(np.percentile(d[hit], 99)) if hit.any() else 0.0
        rows.append((az, frac, mean_hit, p99, float(d.max())))

        stem = f"{args.path.stem}_el{int(args.elev):02d}_az{int(az):03d}"
        plt.imsave(args.out / f"{stem}_global.png", g)
        plt.imsave(args.out / f"{stem}_tiled.png", t)
        # Amplified 20x. The artifact is a thin strip; without gain it is
        # invisible against black.
        plt.imsave(args.out / f"{stem}_diff.png",
                   np.clip(d * 20.0, 0, 1), cmap="inferno")

        print(f"  {az:5.0f} {pg['kept']:9,} {frac:8.2%} "
              f"{mean_hit:9.4f} {p99:8.4f} {d.max():8.4f}   ({dt:.0f}s)")

    rows = np.array(rows)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(rows[:, 0], rows[:, 1] * 100, "o-", color="#c0392b")
    ax[0].set_xlabel("angle from the seam direction (deg)")
    ax[0].set_ylabel("affected pixels (% of lit)")
    ax[0].set_title("how much of the image")
    ax[0].grid(alpha=0.3)
    ax[1].plot(rows[:, 0], rows[:, 2], "o-", label="mean on affected",
               color="#c0392b")
    ax[1].plot(rows[:, 0], rows[:, 3], "o-", label="p99", color="#e67e22")
    ax[1].plot(rows[:, 0], rows[:, 4], "o-", label="max", color="#2980b9")
    ax[1].set_xlabel("angle from the seam direction (deg)")
    ax[1].set_ylabel("per-pixel difference")
    ax[1].set_title("how strong where it happens")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.suptitle(f"{args.path.stem}  patch {args.size}  elevation {args.elev}")
    fig.tight_layout()
    dest = args.out / f"{args.path.stem}_el{int(args.elev):02d}_curve.png"
    fig.savefig(dest, dpi=120)

    print(f"\n  0 deg looks along the seam, 90 looks across it.")
    print(f"  most affected at {rows[np.argmax(rows[:, 1]), 0]:.0f} deg "
          f"({rows[:, 1].max():.2%} of lit pixels)")
    print(f"  strongest at     {rows[np.argmax(rows[:, 4]), 0]:.0f} deg "
          f"(max {rows[:, 4].max():.3f} per pixel)")
    print(f"  wrote {args.out}/")


if __name__ == "__main__":
    main()