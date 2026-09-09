"""Look at the candidate patches before committing to four of them.

Patch choice decides everything downstream: four squares that look like
each other tile into ground, four that do not tile into patchwork. Until
now a scoring function chose them and you never saw what it passed over.

This renders every viable candidate as a labelled thumbnail so the choice
can be made by eye, and prints the index of each. Those indices go straight
into export_wang:

    python scripts/preview_patches.py data/raw/bigsur.ply --preset bigsur
    python scripts/export_wang.py data/raw/bigsur.ply --preset bigsur \
        --patches 3,7,11,2
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bozkir.presets import add_preset_args, apply as apply_preset  # noqa: E402
from bozkir.scene import add_scene_args, scene_from_args  # noqa: E402
from bozkir.graphcut import render_patch  # noqa: E402
from bozkir.patches import appearance, level_patch, pick_patches  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--size", type=float, default=1.5)
    ap.add_argument("--count", type=int, default=24,
                    help="how many candidates to show")
    ap.add_argument("--stride", type=float, default=0.4,
                    help="search step as a fraction of --size; smaller finds "
                         "more candidates and takes longer")
    ap.add_argument("--thickness", type=float, default=None)
    ap.add_argument("--max-tilt", type=float, default=12.0)
    ap.add_argument("--features", action="store_true",
                    help="look for tiles with something in them - a boulder, "
                         "a bush - instead of the flattest ground. Only the "
                         "rim has to be flat; the middle is where the graph "
                         "cut works and where an object can sit.")
    ap.add_argument("--cover-margin", type=float, default=0.10,
                    help="how close to --min-cover an estimate has to be "
                         "before the slow render is used to settle it. "
                         "Larger is more accurate and much slower; 0 never "
                         "renders.")
    ap.add_argument("--min-cover", type=float, default=0.80,
                    help="reject a patch that covers less than this fraction "
                         "of its square; a patch with a hole in it tiles "
                         "with holes")
    ap.add_argument("--extract-margin", type=float, default=0.35,
                    help="cut candidates this much larger than the tile, so "
                         "levelling has material to rotate in from")
    ap.add_argument("--min-separation", type=float, default=1.0,
                    help="how far apart chosen patches must sit, as a "
                         "fraction of --size. Lower it when a scene is small "
                         "and too few candidates survive.")
    ap.add_argument("--edge-flat", type=float, default=0.20,
                    help="with --features: how much relief the rim may have, "
                         "as a fraction of the tile")
    ap.add_argument("--edge-margin", type=float, default=0.22,
                    help="with --features: how wide the rim is, as a fraction "
                         "of the tile")
    ap.add_argument("--max-below", type=float, default=0.5)
    ap.add_argument("--res", type=int, default=180)
    ap.add_argument("--out", type=Path, default=Path("out"))
    add_scene_args(ap)
    add_preset_args(ap)
    args = apply_preset(ap)

    s = scene_from_args(args)
    up = args.up_axis
    thickness = args.thickness if args.thickness else args.size * 0.25

    print(f"  searching for up to {args.count} candidates")
    cands = pick_patches(s, args.size, args.count, up, args.stride,
                         thickness, args.max_tilt, args.max_below,
                         features=args.features, edge_flat=args.edge_flat,
                         edge_margin=args.edge_margin,
                         min_separation=args.min_separation,
                         min_cover=args.min_cover,
                         cover_margin=args.cover_margin,
                         extract_margin=args.extract_margin)

    thumbs, rows = [], []
    for i, (score, (x, y), p, info) in enumerate(cands):
        # Level the wider cut so rotation has material to draw in from,
        # then take the tile-sized middle of the result.
        p, tilt = level_patch(info.get("wide", p), up)
        # Centre on the ground plane the way export_wang does, so the
        # thumbnail shows the square that actually becomes a tile.
        plane2 = [j for j in range(3) if j != up]
        off = np.zeros(3, dtype=np.float32)
        off[plane2] = p.xyz[:, plane2].mean(axis=0)
        p = p.subset(np.arange(len(p)))
        p.xyz = (p.xyz - off).astype(np.float32)
        rgb, alpha = render_patch(p, args.size, args.res, up)
        # Show coverage, so a patch full of holes is obvious.
        thumbs.append(np.clip(rgb * alpha[..., None], 0, 1))
        mean = p.base_rgb.mean(axis=0)
        rows.append((i, x, y, len(p), info["relief"], tilt,
                     float(alpha.mean()), mean, appearance(p),
                     info.get("edge_relief", 0.0),
                     info.get("interior_relief", 0.0)))

    print(f"\n  {'#':>3} {'centre':>16} {'splats':>8} {'relief':>7} "
          f"{'tilt':>6} {'rim':>6} {'mid':>6} {'cover':>6}  mean rgb")
    for (i, x, y, n, relief, tilt, cover, mean, _, rim, mid) in rows:
        print(f"  {i:>3} ({x:6.2f},{y:6.2f}) {n:>8,} {relief:>7.2f} "
              f"{tilt:>5.1f}\u00b0 {rim:>6.2f} {mid:>6.2f} {cover:>6.0%}  "
              f"{mean[0]:.2f} {mean[1]:.2f} {mean[2]:.2f}")

    feats = np.array([r[8] for r in rows])
    print(f"\n  appearance spread across all {len(rows)}: "
          f"{np.linalg.norm(feats - feats.mean(0), axis=1).mean():.3f}")
    print("  four patches that look alike tile into ground; four that do "
          "not tile into patchwork.")

    cols = min(6, max(1, len(thumbs)))
    rws = int(np.ceil(len(thumbs) / cols))
    fig, ax = plt.subplots(rws, cols, figsize=(2.1 * cols, 2.35 * rws))
    ax = np.atleast_1d(ax).ravel()
    for a in ax:
        a.axis("off")
    for k, (img, r) in enumerate(zip(thumbs, rows)):
        ax[k].imshow(img)
        ax[k].set_title(f"{r[0]}   rim {r[9]:.2f}  mid {r[10]:.2f}",
                        fontsize=8)
    fig.suptitle(f"{args.path.stem}  -  candidate patches, "
                 f"{args.size} x {args.size}", fontsize=11)
    fig.tight_layout()

    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f"patches_{args.path.stem}.png"
    fig.savefig(dest, dpi=110)
    print(f"\n  wrote {dest}")
    print("  pick four and pass them as --patches a,b,c,d "
          "(first two become north/south colours, last two east/west)")


if __name__ == "__main__":
    main()