"""Cut patches out of a scene and export them as a tile set for the browser.

A tile set is one .splat file holding several patches back to back, plus a
JSON manifest saying where each begins. The viewer lays them out on a grid,
picking a patch per cell.

Patch selection follows GSWT Section 3.2: choose regions on the ground
plane and keep every Gaussian whose projected position falls inside,
ignoring height. Candidate regions are ranked by how much they look like
ground - dense, flat, and not dominated by one tall object - because a
patch containing a table leg does not tile.

    python scripts/export_tileset.py data/raw/bicycle.ply --clean --size 1.2
    python scripts/export_tileset.py data/raw/bicycle.ply --clean --tiles 8
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bozkir.patches import (clip_slab, ground_level, level_patch,  # noqa: E402
                            pick_patches, score_patch)
from bozkir.presets import add_preset_args, apply as apply_preset  # noqa: E402
from bozkir.scene import add_scene_args, scene_from_args  # noqa: E402
from bozkir.tile import extract_patch  # noqa: E402
from bozkir.transform import rotate, quat_between  # noqa: E402
from bozkir.pack import pack, STRIDE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--size", type=float, default=1.2,
                    help="patch edge length in world units")
    ap.add_argument("--tiles", type=int, default=4,
                    help="how many distinct patches to cut")
    ap.add_argument("--stride", type=float, default=0.5,
                    help="search step as a fraction of --size")
    ap.add_argument("--features", action="store_true",
                    help="look for tiles with something in them - a boulder, "
                         "a bush - instead of the flattest ground. Only the "
                         "rim has to be flat; the middle is where the graph "
                         "cut works and where an object can sit.")
    ap.add_argument("--edge-flat", type=float, default=0.20,
                    help="with --features: how much relief the rim may have, "
                         "as a fraction of the tile")
    ap.add_argument("--edge-margin", type=float, default=0.22,
                    help="with --features: how wide the rim is, as a fraction "
                         "of the tile")
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
                    help="how far apart chosen patches must be, in tiles. "
                         "Keeps the set varied, but on a small scene it "
                         "prunes most of what passed the filters.")
    ap.add_argument("--max-below", type=float, default=0.5,
                    help="reject a patch when more than this fraction of its "
                         "column lies under the detected ground, which means "
                         "the wrong layer was picked. Terrain with relief "
                         "needs a high value: a column through a plateau "
                         "runs down the cliff face beneath it.")
    ap.add_argument("--max-tilt", type=float, default=12.0,
                    help="reject patches whose surface leans more than this "
                         "many degrees from the scene's up axis")
    ap.add_argument("--thickness", type=float, default=None,
                    help="slab height kept above the ground, world units "
                         "(default: a quarter of --size)")
    ap.add_argument("--max-per-tile", type=int, default=None,
                    help="cap splats per patch, most visible kept")
    ap.add_argument("-o", "--out", type=Path, default=None)
    add_scene_args(ap)
    add_preset_args(ap)
    args = apply_preset(ap)

    s = scene_from_args(args)
    up = args.up_axis
    thickness = args.thickness if args.thickness else args.size * 0.25

    print(f"  cutting {args.tiles} patches of {args.size} x {args.size}, "
          f"slab {thickness:.2f} thick")
    chosen = pick_patches(s, args.size, args.tiles, up, args.stride,
                          thickness, args.max_tilt, args.max_below,
                          features=args.features, edge_flat=args.edge_flat,
                          edge_margin=args.edge_margin,
                          min_separation=args.min_separation,
                         min_cover=args.min_cover,
                         cover_margin=args.cover_margin,
                         extract_margin=args.extract_margin)
    if len(chosen) < args.tiles:
        print(f"  only {len(chosen)} patches passed; loosen --max-tilt, "
              f"raise --radius-pct, or shrink --size")

    parts, meta, cursor = [], [], 0
    print(f"\n  {'#':>2} {'centre':>16} {'splats':>9} {'relief':>7} "
          f"{'flat':>8} {'ground':>8} {'tilt':>7} {'below':>6}")
    for i, (sc, (x, y), p, info) in enumerate(chosen):
        if args.max_per_tile and len(p) > args.max_per_tile:
            ink = p.scale.max(axis=1) * p.opacity
            keep = np.argpartition(ink, -args.max_per_tile)[-args.max_per_tile:]
            p = p.subset(np.sort(keep))

        p, tilt = level_patch(p, up)

        parts.append(pack(p))
        meta.append({"start": cursor, "count": len(p), "centre": [x, y],
                     "score": round(sc, 3), "relief": round(info["relief"], 3)})
        cursor += len(p)
        print(f"  {i:>2} ({x:6.2f},{y:6.2f}) {len(p):>9,} "
              f"{info['relief']:>7.2f} {info['planarity']:>8.4f} "
              f"{info['ground']:>+8.2f} {tilt:>6.1f}° "
              f"{info['below']:>5.0%}")

    buf = np.concatenate(parts)
    stem = args.out.stem if args.out else args.path.stem
    dest = args.out or Path("web/data") / f"{stem}.splat"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(buf.tobytes())

    manifest = {"size": args.size, "up_axis": up, "stride": STRIDE,
                "thickness": thickness, "total": cursor, "tiles": meta}
    mpath = dest.with_suffix(".json")
    mpath.write_text(json.dumps(manifest, indent=2))

    print(f"\n  {cursor:,} splats total, {len(buf) / 1e6:.1f} MB")
    print(f"  wrote {dest}")
    print(f"  wrote {mpath}")


if __name__ == "__main__":
    main()