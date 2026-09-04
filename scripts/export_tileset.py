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
from bozkir.scene import add_scene_args, scene_from_args  # noqa: E402
from bozkir.tile import extract_patch  # noqa: E402
from bozkir.transform import rotate, quat_between  # noqa: E402
from export_splat import pack, STRIDE  # noqa: E402


def ground_level(h, thickness):
    """Height of the ground within a patch.

    Not a low percentile: reconstructions leave junk below the surface, and
    a percentile follows it down. The ground is the densest thin horizontal
    layer, so take the tallest bin of a histogram instead. Foliage above is
    diffuse and never out-votes it.
    """
    lo, hi = np.percentile(h, [1, 99])
    if hi - lo < 1e-6:
        return float(lo)
    bins = max(int((hi - lo) / max(thickness / 4.0, 1e-6)), 8)
    counts, edges = np.histogram(h, bins=bins, range=(lo, hi))
    k = int(np.argmax(counts))
    return float((edges[k] + edges[k + 1]) / 2.0)


def mass_below(p, up_axis, g, thickness, below=0.25):
    """Fraction of a column that sits under the layer called 'ground'.

    The histogram vote picks the densest horizontal layer, which under a
    thick tree is the canopy rather than the dirt. Nothing should be beneath
    the ground, so a large fraction here means the wrong layer won. This is
    a local test: unlike comparing against a global height it works on
    terrain with real relief, where the ground is not near zero.
    """
    h = p.xyz[:, up_axis]
    return float((h < g - thickness * below).mean())


def clip_slab(p, up_axis, thickness, below=0.25):
    """Keep a slab around the ground, discarding whatever stands on it.

    GSWT cuts a full column and keeps every Gaussian regardless of height,
    which suits a flat exemplar. On a captured outdoor scene the column runs
    from the dirt up through a whole tree, and a tree does not tile.
    """
    h = p.xyz[:, up_axis]
    g = ground_level(h, thickness)
    keep = (h >= g - thickness * below) & (h <= g + thickness)
    return p.subset(keep), g


def patch_tilt(p, up_axis):
    """Angle between a patch's own surface normal and the scene's up axis."""
    if len(p) < 50:
        return 90.0
    d = p.xyz - p.xyz.mean(axis=0)
    _, evecs = np.linalg.eigh((d.T @ d) / len(d))
    n = evecs[:, 0]
    return float(np.degrees(np.arccos(np.clip(abs(n[up_axis]), 0, 1))))


def level_patch(p, up_axis):
    """Rotate a patch so its own ground is horizontal, then drop it to zero.

    Tiles are laid on a common plane, so each has to agree about which way
    is flat. Without this, patches cut from a sloping scene meet at a step.
    """
    if len(p) < 50:
        return p, 0.0
    d = p.xyz - p.xyz.mean(axis=0)
    _, evecs = np.linalg.eigh((d.T @ d) / len(d))
    n = evecs[:, 0]
    if n[up_axis] < 0:
        n = -n

    target = np.zeros(3, dtype=np.float32)
    target[up_axis] = 1.0
    tilt = float(np.degrees(np.arccos(np.clip(abs(n[up_axis]), 0, 1))))
    if tilt > 0.5:
        p = rotate(p, quat_between(n, target))

    drop = np.zeros(3, dtype=np.float32)
    drop[up_axis] = np.median(p.xyz[:, up_axis])
    out = p.subset(np.arange(len(p)))
    out.xyz = (out.xyz - drop).astype(np.float32)
    return out, tilt


def score_patch(p, up_axis, size):
    """How much a patch looks like tileable ground. Higher is better.

    Three things are wanted: enough splats to render, a surface that is
    flat rather than a wall or an object, and coverage spread across the
    whole square rather than clustered in one corner.
    """
    if len(p) < 2000:
        return -1.0, {}

    plane = [i for i in range(3) if i != up_axis]
    h = p.xyz[:, up_axis]
    lo, hi = np.percentile(h, [5, 95])
    relief = float(hi - lo)

    # How flat the surface actually is, independent of how thick the slab
    # was cut. A patch that is mostly a tree trunk fails this even if the
    # slab clipped it to the right thickness.
    d = p.xyz - p.xyz.mean(axis=0)
    ev = np.linalg.eigvalsh((d.T @ d) / len(d))
    planarity = float(ev[0] / max(ev[2], 1e-30))

    # Occupancy on an 8x8 grid: a patch with a hole in it will not tile.
    g = 8
    ij = np.clip(((p.xyz[:, plane] - p.xyz[:, plane].min(axis=0))
                  / max(size, 1e-6) * g).astype(int), 0, g - 1)
    filled = len(np.unique(ij[:, 0] * g + ij[:, 1])) / (g * g)

    density = len(p) / (size * size)
    flatness = 1.0 / (1.0 + relief / max(size, 1e-6))

    return float(filled * flatness * np.log1p(density)
                 / (1.0 + 20.0 * planarity)), {
        "splats": len(p), "relief": relief, "filled": filled,
        "planarity": planarity,
    }


def pick_patches(s, size, k, up_axis, stride=0.5, thickness=0.3,
                 max_tilt=12.0, max_below=0.5, verbose=True):
    """Search the ground plane for the k best non-overlapping patches."""
    plane = [i for i in range(3) if i != up_axis]
    lo = np.percentile(s.xyz[:, plane], 2, axis=0)
    hi = np.percentile(s.xyz[:, plane], 98, axis=0)

    step = size * stride
    xs = np.arange(lo[0] + size / 2, hi[0] - size / 2 + 1e-6, step)
    ys = np.arange(lo[1] + size / 2, hi[1] - size / 2 + 1e-6, step)
    if len(xs) == 0 or len(ys) == 0:
        raise SystemExit(f"scene is smaller than one {size} patch")

    cands = []
    rejected = {"sparse": 0, "buried": 0, "tilt": 0, "score": 0}
    for x in xs:
        for y in ys:
            p = extract_patch(s, [x, y], size, up_axis=up_axis)
            if len(p) < 2000:
                rejected["sparse"] += 1
                continue
            below = mass_below(p, up_axis, ground_level(p.xyz[:, up_axis],
                                                        thickness), thickness)
            p, g = clip_slab(p, up_axis, thickness)
            if below > max_below:
                rejected["buried"] += 1
                continue
            # Real ground does not need a large correction once the whole
            # scene is already levelled. A patch that does is a wall, a
            # bank, or foliage.
            tilt = patch_tilt(p, up_axis)
            if tilt > max_tilt:
                rejected["tilt"] += 1
                continue
            sc, info = score_patch(p, up_axis, size)
            if sc <= 0:
                rejected["score"] += 1
                continue
            info["ground"] = g
            info["tilt"] = tilt
            info["below"] = below
            cands.append((sc, (float(x), float(y)), p, info))
    if verbose or not cands:
        print(f"  searched {len(xs)}x{len(ys)} positions, "
              f"{len(cands)} viable")
        print(f"  rejected: {rejected['sparse']} too sparse, "
              f"{rejected['buried']} ground buried "
              f"(>{max_below:.0%} of the column below it), "
              f"{rejected['tilt']} too tilted (>{max_tilt:.0f} deg), "
              f"{rejected['score']} low score")
    if not cands:
        raise SystemExit(
            "  nothing passed. The counts above say which filter to loosen:\n"
            "    too sparse   -> larger --size, or higher --radius-pct\n"
            "    ground buried-> thicker --thickness, or higher --max-below\n"
            "    too tilted   -> higher --max-tilt, or the region is a slope")

    cands.sort(key=lambda c: -c[0])

    chosen = []
    for sc, (x, y), p, info in cands:
        # Keep patches apart so the set has genuinely different content.
        if any(abs(x - cx) < size and abs(y - cy) < size
               for _, (cx, cy), _, _ in chosen):
            continue
        chosen.append((sc, (x, y), p, info))
        if len(chosen) >= k:
            break
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--size", type=float, default=1.2,
                    help="patch edge length in world units")
    ap.add_argument("--tiles", type=int, default=4,
                    help="how many distinct patches to cut")
    ap.add_argument("--stride", type=float, default=0.5,
                    help="search step as a fraction of --size")
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
    args = ap.parse_args()

    s = scene_from_args(args)
    up = args.up_axis
    thickness = args.thickness if args.thickness else args.size * 0.25

    print(f"  cutting {args.tiles} patches of {args.size} x {args.size}, "
          f"slab {thickness:.2f} thick")
    chosen = pick_patches(s, args.size, args.tiles, up, args.stride,
                          thickness, args.max_tilt, args.max_below)
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