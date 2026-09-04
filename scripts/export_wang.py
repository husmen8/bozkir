"""Build a Wang tile set from a scene and export it for the browser.

Cuts exemplar patches from the ground, assembles every tile over the given
edge colours, and writes one .splat plus a manifest carrying each tile's
edge codes so the viewer can lay them out aperiodically.

Two colours per axis needs four patches and produces sixteen tiles, which is
the smallest set guaranteeing a tile always fits.

    python scripts/export_wang.py data/raw/bigsur.ply --clean --flip \
        --floater-std 0 --size 1.5 --thickness 0.3 --max-tilt 20 \
        --max-below 0.65 --blend 0.08 --max-per-tile 50000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bozkir.scene import add_scene_args, scene_from_args  # noqa: E402
from bozkir.wang import (build_tile_set, layout, check_layout,  # noqa: E402
                         edge_gaussians)
from export_splat import pack, STRIDE  # noqa: E402
from export_tileset import pick_patches, level_patch  # noqa: E402


def appearance(p):
    """A patch's colour signature: per-channel mean and spread.

    Two patches with similar signatures blend where they meet. Two that do
    not - pale cobbles against dark wet rock - show the diagonal seam no
    matter how wide the feather, because the change is in the material
    rather than in the cut.
    """
    rgb = p.base_rgb
    return np.concatenate([rgb.mean(axis=0), rgb.std(axis=0)])


def select_similar(cands, k, weight=1.0):
    """Choose k patches that score well and look like each other.

    `weight` trades appearance against score: 0 ignores appearance and takes
    the top k by score alone.

    The seed is not simply the best-scoring patch. On a beach the highest
    score can easily be sea foam, and seeding on an outlier drags the whole
    set toward it. The seed is the candidate that is both good and typical -
    closest to the middle of what the scene actually looks like.
    """
    if weight <= 0 or len(cands) <= k:
        return cands[:k]

    feats = [appearance(c[2]) for c in cands]
    scores = np.array([c[0] for c in cands], dtype=np.float64)
    scores = scores / max(scores.max(), 1e-9)

    centre = np.mean(feats, axis=0)
    typicality = np.array([float(np.linalg.norm(f - centre)) for f in feats])
    picked = [int(np.argmax(scores - weight * typicality))]

    while len(picked) < k:
        ref = np.mean([feats[i] for i in picked], axis=0)
        best, best_val = None, -1e30
        for i in range(len(cands)):
            if i in picked:
                continue
            d = float(np.linalg.norm(feats[i] - ref))
            val = scores[i] - weight * d
            if val > best_val:
                best, best_val = i, val
        picked.append(best)
    return [cands[i] for i in picked]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--size", type=float, default=1.5)
    ap.add_argument("--colours", type=int, default=2,
                    help="edge colours per axis; 2 gives 16 tiles from 4 patches")
    ap.add_argument("--blend", type=float, default=0.05,
                    help="width of the feathered band along each diagonal, "
                         "in world units; 0 is a hard cut")
    ap.add_argument("--similarity", type=float, default=1.5,
                    help="how strongly to prefer patches that look alike; "
                         "0 takes the highest-scoring patches regardless, "
                         "which makes the diagonals inside each tile obvious")
    ap.add_argument("--stride", type=float, default=0.5)
    ap.add_argument("--thickness", type=float, default=None)
    ap.add_argument("--max-tilt", type=float, default=12.0)
    ap.add_argument("--max-below", type=float, default=0.5)
    ap.add_argument("--max-per-tile", type=int, default=None)
    ap.add_argument("-o", "--out", type=Path, default=None)
    add_scene_args(ap)
    args = ap.parse_args()

    s = scene_from_args(args)
    up = args.up_axis
    thickness = args.thickness if args.thickness else args.size * 0.25
    need = 2 * args.colours

    print(f"  need {need} patches for {args.colours} colours per axis")
    # Search widely, then narrow on appearance: a patch that scores well but
    # looks nothing like the others produces a tile with a visible X in it.
    chosen = pick_patches(s, args.size, max(need * 4, 12), up, args.stride,
                          thickness, args.max_tilt, args.max_below)
    if len(chosen) < need:
        raise SystemExit(f"  only {len(chosen)} patches passed, need {need}")
    chosen = select_similar(chosen, need, args.similarity)

    feats = np.array([appearance(c[2]) for c in chosen])
    spread = float(np.linalg.norm(feats - feats.mean(axis=0), axis=1).mean())
    print(f"  appearance spread across the {need} chosen: {spread:.3f} "
          f"({'similar' if spread < 0.08 else 'diagonals will show'})")

    patches = []
    print(f"\n  {'#':>2} {'centre':>16} {'splats':>9} {'relief':>7} "
          f"{'tilt':>7} {'mean rgb':>18}")
    for i, (sc, (x, y), p, info) in enumerate(chosen[:need]):
        if args.max_per_tile and len(p) > args.max_per_tile:
            ink = p.scale.max(axis=1) * p.opacity
            keep = np.argpartition(ink, -args.max_per_tile)[-args.max_per_tile:]
            p = p.subset(np.sort(keep))
        p, tilt = level_patch(p, up)
        # Centre on the origin so every tile is assembled in the same frame.
        plane = [j for j in range(3) if j != up]
        off = np.zeros(3, dtype=np.float32)
        off[plane] = p.xyz[:, plane].mean(axis=0)
        q = p.subset(np.arange(len(p)))
        q.xyz = (q.xyz - off).astype(np.float32)
        patches.append(q)
        rgb = q.base_rgb.mean(axis=0)
        print(f"  {i:>2} ({x:6.2f},{y:6.2f}) {len(q):>9,} "
              f"{info['relief']:>7.2f} {tilt:>6.1f}\u00b0   "
              f"{rgb[0]:.2f} {rgb[1]:.2f} {rgb[2]:.2f}")

    h_patches = patches[:args.colours]
    v_patches = patches[args.colours:need]

    print(f"\n  assembling tiles (blend {args.blend})")
    tiles, codes = build_tile_set(h_patches, v_patches, args.size,
                                  blend=args.blend, up_axis=up)

    # The construction is only worth anything if edges really do match.
    ok = True
    for edge, col in (("n", 0), ("e", 1), ("s", 2), ("w", 3)):
        groups = {}
        for t, c in zip(tiles, codes):
            groups.setdefault(c[col], []).append(
                edge_gaussians(t, args.size, up, edge,
                               margin=max(0.03, args.blend / args.size * 1.5)))
        for g in groups.values():
            if not all(np.array_equal(g[0], x) for x in g):
                ok = False
    print(f"  edge check: "
          f"{'matching colours share an identical edge' if ok else 'FAILED'}")

    grid = layout(codes, 16, 16, seed=0)
    print(f"  layout check: {check_layout(codes, grid)} mismatched edges in 16x16")

    parts, meta, cursor = [], [], 0
    for t, c in zip(tiles, codes):
        parts.append(pack(t))
        meta.append({"start": cursor, "count": len(t),
                     "n": int(c[0]), "e": int(c[1]),
                     "s": int(c[2]), "w": int(c[3])})
        cursor += len(t)

    buf = np.concatenate(parts)
    stem = args.out.stem if args.out else args.path.stem
    dest = args.out or Path("web/data") / f"{stem}.splat"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(buf.tobytes())
    dest.with_suffix(".json").write_text(json.dumps({
        "size": args.size, "up_axis": up, "stride": STRIDE,
        "thickness": thickness, "blend": args.blend,
        "colours": args.colours, "wang": True,
        "total": cursor, "tiles": meta}, indent=2))

    print(f"\n  {len(tiles)} tiles, {cursor:,} splats, {len(buf) / 1e6:.1f} MB")
    print(f"  wrote {dest}")
    print(f"  wrote {dest.with_suffix('.json')}")


if __name__ == "__main__":
    main()