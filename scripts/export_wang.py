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
from bozkir.patches import (appearance, level_patch, part_ink,  # noqa: E402
                            pick_patches, select_similar, stratified_keep)
from bozkir.presets import add_preset_args, apply as apply_preset  # noqa: E402
from bozkir.scene import add_scene_args, scene_from_args  # noqa: E402
from bozkir.wang import (build_tile_set, layout, check_layout,  # noqa: E402
                         edge_gaussians)
from bozkir.pack import pack, STRIDE  # noqa: E402
from bozkir.tile import extract_patch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--size", type=float, default=1.5)
    ap.add_argument("--colours", type=int, default=2,
                    help="edge colours per axis; 2 gives 16 tiles from 4 patches")
    ap.add_argument("--blend", type=float, default=0.05,
                    help="width of the feathered band along each diagonal, "
                         "in world units; 0 is a hard cut. Ignored with --cut.")
    ap.add_argument("--cut", action="store_true",
                    help="place each diagonal by graph cut instead of leaving "
                         "it straight: the boundary follows wherever the two "
                         "patches already agree, so the join hides in the "
                         "texture (Kwatra et al. 2003, GSWT 3.2)")
    ap.add_argument("--cut-res", type=int, default=160,
                    help="pixel grid the cut is solved on; higher gives the "
                         "boundary more room to wander and costs more time")
    ap.add_argument("--cut-band", type=float, default=0.14,
                    help="how far from the straight diagonal the boundary may "
                         "move, as a fraction of the tile")
    ap.add_argument("--patches", type=str, default=None,
                    help="use these candidate indices instead of letting the "
                         "scoring choose, e.g. 3,7,11,2. Indices come from "
                         "scripts/preview_patches.py. The first `colours` "
                         "become north/south, the rest east/west.")
    ap.add_argument("--similarity", type=float, default=1.5,
                    help="how strongly to prefer patches that look alike; "
                         "0 takes the highest-scoring patches regardless, "
                         "which makes the diagonals inside each tile obvious")
    ap.add_argument("--stride", type=float, default=0.5)
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
    ap.add_argument("--tile-overhang", type=float, default=None,
                    help="how far past its square a finished tile may reach, "
                         "as a fraction of the tile. Default is automatic: a "
                         "couple of splat widths, which is what it takes to "
                         "cover the seam. Much more and neighbours draw the "
                         "same strip twice and swap which is in front as the "
                         "camera turns; much less and the strip is bare.")
    ap.add_argument("--overhang-splats", type=float, default=2.5,
                    help="with automatic overhang: how many splat widths")
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
    ap.add_argument("--max-per-tile", type=int, default=None)
    ap.add_argument("--no-lod-compensate", dest="lod_compensate",
                    action="store_false",
                    help="do not grow surviving splats to cover for the ones "
                         "a coarse level drops")
    ap.add_argument("--lod", type=int, default=4,
                    help="levels of detail per tile; each holds a quarter of "
                         "the splats of the one before, as in GSWT 3.5")
    ap.add_argument("-o", "--out", type=Path, default=None)
    add_scene_args(ap)
    add_preset_args(ap)
    args = apply_preset(ap)

    s = scene_from_args(args)
    up = args.up_axis
    thickness = args.thickness if args.thickness else args.size * 0.25
    need = 2 * args.colours

    print(f"  need {need} patches for {args.colours} colours per axis")
    # Search widely, then narrow on appearance: a patch that scores well but
    # looks nothing like the others produces a tile with a visible X in it.
    chosen = pick_patches(s, args.size, max(need * 4, 12), up, args.stride,
                          thickness, args.max_tilt, args.max_below,
                          features=args.features, edge_flat=args.edge_flat,
                          edge_margin=args.edge_margin,
                          min_separation=args.min_separation,
                          min_cover=args.min_cover,
                         cover_margin=args.cover_margin,
                          extract_margin=args.extract_margin)
    if len(chosen) < need:
        raise SystemExit(f"  only {len(chosen)} patches passed, need {need}")
    if args.patches:
        want = [int(v) for v in args.patches.replace(" ", "").split(",") if v]
        if len(want) != need:
            raise SystemExit(f"--patches needs exactly {need} indices for "
                             f"{args.colours} colours per axis, got {len(want)}")
        if max(want) >= len(chosen):
            raise SystemExit(f"index {max(want)} is past the {len(chosen)} "
                             f"candidates found; lower --stride to find more")
        chosen = [chosen[i] for i in want]
        print(f"  using candidates {want} as given")
    else:
        chosen = select_similar(chosen, need, args.similarity)

    feats = np.array([appearance(c[2]) for c in chosen])
    spread = float(np.linalg.norm(feats - feats.mean(axis=0), axis=1).mean())
    print(f"  appearance spread across the {need} chosen: {spread:.3f} "
          f"({'similar' if spread < 0.08 else 'diagonals will show'})")

    patches = []
    print(f"\n  {'#':>2} {'centre':>16} {'splats':>9} {'relief':>7} "
          f"{'tilt':>7} {'rim':>6} {'mid':>6} {'mean rgb':>18}")
    for i, (sc, (x, y), p, info) in enumerate(chosen[:need]):
        # Level the wider cut so rotation has material to draw in from,
        # then take the tile-sized middle of the result.
        p, tilt = level_patch(info.get("wide", p), up)
        # Centre on the origin so every tile is assembled in the same frame.
        plane = [j for j in range(3) if j != up]
        off = np.zeros(3, dtype=np.float32)
        off[plane] = p.xyz[:, plane].mean(axis=0)
        q = p.subset(np.arange(len(p)))
        q.xyz = (q.xyz - off).astype(np.float32)

        # The cap goes here, not before levelling: levelling reads the wide
        # cut, so anything trimmed off the narrow one is simply picked up
        # again and the cap does nothing. Spread the survivors rather than
        # taking the brightest, or the tile ends up dense in one corner.
        if args.max_per_tile and len(q) > args.max_per_tile:
            q = q.subset(stratified_keep(q, args.max_per_tile,
                                         args.size * (1.0 + args.extract_margin),
                                         up))
        # The overhang left by levelling is kept: a Gaussian just outside
        # the square still covers ground just inside, and trimming it opens
        # a gap along every edge. label_at gives those a well defined
        # region without needing the pixel grid.
        plane2 = [j for j in range(3) if j != up]
        over = float((np.abs(q.xyz[:, plane2]).max(axis=1)
                      > args.size / 2).mean())
        patches.append(q)
        rgb = q.base_rgb.mean(axis=0)
        print(f"  {i:>2} ({x:6.2f},{y:6.2f}) {len(q):>9,} "
              f"{info['relief']:>7.2f} {tilt:>6.1f}\u00b0 "
              f"{info.get('edge_relief', 0):>6.2f} "
              f"{info.get('interior_relief', 0):>6.2f}  "
              f"{rgb[0]:.2f} {rgb[1]:.2f} {rgb[2]:.2f}"
              f"  overhang {over:>4.0%}")

    h_patches = patches[:args.colours]
    v_patches = patches[args.colours:need]

    if args.cut:
        print(f"\n  assembling tiles (graph cut, {args.cut_res}px, "
              f"band {args.cut_band})")
    else:
        print(f"\n  assembling tiles (feathered, blend {args.blend})")
    tiles, codes = build_tile_set(h_patches, v_patches, args.size,
                                  blend=args.blend, up_axis=up,
                                  cut=args.cut, resolution=args.cut_res,
                                  band=args.cut_band)

    # The construction is only worth anything if edges really do match.
    ok = True
    for edge, col in (("n", 0), ("e", 1), ("s", 2), ("w", 3)):
        groups = {}
        for t, c in zip(tiles, codes):
            groups.setdefault(c[col], []).append(
                edge_gaussians(t, args.size, up, edge,
                               margin=max(0.03, 0.0 if args.cut
                                          else args.blend / args.size * 1.5)))
        for g in groups.values():
            if not all(np.array_equal(g[0], x) for x in g):
                ok = False
    print(f"  edge check: "
          f"{'matching colours share an identical edge' if ok else 'FAILED'}")

    grid = layout(codes, 16, 16, seed=0)
    print(f"  layout check: {check_layout(codes, grid)} mismatched edges in 16x16")

    # Levels of detail. GSWT retrains each level from downsampled images and
    # caps the count at N0 / 4^i; retraining is not available here, so each
    # level keeps the most visible splats of the level above instead -
    # largest area times opacity. Same budget, coarser selection.
    # The wide cut exists so levelling has material; a finished tile only
    # needs enough overhang to cover the seam. Keeping all of it means every
    # boundary strip is drawn by both neighbours, and they swap which is in
    # front as the camera turns. Cutting it to nothing leaves the strip bare,
    # because a Gaussian centred just outside still covers ground just in.
    #
    # The amount that works is a couple of splat widths - a distance set by
    # the material, not by how big the tile happens to be.
    # Whether a Gaussian outside the square is worth keeping depends on how
    # far it reaches, and that varies: sizes have a long tail, and the big
    # ones cover the most ground. A single distance keeps the small ones
    # that barely matter and drops the large ones that do, so each Gaussian
    # is allowed its own reach instead.
    plane2 = [j for j in range(3) if j != up]
    trimmed = []
    for t in tiles:
        if args.tile_overhang is not None:
            reach = args.size * (0.5 + args.tile_overhang)
            keep = np.all(np.abs(t.xyz[:, plane2]) <= reach, axis=1)
        else:
            out = np.abs(t.xyz[:, plane2]).max(axis=1) - args.size / 2.0
            keep = out <= args.overhang_splats * t.scale.max(axis=1)
        trimmed.append(t.subset(keep))

    before = sum(len(t) for t in tiles)
    after = sum(len(t) for t in trimmed)
    if args.tile_overhang is not None:
        how = f"a flat {args.tile_overhang:.1%} of the tile"
    else:
        out = np.concatenate([np.abs(t.xyz[:, plane2]).max(axis=1)
                              - args.size / 2.0 for t in trimmed])
        how = (f"{args.overhang_splats:g} of each Gaussian's own width "
               f"(up to {max(out.max(), 0) / args.size:.1%} of the tile)")
    print(f"  overhang: {how}")
    print(f"  trimmed {before:,} -> {after:,} splats ({after / before:.0%})")
    tiles = trimmed

    parts, meta, cursor = [], [], 0
    total_levels = 0
    for t, c in zip(tiles, codes):
        levels = []
        for lvl in range(max(args.lod, 1)):
            budget = max(len(t) // (4 ** lvl), 64)
            if budget >= len(t):
                part = t
            else:
                ink = part_ink(t)
                keep = stratified_keep(t, budget, args.size, up)
                part = t.subset(keep)
                # Dropping three splats in four leaves holes. GSWT avoids
                # this by retraining each level, so its coarse Gaussians are
                # genuinely larger; retraining is not available here, so the
                # survivors are grown instead until they cover the same
                # total area. Area goes as size squared, hence the sqrt.
                if args.lod_compensate:
                    grow = float(np.sqrt(ink.sum() / max(ink[keep].sum(), 1e-12)))
                    grow = min(grow, 4.0)      # a cap, or level 4 turns to soup
                    part = part.subset(np.arange(len(part)))
                    part.scale = (part.scale * grow).astype(np.float32)
            parts.append(pack(part))
            levels.append([cursor, len(part)])
            cursor += len(part)
            total_levels += 1
        meta.append({"start": levels[0][0], "count": levels[0][1],
                     "levels": levels,
                     "n": int(c[0]), "e": int(c[1]),
                     "s": int(c[2]), "w": int(c[3])})

    buf = np.concatenate(parts)
    stem = args.out.stem if args.out else args.path.stem
    dest = args.out or Path("web/data") / f"{stem}.splat"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(buf.tobytes())
    dest.with_suffix(".json").write_text(json.dumps({
        "size": args.size, "up_axis": up, "stride": STRIDE,
        "thickness": thickness, "blend": args.blend,
        "colours": args.colours, "wang": True, "lod": max(args.lod, 1),
        "cut": bool(args.cut),
        # What produced this file, so a scene can be traced back to the
        # settings that made it without keeping a shell history.
        "settings": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in sorted(vars(args).items())
                     if k not in ("out", "list_presets", "save_preset")},
        "total": cursor, "tiles": meta}, indent=2))

    print(f"\n  {len(tiles)} tiles x {max(args.lod, 1)} levels "
          f"= {total_levels} ranges")
    for lvl in range(max(args.lod, 1)):
        n = sum(m["levels"][lvl][1] for m in meta)
        base = sum(m["levels"][0][1] for m in meta)
        print(f"    level {lvl}: {n:>9,} splats ({n / base:>4.0%} of level 0)")
    print(f"  {cursor:,} splats total, {len(buf) / 1e6:.1f} MB")
    print(f"  wrote {dest}")
    print(f"  wrote {dest.with_suffix('.json')}")


if __name__ == "__main__":
    main()