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
from bozkir.patches import (appearance, apply_settings, auto_pick,  # noqa: E402
                            cached_search, choose, describe_trail,
                            level_patch, rendered_coverage, search_kwargs)
from bozkir import presets as presets_mod  # noqa: E402


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
    ap.add_argument("--auto", dest="auto", action="store_true", default=True,
                    help="when the settings given find fewer than --count "
                         "patches, loosen whichever filter is doing the "
                         "rejecting and say so. On by default")
    ap.add_argument("--no-auto", dest="auto", action="store_false",
                    help="use the settings exactly as given, and report what "
                         "they found even if it is nothing")
    ap.add_argument("--want", type=int, default=4,
                    help="how many patches are actually needed; --auto stops "
                         "loosening once it has this many. Four is a Wang "
                         "set")
    ap.add_argument("--no-search-cache", dest="search_cache",
                    action="store_false", default=True,
                    help="search again rather than recalling a stored "
                         "candidate list for these settings")
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

    # A search of a few thousand positions takes minutes, and printed
    # nothing at all until it finished. One rewriting line is the whole fix:
    # not knowing whether a run is working or hung is its own kind of bug.
    width = [0]

    def progress(done, total, viable):
        msg = (f"\r  {done}/{total} positions, {viable} viable"
               f"   {100.0 * done / total:4.0f}%")
        width[0] = max(width[0], len(msg))
        sys.stdout.write(msg.ljust(width[0]))
        sys.stdout.flush()
        if done == total:
            sys.stdout.write("\r" + " " * width[0] + "\r")
            sys.stdout.flush()

    common = search_kwargs(args, thickness)

    if args.auto:
        want = min(args.want, args.count)
        cands, trail = auto_pick(s, args.size, args.count, up,
                                 progress=progress,
                                 cache=args.search_cache, **common)
        # auto_pick stops at --count; the question that matters is whether
        # it reached --want, since four is what a Wang set needs.
        print(describe_trail(trail, want, args.size))
        # The indices printed below only mean anything alongside the
        # settings that produced them, so record those settings on the
        # namespace: --save-preset then stores what worked, and
        # export_wang --preset reproduces the same candidate list.
        apply_settings(args, trail[-1][1])
        if args.save_preset:
            presets_mod.save(args, args.save_preset, ap)
            print(f"  saved these settings as --preset {args.save_preset}")
        elif len(trail) > 1:
            print("  pass the same settings to export_wang, or re-run with "
                  "--save-preset NAME, or the indices below will point at "
                  "different patches")
        if len(cands) < want:
            _, settings, stats, _ = trail[-1]
            print(f"  rejected at the loosest setting tried: "
                  f"{stats.get('sparse', 0)} too sparse, "
                  f"{stats.get('holes', 0)} too many holes, "
                  f"{stats.get('tilt', 0)} too tilted, "
                  f"{stats.get('buried', 0)} ground buried")
    else:
        print(f"  searching for up to {args.count} candidates")
        sep = common.pop("min_separation", 1.0)
        found, stats, hit = cached_search(s, args.size, up, common,
                                          cache=args.search_cache,
                                          progress=progress, verbose=True)
        cands = choose(found, args.count, args.size, sep)
        for _, _, p, info in cands:
            info["cover"] = rendered_coverage(p, args.size, up)
        if hit:
            print(f"  recalled {len(found)} candidates from a previous "
                  f"search with these settings")

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