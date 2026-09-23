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
from bozkir.patches import (appearance, apply_settings, auto_pick,  # noqa: E402
                            balanced_split, cached_search, choose,
                            class_separation, describe_trail, level_patch,
                            part_ink, rendered_coverage, search_kwargs,
                            select_similar, split_classes, stratified_keep)
from bozkir import presets as presets_mod  # noqa: E402
from bozkir.presets import add_preset_args, apply as apply_preset  # noqa: E402
from bozkir.scene import add_scene_args, scene_from_args  # noqa: E402
from bozkir.wang import (build_tile_set, minimal_codes, layout, check_layout,  # noqa: E402
                         edge_gaussians)
from bozkir.pack import pack, STRIDE  # noqa: E402
from bozkir.catalogue import rebuild as reindex  # noqa: E402


def overlap_report(chosen, size):
    """Warn when the patches of one class share ground.

    Two patches that overlap are, in their shared part, the same ground -
    and an edge colour built from each is a shifted copy of the other. The
    variety more colours were meant to add is then not there, and a
    distinctive feature in the shared part turns up in tile after tile. It
    happens whenever a material covers only a small part of the capture:
    its patches have nowhere else to come from.
    """
    pts = [c[1] for c in chosen]
    pairs, worst = 0, 0.0
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            dx = max(0.0, size - abs(pts[a][0] - pts[b][0]))
            dy = max(0.0, size - abs(pts[a][1] - pts[b][1]))
            share = dx * dy / (size * size)
            if share > 0.01:
                pairs += 1
                worst = max(worst, share)
    if not pairs:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    w = max(xs) - min(xs) + size
    h = max(ys) - min(ys) + size
    print(f"  ! {len(pts)} patches from a {w:.1f} x {h:.1f} area; {pairs} "
          f"pair(s) share ground, up to {worst:.0%}. Overlapping patches are "
          f"shifted copies of one another, so expect repetition in this "
          f"material. A capture with more of it is the fix; fewer colours "
          f"(--colours 2) needs fewer patches.")


def scale_report(s, size):
    """Say so when the tile size and the capture's units disagree.

    --size is in the capture's own units. A drone survey is in metres,
    where a 1.5 tile holds about fifty splats across; a capture trained at
    an arbitrary scale can make 1.5 mean a fingernail or a football pitch,
    and the search then fails in ways that look like the capture's fault.
    """
    m = float(np.median(s.scale.max(axis=1)))
    if m <= 0:
        return
    ratio = size / m
    if 12 <= ratio <= 400:
        return
    guess = 55 * m
    print(f"  ! --size {size:g} is {ratio:.0f}x the typical splat "
          f"({m:.3g}); ground captures in metres sit around 30-150x. If this "
          f"capture is not in metres, try --size {guess:.3g}.")


def prepare_patches(chosen, need, args, up):
    """Cut, level and centre chosen candidates into tile-ready patches.

    Lifted out of main so it can run once per class. Nothing in it is
    class-aware - it is the same work every set of four needs, and
    calling it twice is what makes two tile sets from one capture.
    """
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
            print(f"    patch {i} at ({x:.2f}, {y:.2f}): {len(q):,} splats "
                  f"capped to {args.max_per_tile:,}")
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
    return patches



class _ShortClass(Exception):
    """A class came back with fewer patches than the colours need."""


def _progress_line():
    """One rewriting line of progress: a long search that prints nothing
    looks the same as one that has hung."""
    width = [0]

    def progress(done, total, viable):
        msg = (f"\r  {done}/{total} positions, {viable} viable"
               f"   {100.0 * done / max(total, 1):4.0f}%")
        width[0] = max(width[0], len(msg))
        sys.stdout.write(msg.ljust(width[0]))
        sys.stdout.flush()
        if done == total:
            sys.stdout.write("\r" + " " * width[0] + "\r")
            sys.stdout.flush()
    return progress

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--size", type=float, default=1.5)
    ap.add_argument("--colours", type=int, default=2,
                    help="edge colours per axis; 2 gives 16 tiles from 4 patches")
    ap.add_argument("--minimal", action="store_true",
                    help="build two tiles per (west, south) pair instead of "
                         "every combination (Cohen et al. 2003): 18 tiles at "
                         "3 colours instead of 81. On by default from 3 "
                         "colours up; the variety that fights repetition "
                         "comes from the colours, not the tile count")
    ap.add_argument("--complete", action="store_true",
                    help="build every combination even at 3+ colours")
    ap.add_argument("--blend", type=float, default=0.05,
                    help="width of the feathered band along each diagonal, "
                         "in world units; 0 is a hard cut. Ignored with --cut.")
    ap.add_argument("--cut", action=argparse.BooleanOptionalAction, default=True,
                    help="(on by default; --no-cut for speed) place each diagonal by graph cut instead of leaving "
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
    ap.add_argument("--overhang-splats", type=float, default=0.0,
                    help="how far past the square a tile may reach, in splat "
                         "widths. Zero is the default and means no two tiles "
                         "ever cover the same ground, which is what stops the "
                         "shared strip swapping as the camera turns. It costs "
                         "about a tenth of a percent of coverage at the join, "
                         "because Gaussians still spread over it from their "
                         "own side.")
    ap.add_argument("--extract-margin", type=float, default=0.35,
                    help="cut candidates this much larger than the tile, so "
                         "levelling has material to rotate in from")
    ap.add_argument("--count", type=int, default=0,
                    help="how many candidates to search for before "
                         "choosing. Defaults to four per tile per class. "
                         "The same flag preview_patches takes, so a preset "
                         "saved there carries over")
    ap.add_argument("--exclude", default="",
                    help="candidate indices to drop before choosing, as "
                         "0,3,7. For the ones a score cannot see: a survey "
                         "marker, a bright stone, anything the eye locks "
                         "onto. A landmark inside a tile appears at the "
                         "same spot in every copy of that tile, everywhere")
    ap.add_argument("--classes", type=int, default=1,
                    help="build this many tile sets, one per material found "
                         "in the capture, all sharing the same edge codes so "
                         "any cell can take any class without breaking the "
                         "matching. 1 is the single-material behaviour")
    ap.add_argument("--auto", dest="auto", action="store_true", default=True,
                    help="when the settings given find too few patches, "
                         "loosen whichever filter is doing the rejecting "
                         "and say so. On by default; ignored when --patches "
                         "names the candidates, since those indices depend "
                         "on the settings")
    ap.add_argument("--no-auto", dest="auto", action="store_false",
                    help="use the settings exactly as given")
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
    ap.add_argument("--max-per-tile", type=int, default=60000,
                    help="most splats kept per patch (default 60000; 0 keeps "
                         "all). A dense capture otherwise makes tiles of "
                         "300k+ splats each and a tileset the browser cannot "
                         "draw at speed")
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
    scale_report(s, args.size)
    up = args.up_axis
    thickness = args.thickness if args.thickness else args.size * 0.25
    need = 2 * args.colours

    print(f"  need {need} patches for {args.colours} colours per axis")
    # Search widely, then narrow on appearance: a patch that scores well but
    # looks nothing like the others produces a tile with a visible X in it.
    # Four times what is needed, so appearance has something to choose
    # among - and that much again per class, because a class can only be
    # found if its material is in the pool. Searching for sixteen on a
    # capture whose second material is a tenth of the ground finds sixteen
    # of the first material and splits it into two halves of itself.
    # Four times what is needed, so appearance has something to choose
    # among - and that much again per class, because a class can only be
    # found if its material is in the pool. Searching for sixteen on a
    # capture whose second material is a tenth of the ground finds sixteen
    # of the first material and splits it into two halves of itself.
    want = args.count or max(need * 4 * max(args.classes, 1), 12)
    def gather(want, widen=False):
        common = search_kwargs(args, thickness)
        if widen:
            # More of what was already found, not a new search. The first
            # pass keeps every viable position it saw; only the top few were
            # handed on, and a rarer material is exactly what sits further
            # down that list. Loosening the filters instead - which is what
            # asking auto_pick for more would do - re-searches the whole
            # capture once per setting, minutes each, for candidates that
            # were never missing.
            common.pop("min_separation", None)
            found, _, _ = cached_search(s, args.size, up, common,
                                        cache=args.search_cache, verbose=False)
            chosen = choose(found, want, args.size, 0.5)
            print(f"  taking {len(chosen)} of the {len(found)} viable "
                  f"positions already found")
        elif args.auto and not args.patches:
            # Loosen only when choosing for ourselves. With --patches the
            # indices came from a preview run, and they only mean anything
            # against the settings that produced them - so those settings are
            # used exactly as given, and a short list is an error rather than
            # something to work around.
            chosen, trail = auto_pick(s, args.size, want, up,
                                      cache=args.search_cache,
                                      progress=_progress_line(), **common)
            print(describe_trail(trail, need, args.size))
            apply_settings(args, trail[-1][1])
            if args.save_preset:
                presets_mod.save(args, args.save_preset, ap)
        else:
            sep = common.pop("min_separation", 1.0)
            found, _, hit = cached_search(s, args.size, up, common,
                                          cache=args.search_cache, verbose=True)
            chosen = choose(found, want, args.size, sep)
            for _, _, p, info in chosen:
                info["cover"] = rendered_coverage(p, args.size, up)
            if hit:
                print(f"  recalled {len(found)} candidates from the search "
                      f"preview_patches.py already did")
        if len(chosen) < need:
            raise SystemExit(
                f"  only {len(chosen)} patches passed, need {need}."
                + ("\n  --patches was given, so the settings were used exactly "
                   "as supplied; re-run preview_patches.py and pass the same "
                   "settings, or a preset, to both." if args.patches else
                   "\n  This capture does not yield a tile set at any setting "
                   "tried. preview_patches.py shows what is rejecting."))
        # Dropped by hand, before anything narrows the list. Scoring sees flat,
        # dense and uniform; it has no term for "contains one bright thing", and
        # a survey marker is flat, dense and uniform. Doing this after selection
        # would be excluding from four, which is not a choice.
        if args.exclude.strip():
            drop = {int(x) for x in args.exclude.replace(",", " ").split()}
            before = len(chosen)
            chosen = [c for i, c in enumerate(chosen) if i not in drop]
            print(f"  excluded {sorted(drop)}: "
                  f"{before} candidates -> {len(chosen)}")

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
        elif args.classes <= 1:
            # Narrowing to the four that look most alike is the right move for
            # one tile set and exactly wrong for several: it would hand the
            # split four patches already chosen for being indistinguishable.
            # With classes, split_classes does the same narrowing inside each
            # group instead.
            chosen = select_similar(chosen, need, args.similarity)

        if args.classes <= 1:
            feats = np.array([appearance(c[2]) for c in chosen])
            spread = float(np.linalg.norm(feats - feats.mean(axis=0), axis=1).mean())
            print(f"  appearance spread across the {need} chosen: {spread:.3f} "
                  f"({'similar' if spread < 0.08 else 'diagonals will show'})")

        # One set of four patches per class. With a single class this is the
        # chosen four and nothing changes; with more, the candidates are split
        # into groups that look unlike each other and each group becomes its own
        # tile set.
        if args.classes > 1:
            groups = split_classes(chosen, classes=args.classes, per_class=need)
            sep = class_separation(groups)
            print(f"\n  split into {len(groups)} classes, separation {sep:.2f}")
            # A split of one material into two still finds some gap: noise
            # divided in half always has two halves. On a uniform capture
            # that came out at 1.1, with both "classes" the same colour to
            # two decimals; the desert's real sand and scrub score 7-16.
            # So the bar is a clear ratio and a colour difference an eye
            # would call a different material.
            rgb = [np.mean([appearance(c[2])[:3] for c in g], axis=0)
                   for g in groups if g]
            colour_gap = (float(np.abs(rgb[0] - rgb[1]).max())
                          if len(rgb) > 1 else 0.0)
            if sep < 3.0 or colour_gap < 0.04:
                raise SystemExit(
                    "  the classes are no further apart than they are varied: "
                    "this capture holds one material.\n"
                    "  Building tile sets from it would give two that differ by "
                    "nothing, and a terrain rule with nothing to choose between.")
            short = [i for i, g in enumerate(groups) if len(g) < need]
            if short:
                raise _ShortClass(short)
            class_sets = [g[:need] for g in groups]
        else:
            class_sets = [chosen[:need]]

        return chosen, class_sets

    # A rarer material may not make the first pool at all: search for 48 on
    # ground that is mostly sand and the scrub class comes back short. When
    # choosing for ourselves, widen the search and try again rather than
    # stop and ask the user to guess a bigger --count.
    # A count, from the command line or a preset, is where the search
    # starts, not a promise to stop there: a preset saved for two colours
    # asks for too few candidates for three.
    auto_widen = args.auto and not args.patches
    for attempt in range(4 if auto_widen else 1):
        try:
            chosen, class_sets = gather(want, widen=attempt > 0)
            break
        except _ShortClass as e:
            if attempt == (3 if auto_widen else 0):
                raise SystemExit(
                    f"  class {e.args[0]} still has fewer than {need} patches "
                    f"after searching for {want}. This capture holds too little "
                    f"of that material for {args.colours} colours per axis: "
                    f"try --colours 2, or drop --classes.")
            want *= 2
            print(f"\n  class {e.args[0]} came back short; searching wider, "
                  f"for {want} candidates")

    all_tiles, all_codes, all_class = [], [], []
    for ci, picked in enumerate(class_sets):
        if len(class_sets) > 1:
            f = np.array([appearance(c[2]) for c in picked])
            print(f"\n  class {ci}: rgb {f[:, 0].mean():.2f} "
                  f"{f[:, 1].mean():.2f} {f[:, 2].mean():.2f}")
        overlap_report(picked[:need], args.size)
        patches = prepare_patches(picked, need, args, up)

        # Which patches supply which axis is not arbitrary: put the odd one
        # out on one axis and every boundary running that way is made of
        # different material from the ones running across it.
        (h_patches, v_patches), axis_gap = balanced_split(patches, args.colours)
        naive = float(np.linalg.norm(
            np.mean([appearance(p) for p in patches[:args.colours]], axis=0)
            - np.mean([appearance(p) for p in patches[args.colours:need]],
                      axis=0)))
        print(f"\n  axis balance: {axis_gap:.3f} between the two sets "
              f"({naive:.3f} if split by score)"
              + ("  <- the grid will have a grain" if axis_gap > 0.05 else ""))

        if args.cut:
            print(f"\n  assembling tiles (graph cut, {args.cut_res}px, "
                  f"band {args.cut_band})")
        else:
            print(f"\n  assembling tiles (feathered, blend {args.blend})")
        subset = (minimal_codes(args.colours, args.colours)
                  if (args.minimal or args.colours >= 3) and not args.complete
                  else None)
        if subset is not None and ci == 0:
            print(f"  minimal set: {len(subset)} tiles per class over "
                  f"{args.colours} colours (the complete set would be "
                  f"{args.colours ** 4})")
        t, c = build_tile_set(h_patches, v_patches, args.size,
                              blend=args.blend, up_axis=up,
                              cut=args.cut, resolution=args.cut_res,
                              band=args.cut_band, codes=subset)
        all_tiles += t
        all_codes += c
        all_class += [ci] * len(t)

    # Every class is built over the same edge colours, so each code tuple
    # exists once per class. That is what lets a cell take any class without
    # consulting its neighbours: the matching constraint is satisfied by the
    # code, and the class rides along independently. A class boundary is
    # then a visible change of material with no geometric seam, which is
    # what it should be.
    tiles, codes, tile_class = all_tiles, all_codes, all_class

    # The construction is only worth anything if edges really do match.
    # Grouped by class as well as by colour. Two classes are different
    # material by construction, so their edges are not meant to be
    # identical - only tiles of the same class and colour have to match,
    # and comparing across classes would report a failure that is the
    # whole point of having classes.
    ok = True
    for edge, col in (("n", 0), ("e", 1), ("s", 2), ("w", 3)):
        groups = {}
        for t, c, k in zip(tiles, codes, tile_class):
            groups.setdefault((k, c[col]), []).append(
                edge_gaussians(t, args.size, up, edge,
                               margin=max(0.03, 0.0 if args.cut
                                          else args.blend / args.size * 1.5)))
        for g in groups.values():
            if not all(np.array_equal(g[0], x) for x in g):
                ok = False
    print(f"  edge check: "
          f"{'matching colours share an identical edge' if ok else 'FAILED'}")

    # Layout is checked on one class's codes: every class carries the same
    # set, so a layout valid for one is valid for any, and that is exactly
    # the property that lets the class be chosen per cell afterwards.
    first = [c for c, k in zip(codes, tile_class) if k == 0]
    grid = layout(first, 16, 16, seed=0)
    print(f"  layout check: {check_layout(first, grid)} mismatched edges "
          f"in 16x16")

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
            # Any Gaussian of one tile sitting inside its neighbour's half is
            # a strip both tiles draw. One of them wins, and which one flips
            # when their depths cross - for a whole row at once when the view
            # runs down it.
            #
            # Reaching over on two sides only does not help: the tile that
            # reaches still lands on ground its neighbour also covers.
            # Cutting at the square is the only arrangement where nothing
            # overlaps, and it costs almost nothing, because a Gaussian
            # centred just inside still spreads across the join from its own
            # side.
            allow = args.overhang_splats * t.scale.max(axis=1)
            out = np.abs(t.xyz[:, plane2]).max(axis=1) - args.size / 2.0
            keep = out <= allow
        trimmed.append(t.subset(keep))

    before = sum(len(t) for t in tiles)
    after = sum(len(t) for t in trimmed)
    if args.tile_overhang is not None:
        how = f"a flat {args.tile_overhang:.1%} of the tile"
    else:
        out = np.concatenate([np.abs(t.xyz[:, plane2]).max(axis=1)
                              - args.size / 2.0 for t in trimmed])
        how = ("cut at the square, so no two tiles cover the same ground"
               if args.overhang_splats <= 0 else
               f"{args.overhang_splats:g} of each Gaussian's own width")
    print(f"  overhang: {how}")
    print(f"  trimmed {before:,} -> {after:,} splats ({after / before:.0%})")
    tiles = trimmed

    parts, meta, cursor = [], [], 0
    total_levels = 0
    for t, c, k in zip(tiles, codes, tile_class):
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
                     "s": int(c[2]), "w": int(c[3]),
                     "class": int(k)})

    buf = np.concatenate(parts)
    stem = args.out.stem if args.out else args.path.stem
    dest = args.out or Path("web/data") / f"{stem}.splat"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(buf.tobytes())
    dest.with_suffix(".json").write_text(json.dumps({
        "size": args.size, "up_axis": up, "stride": STRIDE,
        "thickness": thickness, "blend": args.blend,
        "colours": args.colours, "wang": True, "lod": max(args.lod, 1),
        "classes": len(class_sets),
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
    # Keep the viewer's tileset menu honest without anyone editing a list.
    reindex(dest.parent, quiet=True)


if __name__ == "__main__":
    main()