"""Build a Wang tile set from a capture and export it for the browser.

Finds exemplar patches on the ground, optionally splits them into material
classes, builds the tiles (graph-cut seams, minimal sets, rotated patches as
extra colours), and writes one .splat plus a manifest of edge codes, classes
and LOD ranges for the viewer.

    python scripts/export_wang.py data/raw/desert.ply --preset desert3
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bozkir.patches import (appearance, purity, apply_settings, auto_pick,  # noqa: E402
                            balanced_split, cached_search, choose,
                            class_separation, describe_trail, level_patch,
                            rendered_coverage, search_kwargs,
                            select_similar, split_classes, stratified_keep,
                            rank_clean, class_shares, split_materials,
                            material_map, thickness, thickness_map, seen_points,
                            ground_level, tone_gate)
from bozkir import presets as presets_mod  # noqa: E402
from bozkir import frames as frames_mod  # noqa: E402
from bozkir.presets import add_preset_args, apply as apply_preset  # noqa: E402
from bozkir.scene import add_scene_args, scene_from_args, prepare, config_from_args  # noqa: E402
from bozkir.select import cap_height  # noqa: E402
from bozkir.wang import (build_tile_set, minimal_codes, layout, check_layout,  # noqa: E402
                         edge_gaussians)
from bozkir.pack import pack, tile_levels, STRIDE  # noqa: E402
from bozkir.tile import fill_opacity as _fill_opacity, haze_splats, rotate_patch, missing_layer, merge, extract_patch  # noqa: E402
from bozkir.catalogue import rebuild as reindex  # noqa: E402


def overlap_report(chosen, size):
    """Warn when the patches of one class share ground: their colours are
    then shifted copies, and anything distinctive in the shared part repeats
    in tile after tile. Happens when a material covers little of the capture
    (the desert scrub is ~4 x 5 m)."""
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


def apart(cands, n, size):
    """The first `n` of `cands` that share no ground with one another, in
    order; overlapping ones fill in only when too few are apart."""
    picked = []
    for c in cands:
        if all(abs(c[1][0] - p[1][0]) >= size or abs(c[1][1] - p[1][1]) >= size
               for p in picked):
            picked.append(c)
        if len(picked) == n:
            return picked
    ids = {id(p) for p in picked}
    return picked + [c for c in cands if id(c) not in ids][:n - len(picked)]


def apart_alike(cands, n, size, apart_too=True, given=1, only_apart=False, accept=None):
    """`n` of `cands`: the first `given`, then each time the one that
    looks most like those already chosen (appearance), among those sharing
    no ground with them when `apart_too` and there are any. Patches that
    look alike hide the triangle each tile is made of; ones that differ
    show it. `only_apart`: stop when none is left apart, rather than take
    an overlapping one (the most alike is then a shifted copy). `accept`:
    a further test, made only on the ones about to be taken (it may be
    slow); those failing it are passed over."""
    if not cands:
        return []
    feats = [appearance(c[2]) for c in cands]
    ok = accept or (lambda c: True)
    dropped = set()
    picked = []
    for i in range(len(cands)):
        if len(picked) >= min(given, len(cands)):
            break
        if ok(cands[i]):
            picked.append(i)
        else:
            dropped.add(i)
    if not picked:
        return []
    while len(picked) < n:
        ref = np.mean([feats[i] for i in picked], axis=0)
        rest = [i for i in range(len(cands)) if i not in picked and i not in dropped]
        if not rest:
            break
        free = [i for i in rest if all(
            abs(cands[i][1][0] - cands[j][1][0]) >= size
            or abs(cands[i][1][1] - cands[j][1][1]) >= size for j in picked)]
        if only_apart and not free:
            break
        pool = free if (apart_too and free) else rest
        if apart_too and not free:
            # Nothing apart left: the least overlap, not the most alike
            # (that is a shifted copy; desert10 took 1.0 and 1.5 a step apart).
            share = [max(max(0.0, size - abs(cands[i][1][0] - cands[j][1][0]))
                         * max(0.0, size - abs(cands[i][1][1] - cands[j][1][1]))
                         for j in picked) for i in rest]
            pool = [i for i, v in zip(rest, share) if v <= min(share) + 1e-9]
        best = min(pool, key=lambda i: float(np.linalg.norm(feats[i] - ref)))
        if ok(cands[best]):
            picked.append(best)
        else:
            dropped.add(best)
    return [cands[i] for i in picked]


def haze(c, up=2):
    """A candidate's splat layer thickness (m), measured once."""
    if "haze" not in c[3]:
        c[3]["haze"] = thickness(c[2], up)
    return c[3]["haze"]


def weigh_shares(shares, thick, weight):
    """The share each material gets in the viewer: its share of the ground
    times (thinnest layer / its layer) ** weight, so a material the capture
    holds only as haze shows up less without being dropped. 0 keeps the
    ground's shares."""
    ref = max(min(thick), 1e-4)
    w = [sh * (ref / max(t, 1e-4)) ** weight for sh, t in zip(shares, thick)]
    return [v / sum(w) for v in w]


def class_reasons(materials, shares, size, need, min_share, gap=0.04):
    """Why each material cannot be a class of its own (empty when it can):
    it must be a real part of the ground (`min_share` of it), have its own
    colour (mean rgb `gap` from the nearest other) and give `need` patches
    on ground of their own."""
    rgb = [np.mean([appearance(c[2])[:3] for c in g], axis=0) for g in materials]
    out = []
    for i, g in enumerate(materials):
        why = []
        if shares[i] < min_share:
            why.append(f"only {shares[i]:.0%} of the ground")
        near = min((float(np.abs(rgb[i] - rgb[j]).max()) for j in range(len(rgb)) if j != i),
                   default=1.0)
        if near < gap:
            why.append(f"colour only {near:.2f} from another")
        room = []
        for c in g:
            if all(abs(c[1][0] - x) >= size or abs(c[1][1] - y) >= size for x, y in room):
                room.append(c[1])
        if len(room) < need:
            why.append(f"only {len(room)} patches apart, {need} needed")
        out.append(why)
    return out


def choose_classes(s, args, up, thickness_cm, need):
    """How many materials to make tiles for, and why: the most (2 to 4)
    for which every one passes class_reasons. 1 when none does."""
    every = search_kwargs(args, thickness_cm)
    sep = min(every.pop("min_separation", 1.0), 0.25)
    found, st, _ = cached_search(s, args.size, up, every, cache=args.search_cache,
                                 where=True)
    seen = seen_points(found, st) if args.seen_shares else None
    pos = choose(found, len(found), args.size, sep)
    least = args.min_share or 0.1
    print(f"\n  how many materials? each must be {least:.0%} of the ground or more, "
          f"have its own colour and give {need} patches apart:")
    best = 1
    for k in (2, 3, 4):
        mats = split_materials(pos, k)
        sh = class_shares(s, mats, args.size, up, seen=seen)
        why = class_reasons(mats, sh, args.size, need, least)
        bad = [f"material {i} ({sh[i]:.0%}): {', '.join(w)}" for i, w in enumerate(why) if w]
        print(f"    {k}: " + (" / ".join(f"{v:.0%}" for v in sorted(sh, reverse=True))
                              + "  ok" if not bad else "no - " + "; ".join(bad)))
        if not bad:
            best = k
    print(f"  -> {best} material{'s' if best > 1 else ''}")
    return best


def colour_alike(reference):
    """A test: is a candidate's mean colour no further from the `reference`
    patches' mean than the furthest of them, plus 0.04 (a visible step)?
    The material map judges by colour and texture over a whole tile, so the
    capture's dark, hazy rim counts as bush there; this is what keeps it out
    of the bush's patches."""
    if not reference:
        return lambda c: True
    rgb = lambda c: appearance(c[2])[:3]
    mid = np.mean([rgb(e) for e in reference], axis=0)
    reach = max(float(np.abs(rgb(e) - mid).max()) for e in reference) + 0.04
    return lambda c: float(np.abs(rgb(c) - mid).max()) <= reach


def pick_centres(edge, material, count, size, up=2, accept=None):
    """Up to `count` centre patches like the `edge` ones, each on ground of
    its own: a centre's core (the inner 0.7 of the tile, inside the seam
    band) shares none with another's or an edge patch's. Positions a
    quarter tile apart are mostly the same ground shifted, and alike-first
    took exactly those: on desert v1 one material's 18 centres came from a
    3 x 2.4 m spot and the same tuft showed in tile after tile. None
    thinner than the thinnest edge patch, which it sits between: a blurrier
    centre shows as a square. Nor further in colour from the edges' mean
    than the furthest edge patch, plus 0.04: on desert10 the bush took
    brown sand from the capture's rim (0.41 against 0.22) once its own
    ground ran out, and every bush tile had a sand square in it."""
    floor = min(len(e[2]) for e in edge) if edge else 0
    # Nor a thicker haze of splats than the edges have.
    ceiling = max(haze(e, up) for e in edge) if edge else 1e9
    alike = colour_alike(edge)
    spare = sorted((c for c in material if len(c[2]) >= floor and haze(c, up) <= ceiling
                    and alike(c) and all(c is not e for e in edge)),
                   key=lambda c: -len(c[2]))
    return apart_alike(list(edge) + spare, len(edge) + count, 0.7 * size,
                       given=len(edge), only_apart=True, accept=accept)[len(edge):]


_LUM = {}
_T0 = [time.perf_counter()] * 2         # when the run started, and the last step


def took(what):
    """How long the last step took and the run so far, printed: a long run
    says where its time went."""
    now = time.perf_counter()
    print(f"  [{what}: {now - _T0[1]:.0f} s, {now - _T0[0]:.0f} s in all]", flush=True)
    _T0[1] = now


def lum_of(c, ground, size, up=2):
    """A candidate from above as (luminance, coverage), a pixel being 2.5 cm
    of the finished tile; measured once per spot and cut."""
    key = (round(float(c[1][0]), 4), round(float(c[1][1]), 4), round(ground, 4), len(c[2]))
    if key not in _LUM:
        _LUM[key] = frames_mod.luminance(c[2], ground, up, px=0.025 * ground / size)
    return _LUM[key]


def quiet_edges(used, cands, n, ground, inner, size, up=2, accept=None, diamonds=False,
                shape=2.0):
    """The `n` edge patches with the spottiest frames (salient spots outside
    `inner` of the tile, frames.salience) swapped, worst first, for clearly
    quieter ones from the same candidates: a frame repeats in every tile
    with its edge colour, so a spot in it shows at one place tile after
    tile. A swap keeps the patches on ground of their own, alike in colour
    and passing `accept`; one at a time, so a material with just enough
    ground still gets some (taking a quiet subset first left desert v1 with
    too few apart, every time). Returns (patches, the decision)."""
    used = list(used[:n])
    pool = list({id(c): c for c in used + list(cands)}.values())
    if len(pool) <= n:
        return used, "quiet edges: no (no spare patches to choose from)"
    imgs = [lum_of(c, ground, size, up) for c in pool]
    ring = frames_mod.shown(imgs[0][0].shape[0], ground, inner * ground / size, diamonds, shape)
    sal = frames_mod.salience(imgs, [ring & (w > 0.5) for _, w in imgs])
    of = {id(c): v for c, v in zip(pool, sal)}
    before = float(np.mean([of[id(c)] for c in used]))
    ok = accept or (lambda c: True)
    alike = colour_alike(used)
    spare = sorted((c for c in pool if all(c is not u for u in used)), key=lambda c: of[id(c)])
    swaps = 0
    for worst in sorted(used, key=lambda c: -of[id(c)]):
        k = next(i for i, u in enumerate(used) if u is worst)
        tries = 0                                      # hole checks are renders
        for c in spare:
            if of[id(c)] > 0.7 * of[id(worst)] or tries >= 3:
                break                                  # none clearly quieter left
            if all(abs(c[1][0] - u[1][0]) >= ground or abs(c[1][1] - u[1][1]) >= ground
                   for i, u in enumerate(used) if i != k) and alike(c):
                tries += 1
                if ok(c):
                    used[k] = c
                    spare.remove(c)
                    swaps += 1
                    break
    if not swaps:
        return used, (f"quiet edges: no (no clearly quieter patch on ground of its own; "
                      f"salient spots on {before:.1%} of the frames)")
    after = float(np.mean([of[id(c)] for c in used]))
    return used, (f"quiet edges: yes ({swaps} of {n} swapped for quieter: salient spots "
                  f"on {after:.1%} of the frames, {before:.1%} before)")


def sharpness(c, ground, size, geo, up=2, whole=False):
    """How sharp a candidate looks where it is used: (splats per m2, median
    footprint in m) over its centre's region, or with `whole` its whole
    square (an edge patch, which shows all over the tiles). A splat count
    over the whole cut stood in for this before; it says more about the
    cut than about the ground."""
    key = ("sharp", whole)
    if key not in c[3]:
        k = ground / size
        plane = [i for i in range(3) if i != up]
        xy = c[2].xyz[:, plane].astype(np.float64)
        shape = geo.get("shape", 2.0)
        r = frames_mod.norm(xy[:, 0], xy[:, 1], shape)
        inside = np.abs(xy).max(axis=1) <= ground / 2.0
        lo, hi = (0.0, np.inf) if whole else (0.0, geo["outer"] * k)
        sel = inside & (r >= lo) & (r <= hi)
        area = float(frames_mod.zone(64, ground, lo, hi, shape).mean()) * ground * ground
        sc = np.sort(np.asarray(c[2].scale, np.float64)[sel], axis=1)
        foot = float(np.median(np.sqrt(sc[:, 1] * sc[:, 2]))) if len(sc) else np.inf
        c[3][key] = (float(sel.sum()) / max(area, 1e-9), foot)
    return c[3][key]


def pick_pool(edge, material, count, size, ground, geo, up=2, accept=None,
              middles=True, share=0.0, diamonds=False, by_sharpness=False):
    """At most `count` centres for --frames: each edge patch's own middle
    first (its frame never shows it; not with --diamonds, where the middle
    lies on the edge), then discs on ground of their own, clear of every
    edge patch and of one another, the most like the edges first. As
    pick_centres: none with fewer splats or more haze than the edges, nor
    further in colour. With `share` above 0 (--pool-share), where ground of
    its own runs out, centres of which up to that share also shows in a
    frame (frames.shared_with_frames): a material with little ground (a
    bush) gets some variety, its centres turned and away from where the
    frames show the same ground. With `by_sharpness` (--pool-sharpness),
    "thinner than the edges" is measured where it shows (sharpness): none
    sparser, nor with larger splats, than the edge patches, in place of the
    splat count over the whole cut. Returns (centres, edge middles, the
    share of each of the others, why the rest were not taken: a count
    each)."""
    radius = geo["radius"] * ground / size
    reach = geo.get("reach", geo["radius"]) * ground / size     # its corner, if square
    square = geo.get("shape", 2.0) != 2.0
    ok = accept or (lambda c: True)
    pool = [e for e in edge if ok(e)][:count] if middles else []
    mids = len(pool)
    if not edge:
        return pool, mids, [], {}
    floor = min(len(e[2]) for e in edge)
    ceiling = max(haze(e, up) for e in edge)
    alike = colour_alike(edge)
    ref = np.mean([appearance(e[2]) for e in edge], axis=0)
    ids = {id(e) for e in edge}
    others = [c for c in material if id(c) not in ids]
    why = {"candidates": len(others)}
    if by_sharpness:
        # Blur is splat size; how many splats there are is the hole check's
        # business (requiring the edges' density as well left 15 of 53).
        fine = max(sharpness(e, ground, size, geo, up, whole=True)[1] for e in edge)
        thin = [c for c in others if sharpness(c, ground, size, geo, up)[1] > fine]
        why[f"with larger splats than the edges ({fine * 100:.1f} cm)"] = len(thin)
    else:
        thin = [c for c in others if len(c[2]) < floor]
        why[f"with fewer splats than the thinnest edge ({floor:,})"] = len(thin)
    thin_ids = {id(c) for c in thin}
    hazy = [c for c in others if id(c) not in thin_ids and haze(c, up) > ceiling]
    spare = sorted((c for c in others if id(c) not in thin_ids
                    and haze(c, up) <= ceiling and alike(c)),
                   key=lambda c: float(np.linalg.norm(appearance(c[2]) - ref)))
    why["hazier than the edges"] = len(hazy)
    why["further in colour than the edges"] = len(others) - len(thin) - len(hazy) - len(spare)
    shared = []
    if share > 0:
        # Ground of its own first, then the least shared; alike within each.
        of = {id(c): frames_mod.shared_with_frames(c[1], [e[1] for e in edge], ground,
                                                  geo, size, diamonds) for c in spare}
        kept = [c for c in spare if of[id(c)] <= 0.02] + sorted(
            (c for c in spare if 0.02 < of[id(c)] <= share), key=lambda c: of[id(c)])
        why["on the frames' ground"] = len(spare) - len(kept)
        spare = kept
    for k in ("too close to a centre taken", "with holes", "not tried"):
        why[k] = 0
    tries = 0
    for n, c in enumerate(spare):
        if len(pool) >= count:
            break
        if tries >= 3 * count:                           # hole checks are renders
            why["not tried"] = len(spare) - n
            break
        dx = [abs(c[1][0] - p[1][0]) for p in pool]
        dy = [abs(c[1][1] - p[1][1]) for p in pool]
        if any((max(a, b) if square else np.hypot(a, b)) < 2 * radius for a, b in zip(dx, dy)):
            why["too close to a centre taken"] += 1
            continue
        if share > 0 or all(frames_mod.clear_of_frame(c[1], e[1], ground, reach,
                                                       0.05 if middles else None)
                                for e in edge):
            tries += 1
            if ok(c):
                pool.append(c)
                shared.append(of[id(c)] if share > 0 else 0.0)
            else:
                why["with holes"] += 1
        else:
            why["on the frames' ground"] = why.get("on the frames' ground", 0) + 1
    return pool, mids, shared, why


def height_order(groups, up=2, radius=3.0):
    """How strictly the materials, in order (hollows first), follow the
    ground's height in the capture. Each position's ground (its splats'
    lower quartile) against the mean of the positions within `radius`;
    for each two materials in a row, the share of position pairs the wrong
    way round (Mann-Whitney's U over all pairs, ties half). 0: strictly by
    height; 0.5: no relation. None with nothing to compare."""
    from scipy.spatial import cKDTree
    from scipy.stats import rankdata
    xy = np.array([c[1][:2] for g in groups for c in g], np.float64)
    z = np.array([np.percentile(c[2].xyz[:, up], 25) if len(c[2]) else np.nan
                  for g in groups for c in g])
    lab = np.concatenate([[k] * len(g) for k, g in enumerate(groups)]) if len(xy) else []
    ok = np.isfinite(z)
    if ok.sum() < 4:
        return None
    tree = cKDTree(xy[ok])
    rel = np.full(len(z), np.nan)
    rel[ok] = z[ok] - np.array([z[ok][nb].mean() for nb in
                                tree.query_ball_point(xy[ok], radius)])
    wrong = total = 0.0
    for k in range(len(groups) - 1):
        a, b = rel[(lab == k) & ok], rel[(lab == k + 1) & ok]
        if not len(a) or not len(b):
            continue
        r = rankdata(np.concatenate([a, b]))[:len(a)]
        wrong += r.sum() - len(a) * (len(a) + 1) / 2.0        # pairs with a above b
        total += len(a) * len(b)
    return float(wrong / total) if total else None


MATERIAL_RGB = ("#f0a830", "#30b0e0", "#e040a0", "#70d050", "#a070f0")


def look(s, materials, edges, centres, info, size, up, dest, name=""):
    """What the export takes from the capture, in one picture for the
    README and for checking a build: the capture from above with the
    patches used (solid: edge patches k.j, dashed: centre cores, x: left out
    for an odd spot); the material each spot looks like (dots: positions
    the search could use); and how thick the layer of splats is, the
    measure of what looks sharp up close."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Patch, Rectangle
    from scipy.ndimage import gaussian_filter

    plane = [i for i in range(3) if i != up]
    small = np.flatnonzero(s.scale.max(axis=1) < 0.1)   # huge faint ones wash it out
    xy = s.xyz[small][:, plane].astype(np.float64)
    px = 0.03
    lo, hi = np.percentile(xy, 0.5, axis=0), np.percentile(xy, 99.5, axis=0)
    w, h = (np.ceil((hi - lo) / px)).astype(int) + 1
    ij = np.floor((xy - lo) / px).astype(int)
    ok = np.all((ij >= 0) & (ij < [w, h]), axis=1)
    key = ij[ok, 1] * w + ij[ok, 0]
    a = s.opacity[small][ok].astype(np.float64)
    rgb = s.base_rgb[small][ok]
    cov = gaussian_filter(np.bincount(key, a, w * h).reshape(h, w), 1.0)
    img = np.stack([gaussian_filter(np.bincount(key, a * rgb[:, c], w * h).reshape(h, w), 1.0)
                    for c in range(3)], axis=-1) / np.maximum(cov, 1e-9)[..., None]
    extent = (lo[0], lo[0] + w * px, lo[1], lo[1] + h * px)
    grey = np.repeat(img.mean(axis=-1, keepdims=True), 3, axis=-1)

    labels, (mx, my) = material_map(s, materials, size, up, seen=info.get("seen"))
    tint = np.zeros(labels.shape + (4,))
    for k in range(len(materials)):
        tint[labels == k] = matplotlib.colors.to_rgba(MATERIAL_RGB[k % 5], 0.55)
    layer, (tx, ty) = thickness_map(s.subset(small), up)

    fig, axes = plt.subplots(1, 3, figsize=(18, 18 * h / (3 * w) + 3.2))
    for ax, base in zip(axes, (img, grey, grey)):
        ax.imshow(np.clip(base, 0, 1), origin="lower", extent=extent)
        ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
        ax.tick_params(labelsize=7)
    axes[1].imshow(tint, origin="lower", interpolation="nearest",
                   extent=(mx, mx + labels.shape[1] * 0.5, my, my + labels.shape[0] * 0.5))
    viable = info.get("viable", ())
    if len(viable):
        axes[1].plot([p[0] for p in viable], [p[1] for p in viable], ".", color="k", ms=2)
    # Why the search could not use the rest: what keeps patches out of
    # sharp ground.
    why = {}
    for x, y, r in info.get("rejected_at", []):
        why.setdefault(r, []).append((x, y))
    marks = {"sparse": ("s", "#9a9a9a", "too few splats"),
             "holes": ("s", "#e03030", "too many holes"),
             "gaps": ("s", "#e03030", "real gaps (no splats to fill)"),
             "faint": ("s", "#f09030", "faint, a full fill would cover it"),
             "tilt": ("^", "#3060f0", "too steep"),
             "buried": ("v", "#8a5a20", "ground buried"),
             "score": ("o", "#f0e040", "low score"),
             "rim": ("D", "#30e0e0", "rim not flat")}
    for r, pts in sorted(why.items(), key=lambda kv: -len(kv[1])):
        shape, colour, text = marks.get(r, ("o", "w", r))
        axes[2].plot([p[0] for p in pts], [p[1] for p in pts], shape, color=colour, ms=3,
                     mec="k", mew=0.3, label=f"{len(pts)} {text}")
    if why:
        print("  positions the search could not use: " + ", ".join(
            f"{len(v)} {marks.get(r, (0, 0, r))[2]}"
            for r, v in sorted(why.items(), key=lambda kv: -len(kv[1])))
            + f"; {len(viable)} usable")
        axes[2].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.06),
                       frameon=False, ncol=3)
    im = axes[2].imshow(gaussian_filter(np.nan_to_num(layer * 100, nan=0.0), 1.0),
                        origin="lower", interpolation="nearest", cmap="magma",
                        vmin=0, vmax=10, alpha=0.85,
                        extent=(tx, tx + layer.shape[1] * 0.1, ty, ty + layer.shape[0] * 0.1))
    bar = fig.colorbar(im, ax=axes[2], fraction=0.04, pad=0.02)
    bar.set_label("cm", fontsize=7)
    bar.ax.tick_params(labelsize=7)
    cut = info.get("ground") or [size] * len(materials)
    for ax in axes[:2]:
        for k, g in enumerate(edges):
            for j, c in enumerate(g):
                x, y = c[1][0], c[1][1]
                d = cut[k]
                ax.add_patch(Rectangle((x - d / 2, y - d / 2), d, d, fill=False,
                                       ec=MATERIAL_RGB[k % 5], lw=1.6))
                ax.text(x, y, f"{k}.{j}", color="w", fontsize=6, ha="center", va="center")
        disc = info.get("centre_radius")
        for k, g in enumerate(centres):
            for c in g:
                if disc:
                    p = info.get("centre_shape", 2.0)
                    t = np.linspace(0, 2 * np.pi, 97)
                    cs, sn = np.cos(t), np.sin(t)
                    f = disc[k] / frames_mod.norm(cs, sn, p)
                    ax.plot(c[1][0] + f * cs, c[1][1] + f * sn, color=MATERIAL_RGB[k % 5],
                            lw=0.8, ls="--")
                    continue
                r = 0.35 * cut[k]
                ax.add_patch(Rectangle((c[1][0] - r, c[1][1] - r), 2 * r, 2 * r, fill=False,
                                       ec=MATERIAL_RGB[k % 5], lw=0.8, ls="--"))
        odd = info.get("odd", [])
        if odd:
            ax.plot([p[0] for p in odd], [p[1] for p in odd], "x", color="w", ms=4, mew=0.8)
    used, ground, thick = info.get("shares"), info.get("ground_shares"), info.get("thick")
    names = []
    for k, g in enumerate(materials):
        m = np.mean([appearance(c[2])[:3] for c in g], axis=0)
        text = f"material {k}: rgb {m[0]:.2f} {m[1]:.2f} {m[2]:.2f}"
        if ground and thick:
            text += (f", {ground[k]:.0%} of the ground, layer {thick[k] * 100:.1f} cm"
                     f" -> {used[k]:.0%} of the map")
        elif used:
            text += f", {used[k]:.0%} of the ground"
        names.append(Patch(color=MATERIAL_RGB[k % 5], label=text))
    names.append(Patch(color="none", label="no colour: too few splats"
                       + (", or seen too little by the photos" if info.get("seen") else "")))
    axes[1].legend(handles=names, fontsize=7, loc="upper center",
                   bbox_to_anchor=(0.5, -0.06), frameon=False)
    axes[0].set_title("from above: edge patches (k.j), "
                      + ("the centre pool (dashed discs)" if info.get("centre_radius")
                         else "centre cores (dashed)")
                      + ", x left out for an odd spot", fontsize=8)
    axes[1].set_title("the material each spot looks like, judged over a tile; "
                      "dots: usable positions", fontsize=8)
    axes[2].set_title("how thick the layer of splats is: thin looks sharp up close; "
                      "marks: positions the search rejected, and why", fontsize=8)
    import textwrap
    lines = [textwrap.fill(f"material {k}: " + "; ".join(ds), 200, subsequent_indent="    ")
             for k, ds in enumerate(info.get("decisions") or []) if ds]
    if lines:
        fig.text(0.01, 0.01, "decisions\n" + "\n".join(lines), fontsize=7,
                 va="bottom", ha="left", family="monospace")
    if name:
        fig.suptitle(f"{name}: what the export takes from the capture", fontsize=10)
    fig.tight_layout(rect=(0, 0.04 + 0.018 * sum(l.count("\n") + 1 for l in lines), 1, 1))
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=110)
    plt.close(fig)


def holes(p, size, up=2, res=60, min_cm=8.0, ring=None, shape=2.0):
    """See-through holes in a patch or tile from above: spots under half
    covered, at least `min_cm` across, within `ring` (r_min, r_max) of its
    middle if given (a frame, a centre). Returns (count, largest in cm^2,
    mask of the holes)."""
    from bozkir.graphcut import render_patch
    from scipy.ndimage import label
    if not len(p):
        return 1, size * size * 1e4, np.ones((res, res), bool)
    a = render_patch(p, size, res, up)[1]
    see = a < 0.5
    if ring is not None:
        see &= frames_mod.zone(res, size, *ring, shape=shape)
    lab, n = label(see)
    px = (size * 100.0 / res) ** 2
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:] * px
    big = [k + 1 for k, v in enumerate(sizes) if v >= min_cm ** 2]
    mask = np.isin(lab, big)
    return len(big), float(max((sizes[k - 1] for k in big), default=0.0)), mask


# hole_free's answers, kept in data/cache between runs: --look and the
# build after it ask about the same patches, and each answer is two renders.
_HOLED = {"path": None, "data": {}, "new": 0}


def holed_cache(s, thickness, margin, use=True):
    """Load the answers for this prepared capture and cut (none if `use` is off)."""
    import hashlib
    from bozkir.patches import scene_fingerprint
    _HOLED.update(path=None, data={}, new=0)
    if not use:
        return
    key = hashlib.sha1(f"v1|{scene_fingerprint(s)}|{thickness}|{margin}".encode()).hexdigest()
    _HOLED["path"] = Path("data/cache") / f"holes_{key[:16]}.json"
    try:
        _HOLED["data"] = json.loads(_HOLED["path"].read_text())
    except (OSError, ValueError):
        pass


def save_holed():
    if _HOLED["path"] is not None and _HOLED["new"]:
        try:
            _HOLED["path"].parent.mkdir(parents=True, exist_ok=True)
            _HOLED["path"].write_text(json.dumps(_HOLED["data"]))
            _HOLED["new"] = 0
        except OSError:
            pass


_FILL = {"spare_haze": False}       # export --fill-spare-haze


def fill_opacity(t, size, up=2, target=0.98):
    """The fill (bozkir.tile.fill_opacity), sparing large faint splats with
    --fill-spare-haze."""
    return _fill_opacity(t, size, up, target,
                         spare=haze_splats(t, size, up) if _FILL["spare_haze"] else None)


def hole_free(c, size, up=2, fill=0.0):
    """True when a candidate has no see-through hole from above once its
    opacity is filled as its tile's will be (fill_opacity). Holes passed by
    the search (faint, gappy ground) showed in every tile built from the
    patch: desert11's bush 16 of 18 tiles, mid sand 11. Measured once, and
    remembered between runs (holed_cache)."""
    if "holed" not in c[3]:
        key = (f"{c[1][0]:.4f},{c[1][1]:.4f},{size:.4f},{fill:g},{len(c[2])},{up}"
               + (",haze" if _FILL["spare_haze"] else ""))
        if _HOLED["path"] is None:
            _HOLED["data"].pop(key, None)            # no capture to remember it by
        if key not in _HOLED["data"]:
            p = c[2]
            if fill > 0:
                p = fill_opacity(p, size, up, fill)
            _HOLED["data"][key] = bool(holes(p, size, up)[0] > 0)
            _HOLED["new"] += 1
        c[3]["holed"] = _HOLED["data"][key]
    return not c[3]["holed"]


def with_neighbours(tiles, codes, tile_class, size, up=2, band=0.1):
    """Each tile with a `band` of its neighbours' ground past each edge (from
    any tile of its material with that edge colour on the other side; they
    all share that edge): a tile seen alone is thin along its edges, since
    the splats just past them belong to the neighbour, and a faint spot
    there read as a hole that the world does not have."""
    from bozkir.tile import translate
    plane = [i for i in range(3) if i != up]
    h = size / 2.0
    # (edge, its code column, the other side's column, which way, along which axis)
    sides = (("n", 0, 2, 1, 1), ("e", 1, 3, 1, 0), ("s", 2, 0, -1, 1), ("w", 3, 1, -1, 0))
    out = []
    for t, c, k in zip(tiles, codes, tile_class):
        parts = [t]
        for _, mine, theirs, sign, axis in sides:
            other = next((u for u, d, q in zip(tiles, codes, tile_class)
                          if q == k and d[theirs] == c[mine]), None)
            if other is None or not len(other):
                continue
            a = other.xyz[:, plane[axis]] * -sign            # how far in from its edge
            near = np.flatnonzero(a >= h - band)
            if len(near):
                off = np.zeros(3)
                off[plane[axis]] = sign * size
                parts.append(translate(other.subset(near), off))
        out.append(merge(*parts) if len(parts) > 1 else t)
    return out


def hole_report(tiles, codes, tile_class, size, up, dest, ring=None, shape=2.0):
    """Which finished tiles have see-through holes, per material, and a
    picture of them (holes in red). `ring`, one per tile: only there (a
    frame's band; None for a whole tile). Each tile is seen with its
    neighbours' ground just past its edges (with_neighbours)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from bozkir.graphcut import render_patch
    seen = with_neighbours(tiles, codes, tile_class, size, up)
    found = [holes(t, size, up, ring=r, shape=shape)
             for t, r in zip(seen, ring or [None] * len(tiles))]
    print("\n  holes from above (8 cm or more across, under half covered):")
    for k in sorted(set(tile_class)):
        ids = [i for i, c in enumerate(tile_class) if c == k]
        bad = [i for i in ids if found[i][0]]
        worst = max((found[i][1] for i in bad), default=0.0)
        print(f"    class {k}: {len(bad)} of {len(ids)} tiles"
              + (f", largest {worst:.0f} cm2" if bad else ""))
    bad = [i for i, f in enumerate(found) if f[0]]
    if not bad:
        return
    cols = 9
    rows = -(-len(bad) // cols)
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.8))
    ax = np.atleast_2d(ax)
    for a in ax.flat:
        a.axis("off")
    for j, i in enumerate(bad):
        rgb, al = render_patch(seen[i], size, 60, up)
        img = np.clip(rgb * al[..., None], 0, 1)
        img[found[i][2]] = [1.0, 0.1, 0.1]
        ax.flat[j].imshow(img)
        ax.flat[j].set_title(f"class {tile_class[i]} tile {i} {''.join(map(str, codes[i]))}",
                             fontsize=6)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=110)
    plt.close(fig)
    print(f"  the tiles with holes: {dest}")


def even_mean(p):
    """A patch's mean colour from above (opacity-weighted base colour)."""
    w = np.asarray(p.opacity, dtype=np.float64)[:, None]
    return (p.base_rgb * w).sum(axis=0) / max(float(w.sum()), 1e-9)


def even_out(p, level, most):
    """`p` with its mean colour moved to `level`, by at most `most` per
    channel; detail and its own light and dark spots kept. Patches of one
    material that differ in tone show as the triangles and frames of the
    tiles built from them; every tile uses the same corrected patch, so
    edges still match."""
    from bozkir.ply import SH_C0
    shift = np.clip(np.asarray(level) - even_mean(p), -most, most)
    q = p.subset(np.arange(len(p)))
    q.sh_dc = ((np.clip(p.base_rgb + shift, 0, 1) - 0.5) / SH_C0).astype(np.float32)
    return q


def dense_first(cands, at_least, keep=0.5):
    """Densest first, without the patches under `keep` of the material's
    dense (75th percentile) splat count, unless fewer than `at_least` are
    left: on desert v1 thin patches have a quarter of the splats and show
    as pale blurs, and --apart preferred them to dense ones that overlap."""
    order = sorted(cands, key=lambda c: -len(c[2]))
    if not order:
        return order
    floor = keep * float(np.percentile([len(c[2]) for c in order], 75))
    dense = [c for c in order if len(c[2]) >= floor]
    return order[:max(len(dense), at_least)]


def scale_report(s, size):
    """Warn when --size (in the capture's own units) looks wrong for the
    capture's scale, suggesting a size."""
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


def with_missing(wide, raw, centre, args, up):
    """`wide` (a candidate's wider cut, centred on `centre`) with what it
    is missing against the uncleaned capture from above (--missing-layer)."""
    plane = [j for j in range(3) if j != up]
    w = args.size * (1.0 + args.extract_margin)
    d = np.abs(raw.xyz[:, plane] - np.float32(centre)).max(axis=1)
    # Splats centred far off still paint here if they are big enough; ones
    # high above the ground (floaters, sky) do not belong to it.
    near = np.flatnonzero((d <= w / 2 + 3.0 * raw.scale.max(axis=1))
                          & (raw.xyz[:, up] <= wide.xyz[:, up].max() + 0.5))
    cap = raw.subset(near)
    cap.xyz = cap.xyz.copy()
    cap.xyz[:, plane] -= np.float32(centre)
    layer = missing_layer(wide, cap, w, up)
    print(f"    ({centre[0]:.2f}, {centre[1]:.2f}): {len(layer):,} splats "
          f"fill what the capture shows and the patch is missing")
    return merge(wide, layer) if len(layer) else wide


def prepare_patches(chosen, need, args, up, raw=None, scene=None, scale=1.0):
    """Cut, level and centre chosen candidates into tile-ready patches
    (run once per class). With `raw`, each gets its missing layer first.
    With `scale` under 1, each is cut from a square 1/scale as wide in
    `scene` and shrunk to the tile, splats and heights with it (see
    --class-scale)."""
    patches = []
    print(f"\n  {'#':>2} {'centre':>16} {'splats':>9} {'relief':>7} "
          f"{'tilt':>7} {'rim':>6} {'mid':>6} {'mean rgb':>18}")
    for i, (sc, (x, y), p, info) in enumerate(chosen[:need]):
        # Level the wider cut so rotation has material to draw in from,
        # then take the tile-sized middle of the result.
        wide = info.get("wide", p)
        if scale != 1.0 and scene is not None:
            # The slab around the ground's own plane, not a level one: a
            # 2-3 m cut on a slope lost its uphill and downhill sides to a
            # level slab, which showed as gaps (desert10s).
            th = args.thickness or args.size * 0.25
            cut = info.get("cut", args.size / scale)
            raw_w = extract_patch(scene, [x, y], cut * (1.0 + args.extract_margin), up_axis=up)
            plane = [j for j in range(3) if j != up]
            ref = wide.xyz.astype(np.float64)
            # The ground's plane, not the canopy's: fitted to all, then again
            # to the lowest third. Through the bush top, the slab cut away
            # the ground around and under it: holes in every bush tile.
            A = np.c_[ref[:, plane], np.ones(len(ref))]
            coef = np.linalg.lstsq(A, ref[:, up], rcond=None)[0]
            low = ref[:, up] - A @ coef <= np.percentile(ref[:, up] - A @ coef, 33)
            if low.sum() >= 50:
                coef = np.linalg.lstsq(A[low], ref[low, up], rcond=None)[0]
            res = raw_w.xyz[:, up] - (raw_w.xyz[:, plane] @ coef[:2] + coef[2])
            wide = raw_w.subset((res >= -0.25 * th) & (res <= th))
        if raw is not None:
            wide = with_missing(wide, raw, (x, y), args, up)
        p, tilt = level_patch(wide, up)
        # Centre on the origin so every tile is assembled in the same frame.
        plane = [j for j in range(3) if j != up]
        off = np.zeros(3, dtype=np.float32)
        off[plane] = p.xyz[:, plane].mean(axis=0)
        q = p.subset(np.arange(len(p)))
        q.xyz = (q.xyz - off).astype(np.float32)
        if scale != 1.0:
            q.xyz = (q.xyz * scale).astype(np.float32)
            q.scale = (q.scale * scale).astype(np.float32)

        # Cap after levelling (which reads the wide cut), spread evenly.
        if args.max_per_tile and len(q) > args.max_per_tile:
            print(f"    patch {i} at ({x:.2f}, {y:.2f}): {len(q):,} splats "
                  f"capped to {args.max_per_tile:,}")
            q = q.subset(stratified_keep(q, args.max_per_tile,
                                         args.size * (1.0 + args.extract_margin),
                                         up))
        # Overhang kept: a Gaussian just outside still covers ground inside.
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
    ap.add_argument("--purity", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="with --classes, prefer each class's purest patches "
                         "(least ground that looks like another class); on by "
                         "default, --no-purity to rank by the search alone")
    ap.add_argument("--rotations", type=int, default=1, choices=[1, 2, 3, 4],
                    help="make extra edge colours from real patches turned "
                         "by 90 degrees (2 = up to two turns each...). Needs "
                         "fewer distinct patches, so a rarer material still "
                         "gets variety; shadows in a sunny capture turn too")
    ap.add_argument("--flatten-classes", default="",
                    help="flatten per class (in the order the export prints "
                         "them) instead of the whole scene: '0' uses "
                         "--max-height for class 0 only; '0:1,1:4' gives each "
                         "its own height in cm (bare ground 1 cm against its "
                         "spikes, a bush 4 cm: its hazy upright splats lie "
                         "down but its layers keep their heights)")
    ap.add_argument("--half-turns", default="",
                    help="classes whose turned copies are turned by 180 "
                         "degrees, not 90, e.g. 1: a half turn keeps "
                         "branches and ripples running the same way, so they "
                         "do not cross (at most 2 turns: 0 and 180)")
    ap.add_argument("--class-rotations", default="",
                    help="--rotations per class, in the order the export "
                         "prints the classes, e.g. 1,3: real patches only for "
                         "a material with plenty of ground (turned copies "
                         "cross its ripples into an X), turns for a rare one")
    ap.add_argument("--odd-spots", type=float, default=0.0, metavar="CM",
                    help="leave out patches holding a spot at least this wide "
                         "(cm) whose colour the rest of their material does "
                         "not have: a pale hole in scrub, a marker, a lone "
                         "shrub. It would repeat in every tile. Searches "
                         "wider when too few clean patches are left (0 = off)")
    ap.add_argument("--quality-weight", type=float, default=0.0,
                    help="with --min-share, weigh each material's share in the "
                         "viewer by how thin its layer of splats is: share x "
                         "(thinnest / its own) ** this. 0.5 gives a material "
                         "held only as haze (desert v1's bush) a small part "
                         "without dropping it (0 = shares of the ground)")
    ap.add_argument("--pure-share", type=float, default=0.0,
                    help="with --min-share, take a material's patches only where "
                         "at least this much of their ground is that material on "
                         "the material map (e.g. 0.8), when it has enough such "
                         "ground; otherwise mixed ones, and the output says so. "
                         "Mixed patches put another material's triangles in "
                         "every tile (0 = off)")
    ap.add_argument("--class-scale", default="",
                    help="per material, in the order the export prints them: "
                         "cut a larger square and shrink it to the tile, e.g. "
                         "0:0.67 cuts 2.25 m for a 1.5 m tile. Splats shrink "
                         "with it, so up close the material looks as it does "
                         "from 1.5x as far, where a hazy capture tightens up "
                         "(desert v1's bush). One scale per material, so its "
                         "tiles still match; heights shrink too. 'auto': each "
                         "material whose layer of splats is over twice the "
                         "thinnest one's is shrunk by (thinnest / its own) ** "
                         "--quality-weight, no less than 0.5, if it has the "
                         "ground for it")
    ap.add_argument("--seen-shares", action="store_true",
                    help="with --min-share, count only ground the photos saw "
                         "well: not where the search found too few splats, "
                         "faint layers or gaps (desert v1's hazy rim counted "
                         "as bush and inflated it)")
    ap.add_argument("--tone-order", action="store_true",
                    help="with --min-share, order the materials darkest to "
                         "lightest instead of by share. The viewer lays its "
                         "classes from hollow to crest, so tones then meet in "
                         "order and the map shades from one to the next")
    ap.add_argument("--min-share", type=float, default=0.0,
                    help="measure how much of the capture each class covers "
                         "(written to the tileset, where the viewer starts "
                         "from it) and leave out classes under this share, "
                         "e.g. 0.1: one clump of bush gives many good patches "
                         "but is not what the place is made of (0 = off)")
    ap.add_argument("--flatten-bare", type=float, default=0.0, metavar="CM",
                    help="flatten every class of flat ground (its patches' "
                         "median relief under 15 cm) to this height, whatever "
                         "its place in the list; bush and shrubs keep theirs "
                         "(0 = off)")
    ap.add_argument("--even-patches", type=float, default=0.0, metavar="MOST",
                    help="move each patch's mean colour towards its "
                         "material's, by at most this much per channel (e.g. "
                         "0.08); detail kept. Tone differences between a "
                         "material's patches are what makes tile triangles "
                         "and frames show (0 = off)")
    ap.add_argument("--look", action="store_true",
                    help="stop before building and draw what would be used: "
                         "the capture from above coloured by material, and "
                         "the patches picked, to out/classes_<name>.png")
    ap.add_argument("--centres", action="store_true",
                    help="give every tile its own interior from a further "
                         "patch of its material (needs --cut): edges stay as "
                         "they are, so tiles still match, and a closed seam "
                         "replaces the cross of four triangles (a centre "
                         "patch and four boundary strips, DAV-GSWT; omega-"
                         "tiles). Up to one patch per tile instead of six")
    ap.add_argument("--height-order", action="store_true",
                    help="measure how strictly the materials follow the ground's "
                         "height in the capture (with --tone-order); the viewer "
                         "lets the height line between them wander that much")
    ap.add_argument("--pool-sharpness", action="store_true",
                    help="with --frames: a centre must be as dense and fine-grained "
                         "where it shows as the edge patches are, per square metre, in "
                         "place of as many splats as the thinnest edge over its whole cut")
    ap.add_argument("--fill-spare-haze", action="store_true",
                    help="the opacity fill leaves large faint splats (haze) as "
                         "they are: made opaque, they showed as smears from low "
                         "down in the bush")
    ap.add_argument("--diamonds", action="store_true",
                    help="each edge patch sits on the edge it colours (Cohen's "
                         "diamond), so the ground runs on into the neighbour; "
                         "without it a patch sits on the tile's middle and every "
                         "edge joins ground a tile apart")
    ap.add_argument("--frames", action="store_true",
                    help="tiles keep only their frame, the band along the edges; "
                         "each material gets a pool of round centres that the "
                         "viewer picks and turns per cell, joined in a wobbled "
                         "ring (bozkir/frames.py). Replaces --centres")
    ap.add_argument("--centre-size", type=float, default=0.6, metavar="F",
                    help="with --frames: the centre across, as a share of the "
                         "tile's width (default 0.6); 0: as large as fits inside "
                         "the edge strip")
    ap.add_argument("--ring", type=float, default=15.0, metavar="CM",
                    help="with --frames: how wide frame and centre blend into "
                         "each other (default 15 cm)")
    ap.add_argument("--centre-shape", type=float, default=2.0, metavar="P",
                    help="with --frames: 2 (default) a round centre; more, a rounded "
                         "square (the p-norm's ball, e.g. 6) reaching close to the "
                         "edges and corners, so the frame that repeats is a thin "
                         "border; it is turned in quarter turns only")
    ap.add_argument("--pool-share", type=float, default=0.0, metavar="F",
                    help="with --frames: where a material's ground of its own runs "
                         "out, centres of which up to this share also shows in a "
                         "frame (default 0: never). For a material with little "
                         "ground, a bush: some repetition, but turned and elsewhere")
    ap.add_argument("--centre-pool", type=int, default=18, metavar="N",
                    help="with --frames: at most N centres per material (default 18)")
    ap.add_argument("--dense", action="store_true",
                    help="leave thin patches out (under half the material's "
                         "dense splat count), then choose alike ones as "
                         "usual: alike hides the triangle inside each tile")
    ap.add_argument("--varied", action="store_true",
                    help="choose each material's patches densest first "
                         "instead of most alike, so the capture's own "
                         "variety (lighter, darker ground) reaches the map; "
                         "graph cut seams have more to hide")
    ap.add_argument("--apart", action="store_true",
                    help="a material's patches share no ground where it has "
                         "enough: overlapping patches are shifted copies, so "
                         "their footprints and stones repeat across the map")
    ap.add_argument("--missing-layer", action="store_true",
                    help="under each patch, add only what the capture shows "
                         "there from above and the patch is missing (the "
                         "huge faint splats the size cut drops give the "
                         "capture much of its solid look), in the capture's "
                         "own colour, lying on the patch's ground")
    ap.add_argument("--fill-opacity", type=float, default=0.0,
                    help="make each tile's own splats opaque up to this "
                         "coverage from above (e.g. 0.98), so the sky does not "
                         "show through; no splat is added (0 = off)")
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
    ap.add_argument("--faint-fill", type=float, default=0.0, metavar="K",
                    help="judge --min-cover as the ground will be after the "
                         "fill (--fill-opacity) raises opacity at most K times: "
                         "a faint but whole layer passes, real gaps still fail. "
                         "2 is gentle; the fill itself goes up to 6 (0 = off)")
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
    ap.add_argument("--exclude-at", action="append", default=[],
                    metavar="X,Y",
                    help="drop every candidate whose square holds this "
                         "point (a centre from the preview's list). By "
                         "place, not number, so it holds when the export "
                         "searches wider than the preview did. Repeatable")
    ap.add_argument("--classes", type=lambda v: v if v == "auto" else int(v), default=1,
                    help="build this many tile sets, one per material found "
                         "in the capture, all sharing the same edge codes so "
                         "any cell can take any class without breaking the "
                         "matching. 1 is the single-material behaviour; "
                         "'auto' picks the count and prints why")
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
    ap.add_argument("--level-tone", action="store_true",
                    help="match each level of detail's colour to its full tile's as "
                         "seen from low down: the levels keep the largest splats, not "
                         "the ones seen, and a bush's came out paler at a distance")
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
    _FILL["spare_haze"] = bool(args.fill_spare_haze)

    def classes_in(text):
        return {int(v) for v in text.replace(" ", "").split(",") if v}
    half = classes_in(args.half_turns)
    # Class -> cm, from '0' (uses --max-height) or '0:1,1:4'.
    flat_cm = {}
    for item in args.flatten_classes.replace(" ", "").split(","):
        if not item:
            continue
        k, _, cm = item.partition(":")
        flat_cm[int(k)] = float(cm) if cm else args.max_height
    if any(v <= 0 for v in flat_cm.values()):
        raise SystemExit("--flatten-classes needs a height: '0:1' or --max-height")
    if flat_cm:
        # Flattened per class below, after the classes are known.
        scene_cm, args.max_height = args.max_height, 0.0
        s = scene_from_args(args)
        args.max_height = scene_cm
        print("  flattening per class: " + ", ".join(
            f"class {k} to {v:g} cm" for k, v in sorted(flat_cm.items())))
    else:
        s = scene_from_args(args)
    scale_report(s, args.size)
    raw = None
    if args.missing_layer:
        from dataclasses import replace
        raw = prepare(args.path, replace(config_from_args(args), clean=False),
                      cache=not args.no_cache)
    up = args.up_axis
    thickness = args.thickness if args.thickness else args.size * 0.25
    holed_cache(s, thickness, args.extract_margin, args.search_cache)
    took("capture loaded")
    tiles_need = 2 * args.colours
    # With rotations, each real patch also supplies turned copies as further
    # colours, so fewer distinct patches have to be found in the capture.
    need = -(-tiles_need // max(1, args.rotations))
    if args.classes == "auto":
        r = [v for v in args.class_rotations.replace(" ", "").split(",") if v]
        least = min(int(v) for v in r) if r else args.rotations
        args.classes = choose_classes(s, args, up, thickness,
                                      -(-tiles_need // max(1, least)))
        took("materials counted")
        if r:
            args.class_rotations = ",".join((r + r[-1:] * 4)[:max(args.classes, 1)])
    per_class = [int(v) for v in args.class_rotations.replace(" ", "").split(",") if v]
    if len(per_class) == 1:
        per_class = per_class * max(args.classes, 1)     # one value: every class
    if per_class and len(per_class) != max(args.classes, 1):
        raise SystemExit(f"--class-rotations needs {max(args.classes, 1)} "
                         f"values, one per class, got {len(per_class)}")
    if any(r not in (1, 2, 3, 4) for r in per_class):
        raise SystemExit("--class-rotations values must be 1 to 4")

    def rot_of(ci):
        r = per_class[ci] if per_class else args.rotations
        # Half turns give one new copy (180); a second would be the original.
        return min(r, 2) if ci in half else r

    def need_of(ci):
        return -(-tiles_need // max(1, rot_of(ci)))

    shrink = {}
    shrink_auto = args.class_scale.strip() == "auto"
    for item in ("" if shrink_auto else args.class_scale).replace(" ", "").split(","):
        if item:
            k, _, v = item.partition(":")
            shrink[int(k)] = float(v)
    if any(not 0.2 <= v <= 1.0 for v in shrink.values()):
        raise SystemExit("--class-scale values must be 0.2 to 1 (1 = as captured)")

    def ground_of(ci):
        """Ground one patch of class `ci` is cut from (m)."""
        return args.size / shrink.get(ci, 1.0)
    need_max = max(need_of(c) for c in range(max(args.classes, 1)))
    need = min(need_of(c) for c in range(max(args.classes, 1)))

    if per_class:
        print(f"  need {tiles_need} patches for {args.colours} colours per axis; "
              + ", ".join(f"class {c}: {need_of(c)} real"
                          for c in range(len(per_class))))
    else:
        print(f"  need {tiles_need} patches for {args.colours} colours per axis"
              + (f", {need} real and the rest turned copies"
                 if args.rotations > 1 else ""))
    # Search wide (4x what is needed, per class), then narrow on
    # appearance. Too small a pool on ground that is mostly one material
    # finds only that material and splits it into two halves of itself.
    want = args.count or max(need_max * 4 * max(args.classes, 1), 12)
    extra = {}     # what gather measured besides the patches: shares, pools

    def gather(want, widen=False, last=True):
        common = search_kwargs(args, thickness)
        if widen:
            # More from the list already found (a rarer material sits
            # further down it) rather than a new search per setting.
            common.pop("min_separation", None)
            found, _, _ = cached_search(s, args.size, up, common,
                                        cache=args.search_cache, verbose=False)
            chosen = choose(found, want, args.size, 0.5)
            print(f"  taking {len(chosen)} of the {len(found)} viable "
                  f"positions already found")
        elif args.auto and not args.patches:
            # With --patches the indices only mean something under the exact
            # settings of the preview, so nothing is loosened.
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
                if "cover" not in info:
                    info["cover"] = rendered_coverage(p, args.size, up)
            if hit:
                print(f"  recalled {len(found)} candidates from the search "
                      f"preview_patches.py already did")
        if args.min_share > 0 and args.classes > 1:
            # Materials from the whole capture, not from its best-scoring
            # ground: flat bare sand outscores sand with shrubs, so the top
            # candidates were all one material and the others went unseen.
            every = search_kwargs(args, thickness)
            # A quarter tile apart: 0.5 left 63 of 232 positions on desert v1,
            # too few for a material's 6 edge and 18 centre patches.
            sep = min(every.pop("min_separation", 1.0), 0.25)
            found, st, _ = cached_search(s, args.size, up, every,
                                         cache=args.search_cache, verbose=False, where=True)
            extra["rejected_at"] = st.get("rejected_at", [])
            if args.seen_shares:
                extra["seen"] = seen_points(found, st)
            chosen = choose(found, len(found), args.size, sep)
            for _, _, p, info in chosen:
                if "cover" not in info:
                    info["cover"] = rendered_coverage(p, args.size, up)
            print(f"  every viable position counts towards the materials: {len(chosen)}")
            extra["viable"] = [c[1] for c in chosen]
        if len(chosen) < need:
            raise SystemExit(
                f"  only {len(chosen)} patches passed, need {need}."
                + ("\n  --patches was given, so the settings were used exactly "
                   "as supplied; re-run preview_patches.py and pass the same "
                   "settings, or a preset, to both." if args.patches else
                   "\n  This capture does not yield a tile set at any setting "
                   "tried. preview_patches.py shows what is rejecting."))
        # Excluded by hand before narrowing (a survey marker scores as good
        # flat ground).
        if args.exclude.strip():
            drop = {int(x) for x in args.exclude.replace(",", " ").split()}
            before = len(chosen)
            chosen = [c for i, c in enumerate(chosen) if i not in drop]
            print(f"  excluded {sorted(drop)}: "
                  f"{before} candidates -> {len(chosen)}")
        if args.exclude_at:
            pts = [[float(v) for v in p.split(",")] for p in args.exclude_at]
            half = args.size / 2.0
            before = len(chosen)
            chosen = [c for c in chosen
                      if not any(abs(c[1][0] - px) <= half and abs(c[1][1] - py) <= half
                                 for px, py in pts)]
            print(f"  excluded around {pts}: {before} candidates -> {len(chosen)}")
        if args.odd_spots > 0 and not args.patches:
            groups = ((split_materials(chosen, args.classes) if args.min_share > 0
                       else split_classes(chosen, classes=args.classes, per_class=10 ** 9))
                      if args.classes > 1 else [chosen])
            keep = []
            for gi, g in enumerate(groups):
                ordered, clean, spots = rank_clean(g, args.size, args.odd_spots, up)
                n_gi = need_of(gi) if args.classes > 1 else need
                if clean < n_gi and not last:
                    raise _ShortClass([gi])
                keep += ordered[:max(clean, n_gi)]
                odd = ordered[max(clean, n_gi):]
                extra.setdefault("odd", []).extend(c[1] for c in odd)
                if odd:
                    print(f"  material {gi}: {len(odd)} of {len(g)} patches hold an "
                          f"odd spot ({args.odd_spots:g} cm or more), left out: "
                          + ", ".join(f"({c[1][0]:.2f},{c[1][1]:.2f})" for c in odd[:8]))
                if clean < n_gi:
                    print(f"  material {gi}: only {clean} clean patches, "
                          f"{n_gi - clean} with the smallest spots kept")
            ids = {id(c) for c in keep}
            chosen = [c for c in chosen if id(c) in ids]

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
            # With classes, split_classes narrows within each group instead
            # (narrowing first would leave nothing different to split).
            if args.varied:
                chosen = dense_first(chosen, need)[:need * 3]
            elif args.dense:
                chosen = dense_first(chosen, need)
            extra["pools"] = [list(chosen)]
            chosen = (apart(chosen if args.varied else
                            select_similar(chosen, need * 3, args.similarity),
                            need, args.size) if args.apart
                      else chosen[:need] if args.varied
                      else select_similar(chosen, need, args.similarity))

        if args.classes <= 1:
            feats = np.array([appearance(c[2]) for c in chosen])
            spread = float(np.linalg.norm(feats - feats.mean(axis=0), axis=1).mean())
            print(f"  appearance spread across the {need} chosen: {spread:.3f} "
                  f"({'similar' if spread < 0.08 else 'diagonals will show'})")

        # One set of patches per class.
        if args.classes > 1:
            full = (split_materials(chosen, args.classes) if args.min_share > 0
                    else split_classes(chosen, classes=args.classes, per_class=10 ** 9))
            keep = list(range(len(full)))
            if args.min_share > 0:
                sh = class_shares(s, full, args.size, up, seen=extra.get("seen"))
                keep = sorted(keep, key=lambda i: -sh[i])
                if args.tone_order:
                    # Darkest first: the viewer's first class collects in
                    # hollows and the last stays exposed, so tones meet
                    # their neighbours in order (dark, mid, light) and the
                    # map shades instead of jumping from dark to light.
                    keep = sorted(keep, key=lambda i: float(np.mean(
                        [appearance(c[2])[:3] for c in full[i]])))
                print("\n  share of the capture's ground, by material (mean rgb)"
                      + (", darkest first: hollows to crests" if args.tone_order else "")
                      + ":")
                for i in keep:
                    m = np.mean([appearance(c[2])[:3] for c in full[i]], axis=0)
                    print(f"    {m[0]:.2f} {m[1]:.2f} {m[2]:.2f}  {sh[i]:4.0%}  "
                          f"({len(full[i])} positions)")
                left = [i for i in keep if sh[i] < args.min_share]
                keep = [i for i in keep if sh[i] >= args.min_share]
                if left:
                    print(f"  left out under {args.min_share:.0%}: "
                          + ", ".join(f"{sh[i]:.0%}" for i in left))
                total = sum(sh[i] for i in keep)
                ground = [sh[i] / total for i in keep]
                if args.quality_weight > 0:
                    thick = [float(np.median([haze(c, up) for c in full[i]])) for i in keep]
                    used = weigh_shares(ground, thick, args.quality_weight)
                    print(f"\n  in the viewer, weighed by how thin each layer of splats is "
                          f"(weight {args.quality_weight:g}):")
                    for j, i in enumerate(keep):
                        print(f"    material {j}: {ground[j]:4.0%} of the ground, layer "
                              f"{thick[j] * 100:4.1f} cm -> {used[j]:4.0%} of the map")
                    extra["ground_shares"] = [round(v, 3) for v in ground]
                    extra["thick"] = thick
                    ground = used
                extra["shares"] = [round(v, 3) for v in ground]
            full = [full[i] for i in keep]
            extra["materials"] = full
            if args.height_order and args.tone_order:
                p = height_order(full, up)
                if p is not None:
                    extra["height_order"] = round(p, 3)
                    print(f"\n  height: the materials out of height order in {p:.0%} of "
                          f"position pairs (0% strictly by height, 50% unrelated): the "
                          f"viewer's height line wanders by {min(1.0, 2 * p):.0%}")
            # A deeper pool per class when purity is to choose among them.
            if per_class:
                # Each class narrowed to its own pool, exactly as a run with
                # that class's --rotations would, so the other class is
                # unchanged.
                groups = []
                for gi, g in enumerate(full):
                    pool = need_of(gi) * (6 if args.apart else 3) if args.purity else need_of(gi)
                    if args.varied:
                        groups.append(dense_first(g, need_of(gi))[:pool])
                    else:
                        if args.dense:
                            g = dense_first(g, need_of(gi))
                        groups.append(select_similar(g, pool) if len(g) > pool else g)
            elif args.min_share > 0:
                pool = need * 3 if args.purity else need
                groups = [select_similar(g, pool) if len(g) > pool else g for g in full]
            else:
                pool = need * 3 if args.purity else need
                groups = split_classes(chosen, classes=args.classes, per_class=pool)
                groups = [groups[i] for i in keep]
            sep = class_separation(groups) if len(groups) > 1 else 99.0
            print(f"\n  split into {len(groups)} classes, separation {sep:.2f}")
            # Noise split in two still has two halves: a uniform capture
            # scored 1.1 with identical colours, the desert's sand and scrub
            # 7-16. So: a clear ratio and a visible colour difference.
            rgb = [np.mean([appearance(c[2])[:3] for c in g], axis=0)
                   for g in groups if g]
            colour_gap = (float(np.abs(rgb[0] - rgb[1]).max())
                          if len(rgb) > 1 else 1.0)
            # With measured shares, close materials are the capture's own
            # variety (two shades of sand), not a failed split.
            if args.min_share <= 0 and (sep < 3.0 or colour_gap < 0.04):
                raise SystemExit(
                    "  the classes are no further apart than they are varied: "
                    "this capture holds one material.\n"
                    "  Building tile sets from it would give two that differ by "
                    "nothing, and a terrain rule with nothing to choose between.")
            short = [i for i, g in enumerate(groups) if len(g) < need_of(i)]
            if short:
                raise _ShortClass(short)
            if args.min_share > 0:
                # Most typical first: the purest patch of a material is its
                # most extreme one (the one clump of bush in the scrub).
                for gi, g in enumerate(groups):
                    f = np.array([appearance(c[2]) for c in g])
                    mid = np.median(f, axis=0)
                    sd = np.maximum(f.std(axis=0), 1e-6)
                    far = np.linalg.norm((f - mid) / sd, axis=1)
                    # Dense counts as much: the most typical patches of
                    # desert v1's commonest sand had a quarter of the splats.
                    n = np.array([len(c[2]) for c in g], dtype=np.float64)
                    rank = far / max(np.median(far), 1e-9) - n / max(n.max(), 1.0)
                    if args.quality_weight > 0:
                        # Splat count favoured the haze: desert v1's bush has
                        # three times the splats of its sharp sand. A thin
                        # layer is what looks sharp up close.
                        th = np.array([haze(c, up) for c in g])
                        rank = far / max(np.median(far), 1e-9) + th / max(np.median(th), 1e-9)
                    groups[gi] = [g[k] for k in np.argsort(rank, kind="stable")]
                    print(f"  class {gi}: the {need_of(gi)} most typical and "
                          f"{'sharpest' if args.quality_weight > 0 else 'dense'} "
                          f"of {len(g)} patches first")
            elif args.purity:
                # Purest patches first (stable: ties keep the search order).
                for gi, g in enumerate(groups):
                    others = [rgb[k] for k in range(len(rgb)) if k != gi]
                    pur = [purity(c[2], rgb[gi], others, up) for c in g]
                    # Varied: pure enough, then densest (the order given).
                    order = sorted(range(len(g)), key=(lambda k: (pur[k] < 0.98, k))
                                   if args.varied else (lambda k: -pur[k]))
                    groups[gi] = [g[k] for k in order]
                    kept = [pur[k] for k in order[:need_of(gi)]]
                    print(f"  class {gi}: {'densest pure' if args.varied else 'purest'} "
                          f"{need_of(gi)} of {len(g)} patches, "
                          f"purity " + " ".join(f"{v:.2f}" for v in kept))
            class_sets = [apart_alike(g, need_of(gi), ground_of(gi), args.apart) if args.dense
                          else apart(g, need_of(gi), ground_of(gi)) if args.apart
                          else g[:need_of(gi)] for gi, g in enumerate(groups)]
            extra["pools"] = groups
        else:
            class_sets = [chosen[:need]]

        return chosen, class_sets

    # A rarer material may miss the first pool (scrub on mostly-sand ground),
    # so the search widens and retries. --count is where it starts, not a
    # limit (a preset saved for two colours asks too few for three).
    auto_widen = args.auto and not args.patches
    for attempt in range(4 if auto_widen else 1):
        try:
            chosen, class_sets = gather(want, widen=attempt > 0,
                                        last=attempt == (3 if auto_widen else 0))
            break
        except _ShortClass as e:
            if attempt == (3 if auto_widen else 0):
                raise SystemExit(
                    f"  class {e.args[0]} still has fewer than {need_max} patches "
                    f"after searching for {want}. This capture holds too little "
                    f"of that material for {args.colours} colours per axis: "
                    f"try --colours 2, or drop --classes.")
            want *= 2
            print(f"\n  class {e.args[0]} came back short; searching wider, "
                  f"for {want} candidates")

    took("patches searched, odd spots, materials")
    # Per material, what the algorithm does and why: each rule is applied
    # where the capture's measurements call for it and can afford it, and
    # left off otherwise, with the reason printed (and drawn in the picture).
    decisions = [[] for _ in class_sets]
    source = {}         # per class: the candidates its edge patches came from, and their ground
    if args.min_share > 0 and len(class_sets) > 1:
        mats = extra["materials"]
        labels, (mx, my) = material_map(s, mats, args.size, up, seen=extra.get("seen"))

        def purity(c, k, d):
            """Share of the ground under `c` (a `d` square) that is material `k`."""
            i0, i1 = (int(np.floor((c[1][0] - d / 2 - mx) / 0.5)),
                      int(np.ceil((c[1][0] + d / 2 - mx) / 0.5)))
            j0, j1 = (int(np.floor((c[1][1] - d / 2 - my) / 0.5)),
                      int(np.ceil((c[1][1] + d / 2 - my) / 0.5)))
            w = labels[max(j0, 0):max(j1, 0), max(i0, 0):max(i1, 0)]
            w = w[w >= 0]
            return float((w == k).mean()) if w.size else 0.0

        def apart_count(cands, d):
            room = []
            for c in cands:
                if all(abs(c[1][0] - x) >= d or abs(c[1][1] - y) >= d for x, y in room):
                    room.append(c[1])
            return len(room)

        thick = extra.get("thick") or [float(np.median([haze(c, up) for c in g])) for g in mats]
        cpool = list(mats)              # where centres come from, per material
        weight = args.quality_weight or 0.5
        floor = args.pure_share or 0.8
        for ci in range(len(class_sets)):
            n_c = need_of(ci)
            tone = tone_gate(mats[ci])
            in_tone = lambda c: tone(c)[0]
            solid = lambda c, d=args.size: hole_free(c, d, up, args.fill_opacity)
            rel = float(np.median([c[3].get("relief", 0.0) for c in class_sets[ci][:n_c]]))
            decisions[ci].append(f"flatten: {'yes' if args.flatten_bare and rel < 0.15 else 'no'}"
                                 f" (relief {rel * 100:.0f} cm)")
            # Shrink a hazy material (--class-scale).
            want = shrink.get(ci)
            ratio = min(thick) / max(thick[ci], 1e-6)
            if shrink_auto and want is None:
                want = max(0.5, round(ratio ** weight, 2)) if ratio < 0.5 else None
                if want is None:
                    decisions[ci].append(f"shrink: no (layer {thick[ci] * 100:.1f} cm, within "
                                         f"twice the thinnest)")
            if want is not None and want < 1.0:
                # The strongest shrink the ground can afford: from the asked
                # one towards 1 in steps (auto only), each searched at its
                # own size. Half-turned copies stand in where only half the
                # distinct patches are there, as elsewhere when ground runs out.
                ladder = [want] + ([v for v in np.arange(want + 0.125, 0.95, 0.125)]
                                   if shrink_auto else [])
                tried, done_ = [], False
                alike = colour_alike(class_sets[ci][:n_c])
                for want in ladder:
                    want = round(float(want), 2)
                    d = args.size / want
                    every = search_kwargs(args, thickness)
                    sep = min(every.pop("min_separation", 1.0), 0.25)
                    print(f"\n  class {ci}: searching {d:.2f} m squares to shrink to "
                          f"{args.size:g} m")
                    found_big, _, _ = cached_search(s, d, up, every, cache=args.search_cache,
                                                    progress=_progress_line())
                    big = [c for c in choose(found_big, len(found_big), d, sep)
                           if purity(c, ci, d) >= floor and alike(c) and in_tone(c)]
                    if args.odd_spots > 0 and big:
                        ordered, clean, _ = rank_clean(big, d, args.odd_spots, up)
                        big = ordered[:clean]
                    for c in big:
                        c[3]["cut"] = d
                    if not big:
                        tried.append(f"x{want:g}: no cuts")
                        continue
                    f = np.array([appearance(c[2]) for c in mats[ci]])
                    mid, sd = np.median(f, axis=0), np.maximum(f.std(axis=0), 1e-6)
                    far = np.array([np.linalg.norm((appearance(c[2]) - mid) / sd) for c in big])
                    th = np.array([haze(c, up) for c in big])
                    rank = far / max(np.median(far), 1e-9) + th / max(np.median(th), 1e-9)
                    big = [big[k] for k in np.argsort(rank, kind="stable")]
                    # Ground of their own and no holes, nothing else: shifted
                    # copies repeated the same tufts tile after tile, and a
                    # hole in one patch showed in every tile built from it.
                    real = apart_alike(big, n_c, d, True, only_apart=True,
                                       accept=lambda c, d=d: solid(c, d))
                    holed = sum(1 for c in big if c[3].get("holed"))
                    half_n = -(-n_c // 2)
                    if len(real) < half_n:
                        tried.append(f"x{want:g}: {len(big)} cuts, {len(real)} apart and "
                                     f"without holes")
                        continue
                    turned = len(real) < n_c
                    if turned:
                        # The real ones, each also half turned.
                        if per_class:
                            per_class[ci] = 2
                        half.add(ci)
                        n_c = need_of(ci)
                    shrink[ci] = want
                    class_sets[ci] = real[:n_c]
                    source[ci] = (big, d)
                    cpool[ci] = big
                    decisions[ci].append(
                        f"shrink: x{want:g} (layer {thick[ci] * 100:.1f} cm against "
                        f"{min(thick) * 100:.1f}; {d:.2f} m cuts, {len(big)} at least "
                        f"{floor:.0%} this material, {len(real)} apart and without holes"
                        + (f", {holed} passed over for holes" if holed else "")
                        + (", half-turned copies for the rest" if turned else "")
                        + (f"; tried {', '.join(tried)}" if tried else "") + ")")
                    done_ = True
                    break
                if not done_:
                    shrink.pop(ci, None)
                    decisions[ci].append(f"shrink: no (not enough cuts {floor:.0%} this "
                                         f"material: {'; '.join(tried)})")
                    want = None
            # Pure patches (--pure-share), for materials not shrunk.
            if ci not in shrink and args.pure_share > 0:
                pool = extra["pools"][ci]
                pure = [c for c in pool if purity(c, ci, args.size) >= floor and in_tone(c)]
                apart_n = apart_count(pure, args.size)
                if apart_n >= n_c:
                    class_sets[ci] = apart_alike(pure, n_c, args.size, True, accept=solid)
                    source[ci] = (pure, args.size)
                    cpool[ci] = [c for c in mats[ci]
                                 if purity(c, ci, args.size) >= floor and in_tone(c)]
                    decisions[ci].append(f"pure patches: yes ({apart_n} apart at "
                                         f"{floor:.0%} or more, in the common tone)")
                else:
                    decisions[ci].append(f"pure patches: no (only {apart_n} apart at "
                                         f"{floor:.0%}; mixed ones kept)")
                    toned = [c for c in pool if in_tone(c)]
                    if apart_count(toned, args.size) >= n_c:
                        class_sets[ci] = apart_alike(toned, n_c, args.size, True, accept=solid)
                        source[ci] = (toned, args.size)
                        cpool[ci] = [c for c in mats[ci] if in_tone(c)]
            # No holes, for every material however its patches were chosen.
            if ci not in shrink and not all(solid(c) for c in class_sets[ci][:n_c]):
                pool = [c for c in extra["pools"][ci] if in_tone(c)] or extra["pools"][ci]
                class_sets[ci] = apart_alike(pool, n_c, args.size, True, accept=solid)
                source[ci] = (pool, args.size)
            if len(class_sets[ci]) < n_c:
                # Too few without holes: the most alike of the rest, holes and
                # all, rather than too few for the tile set (said below).
                have = {id(c) for c in class_sets[ci]}
                rest = apart_alike([c for c in extra["pools"][ci] if id(c) not in have],
                                   n_c - len(class_sets[ci]), args.size, True)
                class_sets[ci] = list(class_sets[ci]) + rest
            bad = [c for c in class_sets[ci][:n_c]
                   if not solid(c, c[3].get("cut", args.size))]
            passed = sum(1 for c in extra["pools"][ci] if c[3].get("holed"))
            decisions[ci].append(
                "holes: " + (f"{len(bad)} used with holes, none to spare" if bad
                             else "none in the patches used")
                + (f"; {passed} passed over" if passed else ""))
            cpool[ci] = [c for c in cpool[ci] if in_tone(c)]       # centres too
            # What the tone test left out, and why (bluer, greener, mixed...).
            used = class_sets[ci][:n_c]
            off = [tone(c)[1] for c in used if not tone(c)[0]]
            out = {}
            for c in (extra["pools"][ci] if ci not in shrink else cpool[ci]):
                ok, why = tone(c)
                if not ok:
                    out[why] = out.get(why, 0) + 1
            decisions[ci].append(
                "common tone: " + ("all patches in it" if not off else
                                   f"{len(off)} used outside it ({', '.join(off)}), none "
                                   f"to spare")
                + (f"; left out {', '.join(f'{v} {k}' for k, v in out.items())}" if out else ""))
            if extra.get("ground_shares"):
                decisions[ci].append(f"map share: {extra['ground_shares'][ci]:.0%} of the "
                                     f"ground -> {extra['shares'][ci]:.0%} (layer "
                                     f"{thick[ci] * 100:.1f} cm)")
        extra["centre_pool"] = cpool
    if any(decisions):
        took("decisions (shrink, purity, holes, tone)")
    geo, turns, whole = None, [], set()     # whole: classes keeping whole tiles
    if args.frames:
        # Frames repeat in every tile with their edge colour, so the quietest
        # edge patches make them; the centres come from a pool, turned in
        # the viewer as far as the material's texture allows.
        try:
            geo = frames_mod.geometry(args.size, args.centre_size, args.ring, args.centre_shape)
        except ValueError as e:
            raise SystemExit(f"  {e}")
        material = (extra.get("centre_pool") or extra.get("materials")
                    or extra.get("pools") or [chosen] * len(class_sets))
        extra["frame_pools"] = []
        for ci in range(len(class_sets)):
            n_c, d = need_of(ci), ground_of(ci)
            solid_d = lambda c, d=d: hole_free(c, d, up, args.fill_opacity)
            cands = source.get(ci, ((extra.get("pools") or class_sets)[ci], d))[0]
            rule = frames_mod.turn_rule([lum_of(c, d, args.size, up)
                                         for c in list(class_sets[ci][:n_c]) + list(cands)[:12]])
            turns.append(rule)
            if not rule["any"]:
                # Ripples meet a turned centre out of step all round the ring,
                # which shows as a circle; the graph-cut tile keeps its seams
                # where the patches agree. Its patches stay as chosen: to the
                # spot measure, ripples themselves are spots.
                whole.add(ci)
                extra["frame_pools"].append([])
                decisions[ci].append(
                    f"centres: no, whole tiles kept (its texture runs one way, coherence "
                    f"{rule['coherence']:.2f}: ripples would meet a centre out of step "
                    f"all round the ring)")
                continue
            edge, line = quiet_edges(class_sets[ci][:n_c], cands, n_c, d, geo["inner"],
                                     args.size, up, solid_d, args.diamonds, geo["shape"])
            class_sets[ci] = list(edge) + [c for c in class_sets[ci]
                                           if all(c is not e for e in edge)]
            decisions[ci].append(line)
            pool, mids, shared, why = pick_pool(class_sets[ci][:n_c], material[ci], args.centre_pool,
                                         args.size, d, geo, up, solid_d, not args.diamonds,
                                         args.pool_share, args.diamonds,
                                         args.pool_sharpness)
            if not pool:
                # A frame with nothing to put in it would leave a hole.
                whole.add(ci)
                extra["frame_pools"].append([])
                decisions[ci].append(
                    "centres: no, whole tiles kept (no disc of its ground clear of the "
                    "edge patches, nor whole and alike)")
                decisions[ci].append("turns: " + rule["why"])
                continue
            extra["frame_pools"].append(pool)
            square = geo["shape"] != 2.0
            decisions[ci].append(
                f"centres: a pool of {len(pool)} "
                + ("rounded squares" if square else "discs")
                + f" {2 * geo['radius']:.2f} m across ({2 * geo['radius'] / args.size:.0%} of the tile), "
                + (f"{mids} the edge patches' own middles, " if not args.diamonds else "")
                + f"{sum(v <= 0.02 for v in shared)} on ground of their own"
                + (f", {sum(v > 0.02 for v in shared)} sharing up to {max(shared):.0%} of "
                   f"theirs with the frames (--pool-share {args.pool_share:g})"
                   if any(v > 0.02 for v in shared) else "")
                + f"; joined in a {args.ring:g} cm ring"
                + (f" (short of {args.centre_pool}: of {why['candidates']} candidates, "
                   + ", ".join(f"{v} {k}" for k, v in why.items()
                               if k != "candidates" and v) + ")"
                   if len(pool) < args.centre_pool and why else ""))
            decisions[ci].append(
                "turns: " + ("quarter turns (a rounded square keeps its shape only so); "
                             "its texture: " if square else "") + rule["why"])
    save_holed()
    if args.frames:
        took("frames (quiet edges, centre pool, turns)")
    if any(decisions):
        print("\n  decisions, per material:")
        for ci, ds in enumerate(decisions):
            for line in ds:
                print(f"    class {ci}  {line}")
    extra["decisions"] = decisions

    if args.look or args.min_share > 0:
        count = (len(minimal_codes(args.colours, args.colours))
                 if (args.minimal or args.colours >= 3) and not args.complete
                 else args.colours ** 4)
        mats = extra.get("materials") or [chosen]
        pool = extra.get("centre_pool") or mats
        centres = [pick_centres(g[:need_of(ci)], pool[ci], count, ground_of(ci), up,
                                accept=lambda c, d=ground_of(ci): hole_free(
                                    c, d, up, args.fill_opacity))
                   if args.centres and not args.frames else []
                   for ci, g in enumerate(class_sets)]
        if args.frames:
            centres = extra["frame_pools"]
            extra["centre_radius"] = [geo["radius"] * ground_of(ci) / args.size
                                      for ci in range(len(class_sets))]
            extra["centre_shape"] = geo["shape"]
        elif args.centres:
            for ci, c in enumerate(centres):
                if ci < len(decisions):
                    decisions[ci].append(
                        "centres: none on ground of its own, four triangles kept" if not c
                        else f"centres: {len(c)} on ground of their own"
                        + (", the rest turned copies" if len(c) < count else ""))
        name = (args.out.stem if args.out else args.path.stem).removesuffix("_base")
        dest = Path(__file__).resolve().parents[1] / "out" / f"classes_{name}.png"
        extra["ground"] = [ground_of(ci) for ci in range(len(class_sets))]
        look(s, mats, [g[:need_of(ci)] for ci, g in enumerate(class_sets)], centres,
             extra, args.size, up, dest, name)
        took("the picture")
        if args.look:
            for ci, c in enumerate(centres):
                if c:
                    print(f"  class {ci}: {len(c)} centre patches on ground of their own "
                          + ("in the pool" if args.frames else f"for {count} tiles"))
            print(f"\n  nothing built; the picture: {dest}")
            return
        print(f"\n  the materials and patches used: {dest}")

    all_tiles, all_codes, all_class = [], [], []
    pool_patches, pool_class = [], []          # --frames: the centres, per class
    for ci, picked in enumerate(class_sets):
        if len(class_sets) > 1:
            f = np.array([appearance(c[2]) for c in picked])
            print(f"\n  class {ci}: rgb {f[:, 0].mean():.2f} "
                  f"{f[:, 1].mean():.2f} {f[:, 2].mean():.2f}")
        n_c, r_c = need_of(ci), rot_of(ci)
        if args.flatten_bare > 0 and ci not in flat_cm:
            relief = float(np.median([c[3].get("relief", 0.0) for c in picked[:n_c]]))
            if relief < 0.15:
                flat_cm[ci] = args.flatten_bare
                print(f"  class {ci} is flat ground (relief {relief * 100:.0f} cm): "
                      f"flattened to {args.flatten_bare:g} cm")
        overlap_report(picked[:n_c], ground_of(ci))
        if ci in shrink and shrink[ci] != 1.0:
            print(f"  class {ci}: cut at {ground_of(ci):.2f} m and shrunk to {args.size:g} m "
                  f"(x{shrink[ci]:g}): up close it looks as from {1 / shrink[ci]:.2g}x as far")
        patches = prepare_patches(picked, n_c, args, up, raw, s, shrink.get(ci, 1.0))
        if ci in flat_cm:
            patches = [cap_height(p, flat_cm[ci] / 100.0, up) for p in patches]
            print(f"  class {ci} flattened to {flat_cm[ci]:g} cm")
        level = None
        if args.even_patches > 0:
            level = np.mean([even_mean(p) for p in patches], axis=0)
            patches = [even_out(p, level, args.even_patches) for p in patches]
            print(f"  class {ci}: each patch's mean colour moved towards the "
                  f"material's ({level[0]:.2f} {level[1]:.2f} {level[2]:.2f}), "
                  f"at most {args.even_patches:g}")
        if r_c > 1:
            # Real patches first, then quarter turns, then half turns: turned
            # copies only where the capture ran out of distinct ground.
            step = 2 if ci in half else 1
            patches = [rotate_patch(p, r * step, up) for r in range(r_c)
                       for p in patches][:tiles_need]
            print(f"  {n_c} real patches, {len(patches) - n_c} turned copies"
                  + (" (half turns)" if step == 2 else ""))

        # Balance the axes, or boundaries one way differ from the other.
        (h_patches, v_patches), axis_gap = balanced_split(patches, args.colours)
        naive = float(np.linalg.norm(
            np.mean([appearance(p) for p in patches[:args.colours]], axis=0)
            - np.mean([appearance(p) for p in patches[args.colours:tiles_need]],
                      axis=0)))
        if r_c > 1:
            # With turned copies both axes hold the same sources: nothing
            # to measure.
            print("\n  axis balance: not meaningful with --rotations "
                  "(both axes share the same sources)")
        else:
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
        centres = None
        if args.frames and ci not in whole:
            chosen_c = extra["frame_pools"][ci]
            print(f"\n  centre pool: {len(chosen_c)} "
                  + ("rounded squares" if geo["shape"] != 2.0 else "discs")
                  + " of this material's ground")
            if chosen_c:
                pool = prepare_patches(chosen_c, len(chosen_c), args, up, raw, s,
                                       shrink.get(ci, 1.0))
                if ci in flat_cm:
                    pool = [cap_height(p, flat_cm[ci] / 100.0, up) for p in pool]
                if level is not None:
                    pool = [even_out(p, level, args.even_patches) for p in pool]
                pool_patches += pool
                pool_class += [ci] * len(pool)
        elif args.centres and args.cut:
            count = len(subset) if subset is not None else args.colours ** 4
            material = (extra.get("centre_pool") or extra.get("materials")
                        or extra.get("pools", [picked]))[ci]
            chosen_c = pick_centres(picked[:n_c], material, count, ground_of(ci), up,
                                    accept=lambda c, d=ground_of(ci): hole_free(
                                        c, d, up, args.fill_opacity))
            if chosen_c:
                print(f"\n  centre patches, one per tile: {len(chosen_c)} on ground "
                      f"of their own for {count} tiles")
                centres = prepare_patches(chosen_c, len(chosen_c), args, up, raw, s,
                                          shrink.get(ci, 1.0))
                if ci in flat_cm:
                    centres = [cap_height(p, flat_cm[ci] / 100.0, up) for p in centres]
                if level is not None:
                    centres = [even_out(p, level, args.even_patches) for p in centres]
                if len(centres) < count:
                    # Half turns keep ripples running the same way.
                    centres += [rotate_patch(p, 2, up) for p in centres]
                    print(f"  and {min(count, len(centres)) - len(chosen_c)} of them "
                          f"again, half turned"
                          + (", then repeated" if len(centres) < count else ""))
            else:
                print("\n  centre patches: no ground of this material's own left; "
                      "its tiles keep their four triangles")
        # Holes the source patches bring in (shown before they reach a tile).
        for kind, group in (("edge", patches[:n_c]), ("centre", centres or [])):
            for j, p in enumerate(group):
                n_h, big_h, _ = holes(p, args.size, up)
                if n_h:
                    print(f"  class {ci} {kind} patch {j}: {n_h} hole(s) from above, "
                          f"largest {big_h:.0f} cm2")
        t, c = build_tile_set(h_patches, v_patches, args.size,
                              blend=args.blend, up_axis=up,
                              cut=args.cut, resolution=args.cut_res,
                              band=args.cut_band, codes=subset, centres=centres,
                              diamonds=args.diamonds)
        all_tiles += t
        all_codes += c
        all_class += [ci] * len(t)

    took("tiles assembled")
    # Every class uses the same edge codes, so a cell can take any class
    # without breaking matching: the code satisfies the neighbours, the
    # class is chosen independently by the rule.
    tiles, codes, tile_class = all_tiles, all_codes, all_class

    # Check that edges really match, within each class (different classes
    # are different material by design).
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

    # Layout checked on one class's codes (every class has the same set).
    first = [c for c, k in zip(codes, tile_class) if k == 0]
    grid = layout(first, 16, 16, seed=0)
    print(f"  layout check: {check_layout(first, grid)} mismatched edges "
          f"in 16x16")

    # Levels of detail. GSWT retrains each level with N0 / 4^i splats; with
    # no retraining here, each level keeps the most visible splats of the one
    # above (area times opacity), spread over the tile.
    #
    # Overhang: kept whole, boundary strips are drawn by both neighbours and
    # swap as the camera turns. Optionally each Gaussian gets its own reach
    # (--overhang-splats) or a flat margin (--tile-overhang).
    plane2 = [j for j in range(3) if j != up]
    trimmed = []
    for t in tiles:
        if args.tile_overhang is not None:
            reach = args.size * (0.5 + args.tile_overhang)
            keep = np.all(np.abs(t.xyz[:, plane2]) <= reach, axis=1)
        else:
            # Default: cut at the square, the only arrangement where no strip
            # is drawn by both tiles (whose winner flips for a whole row when
            # depths cross). Gaussians just inside still spread over the join.
            allow = args.overhang_splats * t.scale.max(axis=1)
            out = np.abs(t.xyz[:, plane2]).max(axis=1) - args.size / 2.0
            keep = out <= allow
        trimmed.append(t.subset(keep))

    before = sum(len(t) for t in tiles)
    after = sum(len(t) for t in trimmed)
    if args.tile_overhang is not None:
        how = f"a flat {args.tile_overhang:.1%} of the tile"
    else:
        how = ("cut at the square, so no two tiles cover the same ground"
               if args.overhang_splats <= 0 else
               f"{args.overhang_splats:g} of each Gaussian's own width")
    print(f"  overhang: {how}")
    print(f"  trimmed {before:,} -> {after:,} splats ({after / before:.0%})")
    tiles = trimmed
    if args.fill_opacity > 0:
        from bozkir.graphcut import render_patch
        cover = lambda ts: np.mean([render_patch(t, args.size, 48, up)[1].mean()
                                    for t in ts])
        was = cover(tiles)
        if args.fill_spare_haze:
            n = sum(int(haze_splats(t, args.size, up).sum()) for t in tiles)
            print(f"  fill: {n:,} large faint splats ({n / max(1, sum(len(t) for t in tiles)):.0%}) "
                  f"left as they are (--fill-spare-haze)")
        tiles = [fill_opacity(t, args.size, up, args.fill_opacity) for t in tiles]
        print(f"  fill opacity: cover from above {was:.0%} -> {cover(tiles):.0%}")
    name = (args.out.stem if args.out else args.path.stem).removesuffix("_base")
    hole_report(tiles, codes, tile_class, args.size, up,
                Path(__file__).resolve().parents[1] / "out" / f"holes_{name}.png",
                ring=[(geo["inner"], np.inf) if args.frames and k not in whole else None
                      for k in tile_class], shape=geo["shape"] if geo else 2.0)
    if args.frames:
        # Filled whole, then cut: a fill at the cut's rim would thicken it.
        if args.fill_opacity > 0:
            pool_patches = [fill_opacity(p, args.size, up, args.fill_opacity)
                            for p in pool_patches]
        for k in sorted(set(pool_class)):
            ids = [i for i, c in enumerate(pool_class) if c == k]
            bad = [i for i in ids if holes(pool_patches[i], args.size, up,
                                           ring=(0.0, geo["radius"]), shape=geo["shape"])[0]]
            print(f"    class {k} centres: {len(bad)} of {len(ids)} with holes")
        was = sum(len(t) for t in tiles)
        tiles = [t if k in whole else frames_mod.frame_of(t, geo["inner"], up, geo["shape"])
                 for t, k in zip(tiles, tile_class)]
        pool_patches = [frames_mod.centre_of(p, geo["outer"], up, geo["shape"])
                        for p in pool_patches]
        print(f"  frames: {sum(len(t) for t in tiles):,} of the tiles' {was:,} splats, "
              f"from {geo['inner']:.2f} m out"
              + (f" (whole tiles for class {', '.join(map(str, sorted(whole)))})" if whole else "")
              + f"; centres: {len(pool_patches)} "
              + ("rounded squares" if geo["shape"] != 2.0 else "discs")
              + f" to {geo['outer']:.2f} m, "
              f"{sum(len(p) for p in pool_patches):,} splats")

    parts, meta, cursor = [], [], 0
    total_levels = 0
    shift = {}           # --level-tone: per class, how much brighter levels came out
    for t, c, k in zip(tiles, codes, tile_class):
        levels = []
        for part in tile_levels(t, args.lod, args.size, up, args.lod_compensate,
                                args.level_tone, shift.setdefault(k, [])):
            parts.append(pack(part))
            levels.append([cursor, len(part)])
            cursor += len(part)
            total_levels += 1
        meta.append({"start": levels[0][0], "count": levels[0][1],
                     "levels": levels,
                     "n": int(c[0]), "e": int(c[1]),
                     "s": int(c[2]), "w": int(c[3]),
                     "class": int(k)})

    if args.level_tone:
        print("\n  level tone: each level matched to its full tile's colour from 20 deg; "
              "before, levels were brighter by " + ", ".join(
                  f"{np.mean(v) * 100:+.1f}% (class {k})" for k, v in sorted(shift.items()) if v))
    centre_meta = []
    for p, k in zip(pool_patches, pool_class):
        levels = []
        for part in tile_levels(p, args.lod, args.size, up, args.lod_compensate,
                                args.level_tone):
            parts.append(pack(part))
            levels.append([cursor, len(part)])
            cursor += len(part)
        centre_meta.append({"start": levels[0][0], "count": levels[0][1],
                            "levels": levels, "class": int(k)})

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
        **({"shares": extra["shares"]} if extra.get("shares") else {}),
        **({"ground_shares": extra["ground_shares"]} if extra.get("ground_shares") else {}),
        **({"height_order": extra["height_order"]} if "height_order" in extra else {}),
        "cut": bool(args.cut),
        **({"frames": frames_mod.block(geo, turns), "centres": centre_meta}
           if args.frames else {}),
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
    if centre_meta:
        print(f"  and {len(centre_meta)} centres, {sum(m['count'] for m in centre_meta):,} "
              f"splats at level 0")
    print(f"  {cursor:,} splats total, {len(buf) / 1e6:.1f} MB")
    took("fill, holes, levels, written")
    print(f"  wrote {dest}")
    print(f"  wrote {dest.with_suffix('.json')}")
    # Keep the viewer's tileset menu honest without anyone editing a list.
    reindex(dest.parent, quiet=True)
    save_holed()


if __name__ == "__main__":
    main()
