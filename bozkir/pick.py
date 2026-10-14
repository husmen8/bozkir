"""Which candidates a material's tiles are made of: edge patches on ground
of their own and alike, frames without salient spots, and a pool of
centres (image quilting's choice by fit, Efros and Freeman 2001, is the
idea behind choosing among candidates rather than taking the best)."""

import numpy as np

from . import frames as frames_mod
from .materials import haze
from .patches import appearance


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
            # (that is a shifted copy, 1.0 and 1.5 a step apart).
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
    took exactly those: on the desert one material's 18 centres came from a
    3 x 2.4 m spot and the same tuft showed in tile after tile. None
    thinner than the thinnest edge patch, which it sits between: a blurrier
    centre shows as a square. Nor further in colour from the edges' mean
    than the furthest edge patch, plus 0.04: on the desert the bush took
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


def dense_first(cands, at_least, keep=0.5):
    """Densest first, without the patches under `keep` of the material's
    dense (75th percentile) splat count, unless fewer than `at_least` are
    left: on the desert thin patches have a quarter of the splats and show
    as pale blurs, and --apart preferred them to dense ones that overlap."""
    order = sorted(cands, key=lambda c: -len(c[2]))
    if not order:
        return order
    floor = keep * float(np.percentile([len(c[2]) for c in order], 75))
    dense = [c for c in order if len(c[2]) >= floor]
    return order[:max(len(dense), at_least)]


_LUM = {}


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
    ground still gets some (taking a quiet subset first left the desert with
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
