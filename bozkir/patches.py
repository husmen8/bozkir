"""Choosing which squares of a scene become tiles.

Everything here decides what goes into a tile set: where to cut, how thick a
slab to keep, whether a square is ground or a wall, whether four patches
look enough alike to sit next to each other, and which splats survive into a
coarser level of detail.

It used to live inside scripts/export_tileset.py, which made it library code
in a command-line tool - awkward to import and invisible to an editor.
"""

import numpy as np

from .tile import extract_patch
from .transform import rotate, quat_between


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

def ground_field(p, size, up_axis=2, grid=12, low_pct=20.0):
    """A coarse height map of the ground across a patch.

    clip_slab uses one level for the whole patch, which is fine on a flat
    square and wrong on anything that undulates: a flat band cuts into the
    high ground and takes air over the low ground. Estimating a height per
    bin instead lets the band follow the surface.

    Bins take a low percentile rather than a minimum, since reconstructions
    leave junk underneath, and the result is median-filtered because a bin
    holding only a few splats is not to be trusted. Empty bins are filled
    from the patch as a whole.

    Returns a (grid, grid) array of heights.
    """
    plane = [i for i in range(3) if i != up_axis]
    half = size / 2.0
    xy = p.xyz[:, plane]
    h = p.xyz[:, up_axis]

    ij = np.clip(((xy + half) / max(size, 1e-9) * grid).astype(int), 0, grid - 1)
    key = ij[:, 0] * grid + ij[:, 1]

    field = np.full(grid * grid, np.nan)
    order = np.lexsort((h, key))
    ks, starts = np.unique(key[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    for k, a, b in zip(ks, starts, ends):
        if b - a >= 8:
            field[k] = np.percentile(h[order[a:b]], low_pct)

    field = field.reshape(grid, grid)
    fallback = float(np.nanmedian(field)) if np.isfinite(field).any() \
        else float(np.percentile(h, low_pct))
    field = np.where(np.isfinite(field), field, fallback)

    # 3x3 median, so one odd bin cannot pull the surface with it.
    padded = np.pad(field, 1, mode="edge")
    stack = np.stack([padded[i:i + grid, j:j + grid]
                      for i in range(3) for j in range(3)])
    return np.median(stack, axis=0)


def height_above_ground(p, field, size, up_axis=2):
    """Each Gaussian's height above the local ground, by bilinear lookup."""
    plane = [i for i in range(3) if i != up_axis]
    grid = field.shape[0]
    half = size / 2.0
    u = np.clip((p.xyz[:, plane[0]] + half) / max(size, 1e-9) * grid - 0.5,
                0, grid - 1)
    v = np.clip((p.xyz[:, plane[1]] + half) / max(size, 1e-9) * grid - 0.5,
                0, grid - 1)
    i0, j0 = u.astype(int), v.astype(int)
    i1 = np.minimum(i0 + 1, grid - 1)
    j1 = np.minimum(j0 + 1, grid - 1)
    fu, fv = u - i0, v - j0
    g = (field[i0, j0] * (1 - fu) * (1 - fv) + field[i1, j0] * fu * (1 - fv)
         + field[i0, j1] * (1 - fu) * fv + field[i1, j1] * fu * fv)
    return p.xyz[:, up_axis] - g






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

def patch_tilt(p, up_axis):
    """Angle between a patch's own surface normal and the scene's up axis."""
    if len(p) < 50:
        return 90.0
    d = p.xyz - p.xyz.mean(axis=0)
    _, evecs = np.linalg.eigh((d.T @ d) / len(d))
    n = evecs[:, 0]
    return float(np.degrees(np.arccos(np.clip(abs(n[up_axis]), 0, 1))))

def band_stats(p, up_axis, size, margin=0.22):
    """Relief and tilt measured separately for the edge band and the middle.

    A tile can carry a boulder or a bush and still tile, as long as its
    edges stay flat: the edge strips are what two neighbouring tiles have to
    agree on, and the graph cut only ever moves boundaries in the interior.
    Judging the whole patch at once throws away every interesting square,
    which is why the terrain ends up as the same flat ground repeated.

    Returns (edge_relief, edge_tilt, interior_relief).
    """
    plane = [i for i in range(3) if i != up_axis]
    h = p.xyz[:, up_axis]
    r = np.max(np.abs(p.xyz[:, plane]), axis=1) / max(size / 2.0, 1e-9)
    edge = r > 1.0 - margin
    mid = ~edge

    def relief(sel):
        if sel.sum() < 50:
            return 0.0
        lo, hi = np.percentile(h[sel], [5, 95])
        return float(hi - lo)

    # Tilt is about whether the ground at the rim is level, so the plane is
    # fitted to the lower part of the band only. Fitting the whole band
    # measures the shape of a shell as tall as the slab, which says nothing
    # about the ground and reports a near-random angle once the slab is
    # thick enough to hold anything standing up.
    tilt = 90.0
    if edge.sum() >= 50:
        eh = h[edge]
        low = eh <= np.percentile(eh, 40.0)
        q = p.subset(np.flatnonzero(edge)[low])
        if len(q) >= 30:
            d = q.xyz - q.xyz.mean(axis=0)
            _, evecs = np.linalg.eigh((d.T @ d) / len(d))
            n = evecs[:, 0]
            tilt = float(np.degrees(np.arccos(np.clip(abs(n[up_axis]), 0, 1))))

    return relief(edge), tilt, relief(mid)

def part_ink(p):
    """How much screen a splat can cover: its area times its opacity.

    Used to decide which splats survive into a coarser level. Keeping the
    largest and most opaque preserves the overall look while dropping the
    fine detail that a distant tile could not resolve anyway.
    """
    sc = np.sort(p.scale, axis=1)
    return sc[:, 2] * sc[:, 1] * p.opacity

def stratified_keep(p, budget, size, up_axis=2):
    """Pick `budget` splats spread evenly over the tile.

    Taking the highest-ink splats globally concentrates them wherever the
    tile happens to be brightest, leaves holes elsewhere, and - because
    every tile is built from the same few source patches - keeps nearly the
    same set in every tile, so the tiling looks repetitive at distance.

    Binning the tile into a grid and keeping the best from each bin fixes
    both: coverage stays even, and each tile keeps what is locally
    distinctive about it rather than what is globally brightest.
    """
    if budget >= len(p):
        return np.arange(len(p))

    ink = part_ink(p)
    plane = [i for i in range(3) if i != up_axis]
    g = max(int(np.ceil(np.sqrt(budget))), 1)
    xy = p.xyz[:, plane]
    ij = np.clip(((xy + size / 2.0) / max(size, 1e-9) * g).astype(int), 0, g - 1)
    cell = ij[:, 0] * g + ij[:, 1]

    # Best splat per occupied bin, found by sorting once.
    order = np.lexsort((-ink, cell))
    first = np.ones(len(order), dtype=bool)
    first[1:] = cell[order][1:] != cell[order][:-1]
    keep = order[first]

    if len(keep) > budget:                      # more bins than budget
        keep = keep[np.argsort(-ink[keep])[:budget]]
    elif len(keep) < budget:                    # empty bins, top up globally
        rest = np.setdiff1d(np.argsort(-ink), keep, assume_unique=False)
        keep = np.concatenate([keep, rest[:budget - len(keep)]])
    return np.sort(keep)

def rendered_coverage(p, size, up_axis=2, resolution=80, cap=12_000):
    """What fraction of the tile a top-down render actually fills.

    The bin estimate below is cheap and roughly right, but it and a render
    can disagree by 0.14 - it reported over 80% for patches a render showed
    as half empty, so patches with a hole in them passed the filter.

    The rasterizer loops per splat in Python, so a dense patch costs
    seconds. Rendering a random subset instead is proportionally faster,
    and the density it loses can be put back: a pixel covered with
    probability 1 - exp(-d) at full density reads 1 - exp(-f*d) with a
    fraction f of the splats, so raising (1 - alpha) to the power 1/f
    recovers it. Per pixel, not overall - correcting the average instead
    pushes empty regions toward one and reports a half-empty patch as full.
    """
    from .graphcut import render_patch
    n = len(p)
    if not n:
        return 0.0
    if n <= cap:
        return float(render_patch(p, size, resolution, up_axis)[1].mean())

    idx = np.sort(np.random.default_rng(0).choice(n, size=cap, replace=False))
    alpha = render_patch(p.subset(idx), size, resolution, up_axis)[1]
    f = cap / n
    return float(np.mean(1.0 - (1.0 - np.minimum(alpha, 0.999)) ** (1.0 / f)))


def coverage(p, size, up_axis=2, grid=16):
    """Roughly what fraction of the tile square a render would fill.

    Two wrong ways to do this. Counting how many bins hold at least one
    splat says a bin with three is as good as one with three thousand.
    Summing all the splat area and dividing by the tile area never looks at
    where the splats are, so a patch with everything crammed into half the
    square scores the same as an even one - which is the case that matters,
    because that is what a half-reconstructed patch looks like.

    So: bin the area, and ask per bin how likely a point in it is covered,
    treating the splats inside as scattered at random. Averaging those gives
    a number that tracks a render.
    """
    if not len(p):
        return 0.0
    plane = [i for i in range(3) if i != up_axis]
    half = size / 2.0
    xy = p.xyz[:, plane]
    inside = np.all(np.abs(xy) <= half, axis=1)
    if not inside.any():
        return 0.0

    sc = np.sort(p.scale[inside], axis=1)
    # Two-sigma footprint, weighted by how opaque the splat is.
    area = np.pi * (2 * sc[:, 2]) * (2 * sc[:, 1]) * p.opacity[inside]

    ij = np.clip(((xy[inside] + half) / size * grid).astype(int), 0, grid - 1)
    binned = np.bincount(ij[:, 0] * grid + ij[:, 1], weights=area,
                         minlength=grid * grid)
    cell = (size / grid) ** 2
    return float(np.mean(1.0 - np.exp(-binned / cell)))

def score_patch(p, up_axis, size, features=False, edge_margin=0.22):
    """How much a patch looks like tileable ground. Higher is better.

    Three things are wanted: enough splats to render, a surface that is
    flat rather than a wall or an object, and coverage spread across the
    whole square rather than clustered in one corner.
    """
    if len(p) < 2000:
        # Same keys as a real result: callers read these before checking
        # the score, and a bare dict turns a sparse patch into a crash.
        return -1.0, {"splats": len(p), "relief": 0.0, "filled": 0.0,
                      "planarity": 1.0, "edge_relief": 0.0,
                      "edge_tilt": 90.0, "interior_relief": 0.0, "cover": 0.0}

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

    edge_rel, edge_tilt, mid_rel = band_stats(p, up_axis, size, edge_margin)
    cover = coverage(p, size, up_axis)
    info = {"splats": len(p), "relief": relief, "filled": filled,
            "planarity": planarity, "edge_relief": edge_rel,
            "edge_tilt": edge_tilt, "interior_relief": mid_rel,
            "cover": cover}

    if features:
        # Reward what stands in the middle, punish anything at the rim.
        # The scoring above does the opposite, which is correct for a plain
        # ground tile and wrong for one meant to carry something.
        interest = 1.0 + 3.0 * mid_rel / max(size, 1e-9)
        rim = 1.0 + 12.0 * edge_rel / max(size, 1e-9)
        return float(cover * filled * np.log1p(density) * interest / rim), info

    return float(cover * filled * flatness * np.log1p(density)
                 / (1.0 + 20.0 * planarity)), info

def pick_patches(s, size, k, up_axis, stride=0.5, thickness=0.3,
                 max_tilt=12.0, max_below=0.5, features=False,
                 edge_flat=0.20, edge_margin=0.22, min_separation=1.0,
                 min_cover=0.80, extract_margin=0.35, cover_margin=0.10,
                 verbose=True):
    """Search the ground plane for the k best non-overlapping patches.

    Each candidate is cut larger than the tile it will become, by
    `extract_margin`. Levelling rotates a patch to sit flat, and rotating a
    square leaves empty wedges along its edges - so the extra ring is the
    material those wedges are filled from. Scoring still looks only at the
    tile-sized middle, since that is what ends up on screen.
    """
    plane = [i for i in range(3) if i != up_axis]
    lo = np.percentile(s.xyz[:, plane], 2, axis=0)
    hi = np.percentile(s.xyz[:, plane], 98, axis=0)

    step = size * stride
    xs = np.arange(lo[0] + size / 2, hi[0] - size / 2 + 1e-6, step)
    ys = np.arange(lo[1] + size / 2, hi[1] - size / 2 + 1e-6, step)
    if len(xs) == 0 or len(ys) == 0:
        raise SystemExit(f"scene is smaller than one {size} patch")

    cands = []
    rendered = 0
    rejected = {"sparse": 0, "buried": 0, "tilt": 0, "rim": 0,
                "holes": 0, "score": 0}
    for x in xs:
        for y in ys:
            # Cut wide, judge narrow.
            wide = extract_patch(s, [x, y], size * (1.0 + extract_margin),
                                 up_axis=up_axis)
            p = extract_patch(wide, [0.0, 0.0], size, up_axis=up_axis,
                              recentre=False)
            if len(p) < 2000:
                rejected["sparse"] += 1
                continue
            below = mass_below(p, up_axis, ground_level(p.xyz[:, up_axis],
                                                        thickness), thickness)
            g_level = ground_level(p.xyz[:, up_axis], thickness)
            p, g = clip_slab(p, up_axis, thickness)
            # The wide version gets the same slab, using the middle's ground
            # level so both keep the same band.
            h_wide = wide.xyz[:, up_axis]
            wide = wide.subset((h_wide >= g_level - thickness * 0.25)
                               & (h_wide <= g_level + thickness))
            if below > max_below:
                rejected["buried"] += 1
                continue
            tilt = (band_stats(p, up_axis, size, edge_margin)[1] if features
                    else patch_tilt(p, up_axis))
            if tilt > max_tilt:
                rejected["tilt"] += 1
                continue
            sc, info = score_patch(p, up_axis, size, features, edge_margin)
            if sc <= 0:
                rejected["score"] += 1
                continue
            # The bin estimate is far faster but can be off by 0.14, which
            # is enough to let a holed patch through. It is only used to
            # throw out the hopeless cases; anything that might pass gets
            # measured properly.
            est = coverage(p, size, up_axis)
            if est < min_cover - cover_margin:
                rejected["holes"] += 1
                continue
            info["cover"] = rendered_coverage(p, size, up_axis)
            rendered += 1
            if info["cover"] < min_cover:
                rejected["holes"] += 1
                continue
            if features and info["edge_relief"] > edge_flat * size:
                rejected["rim"] += 1
                continue
            info["ground"] = g
            info["tilt"] = tilt
            info["below"] = below
            info["wide"] = wide
            cands.append((sc, (float(x), float(y)), p, info))
    if verbose or not cands:
        print(f"  searched {len(xs)}x{len(ys)} positions, "
              f"{len(cands)} viable")
        print(f"  rejected: {rejected['sparse']} too sparse, "
              f"{rejected['buried']} ground buried "
              f"(>{max_below:.0%} of the column below it), "
              f"{rejected['tilt']} too tilted (>{max_tilt:.0f} deg), "
              f"{rejected['rim']} rim not flat, "
              f"{rejected['holes']} too many holes (<{min_cover:.0%} covered), "
              f"{rejected['score']} low score")
        print(f"  coverage: {rendered} of {len(xs) * len(ys)} positions "
              f"measured by render, the rest ruled out by estimate")
    if not cands:
        raise SystemExit(
            "  nothing passed. The counts above say which filter to loosen:\n"
            "    too sparse   -> larger --size, or higher --radius-pct\n"
            "    ground buried-> thicker --thickness, or higher --max-below\n"
            "    too tilted   -> higher --max-tilt, or the region is a slope\n"
            "    rim not flat -> higher --edge-flat, or a smaller --size\n"
            "    holes        -> lower --min-cover, or a smaller --size")

    cands.sort(key=lambda c: -c[0])

    chosen = []
    gap = size * max(min_separation, 0.0)
    for sc, (x, y), p, info in cands:
        # Keep patches apart so the set has genuinely different content.
        # A full tile apart is the safe default, but on a small scene it
        # prunes away most of what passed the filters, so it is adjustable.
        if any(abs(x - cx) < gap and abs(y - cy) < gap
               for _, (cx, cy), _, _ in chosen):
            continue
        chosen.append((sc, (x, y), p, info))
        if len(chosen) >= k:
            break

    if verbose and len(chosen) < min(k, len(cands)):
        print(f"  {len(cands)} viable -> {len(chosen)} after keeping centres "
              f"{gap:.2f} apart (lower --min-separation for more)")
    return chosen

def balanced_split(patches, colours):
    """Split the chosen patches into the two axes so both look alike.

    One set supplies the north and south edge colours, the other east and
    west. Taking them in score order puts the best two on one axis and the
    rest on the other, so if any patch is darker or rougher than its
    fellows, every boundary running one way is made of different material
    from every boundary running the other. The grid then has a grain, and
    the scene changes character each quarter turn of the camera.

    Every way of dealing the patches into two sets is tried, and the one
    whose halves match closest is kept. With four patches that is three
    splits; it stays small for the sizes a tile set is ever built at.
    """
    import itertools

    n = len(patches)
    if n != 2 * colours or colours < 1:
        return patches[:colours], patches[colours:2 * colours]

    feats = [appearance(p) for p in patches]
    best, best_gap = None, None
    for combo in itertools.combinations(range(n), colours):
        if 0 not in combo:          # each split appears twice; keep one
            continue
        rest = [i for i in range(n) if i not in combo]
        gap = float(np.linalg.norm(np.mean([feats[i] for i in combo], axis=0)
                                   - np.mean([feats[i] for i in rest], axis=0)))
        if best_gap is None or gap < best_gap:
            best, best_gap = (combo, rest), gap

    combo, rest = best
    return ([patches[i] for i in combo], [patches[i] for i in rest]), best_gap


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