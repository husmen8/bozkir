"""Choosing which squares of a scene become tiles.

Where to cut, how thick a slab to keep, whether a square is ground or a
wall, which patches look alike enough to sit side by side, which splats
survive into coarser levels of detail, and splitting a capture into
material classes.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .tile import PlaneIndex, extract_patch
from .transform import rotate, quat_between


def ground_level(h, thickness):
    """Height of the ground within a patch: the tallest bin of a height
    histogram (the densest thin layer). A low percentile would follow the
    junk reconstructions leave below the surface."""
    lo, hi = np.percentile(h, [1, 99])
    if hi - lo < 1e-6:
        return float(lo)
    bins = max(int((hi - lo) / max(thickness / 4.0, 1e-6)), 8)
    counts, edges = np.histogram(h, bins=bins, range=(lo, hi))
    k = int(np.argmax(counts))
    return float((edges[k] + edges[k + 1]) / 2.0)

def mass_below(p, up_axis, g, thickness, below=0.25):
    """Fraction of a column below the layer taken as ground. Large means the
    wrong layer won the vote (under a tree, the canopy). Local, so it works
    on sloping terrain."""
    h = p.xyz[:, up_axis]
    return float((h < g - thickness * below).mean())

def clip_slab(p, up_axis, thickness, below=0.25):
    """Keep a slab around the ground. GSWT keeps the whole column, fine for
    a flat exemplar; outdoors the column runs up through trees."""
    h = p.xyz[:, up_axis]
    g = ground_level(h, thickness)
    keep = (h >= g - thickness * below) & (h <= g + thickness)
    return p.subset(keep), g

def level_patch(p, up_axis):
    """Rotate a patch so its ground is horizontal, then drop it to zero, so
    patches cut from a slope do not meet at a step."""
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
    """Relief and tilt of the edge band, and relief of the middle.

    A tile can carry a bush and still tile if its edges are flat - edges are
    what neighbours must agree on, and the graph cut only moves the interior.
    Judging the whole patch rejects every interesting square.

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

    # Tilt from the lower 40% of the band only: fitted to the whole band it
    # measures a shell as tall as the slab and reports a near-random angle.
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
    """Screen a splat can cover: area times opacity. Coarser LOD levels keep
    the largest, most opaque splats."""
    sc = np.sort(p.scale, axis=1)
    return sc[:, 2] * sc[:, 1] * p.opacity

def stratified_keep(p, budget, size, up_axis=2):
    """Pick `budget` splats spread evenly over the tile: the best per grid
    bin. Taking the highest-ink splats globally left holes, and kept the
    same splats in every tile built from the same patches (repetitive at
    distance)."""
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
    """Fraction of the tile a top-down render fills.

    The cheap estimate (coverage) can be off by 0.14 and let holed patches
    through. The render loops per splat in Python, so dense patches render a
    random subset of `cap` splats and the density is restored per pixel:
    1 - (1 - alpha)^(1/f) for a fraction f. Per pixel, not on the average,
    which would push empty regions towards full.
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
    """Fast estimate of the fraction of the tile a render would fill.

    Splat area (opacity-weighted) binned on a grid, and per bin the chance a
    point is covered if the splats inside were scattered at random:
    1 - exp(-area / cell). Counting occupied bins, or total area over tile
    area, both miss a patch with everything crammed into one half.
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
    """How much a patch looks like tileable ground; higher is better.
    Wants enough splats, a flat surface, and even coverage."""
    if len(p) < 2000:
        # Same keys as a real result, so callers can read them safely.
        return -1.0, {"splats": len(p), "relief": 0.0, "filled": 0.0,
                      "planarity": 1.0, "edge_relief": 0.0,
                      "edge_tilt": 90.0, "interior_relief": 0.0, "cover": 0.0,
                      "salience": 0.0}

    plane = [i for i in range(3) if i != up_axis]
    h = p.xyz[:, up_axis]
    lo, hi = np.percentile(h, [5, 95])
    relief = float(hi - lo)

    # Flatness of the surface itself, independent of the slab thickness
    # (a patch that is mostly a trunk fails this).
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
    sal = salience(p, size, up_axis)
    info = {"splats": len(p), "relief": relief, "filled": filled,
            "planarity": planarity, "edge_relief": edge_rel,
            "edge_tilt": edge_tilt, "interior_relief": mid_rel,
            "cover": cover, "salience": sal}

    if features:
        # Feature tiles: reward what stands in the middle, punish the rim.
        interest = 1.0 + 3.0 * mid_rel / max(size, 1e-9)
        rim = 1.0 + 12.0 * edge_rel / max(size, 1e-9)
        return float(cover * filled * np.log1p(density) * interest / rim), info

    # Landmarks repeat recognisably, texture does not. Texture reads 2-3 on
    # salience and is untouched; one isolated spot reads tens and is pushed
    # well down, so auto-pick stops choosing patches I was excluding by hand.
    landmark = 1.0 + max(0.0, sal - 4.0) / 4.0
    return float(cover * filled * flatness * np.log1p(density)
                 / (1.0 + 20.0 * planarity) / landmark), info

def pick_patches(s, size, k, up_axis, stride=0.5, thickness=0.3,
                 max_tilt=12.0, max_below=0.5, features=False,
                 edge_flat=0.20, edge_margin=0.22, min_separation=1.0,
                 min_cover=0.80, extract_margin=0.35, cover_margin=0.10,
                 verbose=True, stats=None, progress=None, screen_cap=4_000):
    """Search the ground plane for the k best non-overlapping patches.

    Candidates are cut larger by `extract_margin`, since levelling a square
    leaves empty wedges at the edges that the extra ring fills; scoring looks
    only at the tile-sized middle.

    `stats` receives the rejection tally (which filter rejects most - what
    auto_pick reads to loosen the right one). `progress(done, total,
    viable)` reports, since a search can run for minutes.

    `screen_cap`: splats used by the coverage render while searching (the
    render dominates the search). Survivors are re-measured exactly. On a
    891k-splat scene (~25k splats per tile):

        screen_cap    search      patches
          2,000       19.2 s        22
          4,000       24.7 s        22      <- default
         12,000       49.8 s        22
          exact       91.0 s        22

    Same patches and coverage; 4,000 leaves margin for sparser captures.
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
    records = []
    rendered = 0
    rejected = {"sparse": 0, "buried": 0, "tilt": 0, "rim": 0,
                "holes": 0, "score": 0}
    total = len(xs) * len(ys)
    done = 0
    index = PlaneIndex(s, up_axis, cell=size / 2.0)
    for x in xs:
        for y in ys:
            done += 1
            if progress is not None and (done % 25 == 0 or done == total):
                progress(done, total, len(cands))
            # Cut wide, judge narrow.
            wide = extract_patch(s, [x, y], size * (1.0 + extract_margin),
                                 up_axis=up_axis, index=index)
            p = extract_patch(wide, [0.0, 0.0], size, up_axis=up_axis,
                              recentre=False)
            if len(p) < 2000:
                rejected["sparse"] += 1
                continue
            below = mass_below(p, up_axis, ground_level(p.xyz[:, up_axis],
                                                        thickness), thickness)
            g_level = ground_level(p.xyz[:, up_axis], thickness)
            p, g = clip_slab(p, up_axis, thickness)
            # Same slab for the wide cut, from the middle's ground level.
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
            # Estimate first to throw out hopeless cases; anything that
            # might pass is rendered.
            est = coverage(p, size, up_axis)
            if est < min_cover - cover_margin:
                rejected["holes"] += 1
                continue
            info["cover"] = rendered_coverage(p, size, up_axis,
                                              cap=screen_cap)
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
            # Without the splats, for the search cache.
            records.append({"score": float(sc), "x": float(x), "y": float(y),
                            "info": {k: float(v) for k, v in info.items()
                                     if isinstance(v, (int, float))}})
    if stats is not None:
        stats.update(rejected)
        stats["positions"] = total
        stats["viable"] = len(cands)
        stats["rendered"] = rendered
        stats["records"] = records
    # Quiet unless asked (a sweep reports its own trail); the failure
    # message below travels with the exception either way.
    if verbose:
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
        # Keep patches apart so they are different ground (adjustable: a
        # full tile apart prunes most of a small scene).
        if any(abs(x - cx) < gap and abs(y - cy) < gap
               for _, (cx, cy), _, _ in chosen):
            continue
        chosen.append((sc, (x, y), p, info))
        if len(chosen) >= k:
            break

    # Re-measure the survivors exactly.
    for _, _, p, info in chosen:
        info["cover"] = rendered_coverage(p, size, up_axis)

    if stats is not None:
        stats["chosen"] = len(chosen)
    if verbose and len(chosen) < min(k, len(cands)):
        print(f"  {len(cands)} viable -> {len(chosen)} after keeping centres "
              f"{gap:.2f} apart (lower --min-separation for more)")
    return chosen


# What to loosen when one filter does most of the rejecting: tally key,
# setting, values to try (first is the default). min_cover stops at 0.55,
# about where a patch starts reading as lace rather than ground.
RELAXATIONS = [
    ("holes", "min_cover", [0.80, 0.70, 0.60, 0.55]),
    ("sparse", "size", None),          # handled separately: size changes the grid
    ("tilt", "max_tilt", [12.0, 18.0, 25.0]),
    ("buried", "max_below", [0.5, 0.65, 0.8]),
]

SETTING_NOTE = {
    "min_cover": "coverage",
    "max_tilt": "tilt limit",
    "max_below": "buried-ground limit",
    "min_separation": "separation",
    "size": "tile size",
}


# --- remembering a search ------------------------------------------------
#
# The search is the slow part (ten minutes on a two-million-splat scene),
# and preview_patches and export_wang used to run the same one twice. The
# accepted positions are stored - score, centre, measurements, not splats -
# and patches are simply cut again. The key covers the scene and every
# setting that moves a candidate: a stale list would make `--patches 2`
# silently mean another patch.

# 2: scores divided by the landmark penalty (salience).
# 3: salience stopped counting texture and material mixes as landmarks
#    (version 2 starved the scrub class).
SEARCH_CACHE_VERSION = 3


def scene_fingerprint(s, sample=4096):
    """A short hash of a prepared scene: the count plus positions sampled
    across the whole array."""
    n = len(s)
    if not n:
        return "empty"
    idx = np.linspace(0, n - 1, min(n, sample)).astype(np.int64)
    h = hashlib.sha1()
    h.update(f"{n}|".encode())
    h.update(np.ascontiguousarray(s.xyz[idx], dtype=np.float32).tobytes())
    return h.hexdigest()[:12]


def search_key(s, size, up_axis, settings):
    """Cache key for one search. `min_separation` is left out: it chooses
    among accepted candidates, so changing it reuses the list."""
    parts = [f"v={SEARCH_CACHE_VERSION}", f"scene={scene_fingerprint(s)}",
             f"size={size}", f"up={up_axis}"]
    parts += sorted(f"{k}={v}" for k, v in settings.items()
                    if k != "min_separation")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def load_search(key, cache_dir="data/cache"):
    """The stored candidate list for a key, or None."""
    path = Path(cache_dir) / f"search_{key}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            blob = json.load(f)
    except (OSError, ValueError):
        return None
    if blob.get("version") != SEARCH_CACHE_VERSION:
        return None
    return blob


def save_search(key, cands, stats, cache_dir="data/cache"):
    """Store a candidate list. Failure to write is not failure to search."""
    path = Path(cache_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / f"search_{key}.json", "w") as f:
            json.dump({"version": SEARCH_CACHE_VERSION,
                       "stats": stats,
                       "cands": cands}, f)
    except OSError:
        pass


# The settings that define a search. Every script must pass the same ones:
# `--patches 0,1,2,4` indexes a list only those settings produce.
SEARCH_KEYS = ("stride", "thickness", "max_tilt", "max_below", "features",
               "edge_flat", "edge_margin", "min_separation", "min_cover",
               "cover_margin", "extract_margin")


def search_kwargs(args, thickness=None):
    """The search settings out of a parsed argparse namespace."""
    kw = {k: getattr(args, k) for k in SEARCH_KEYS if hasattr(args, k)}
    if thickness is not None:
        kw["thickness"] = thickness
    return kw


def apply_settings(args, settings):
    """Write a search's resolved settings back onto the namespace, so
    --save-preset records what worked, not what was asked for."""
    for key, value in settings.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def rebuild_candidates(s, records, size, up_axis, thickness,
                       extract_margin=0.35):
    """Cut the scene again at stored centres, in stored order (the cheap
    half of a search)."""
    out = []
    index = PlaneIndex(s, up_axis, cell=size / 2.0) if len(records) > 4 else None
    for rec in records:
        x, y = rec["x"], rec["y"]
        wide = extract_patch(s, [x, y], size * (1.0 + extract_margin),
                             up_axis=up_axis, index=index)
        p = extract_patch(wide, [0.0, 0.0], size, up_axis=up_axis,
                          recentre=False)
        g_level = ground_level(p.xyz[:, up_axis], thickness)
        p, g = clip_slab(p, up_axis, thickness)
        h_wide = wide.xyz[:, up_axis]
        wide = wide.subset((h_wide >= g_level - thickness * 0.25)
                           & (h_wide <= g_level + thickness))
        info = dict(rec["info"])
        info["ground"] = g
        info["wide"] = wide
        out.append((rec["score"], (x, y), p, info))
    return out


def choose(cands, k, size, min_separation=1.0):
    """Take k candidates whose centres are at least `min_separation` tiles
    apart. Separate from the search so it can be re-run for free."""
    ordered = sorted(cands, key=lambda c: -c[0])
    chosen = []
    gap = size * max(min_separation, 0.0)
    for cand in ordered:
        x, y = cand[1]
        if any(abs(x - cx) < gap and abs(y - cy) < gap
               for _, (cx, cy), _, _ in chosen):
            continue
        chosen.append(cand)
        if len(chosen) >= k:
            break
    return chosen


def cached_search(s, size, up_axis, settings, cache_dir="data/cache",
                  cache=True, progress=None, verbose=False):
    """The candidate list for these settings, searched or recalled.
    Returns (cands, stats, hit)."""
    key = search_key(s, size, up_axis, settings)
    if cache:
        blob = load_search(key, cache_dir)
        if blob is not None:
            thickness = settings.get("thickness", size * 0.25)
            cands = rebuild_candidates(
                s, blob["cands"], size, up_axis, thickness,
                settings.get("extract_margin", 0.35))
            stats = dict(blob["stats"])
            stats["records"] = blob["cands"]
            return cands, stats, True

    stats = {}
    # Ask for everything and let choose() narrow it, so one stored list
    # serves every caller.
    pick_patches(s, size, 10 ** 9, up_axis, verbose=verbose, stats=stats,
                 progress=progress, **settings)
    cands = rebuild_candidates(
        s, stats.get("records", []), size, up_axis,
        settings.get("thickness", size * 0.25),
        settings.get("extract_margin", 0.35))
    if cache:
        storable = {k: v for k, v in stats.items() if k != "records"}
        save_search(key, stats.get("records", []), storable, cache_dir)
    return cands, stats, False


def auto_pick(s, size, k, up_axis, verbose=True, progress=None,
              cache=True, cache_dir="data/cache", **kw):
    """Find k patches, loosening whichever filter is doing the rejecting.

    The defaults suit a clean capture of solid ground; a sparser or steeper
    one fails them quietly. The tally says which filter rejected most, and
    each filter has one setting, so the loosening is done here and every
    step is printed with what it produced.

    Returns (chosen, trail): trail is (label, settings, stats, chosen)
    per attempt, first to last.
    """
    trail = []
    settings = dict(kw)
    settings.setdefault("min_cover", 0.80)
    settings.setdefault("max_tilt", 12.0)
    settings.setdefault("max_below", 0.5)
    settings.setdefault("min_separation", 1.0)

    def attempt(label):
        # Separation only re-selects from the list in hand; everything else
        # searches again (from cache when seen before).
        sep = settings.get("min_separation", 1.0)
        search = {kk: vv for kk, vv in settings.items()
                  if kk != "min_separation"}
        try:
            cands, stats, hit = cached_search(
                s, size, up_axis, search, cache_dir=cache_dir, cache=cache,
                progress=None if label.startswith("separation") else progress)
            chosen = choose(cands, k, size, sep)
            for _, _, p, info in chosen:
                info["cover"] = rendered_coverage(p, size, up_axis)
        except SystemExit:
            chosen, stats, hit = [], {}, False
        trail.append((label, dict(settings), stats, len(chosen)))
        if verbose:
            got = f"{len(chosen)} of {k}"
            note = " (recalled)" if hit else ""
            print(f"  {label:<28} {got:>10}   "
                  f"({stats.get('viable', 0)} viable){note}")
        return chosen, stats

    if verbose:
        print("  searching")
    chosen, stats = attempt("defaults")
    if len(chosen) >= k:
        return chosen, trail

    # Separation first: it costs only variety, every other step costs quality.
    if stats.get("viable", 0) > len(chosen):
        for sep in (0.75, 0.5):
            settings["min_separation"] = sep
            chosen, stats = attempt(f"separation {sep}")
            if len(chosen) >= k:
                return chosen, trail

    # Then whichever filter rejects most, as ranked by the tally. A local
    # copy, since exhausted filters are retired from it.
    remaining = [r for r in RELAXATIONS if r[2]]
    while remaining:
        tally = {key: stats.get(key, 0) for key, _, _ in remaining}
        if not any(tally.values()):
            break
        worst = max(tally, key=tally.get)
        setting, steps = next((st, sp) for kk, st, sp in remaining
                              if kk == worst)
        cur = settings.get(setting)
        # min_cover loosens downward, every other setting upward.
        later = [v for v in steps
                 if (v < cur if setting == "min_cover" else v > cur)]
        if not later:
            # As loose as it goes: retire it and try the next worst.
            remaining = [r for r in remaining if r[0] != worst]
            continue
        settings[setting] = later[0]
        chosen, stats = attempt(f"{SETTING_NOTE[setting]} {later[0]}")
        if len(chosen) >= k:
            return chosen, trail

    return chosen, trail


def describe_trail(trail, k, size):
    """One paragraph on what was tried and what it cost, with the flags to
    repeat it by hand."""
    if not trail:
        return "nothing was searched"
    label, settings, stats, got = trail[-1]
    lines = []
    if len(trail) == 1:
        lines.append(f"  the defaults gave {got} patches")
    else:
        first = trail[0][3]
        lines.append(f"  the defaults gave {first}; "
                     f"{len(trail) - 1} setting(s) were loosened to reach {got}")
        changed = []
        base = trail[0][1]
        for key, val in settings.items():
            if base.get(key) != val:
                changed.append(f"--{key.replace('_', '-')} {val}")
        if changed:
            lines.append("  to repeat this by hand: " + " ".join(changed))
    if got < k:
        lines.append(f"  still short of {k}. The tally below says what is "
                     f"rejecting; a capture that cannot reach {k} at any "
                     f"setting is not a tiling exemplar.")
    return "\n".join(lines)


def balanced_split(patches, colours):
    """Split patches between the two axes so both halves look alike.

    In score order the best land on one axis, and if they differ every
    boundary one way is a different material from every boundary the other
    way: the grid gets a grain. Every split is tried and the closest-matched
    kept (three splits for four patches).
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


def salience(p, size, up_axis=2, grid=16, min_count=6):
    """How much a patch contains something the eye will find again.

    Repetition is noticed through landmarks (a pale opening in scrub, a
    stone, a marker), not texture. A landmark sits at the same place in
    every tile built from its patch, so across a grid it forms a lattice.

    Measured on a coarse top-down luminance grid: how far the most extreme
    3% of cells stand from the median, in robust units (MAD), faded out when
    many cells stand out. Texture scores ~1-3, one isolated spot tens, and a
    busy or mixed patch low again.
    """
    plane = [i for i in range(3) if i != up_axis]
    xy = p.xyz[:, plane]
    if len(xy) < 50:
        return 0.0
    lo = xy.min(axis=0)
    ij = np.clip(((xy - lo) / max(size, 1e-6) * grid).astype(int), 0, grid - 1)
    cell = ij[:, 0] * grid + ij[:, 1]
    rgb = p.base_rgb
    lum = rgb @ np.array([0.299, 0.587, 0.114])
    count = np.bincount(cell, minlength=grid * grid)
    total = np.bincount(cell, weights=lum, minlength=grid * grid)
    ok = count >= min_count
    if ok.sum() < grid:
        return 0.0
    mean = total[ok] / count[ok]
    med = float(np.median(mean))
    mad = float(np.median(np.abs(mean - med))) * 1.4826
    z = np.abs(mean - med) / max(mad, 1e-4)
    # The worst 3% of cells: one small spot, not a single noisy cell.
    k = max(2, int(round(0.03 * len(z))))
    top = float(np.sort(z)[-k:].mean())
    # A landmark is rare as well as different. In scrub (dark bushes on
    # sand) a third of cells are outliers; that is texture, and scoring it
    # starved the scrub class. Fade out past what one spot can occupy.
    share = float((z > 4.0).mean())
    return top * min(1.0, max(0.0, (0.12 - share) / 0.08))


def purity(p, own, others, up_axis=2, grid=12, min_count=4):
    """Share of a patch's ground that looks like its own material, 0..1.

    The desert's scrub patches each held a pale opening of bare sand, which
    became what repeated once rotations made colours cheap. Top-down cells
    are assigned to the nearest class mean colour (`own` vs `others`); the
    share nearest `own` lets a class prefer its purest patches.
    """
    plane = [i for i in range(3) if i != up_axis]
    xy = p.xyz[:, plane]
    if len(xy) < 50:
        return 0.0
    lo = xy.min(axis=0)
    span = max(float((xy.max(axis=0) - lo).max()), 1e-6)
    ij = np.clip(((xy - lo) / span * grid).astype(int), 0, grid - 1)
    cell = ij[:, 0] * grid + ij[:, 1]
    w = np.asarray(p.opacity, dtype=np.float64)
    rgb = p.base_rgb
    count = np.bincount(cell, minlength=grid * grid)
    wsum = np.bincount(cell, weights=w, minlength=grid * grid)
    ok = (count >= min_count) & (wsum > 0)
    if not ok.any():
        return 0.0
    mean = np.stack([np.bincount(cell, weights=w * rgb[:, c],
                                 minlength=grid * grid)[ok] / wsum[ok]
                     for c in range(3)], axis=1)
    d_own = np.linalg.norm(mean - np.asarray(own), axis=1)
    d_other = np.min([np.linalg.norm(mean - np.asarray(o), axis=1)
                      for o in others], axis=0)
    return float((d_own <= d_other).mean())


def appearance(p):
    """Colour signature: per-channel mean and spread. Patches that differ
    here show a seam however wide the feather."""
    rgb = p.base_rgb
    return np.concatenate([rgb.mean(axis=0), rgb.std(axis=0)])

def split_classes(cands, classes=2, per_class=4, weight=1.0, seed=0):
    """Split candidates into groups that look unlike each other (material
    classes), the opposite of select_similar.

    The capture usually holds both: the desert's candidates spread 0.129 in
    appearance, the four chosen 0.043 - the similarity filter had discarded
    the scrub. k-means on the colour signature, seeded on the two furthest
    apart, so it is deterministic and `--patches` indices stay valid.

    Returns `classes` lists, largest first, each narrowed to its
    `per_class` most alike. Small groups are returned anyway so a
    one-material capture can say so.
    """
    if classes < 2 or len(cands) < classes:
        return [cands[:per_class]]

    feats = np.array([appearance(c[2]) for c in cands], dtype=np.float64)
    # Standardised so the means do not dominate the spreads - with a floor
    # at a tenth of the largest deviation: the spreads barely vary, and
    # dividing by their tiny deviation amplified noise until it put a dark
    # patch in with the pale ones.
    sd = feats.std(axis=0)
    sd = np.maximum(sd, 0.1 * max(sd.max(), 1e-9))
    z = (feats - feats.mean(axis=0)) / sd

    # Seed on the pair furthest apart, then furthest from all seeds.
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    a, b = np.unravel_index(int(np.argmax(d2)), d2.shape)
    centres = [z[a], z[b]]
    while len(centres) < classes:
        # Furthest from everything seeded so far.
        dist = np.min([np.linalg.norm(z - c, axis=1) for c in centres], axis=0)
        centres.append(z[int(np.argmax(dist))])

    labels = np.zeros(len(cands), dtype=np.int64)
    for _ in range(50):
        dist = np.stack([np.linalg.norm(z - c, axis=1) for c in centres])
        new = dist.argmin(axis=0)
        if (new == labels).all() and _ > 0:
            break
        labels = new
        for ci in range(len(centres)):
            members = z[labels == ci]
            if len(members):
                centres[ci] = members.mean(axis=0)

    groups = []
    for ci in range(len(centres)):
        members = [cands[i] for i in range(len(cands)) if labels[i] == ci]
        groups.append(members)
    groups.sort(key=len, reverse=True)

    return [select_similar(g, per_class, weight) if len(g) > per_class
            else g for g in groups]


def class_separation(groups):
    """Gap between two groups' mean appearance over the spread within them.
    Below ~1 the split is noise, not two materials."""
    means = []
    spreads = []
    for g in groups:
        if not g:
            continue
        f = np.array([appearance(c[2]) for c in g], dtype=np.float64)
        means.append(f.mean(axis=0))
        spreads.append(float(np.linalg.norm(f - f.mean(axis=0), axis=1).mean())
                       if len(f) > 1 else 0.0)
    if len(means) < 2:
        return 0.0
    gap = float(np.linalg.norm(means[0] - means[1]))
    within = max(float(np.mean(spreads)), 1e-9)
    return gap / within


def select_similar(cands, k, weight=1.0):
    """Choose k patches that score well and look like each other. `weight`
    trades appearance against score (0 = top k by score).

    The seed is good and typical, not simply the best: on a beach the top
    score can be sea foam, and an outlier seed drags the set towards it.
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
