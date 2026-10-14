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

def rendered_coverage(p, size, up_axis=2, resolution=80, cap=12_000, boost=1.0):
    """Fraction of the tile a top-down render fills.

    The cheap estimate (coverage) can be off by 0.14 and let holed patches
    through. The render loops per splat in Python, so dense patches render a
    random subset of `cap` splats and the density is restored per pixel:
    1 - (1 - alpha)^(1/f) for a fraction f. Per pixel, not on the average,
    which would push empty regions towards full.

    `boost`: the coverage after each opacity is
    raised to 1 - (1 - a)^k (what fill_opacity does), from one render. Exact
    per pixel: compositing multiplies the (1 - a) of the splats. A tuple
    gives a list, one per value, from the same render.
    """
    from .graphcut import render_patch
    n = len(p)
    one = isinstance(boost, (int, float))
    ks = (boost,) if one else tuple(boost)
    if not n:
        return 0.0 if one else [0.0] * len(ks)
    f = 1.0
    if n <= cap:
        alpha = render_patch(p, size, resolution, up_axis)[1]
    else:
        idx = np.sort(np.random.default_rng(0).choice(n, size=cap, replace=False))
        alpha = render_patch(p.subset(idx), size, resolution, up_axis)[1]
        f = cap / n
    keep = 1.0 - np.minimum(alpha, 0.999)
    out = [float(alpha.mean()) if k / f == 1.0 else float(np.mean(1.0 - keep ** (k / f)))
           for k in ks]
    return out[0] if one else out


def coverage(p, size, up_axis=2, grid=16, boost=1.0):
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
    # Two-sigma footprint, weighted by how opaque the splat is (after a
    # fill of `boost`, see rendered_coverage).
    op = p.opacity[inside].astype(np.float64)
    if boost != 1.0:
        op = 1.0 - (1.0 - np.minimum(op, 0.999)) ** boost
    area = np.pi * (2 * sc[:, 2]) * (2 * sc[:, 1]) * op

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
    # well down, below patches with no landmark.
    landmark = 1.0 + max(0.0, sal - 4.0) / 4.0
    return float(cover * filled * flatness * np.log1p(density)
                 / (1.0 + 20.0 * planarity) / landmark), info

def pick_patches(s, size, k, up_axis, stride=0.5, thickness=0.3,
                 max_tilt=12.0, max_below=0.5, features=False,
                 edge_flat=0.20, edge_margin=0.22, min_separation=1.0,
                 min_cover=0.80, extract_margin=0.35, cover_margin=0.10,
                 verbose=True, stats=None, progress=None, screen_cap=4_000,
                 faint_fill=0.0):
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

    `faint_fill`: judge coverage as it will be after the export's fill of at
    most this many times the opacity (0 = as captured). A faint but whole
    layer then passes; real gaps, with no splat to raise, still fail. Holes
    are recorded as "faint" (the export's full fill, 6x, would cover them)
    or "gaps" (it would not).
    """
    full_fill = 6.0
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
    where = []          # (x, y, reason) of each rejected position
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
                where.append((float(x), float(y), "sparse"))
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
                where.append((float(x), float(y), "buried"))
                continue
            tilt = (band_stats(p, up_axis, size, edge_margin)[1] if features
                    else patch_tilt(p, up_axis))
            if tilt > max_tilt:
                rejected["tilt"] += 1
                where.append((float(x), float(y), "tilt"))
                continue
            sc, info = score_patch(p, up_axis, size, features, edge_margin)
            if sc <= 0:
                rejected["score"] += 1
                where.append((float(x), float(y), "score"))
                continue
            # Estimate first to throw out hopeless cases; anything that
            # might pass is rendered.
            est = coverage(p, size, up_axis, boost=faint_fill or 1.0)
            if est < min_cover - cover_margin:
                rejected["holes"] += 1
                full = coverage(p, size, up_axis, boost=full_fill)
                where.append((float(x), float(y),
                              "faint" if full >= min_cover else "gaps"))
                continue
            info["cover"], judged, full = rendered_coverage(
                p, size, up_axis, cap=screen_cap,
                boost=(1.0, faint_fill or 1.0, full_fill))
            rendered += 1
            if judged < min_cover:
                rejected["holes"] += 1
                where.append((float(x), float(y),
                              "faint" if full >= min_cover else "gaps"))
                continue
            if features and info["edge_relief"] > edge_flat * size:
                rejected["rim"] += 1
                where.append((float(x), float(y), "rim"))
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
        stats["rejected_at"] = where
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
               "cover_margin", "extract_margin", "faint_fill")


def search_kwargs(args, thickness=None):
    """The search settings out of a parsed argparse namespace."""
    kw = {k: getattr(args, k) for k in SEARCH_KEYS if hasattr(args, k)}
    if not kw.get("faint_fill"):
        kw.pop("faint_fill", None)       # off: same key as before it existed
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


_RECALLED = {}          # this run's recalled searches, by key


def cached_search(s, size, up_axis, settings, cache_dir="data/cache",
                  cache=True, progress=None, verbose=False, where=False):
    """The candidate list for these settings, searched or recalled.
    Returns (cands, stats, hit). `where`: the stats must say where each
    rejected position was (older caches do not, and are searched again)."""
    key = search_key(s, size, up_axis, settings)
    if cache and key in _RECALLED:
        # Cut once per run: rebuilding 800 candidates took seconds, and one
        # export asks for the same list three times.
        cands, stats = _RECALLED[key]
        if not (where and "rejected_at" not in stats):
            return list(cands), dict(stats), True
    if cache:
        blob = load_search(key, cache_dir)
        if blob is not None and where and (
                "rejected_at" not in blob["stats"]
                or any(r[2] == "holes" for r in blob["stats"]["rejected_at"])):
            blob = None                  # no record of where, or holes unsplit
        if blob is not None:
            thickness = settings.get("thickness", size * 0.25)
            cands = rebuild_candidates(
                s, blob["cands"], size, up_axis, thickness,
                settings.get("extract_margin", 0.35))
            stats = dict(blob["stats"])
            stats["records"] = blob["cands"]
            _RECALLED[key] = (cands, stats)
            return list(cands), dict(stats), True

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
        _RECALLED[key] = (cands, stats)
    return list(cands), stats, False


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
            # The search measured it; a recalled list keeps it. Rendering it
            # again took most of a minute on the desert, for nothing new.
            for _, _, p, info in chosen:
                if "cover" not in info:
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


def split_materials(cands, classes=3, tries=24, seed=0, peers=6):
    """Candidates in `classes` groups by colour signature, as
    split_classes, but robust to a few odd patches: seeding on the
    furthest-apart pair, as split_classes does, lets three markers become
    a material of their own. k-means from many seeded starts, keeping the
    tightest; a group of fewer than `peers` patches (a material needs that
    many anyway) is set aside as odd and the rest clustered again, and the
    odd ones join their nearest group at the end. Judged by group size, not
    by how tightly packed a patch's neighbours are: with 800 positions the
    sand packed so tightly that all of the desert's bush counted as odd.
    Deterministic. Groups largest first, not narrowed."""
    if len(cands) < classes:
        return [list(cands)]
    feats = np.array([appearance(c[2]) for c in cands], dtype=np.float64)
    sd = feats.std(axis=0)
    sd = np.maximum(sd, 0.1 * max(sd.max(), 1e-9))
    everyone = (feats - feats.mean(axis=0)) / sd

    def kmeans(z):
        rng = np.random.default_rng(seed)
        best, best_cost = None, np.inf
        for _ in range(tries):
            # k-means++ start: each next centre far from those already chosen.
            centres = [z[rng.integers(len(z))]]
            while len(centres) < classes:
                d = np.min([((z - c) ** 2).sum(1) for c in centres], axis=0)
                centres.append(z[rng.choice(len(z), p=d / max(d.sum(), 1e-12))])
            centres = np.array(centres)
            for _ in range(100):
                labels = np.argmin(((z[:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
                new = np.array([z[labels == k].mean(0) if (labels == k).any() else centres[k]
                                for k in range(classes)])
                if np.allclose(new, centres):
                    break
                centres = new
            cost = float(((z - centres[labels]) ** 2).sum())
            if cost < best_cost - 1e-9:
                best, best_cost = centres, cost
        return best

    use = np.arange(len(everyone))
    for _ in range(len(everyone)):
        centres = kmeans(everyone[use])
        labels = np.argmin(((everyone[use][:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
        small = np.bincount(labels, minlength=classes) < peers
        if not small.any() or (~small[labels]).sum() < classes:
            break
        use = use[~small[labels]]
    best = np.argmin(((everyone[:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
    groups = [[cands[i] for i in np.flatnonzero(best == k)] for k in range(classes)]
    return sorted([g for g in groups if g], key=len, reverse=True)


def thickness_map(p, up_axis=2, cell=0.1):
    """How thick the layer of splats is at each `cell` square (m): the
    spread of their heights, weighted by how much each one shows from above
    (opacity times footprint). Returns (grid, (x0, y0)); NaN where fewer
    than 3 splats. Ground the photos saw well is a thin crust of splats; a
    thick haze of see-through ones looks sharp only from afar, as desert
    v1's bush does."""
    plane = [i for i in range(3) if i != up_axis]
    xy = p.xyz[:, plane].astype(np.float64)
    z = p.xyz[:, up_axis].astype(np.float64)
    sc = np.sort(p.scale, axis=1).astype(np.float64)
    w = p.opacity * np.minimum(sc[:, 2] * sc[:, 1], cell * cell)
    lo = xy.min(axis=0)
    ij = np.floor((xy - lo) / cell).astype(np.int64)
    nx, ny = ij.max(axis=0) + 1
    key = ij[:, 1] * nx + ij[:, 0]
    W = np.bincount(key, w, nx * ny)
    m = np.bincount(key, w * z, nx * ny) / np.maximum(W, 1e-12)
    v = np.bincount(key, w * z * z, nx * ny) / np.maximum(W, 1e-12) - m * m
    out = np.sqrt(np.maximum(v, 0.0))
    out[np.bincount(key, minlength=nx * ny) < 3] = np.nan
    return out.reshape(ny, nx), (lo[0], lo[1])


def thickness(p, up_axis=2):
    """A patch's splat layer thickness (m), the median over its cells."""
    g = thickness_map(p, up_axis)[0]
    return float(np.nanmedian(g)) if np.isfinite(g).any() else 0.0


def _material_space(groups):
    """Each group's mean signature and the scale the distances use."""
    feats = [np.array([appearance(c[2]) for c in g], dtype=np.float64) for g in groups]
    every = np.concatenate([f for f in feats if len(f)])
    sd = every.std(axis=0)
    sd = np.maximum(sd, 0.1 * max(sd.max(), 1e-9))
    means = np.array([f.mean(axis=0) / sd if len(f) else np.full(6, np.inf) for f in feats])
    return means, sd


def _blurred(log_scale, ok, blur):
    """Cells whose splats are on average `blur` times the usual size: the
    capture's smeared rim, seen by few photos. On the desert its dark smear
    counted as bush."""
    if not ok.any():
        return np.zeros_like(ok)
    return log_scale > np.median(log_scale[ok]) + np.log(blur)


POORLY_SEEN = ("sparse", "faint", "gaps", "holes")


def seen_points(found, stats):
    """Search positions as (well seen, poorly seen): usable ones and those
    turned down for their shape (too steep, buried, ...) against those
    turned down because the photos saw little there (too few splats,
    faint, gaps). The desert's hazy rim is the second kind."""
    where = stats.get("rejected_at", [])
    good = [c[1] for c in found] + [(x, y) for x, y, r in where if r not in POORLY_SEEN]
    bad = [(x, y) for x, y, r in where if r in POORLY_SEEN]
    return good, bad


def _well_seen(xy, seen, reach):
    """Which points of `xy` lie nearer a well-seen search position than a
    poorly seen one, and within `reach` of it."""
    from scipy.spatial import cKDTree
    good, bad = seen
    if not good:
        return np.ones(len(xy), dtype=bool)
    dg = cKDTree(np.asarray(good, dtype=np.float64)).query(xy)[0]
    db = (cKDTree(np.asarray(bad, dtype=np.float64)).query(xy)[0] if bad
          else np.full(len(xy), np.inf))
    return (dg <= db) & (dg <= reach)


def material_map(s, groups, size, up_axis=2, step=0.5, min_splats=2000, blur=2.0,
                 seen=None):
    """The material every spot of the capture looks like, as class_shares
    decides it, on a `step` grid: each value judged over the `size` square
    around it. Returns (labels, (x0, y0)); -1 where too few splats,
    blurred (see _blurred) or, given `seen` (seen_points), poorly seen."""
    from scipy.ndimage import uniform_filter
    means, sd = _material_space(groups)
    plane = [i for i in range(3) if i != up_axis]
    xy = s.xyz[:, plane].astype(np.float64)
    lo = xy.min(axis=0)
    ij = np.floor((xy - lo) / step).astype(np.int64)
    w, h = ij.max(axis=0) + 1
    key = ij[:, 1] * w + ij[:, 0]
    rgb = s.base_rgb.astype(np.float64)
    k = max(1, int(round(size / step)))
    box = lambda v: uniform_filter(np.bincount(key, v, w * h).reshape(h, w), k,
                                   mode="constant") * k * k
    n = box(np.ones(len(key)))
    m = np.stack([box(rgb[:, c]) for c in range(3)], axis=-1) / np.maximum(n, 1)[..., None]
    m2 = np.stack([box(rgb[:, c] ** 2) for c in range(3)], axis=-1) / np.maximum(n, 1)[..., None]
    f = np.concatenate([m, np.sqrt(np.maximum(m2 - m * m, 0.0))], axis=-1) / sd
    labels = np.argmin([np.linalg.norm(f - mu, axis=-1) for mu in means], axis=0)
    ok = n >= min_splats
    ls = box(np.log(s.scale.max(axis=1).astype(np.float64))) / np.maximum(n, 1)
    if seen is not None:
        gy, gx = np.mgrid[0:h, 0:w]
        at = np.c_[lo[0] + (gx.ravel() + 0.5) * step, lo[1] + (gy.ravel() + 0.5) * step]
        ok &= _well_seen(at, seen, size).reshape(h, w)
    labels[~ok | _blurred(ls, ok, blur)] = -1
    return labels, (lo[0], lo[1])


def class_shares(s, groups, size, up_axis=2, min_splats=2000, blur=2.0, seen=None):
    """How much of the capture's ground looks like each group: every
    `size` cell of the scene (on four half-shifted grids) goes to the
    group whose mean colour signature is nearest, in the units
    split_classes uses. A material the capture holds little of, a single
    clump of scrub, scores low however many good patches it gave. The
    smeared rim (_blurred) is not ground and is not counted, nor, given
    `seen` (seen_points), ground the photos saw too little of."""
    means, sd = _material_space(groups)
    plane = [i for i in range(3) if i != up_axis]
    xy = s.xyz[:, plane].astype(np.float64)
    rgb = s.base_rgb.astype(np.float64)
    logs = np.log(s.scale.max(axis=1).astype(np.float64))
    counts = np.zeros(len(groups))
    for shift in ((0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)):
        ij = np.floor((xy - xy.min(axis=0)) / size + np.array(shift)).astype(np.int64)
        key = ij[:, 0] * (ij[:, 1].max() + 1) + ij[:, 1]
        cells, inv, n = np.unique(key, return_inverse=True, return_counts=True)
        ok = n >= min_splats
        ok &= ~_blurred(np.bincount(inv, logs) / n, ok, blur)
        if seen is not None:
            at = np.c_[np.bincount(inv, xy[:, 0]) / n, np.bincount(inv, xy[:, 1]) / n]
            ok &= _well_seen(at, seen, size)
        if not ok.any():
            continue
        m = np.stack([np.bincount(inv, rgb[:, c]) for c in range(3)], axis=1) / n[:, None]
        m2 = np.stack([np.bincount(inv, rgb[:, c] ** 2) for c in range(3)], axis=1) / n[:, None]
        f = np.concatenate([m, np.sqrt(np.maximum(m2 - m * m, 0.0))], axis=1)[ok] / sd
        near = np.argmin([np.linalg.norm(f - mu, axis=1) for mu in means], axis=0)
        counts += np.bincount(near, minlength=len(groups))
    total = counts.sum()
    return (counts / total).tolist() if total else [1.0 / len(groups)] * len(groups)


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


# Brightness, red against green, yellow against blue.
OPPONENT = np.array([[0.299, 0.587, 0.114],
                     [1.0, -1.0, 0.0],
                     [0.5, 0.5, -1.0]], dtype=np.float64).T


def _opponent(p):
    """A patch's splats in opponent colour (brightness, red-green,
    yellow-blue) and their weights (opacity)."""
    return (p.base_rgb.astype(np.float64) @ OPPONENT,
            np.asarray(p.opacity, dtype=np.float64))


def tone_gate(material, core=0.5, pad=0.01, fit_min=0.8, step=0.04, keep=0.9):
    """A test for one material's patches: do they share its common tone?

    Two parts. The patch's mean cast (red-green and yellow-blue) must lie
    within the range of the material's core, the most typical half of its
    positions, widened by `pad`: the desert's bluish grass had the bush's
    brightness and passed every other test. A range, not a distance, so a
    material that varies one way (yellower bush) keeps that variety. And at
    least `fit_min` of what it shows from above must be colours the core
    holds (the bins holding `keep` of the core's colour): a shrub in a sand
    tile is a small part and passes; a tile half of something else does not.

    Returns test(cand) -> (ok, why)."""
    if len(material) < 4:
        return lambda c: (True, "")
    means = []
    for c in material:
        o, w = _opponent(c[2])
        means.append((o * w[:, None]).sum(0) / max(w.sum(), 1e-9))
    means = np.array(means)
    mid = np.median(means, axis=0)
    sd = np.maximum(1.4826 * np.median(np.abs(means - mid), axis=0), 0.01)
    dist = np.linalg.norm((means - mid) / sd, axis=1)
    core_ids = np.argsort(dist)[:max(3, int(round(core * len(material))))]
    lo = means[core_ids, 1:].min(axis=0) - pad
    hi = means[core_ids, 1:].max(axis=0) + pad
    # The core's palette: colour bins holding `keep` of its weight.
    mass = {}
    for i in core_ids:
        o, w = _opponent(material[i][2])
        keys = [tuple(k) for k in np.floor(o / step).astype(np.int64)]
        w = w / max(w.sum(), 1e-9)
        for k, v in zip(keys, w):
            mass[k] = mass.get(k, 0.0) + v
    order = sorted(mass.values(), reverse=True)
    total, run, cut = sum(order), 0.0, 0.0
    for v in order:
        run += v
        cut = v
        if run >= keep * total:
            break
    common = {k for k, v in mass.items() if v >= cut}

    def test(c):
        o, w = _opponent(c[2])
        m = (o * w[:, None]).sum(0) / max(w.sum(), 1e-9)
        names = ("redder" if m[1] > hi[0] else "greener" if m[1] < lo[0] else "",
                 "yellower" if m[2] > hi[1] else "bluer" if m[2] < lo[1] else "")
        off = [n for n in names if n]
        if off:
            return False, " and ".join(off)
        keys = np.floor(o / step).astype(np.int64)
        inside = np.array([tuple(k) in common for k in keys])
        fit = float(w[inside].sum() / max(w.sum(), 1e-9))
        if fit < fit_min:
            return False, f"{1 - fit:.0%} of it in other colours"
        return True, ""
    return test


def cell_colours(p, size, up_axis=2, grid=16, min_count=4, centre=(0.0, 0.0)):
    """A patch from above as grid x grid cells of opponent colour, NaN
    where too few splats land. Candidates are cut centred on the origin."""
    plane = [i for i in range(3) if i != up_axis]
    xy = p.xyz[:, plane] - (np.asarray(centre, dtype=np.float64) - size / 2.0)
    ij = np.floor(xy / size * grid).astype(np.int64)
    inside = np.all((ij >= 0) & (ij < grid), axis=1)
    cell = ij[inside, 1] * grid + ij[inside, 0]
    w = np.asarray(p.opacity, dtype=np.float64)[inside]
    col = p.base_rgb[inside].astype(np.float64) @ OPPONENT
    count = np.bincount(cell, minlength=grid * grid)
    wsum = np.bincount(cell, weights=w, minlength=grid * grid)
    out = np.full((grid * grid, 3), np.nan)
    ok = (count >= min_count) & (wsum > 0)
    for c in range(3):
        out[ok, c] = np.bincount(cell, weights=w * col[:, c],
                                 minlength=grid * grid)[ok] / wsum[ok]
    return out.reshape(grid, grid, 3)


def odd_spots(cells, centres, size, rare=4.0, k=5):
    """For patches of one material (cells from cell_colours, centres in the
    scene), the largest spot in each whose colour the rest of the material
    does not have: a pale hole in scrub, a marker on sand, one shrub on bare
    ground. A spot like that repeats in every tile built from its patch.

    Rarity, not distance from an average: ordinary blotches have matches in
    other patches, an odd spot does not. Each cell is compared with the
    cells of every patch that does not overlap its own (overlapping patches
    share the spot), and is odd when its k-th nearest colour there is
    `rare` times further than is usual between the material's own cells.
    Returns, per patch, the cell count of its largest odd spot
    (8-connected)."""
    from scipy.ndimage import label
    from scipy.spatial import cKDTree
    flat = [c.reshape(-1, 3) for c in cells]
    pool = np.concatenate(flat)
    pool = pool[np.all(np.isfinite(pool), axis=1)]
    if len(pool) < 10 * k:
        return [0] * len(cells)
    mid = np.median(pool, axis=0)
    unit = np.maximum(1.4826 * np.median(np.abs(pool - mid), axis=0), 1e-3)
    flat = [(f - mid) / unit for f in flat]
    xy = np.asarray(centres, dtype=np.float64)
    out = []
    for i, f in enumerate(flat):
        apart = np.any(np.abs(xy - xy[i]) >= size, axis=1)
        ref = np.concatenate([flat[j] for j in np.flatnonzero(apart)] or [np.zeros((0, 3))])
        ref = ref[np.all(np.isfinite(ref), axis=1)]
        if len(ref) < 10 * k:
            out.append(0)
            continue
        tree = cKDTree(ref)
        # Every core (the same answer): one query per patch over all the
        # others' cells is quadratic, 4 minutes for the desert's 534.
        usual = float(np.median(tree.query(ref, k + 1, workers=-1)[0][:, -1]))
        ok = np.all(np.isfinite(f), axis=1)
        d = np.zeros(len(f))
        if ok.any():
            d[ok] = tree.query(f[ok], k, workers=-1)[0][:, -1]
        odd = (d > rare * max(usual, 1e-6)).reshape(cells[i].shape[:2])
        lab, n = label(odd, structure=np.ones((3, 3)))
        out.append(int(np.bincount(lab.ravel())[1:].max()) if n else 0)
    return out


def rank_clean(cands, size, across_cm, up_axis=2, rare=4.0, grid=16):
    """`cands` (one material) with the patches holding an odd spot at least
    `across_cm` wide moved to the back, worst last; the order is kept
    otherwise. Returns (ordered, how many are clean, spot size per patch in
    cells)."""
    cell_cm = size * 100.0 / grid
    min_cells = max(2, int(round(np.pi * (across_cm / 2.0) ** 2 / cell_cm ** 2)))
    spots = odd_spots([cell_colours(c[2], size, up_axis, grid) for c in cands],
                      [c[1] for c in cands], size, rare)
    bad = [s >= min_cells for s in spots]
    order = sorted(range(len(cands)), key=lambda i: (bad[i], spots[i] if bad[i] else 0))
    return ([cands[i] for i in order], len(cands) - sum(bad),
            [spots[i] for i in order])


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
