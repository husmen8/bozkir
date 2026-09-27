"""Generating terrain that has been rained on.

A survey covers a few dozen metres and a game map is kilometres, so the
terrain is generated - by the process that made the real one. Noise alone
gives hills with no channels; water run over it gives drainage.

The stream power law of geomorphology,

    dh/dt  =  U  -  K * A^m * S  +  D * laplacian(h)

(uplift, fluvial incision by drainage area A and slope S, hillslope
diffusion), solved the FastScape way (Braun and Willett 2013): steepest-
descent receivers, implicit incision, updates receivers-first, so any step
is stable and the cost is linear. The same equation as Cordonnier et al.
2016 and Schott et al. 2023. docs/terrain.md maps each function to its
source.

Simplifications:
  - Pits are filled at the start and the end (Priority-Flood, Barnes et
    al. 2014), not routed every step (Cordonnier, Bovy and Braun 2019).
  - Sediment is a proxy: eroded material passed downstream, settling where
    the ground flattens. Yuan et al. 2019 is the real deposition model.

web/terrain.js does the same arithmetic in the same order (integer hashing
instead of an RNG, fixed evaluation order); tests/test_pipeline.py holds the
two bit-identical.
"""

import numpy as np

__all__ = ["fbm", "ridged", "erode", "generate", "terrace", "PROFILES",
           "base_surface", "fill_depressions"]

_M32 = np.uint64(0xFFFFFFFF)


# Fraction of each side simulated but not returned. See generate().
CROP_MARGIN = 0.15

# --------------------------------------------------------------- profiles

# Named terrains: a noise recipe, optional terraces, and the erosion run on
# it. With `uplift` the noise is an uplift map and relief grows (valleys
# emerge); without, erosion only cuts into the noise.
PROFILES = {
    # Low relief, wide valleys: open country.
    "plains":    dict(dome=0.5, octaves=5, freq=2, ridge=0.0, relief=0.35,
                      iterations=40, incision=0.15, diffusion=0.20),
    "rolling":   dict(dome=0.5, octaves=6, freq=3, ridge=0.15, relief=0.6,
                      iterations=50, incision=0.25, diffusion=0.18),
    "hills":     dict(dome=0.5, octaves=6, freq=4, ridge=0.3, relief=0.9,
                      iterations=60, incision=0.35, diffusion=0.14),
    # The desert scene's character: mid relief, clear drainage.
    "desert":    dict(dome=0.25, octaves=6, freq=4, ridge=0.6, relief=1.0,
                      iterations=60, incision=0.4, diffusion=0.10),
    # Fine, dense gullies and very little creep to soften them.
    "badlands":  dict(dome=0.45, octaves=7, freq=6, ridge=0.5, relief=1.0,
                      iterations=80, incision=0.8, diffusion=0.03),
    # One deep network cut into high ground.
    "canyon":    dict(dome=0.25, octaves=7, freq=2, ridge=0.85, relief=1.2,
                      iterations=90, incision=0.9, diffusion=0.05),
    # Flat tops with steep steps: bench, face and floor all present.
    "mesa":      dict(dome=0.4, octaves=6, freq=3, ridge=0.45, relief=1.0,
                      iterations=55, incision=0.5, diffusion=0.06,
                      terraces=5),
    "plateau":   dict(dome=0.35, octaves=5, freq=2, ridge=0.3, relief=1.0,
                      iterations=50, incision=0.45, diffusion=0.08,
                      terraces=2),
    # Relief grown by uplift: ranges and dendritic valleys that emerge.
    "foothills": dict(octaves=6, freq=3, ridge=0.4, relief=0.3,
                      iterations=120, incision=0.35, diffusion=0.10,
                      uplift=0.012),
    "ridges":    dict(octaves=6, freq=3, ridge=0.9, relief=0.3,
                      iterations=150, incision=0.45, diffusion=0.06,
                      uplift=0.02),
    "alpine":    dict(octaves=7, freq=2, ridge=0.75, relief=0.4,
                      iterations=180, incision=0.55, diffusion=0.04,
                      uplift=0.03),
    # Rugged ground: almost no creep (diffusion is what rounds a landscape),
    # so cut ridges and steps stay sharp.
    "crags":     dict(dome=0.3, octaves=7, freq=5, ridge=0.95, relief=1.1,
                      iterations=60, incision=0.6, diffusion=0.01),
    "buttes":    dict(dome=0.35, octaves=6, freq=4, ridge=0.6, relief=1.0,
                      iterations=55, incision=0.7, diffusion=0.02,
                      terraces=4),
    "gorge":     dict(dome=0.2, octaves=7, freq=3, ridge=0.9, relief=1.3,
                      iterations=100, incision=1.2, diffusion=0.02),
    # Ground falling away from a range: parallel valleys draining one way.
    "piedmont":  dict(octaves=6, freq=3, ridge=0.4, relief=1.0,
                      iterations=70, incision=0.45, diffusion=0.08,
                      tilt=0.55),
    # Bozkir, the steppe the project is named for: wide, nearly level
    # ground with long shallow draws.
    "steppe":    dict(dome=0.5, octaves=5, freq=2, ridge=0.1, relief=0.25,
                      iterations=45, incision=0.3, diffusion=0.12),
    # Hills cut into steps: resistant beds left standing as benches.
    "terraced":  dict(dome=0.45, octaves=6, freq=3, ridge=0.3, relief=0.9,
                      iterations=50, incision=0.35, diffusion=0.05,
                      terraces=7),
    # Long parallel crests: noise stretched along one axis.
    "dunes":     dict(dome=0.3, octaves=5, freq=5, ridge=0.9, relief=0.5,
                      iterations=20, incision=0.1, diffusion=0.02,
                      stretch=0.3),
    # Steep slopes of loose rock: fine, dense ridges and little softening.
    "scree":     dict(dome=0.35, octaves=7, freq=7, ridge=0.7, relief=1.0,
                      iterations=70, incision=0.9, diffusion=0.01),
}


# ------------------------------------------------------------------ noise

def _imul(a, b):
    """Low 32 bits of a product, as JavaScript's Math.imul gives them."""
    return (a.astype(np.uint64) * np.uint64(b)) & _M32


def _hash_u32(x, y, s):
    """A 32-bit integer hash of lattice point (x, y) under seed s, in 32-bit
    steps so web/terrain.js gets the same bits (numpy's RNG cannot be
    reproduced in a browser)."""
    x = np.asarray(x, dtype=np.uint64) & _M32
    y = np.asarray(y, dtype=np.uint64) & _M32
    h = _imul(x, 0x27D4EB2D) ^ _imul(y, 0x165667B1) \
        ^ ((np.uint64(s) * np.uint64(0x9E3779B1)) & _M32)
    h = _imul(h ^ (h >> np.uint64(15)), 0x85EBCA6B)
    h = _imul(h ^ (h >> np.uint64(13)), 0xC2B2AE35)
    return h ^ (h >> np.uint64(16))


# Eight fixed gradient directions: no sin/cos, whose rounding can differ
# between languages.
_GX = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0])
_GY = np.array([1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0])


def _gradient_noise(h, w, f, s, stretch=1.0):
    """One octave of gradient (Perlin) noise, f lattice cells across, 0..1.
    Not value noise, whose axis-aligned blocks eroded into rectangular
    valleys. Quintic fade, as in Perlin's improved noise. `stretch` scales
    the lattice along x (anisotropic noise, for dunes)."""
    i = np.arange(w, dtype=np.float64)
    j = np.arange(h, dtype=np.float64)
    x = (i * (f * stretch)) / w
    y = (j * f) / h
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = (x - x0)[None, :]
    fy = (y - y0)[:, None]
    X0 = x0.astype(np.int64)[None, :]
    Y0 = y0.astype(np.int64)[:, None]

    def corner(dx, dy):
        g = (_hash_u32(X0 + dx, Y0 + dy, s) & np.uint64(7)).astype(np.int64)
        return _GX[g] * (fx - dx) + _GY[g] * (fy - dy)

    ux = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0)
    uy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0)
    a = corner(0, 0)
    b = corner(1, 0)
    c = corner(0, 1)
    d = corner(1, 1)
    top = a + (b - a) * ux
    bot = c + (d - c) * ux
    # The raw value lies in about -1..1; halved and shifted into 0..1.
    return (top + (bot - top) * uy) * 0.5 + 0.5


def fbm(shape, octaves=6, freq=4, gain=0.5, lacunarity=2.0, seed=0,
        stretch=1.0):
    """Fractal noise: octaves at doubling frequency and halving amplitude."""
    h, w = int(shape[0]), int(shape[1])
    out = np.zeros((h, w))
    amp, f, norm = 1.0, float(freq), 0.0
    for k in range(max(1, int(octaves))):
        out = out + amp * _gradient_noise(h, w, f, int(seed) * 1013 + k, stretch)
        norm = norm + amp
        amp = amp * gain
        f = f * lacunarity
    return out / max(norm, 1e-9)


def ridged(shape, octaves=6, freq=4, gain=0.5, lacunarity=2.0, seed=0,
           stretch=1.0):
    """Ridged fractal noise: each octave folded about its midpoint and
    inverted, so maxima become creases (Musgrave's ridged multifractal,
    simplified). Mountains read as ranges, not mounds."""
    h, w = int(shape[0]), int(shape[1])
    out = np.zeros((h, w))
    amp, f, norm = 1.0, float(freq), 0.0
    for k in range(max(1, int(octaves))):
        v = _gradient_noise(h, w, f, int(seed) * 1013 + 7919 + k, stretch)
        n = 1.0 - np.abs(2.0 * v - 1.0)
        out = out + amp * (n * n)
        norm = norm + amp
        amp = amp * gain
        f = f * lacunarity
    return out / max(norm, 1e-9)


def _unit(a):
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(a)


def terrace(z, steps, sharpness=0.5):
    """Pull heights towards `steps` levels: flats with risers between, as
    resistant beds erode. `sharpness` 0 leaves the surface, 1 makes every
    level flat. Rounds half up explicitly (Python and JS round 0.5
    differently)."""
    steps = max(1, int(steps))
    k = np.asarray(z, dtype=np.float64) * steps
    base = np.floor(k)
    frac = k - base
    snap = (frac >= 0.5).astype(np.float64)
    shaped = base + frac * frac * (3.0 - 2.0 * frac) * (1.0 - sharpness) \
        + sharpness * snap
    return np.clip(shaped / steps, 0.0, 1.0)


def base_surface(size, seed=0, octaves=6, freq=4, ridge=0.5, terraces=0,
                 dome=0.0, tilt=0.0, stretch=1.0):
    """The noise a profile starts from, 0..1, before erosion.

    `dome` raises the middle against the border (base level), so low
    patches drain out instead of filling into dead-flat lakes. `tilt` lowers
    the field towards one edge, for parallel drainage.
    """
    n = int(size)
    shape = (n, n)
    z = fbm(shape, octaves=octaves, freq=freq, seed=seed, stretch=stretch)
    if ridge > 0:
        r = ridged(shape, octaves=octaves, freq=freq, seed=seed, stretch=stretch)
        z = z * (1.0 - ridge) + r * ridge
    z = _unit(z)
    if dome > 0:
        # The rise sits in the band generate() crops away, so the visible
        # ground is not tilted towards its middle.
        band = CROP_MARGIN / (1.0 + 2.0 * CROP_MARGIN)
        z = _unit(z * (1.0 - dome) + _edge_taper(n, n, frac=band) * dome)
    if tilt > 0:
        c = np.arange(n, dtype=np.float64)
        ramp = 1.0 - c / (n - 1)
        z = _unit(z * (1.0 - tilt) + ramp[:, None] * tilt)
    if terraces:
        z = terrace(z, terraces)
    return z


# ---------------------------------------------------------------- erosion

_NEIGHBOURS = [(b, a) for b in (-1, 0, 1) for a in (-1, 0, 1)
               if not (a == 0 and b == 0)]


def fill_depressions(h, eps=None):
    """Raise every pit until it drains to the border (Priority-Flood,
    Barnes, Lehman and Mulla 2014).

    Without it, noise drains into hundreds of puddles and erosion has
    nothing to cut along. Flooded inwards from the border by height, each
    cell lifted a hair above the one it was reached from. Run before and
    after erosion (creep can leave shallow pits). Ties break by index so JS
    floods in the same order.
    """
    import heapq
    g = np.array(h, dtype=np.float64, copy=True)
    if eps is None:
        # Enough tilt across a filled floor for incision to find a path,
        # too little to see; scaled so it means the same at every size.
        eps = 1e-4 * max(float(g.max() - g.min()), 1e-9) * 256.0 / max(g.shape)
    n0, n1 = g.shape
    flat = g.ravel()
    done = np.zeros(flat.size, bool)
    heap = []
    for r in range(n0):
        for c in range(n1):
            if r == 0 or c == 0 or r == n0 - 1 or c == n1 - 1:
                k = r * n1 + c
                heap.append((flat[k], k))
                done[k] = True
    heapq.heapify(heap)
    offs = [(b, a) for b, a in _NEIGHBOURS]
    while heap:
        v, k = heapq.heappop(heap)
        r, c = divmod(k, n1)
        for b, a in offs:
            rr, cc = r + b, c + a
            if rr < 0 or cc < 0 or rr >= n0 or cc >= n1:
                continue
            q = rr * n1 + cc
            if done[q]:
                continue
            done[q] = True
            if flat[q] <= v:
                flat[q] = v + eps
            heapq.heappush(heap, (flat[q], q))
    return flat.reshape(n0, n1)


def _receivers(h):
    """Steepest-descent receiver and distance for every cell. Border cells
    are base level; a cell with no lower neighbour is its own receiver.
    Fixed neighbour order and strict comparison, so ties resolve the same
    in both languages."""
    n0, n1 = h.shape
    N = n0 * n1
    idx = np.arange(N).reshape(n0, n1)
    rec = idx.copy()
    dist = np.ones((n0, n1))
    best = np.zeros((n0, n1))
    inner = (slice(1, -1), slice(1, -1))
    c = h[inner]
    for b, a in _NEIGHBOURS:
        d = np.sqrt(2.0) if (a and b) else 1.0
        nb = h[1 + b:n0 - 1 + b, 1 + a:n1 - 1 + a]
        s = (c - nb) / d
        take = s > best[inner]
        best[inner] = np.where(take, s, best[inner])
        rec[inner] = np.where(take, idx[1 + b:n0 - 1 + b, 1 + a:n1 - 1 + a],
                              rec[inner])
        dist[inner] = np.where(take, d, dist[inner])
    return rec.ravel(), dist.ravel()


def _edge_taper(n0, n1, frac=0.12):
    """1 in the interior, easing to 0 over the outer `frac` of each side."""
    def ramp(n):
        k = np.arange(n, dtype=np.float64)
        d = np.minimum(k, (n - 1) - k) / max(1.0, frac * (n - 1))
        d = np.minimum(1.0, d)
        return d * d * (3.0 - 2.0 * d)
    return ramp(n0)[:, None] * ramp(n1)[None, :]


def _levels(rec):
    """Order cells receivers-first: by distance to the outlet (pointer
    jumping, log(depth) passes), then index. Cheap in JS too, and it fixes
    the order floating-point sums happen in."""
    N = rec.size
    ar = np.arange(N)
    d = (rec != ar).astype(np.int64)
    p = rec.copy()
    while True:
        nd = d + d[p]
        np_ = p[p]
        if np.array_equal(np_, p):
            d = nd if not np.array_equal(nd, d) else d
            break
        d, p = nd, np_
    order = np.argsort(d, kind="stable")
    counts = np.bincount(d, minlength=int(d.max()) + 1)
    starts = np.concatenate(([0], np.cumsum(counts)))
    return d, order, starts


def erode(z, iterations=40, incision=0.3, diffusion=0.1, uplift=0.0,
          uplift_map=None, area_exp=0.5, dt=1.0, track=True):
    """Run the stream power law (see the module docstring).

    `incision` is K, `diffusion` D, `uplift` U (times the uplift map, by
    default the starting surface). Drainage area is a fraction of the field,
    so K means the same at any size. Returns (height, sediment), or height
    alone without `track`.
    """
    h = fill_depressions(np.array(z, dtype=np.float64, copy=True))
    n0, n1 = h.shape
    N = h.size
    flat = h.ravel()
    umap = (np.asarray(uplift_map, dtype=np.float64).ravel()
            if uplift_map is not None else _unit(flat.copy()))
    # Uplift fades out at the border (base level), or the field ends in a
    # cliff on all four sides.
    umap = umap * _edge_taper(n0, n1).ravel()
    border = np.zeros((n0, n1), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    border = border.ravel()
    interior = ~border
    sediment = np.zeros(N)
    exact_sqrt = area_exp == 0.5

    for _ in range(max(1, int(iterations))):
        rec, dist = _receivers(flat.reshape(n0, n1))
        depth, order, starts = _levels(rec)
        levels = len(starts) - 1

        # Drainage area in cells, deepest level first (whole numbers, so
        # addition order does not matter).
        area = np.ones(N)
        for L in range(levels - 1, 0, -1):
            idx = order[starts[L]:starts[L + 1]]
            np.add.at(area, rec[idx], area[idx])
        a = area / N
        pw = np.sqrt(a) if exact_sqrt else np.power(a, area_exp)

        if uplift:
            flat = flat + np.where(interior, uplift * dt * umap, 0.0)
        before = flat.copy()

        # Implicit incision, receivers first (Braun and Willett 2013, n = 1):
        #   h_i <- (h_i + F h_r) / (1 + F),   F = K dt A^m / distance
        F = (incision * dt) * pw / dist
        for L in range(1, levels):
            idx = order[starts[L]:starts[L + 1]]
            f = F[idx]
            flat[idx] = (flat[idx] + f * flat[rec[idx]]) / (1.0 + f)

        eroded = np.maximum(before - flat, 0.0)
        slope = np.maximum(flat - flat[rec], 0.0) / dist

        # Hillslope creep, explicit, interior only, terms in a fixed order.
        if diffusion:
            g = flat.reshape(n0, n1)
            lap = np.zeros((n0, n1))
            lap[1:-1, 1:-1] = ((((g[:-2, 1:-1] + g[2:, 1:-1]) + g[1:-1, :-2])
                                + g[1:-1, 2:]) - 4.0 * g[1:-1, 1:-1])
            flat = flat + (diffusion * dt) * lap.ravel()

        if track:
            # Sediment proxy: passed downstream, settling where it flattens.
            qs = eroded.copy()
            settle = 0.5 * (0.02 / (0.02 + slope))
            for L in range(levels - 1, 0, -1):
                idx = order[starts[L]:starts[L + 1]]
                dep = qs[idx] * settle[idx]
                sediment[idx] = sediment[idx] + dep
                np.add.at(qs, rec[idx], qs[idx] - dep)
            roots = order[starts[0]:starts[1]]
            sediment[roots] = sediment[roots] + qs[roots]

    # Filled once more: creep can leave shallow hollows in channel floors,
    # which would break the drainage the material rule reads.
    h = fill_depressions(flat.reshape(n0, n1))
    return (h, sediment.reshape(n0, n1)) if track else h


def generate(size=256, seed=0, profile=None, octaves=6, freq=4, ridge=0.5,
             relief=1.0, iterations=40, incision=0.3, diffusion=0.1,
             uplift=0.0, terraces=0, dome=0.0, tilt=0.0, stretch=1.0):
    """An eroded height field and its sediment map, both 0..1.

    `profile` names one of PROFILES; any argument passed explicitly still
    wins, so a profile can be nudged without copying it.
    """
    given = dict(octaves=octaves, freq=freq, ridge=ridge, relief=relief,
                 iterations=iterations, incision=incision,
                 diffusion=diffusion, uplift=uplift, terraces=terraces,
                 dome=dome, tilt=tilt, stretch=stretch)
    if profile is not None:
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r}; "
                             f"have {', '.join(sorted(PROFILES))}")
        p = dict(PROFILES[profile])
        defaults = generate.__defaults__
        names = ("size", "seed", "profile", "octaves", "freq", "ridge",
                 "relief", "iterations", "incision", "diffusion", "uplift",
                 "terraces", "dome", "tilt", "stretch")
        dflt = dict(zip(names, defaults))
        for k, v in given.items():
            if v != dflt[k]:
                p[k] = v
        given = {**{k: dflt[k] for k in given}, **p}
    s = given
    # Simulated larger and cropped to the middle: everything drains to the
    # simulation border, so the rule would draw a frame of collected
    # material round every map.
    n = int(size)
    # Rounded half up by hand (Python rounds halves to even, JS up).
    big = n + 2 * int(np.floor(n * CROP_MARGIN + 0.5))
    k = (big - n) // 2
    z0 = base_surface(big, seed, s["octaves"], s["freq"], s["ridge"],
                      s["terraces"], s["dome"], s["tilt"], s["stretch"])
    z, sed = erode(z0 * s["relief"], iterations=s["iterations"],
                   incision=s["incision"], diffusion=s["diffusion"],
                   uplift=s["uplift"] * s["relief"], uplift_map=z0)
    return _unit(z[k:k + n, k:k + n]), _unit(sed[k:k + n, k:k + n])