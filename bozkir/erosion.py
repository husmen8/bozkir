"""Generating terrain that has been rained on.

The surface under the tiles has been two sine waves, or a borrowed survey.
Neither is satisfactory for long. Sine waves have no drainage at all, and a
survey covers the few dozen metres somebody flew over - a game map is
kilometres, and there is no capture of it.

So: generate the terrain, and generate it by the process that made the real
one. Fractal noise alone gives hills, but hills of a particular wrong kind -
every slope the same, no channels, ridges that wander without joining.
Running water over it fixes that, and fixes it for the right reason.

The model is the stream power law of geomorphology,

    dh/dt  =  U  -  K * A^m * S  +  D * laplacian(h)

uplift, minus fluvial incision (drainage area A times slope S), plus
hillslope diffusion. It is solved the way FastScape solves it (Braun and
Willett, Geomorphology 2013): each step, every cell drains to its steepest
neighbour, the incision term is made implicit, and the heights are updated
in drainage order - receivers before donors - so the step is stable however
large it is and the cost is linear in the number of cells. The same
equation drives the graphics work this descends from (Cordonnier et al.,
Eurographics 2016; Schott et al., ACM TOG 2023). docs/terrain.md maps each
function here to its source.

Two simplifications, stated so nobody has to discover them:

  - Pits are filled at the start and the end (Priority-Flood, Barnes et
    al. 2014), not routed through at every step. Depression routing during
    the run (Cordonnier, Bovy and Braun 2019) is the thorough version.
  - The sediment map is a proxy, not a deposition model. Eroded material is
    passed downstream and settles in proportion to how flat the ground is.
    Yuan et al. (JGR Earth Surface 2019) is the principled version.

Everything here is written so web/terrain.js can do exactly the same
arithmetic in the same order - integer hashing for noise instead of a
random number generator, explicit evaluation order everywhere - and
tests/test_pipeline.py holds the two to the same output.
"""

import numpy as np

__all__ = ["fbm", "ridged", "erode", "generate", "terrace", "PROFILES",
           "base_surface", "fill_depressions"]

_M32 = np.uint64(0xFFFFFFFF)


# Fraction of each side simulated but not returned. See generate().
CROP_MARGIN = 0.15

# --------------------------------------------------------------- profiles

# Named terrains. Each is a noise recipe, an optional terracing, and the
# erosion that is run on it. `uplift` above zero grows relief from the noise
# as an uplift map, the way the graphics papers above do it - valleys then
# emerge rather than being carved into a surface that was already there.
# At zero, the noise is the starting surface and erosion only cuts into it.
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
    # Ground falling away from a range to one side: parallel valleys all
    # running the same way, the way an apron below mountains drains.
    "piedmont":  dict(octaves=6, freq=3, ridge=0.4, relief=1.0,
                      iterations=70, incision=0.45, diffusion=0.08,
                      tilt=0.55),
}


# ------------------------------------------------------------------ noise

def _imul(a, b):
    """Low 32 bits of a product, as JavaScript's Math.imul gives them."""
    return (a.astype(np.uint64) * np.uint64(b)) & _M32


def _hash_u32(x, y, s):
    """A 32-bit integer hash of lattice point (x, y) under seed s.

    Written out in 32-bit steps rather than drawn from a random number
    generator so that web/terrain.js produces the same lattice bit for bit:
    numpy's generator cannot be reproduced in a browser.
    """
    x = np.asarray(x, dtype=np.uint64) & _M32
    y = np.asarray(y, dtype=np.uint64) & _M32
    h = _imul(x, 0x27D4EB2D) ^ _imul(y, 0x165667B1) \
        ^ ((np.uint64(s) * np.uint64(0x9E3779B1)) & _M32)
    h = _imul(h ^ (h >> np.uint64(15)), 0x85EBCA6B)
    h = _imul(h ^ (h >> np.uint64(13)), 0xC2B2AE35)
    return h ^ (h >> np.uint64(16))


# Eight gradient directions, picked by the hash. Fixed vectors rather than
# an angle so no trigonometry is involved: sin and cos are not required to
# round the same way in every language, and these are.
_GX = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0])
_GY = np.array([1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0])


def _gradient_noise(h, w, f, s):
    """One octave of gradient (Perlin) noise, f lattice cells across, 0..1.

    Gradient rather than value noise: value noise interpolates random
    heights on a square lattice and the lattice shows, as flat-topped
    blocks aligned with the axes - which a terrain then erodes into
    rectangular valleys. Gradient noise has its extremes between lattice
    points instead of on them. Quintic fade, as in Perlin's improved noise.
    """
    i = np.arange(w, dtype=np.float64)
    j = np.arange(h, dtype=np.float64)
    x = (i * f) / w
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


def fbm(shape, octaves=6, freq=4, gain=0.5, lacunarity=2.0, seed=0):
    """Fractal noise: octaves at doubling frequency and halving amplitude.

    On its own it makes lumpy ground - fine for a distant hillside, wrong
    anywhere the eye can see that water has never run across it.
    """
    h, w = int(shape[0]), int(shape[1])
    out = np.zeros((h, w))
    amp, f, norm = 1.0, float(freq), 0.0
    for k in range(max(1, int(octaves))):
        out = out + amp * _gradient_noise(h, w, f, int(seed) * 1013 + k)
        norm = norm + amp
        amp = amp * gain
        f = f * lacunarity
    return out / max(norm, 1e-9)


def ridged(shape, octaves=6, freq=4, gain=0.5, lacunarity=2.0, seed=0):
    """Ridged fractal noise: sharp crests instead of round lumps.

    Each octave is folded about its midpoint and inverted, which turns the
    smooth maxima into creases (Musgrave's ridged multifractal, simplified).
    Mountains read as ranges rather than as a field of mounds.
    """
    h, w = int(shape[0]), int(shape[1])
    out = np.zeros((h, w))
    amp, f, norm = 1.0, float(freq), 0.0
    for k in range(max(1, int(octaves))):
        v = _gradient_noise(h, w, f, int(seed) * 1013 + 7919 + k)
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
    """Pull heights towards `steps` evenly spaced levels.

    Flats separated by steep risers, which is what a stack of resistant beds
    erodes into. `sharpness` 0 leaves the surface alone and 1 makes every
    level flat; in between the riser keeps some slope for erosion to cut.
    Rounds half up, explicitly, because Python and JavaScript disagree about
    rounding 0.5 and this has to come out the same in both.
    """
    steps = max(1, int(steps))
    k = np.asarray(z, dtype=np.float64) * steps
    base = np.floor(k)
    frac = k - base
    snap = (frac >= 0.5).astype(np.float64)
    shaped = base + frac * frac * (3.0 - 2.0 * frac) * (1.0 - sharpness) \
        + sharpness * snap
    return np.clip(shaped / steps, 0.0, 1.0)


def base_surface(size, seed=0, octaves=6, freq=4, ridge=0.5, terraces=0,
                 dome=0.0, tilt=0.0):
    """The noise a profile starts from, 0..1, before any erosion.

    `dome` raises the middle against the border. The border is base level -
    where water leaves the map - and noise knows nothing about that, so its
    broad low patches sit in the interior as closed basins. Filled, they
    become lakes of dead flat ground. A gentle rise towards the middle
    gives them somewhere to drain instead.

    `tilt` lowers the field towards one edge, so every valley drains the
    same way - parallel drainage, which reads nothing like the branching
    networks the other profiles make.
    """
    n = int(size)
    shape = (n, n)
    z = fbm(shape, octaves=octaves, freq=freq, seed=seed)
    if ridge > 0:
        r = ridged(shape, octaves=octaves, freq=freq, seed=seed)
        z = z * (1.0 - ridge) + r * ridge
    z = _unit(z)
    if dome > 0:
        # The rise sits in the outer band that generate() crops away, so
        # the drainage it gives reaches the border without tilting the
        # ground anyone sees towards its own middle.
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
    """Raise every pit until it can drain to the border (Priority-Flood).

    Noise is full of small closed hollows. Water that runs into one has
    nowhere to go, so without this the drainage network is a few hundred
    disconnected puddles and erosion has nothing to cut along. Flooding
    inwards from the border in order of height (Barnes, Lehman and Mulla,
    Computers & Geosciences 2014), and lifting each cell to a hair above
    the one it was reached from, gives every cell a downhill path out.

    Run on the starting surface and again on the result. Incision keeps a
    cell above its receiver, so it cannot make new pits; creep can make
    shallow ones, which the second pass removes. Ties are broken by index so
    JavaScript floods in the same order.
    """
    import heapq
    g = np.array(h, dtype=np.float64, copy=True)
    if eps is None:
        # Enough tilt across a filled floor for incision to find a line
        # across it, too little to see: a ten-thousandth of the relief per
        # cell, scaled to the field so it means the same at every size.
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
    """Steepest-descent receiver and distance for every cell.

    Border cells are base level and receive nothing; a cell with no lower
    neighbour is its own receiver (a pit). Neighbours are tried in a fixed
    order and only a strictly steeper one replaces the best so far, so ties
    resolve identically in both languages.
    """
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
    """Order cells receivers-first: by distance from their outlet, then index.

    Distance to the outlet by pointer jumping, which needs log(depth) array
    passes instead of a walk per cell. Any order with receivers first gives
    the same implicit update; this one is also cheap to build in JavaScript
    and fixes the order floating-point sums happen in.
    """
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
    """Run the stream power law over a height field. See the module docstring.

    `incision` is K, `diffusion` D, `uplift` U (per step, times the uplift
    map, which defaults to the starting surface itself). Drainage area is a
    fraction of the whole field so the same K means the same thing at 256
    and at 1024. Returns `(height, sediment)`, or height alone without
    `track`.
    """
    h = fill_depressions(np.array(z, dtype=np.float64, copy=True))
    n0, n1 = h.shape
    N = h.size
    flat = h.ravel()
    umap = (np.asarray(uplift_map, dtype=np.float64).ravel()
            if uplift_map is not None else _unit(flat.copy()))
    # Uplift fades to nothing at the border, which is base level. Without
    # the fade the interior rises past a border that cannot move and the
    # field ends in a cliff on all four sides.
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

        # Drainage area in cells, deepest level first. Whole numbers, so the
        # order of the additions cannot change the result.
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

        # Hillslope creep, explicit, interior only. Terms summed in a fixed
        # order so JavaScript can repeat it exactly.
        if diffusion:
            g = flat.reshape(n0, n1)
            lap = np.zeros((n0, n1))
            lap[1:-1, 1:-1] = ((((g[:-2, 1:-1] + g[2:, 1:-1]) + g[1:-1, :-2])
                                + g[1:-1, 2:]) - 4.0 * g[1:-1, 1:-1])
            flat = flat + (diffusion * dt) * lap.ravel()

        if track:
            # Material passed downstream, settling where the ground flattens.
            # A proxy for deposition, not a model of it.
            qs = eroded.copy()
            settle = 0.5 * (0.02 / (0.02 + slope))
            for L in range(levels - 1, 0, -1):
                idx = order[starts[L]:starts[L + 1]]
                dep = qs[idx] * settle[idx]
                sediment[idx] = sediment[idx] + dep
                np.add.at(qs, rec[idx], qs[idx] - dep)
            roots = order[starts[0]:starts[1]]
            sediment[roots] = sediment[roots] + qs[roots]

    # Once more at the end. Creep smooths channel floors and can leave
    # shallow closed hollows in them; anything that reads drainage off the
    # result - the material rule does - would see a network broken into
    # puddles. Hydrological conditioning, as surveys get before analysis.
    h = fill_depressions(flat.reshape(n0, n1))
    return (h, sediment.reshape(n0, n1)) if track else h


def generate(size=256, seed=0, profile=None, octaves=6, freq=4, ridge=0.5,
             relief=1.0, iterations=40, incision=0.3, diffusion=0.1,
             uplift=0.0, terraces=0, dome=0.0, tilt=0.0):
    """A height field, eroded, and where its sediment settled; both 0..1.

    `profile` names one of PROFILES and fills in the rest. An argument
    passed explicitly still wins, so a profile can be nudged without
    copying it - which is why the defaults here are only used when there is
    no profile at all.
    """
    given = dict(octaves=octaves, freq=freq, ridge=ridge, relief=relief,
                 iterations=iterations, incision=incision,
                 diffusion=diffusion, uplift=uplift, terraces=terraces,
                 dome=dome, tilt=tilt)
    if profile is not None:
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r}; "
                             f"have {', '.join(sorted(PROFILES))}")
        p = dict(PROFILES[profile])
        defaults = generate.__defaults__
        names = ("size", "seed", "profile", "octaves", "freq", "ridge",
                 "relief", "iterations", "incision", "diffusion", "uplift",
                 "terraces", "dome", "tilt")
        dflt = dict(zip(names, defaults))
        for k, v in given.items():
            if v != dflt[k]:
                p[k] = v
        given = {**{k: dflt[k] for k in given}, **p}
    s = given
    # Simulated on a larger field and cropped to its middle. The border of
    # the simulation is base level - where water leaves - so everything
    # drains towards it, and a rule that places material in low, collected
    # ground would draw a frame round every map. Keeping the outer
    # CROP_MARGIN of each side out of the result puts that border outside
    # what anyone sees.
    n = int(size)
    # Rounded half up by hand: Python rounds halves to even, JavaScript
    # up, and at n = 30 the two would crop different fields.
    big = n + 2 * int(np.floor(n * CROP_MARGIN + 0.5))
    k = (big - n) // 2
    z0 = base_surface(big, seed, s["octaves"], s["freq"], s["ridge"],
                      s["terraces"], s["dome"], s["tilt"])
    z, sed = erode(z0 * s["relief"], iterations=s["iterations"],
                   incision=s["incision"], diffusion=s["diffusion"],
                   uplift=s["uplift"] * s["relief"], uplift_map=z0)
    return _unit(z[k:k + n, k:k + n]), _unit(sed[k:k + n, k:k + n])