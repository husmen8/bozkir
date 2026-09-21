"""Generating terrain that has been rained on.

The surface under the tiles has been two sine waves, or a borrowed survey.
Neither is satisfactory for long. Sine waves have no drainage at all, and a
survey covers the few dozen metres somebody flew over - a game map is
kilometres, and there is no capture of it.

So: generate the terrain, and generate it by the process that made the real
one. Fractal noise alone gives hills, but hills of a particular wrong kind -
every slope the same, no channels, ridges that wander without joining.
Running water over it fixes that, and fixes it for the right reason.

    height = ridged_noise(...)
    height = erode(height, iterations=...)

What erosion buys, beyond looking right:

  - Channels that branch. A drainage network is a tree, and a tree is what
    the eye reads as a landscape rather than as noise.
  - Slopes that grade into valley floors, because material removed from the
    steep part is deposited in the flat part. Noise has no memory of what
    it took from where.
  - **A sediment map**, which is the part that matters most here. The rule
    placing captured material currently infers where loose material would
    collect, from the shape of the ground. After erosion that is not an
    inference: the simulation moved sediment and recorded where it settled.
    The rule stops being a plausible story about geology and becomes a
    readout of what the generator did.

The method is the standard droplet simulation - Musgrave's hydraulic
erosion, in the form most implementations use. A droplet is dropped at
random, runs downhill, picks up material where it accelerates and drops it
where it slows, and dies. Thousands of them carve a landscape. It is not a
physical model of anything; it is a procedure that produces the right
shapes, which is what a terrain generator is for.
"""

import numpy as np

__all__ = ["fbm", "ridged", "erode", "generate"]


def _value_noise(shape, freq, rng):
    """One octave of smooth noise, by bilinear upsampling of a coarse grid.

    Value noise rather than Perlin: the difference is visible on a single
    octave and invisible once several are summed, and this needs no
    gradient table.
    """
    h, w = shape
    gh, gw = max(2, int(freq) + 1), max(2, int(freq) + 1)
    g = rng.random((gh, gw))

    ys = np.linspace(0, gh - 1, h)
    xs = np.linspace(0, gw - 1, w)
    y0 = np.clip(ys.astype(int), 0, gh - 2)
    x0 = np.clip(xs.astype(int), 0, gw - 2)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    # Smoothstep, so octaves do not show their grid as a diamond lattice.
    fy = fy * fy * (3 - 2 * fy)
    fx = fx * fx * (3 - 2 * fx)

    a = g[np.ix_(y0, x0)]
    b = g[np.ix_(y0, x0 + 1)]
    c = g[np.ix_(y0 + 1, x0)]
    d = g[np.ix_(y0 + 1, x0 + 1)]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def fbm(shape, octaves=6, freq=4, gain=0.5, lacunarity=2.0, seed=0):
    """Fractal noise: octaves at doubling frequency and halving amplitude.

    The usual starting point. On its own it makes lumpy ground - fine for a
    distant hillside, wrong anywhere the eye can see that water has never
    run across it.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(shape)
    amp, f, norm = 1.0, float(freq), 0.0
    for _ in range(max(1, octaves)):
        out += amp * _value_noise(shape, f, rng)
        norm += amp
        amp *= gain
        f *= lacunarity
    return out / max(norm, 1e-9)


def ridged(shape, octaves=6, freq=4, gain=0.5, lacunarity=2.0, seed=0):
    """Ridged fractal noise: sharp crests instead of round lumps.

    Each octave is folded about its midpoint and inverted, which turns the
    smooth maxima into creases. Mountains read as ranges rather than as a
    field of mounds, and erosion then has ridgelines to cut between.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(shape)
    amp, f, norm = 1.0, float(freq), 0.0
    for _ in range(max(1, octaves)):
        n = 1.0 - np.abs(2.0 * _value_noise(shape, f, rng) - 1.0)
        out += amp * n * n
        norm += amp
        amp *= gain
        f *= lacunarity
    return out / max(norm, 1e-9)


def _flow_and_slope(z, spacing=1.0):
    """Drainage area and slope, sharing one pass over the neighbourhood."""
    from .terrain import flow_accumulation, slope as _slope
    return flow_accumulation(z, spacing), _slope(z, spacing)


def erode(z, iterations=40, rain=1.0, incision=0.06, area_exp=0.5,
          slope_exp=1.0, diffusion=0.25, spacing=1.0, track=True,
          **_ignored):
    """Cut a drainage network into a height field.

    Stream power: a river's ability to cut its bed goes as the water
    passing through it times how steeply it falls. Written as an erosion
    rate,

        dz/dt  =  -K * A^m * S^n  +  D * laplacian(z)

    where A is the drainage area above a point - which is exactly the flow
    accumulation already computed for placing material - S the local slope,
    and the second term is hillslope diffusion, the slow creep that rounds
    ridges and fills hollows between the channels.

    This is the standard landscape evolution model, and it is used here in
    preference to the droplet simulation that is more common in graphics
    for one practical reason: it cuts channels reliably. A droplet method
    depends on thousands of random walks finding the same route, and on a
    noisy surface most of them die in the nearest pit instead - the result
    is a smoothed version of what it started with, not a landscape. Driving
    erosion from the drainage network directly means the network is an
    input rather than something that has to emerge by luck.

    The two terms have to be balanced against each other, and that balance
    is what the terrain looks like: incision alone gives bare canyons,
    diffusion alone gives rolling hills with no drainage. Their ratio sets
    how far apart the valleys sit, which is the single most recognisable
    property of a landscape.

    Returns `(height, sediment)` when `track`. Sediment is where material
    came to rest: high where a lot of water arrives but the ground is too
    flat to carry anything further, which is where loose material really
    does collect. That map is the reason this belongs here rather than in a
    texture - the rule placing captured material can read what the
    generator did instead of inferring it from the shape afterwards.
    """
    z = np.array(z, dtype=np.float64, copy=True)
    sediment = np.zeros_like(z)

    for _ in range(max(1, int(iterations))):
        area, grad = _flow_and_slope(z, spacing)
        # Normalised so the rate does not depend on the grid size: a
        # 512-wide field would otherwise erode far harder than a 128 for
        # no reason but having more cells upstream.
        a = (area / area.max()) if area.max() > 0 else area
        cut = incision * np.power(a, area_exp) * np.power(grad, slope_exp)

        # Hillslope diffusion: the Laplacian, which moves material from
        # convex ground to concave.
        p = np.pad(z, 1, mode="edge")
        lap = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
               - 4.0 * z)

        z = z - rain * cut + diffusion * lap
        if track:
            # What the diffusion put down, where the water had already
            # arrived: deposition is the positive part of the Laplacian
            # weighted by how much drains through.
            sediment += np.maximum(lap, 0.0) * (0.25 + 0.75 * a)

    return (z, sediment) if track else z


def generate(size=256, seed=0, octaves=6, freq=4, ridge=0.5,
             iterations=40, **kw):
    """A height field, eroded, normalised to 0..1.

    `ridge` mixes ridged noise into fractal noise: 0 is rolling hills, 1 is
    a mountain range. The mixture rather than either alone, because pure
    ridged noise is all crest and no basin, and pure fractal noise has
    nothing for erosion to cut between.

    Returns `(height, sediment)`, both normalised, so the second can be
    stored beside the first and read as "where loose material settled".
    """
    shape = (int(size), int(size))
    base = fbm(shape, octaves=octaves, freq=freq, seed=seed)
    if ridge > 0:
        crest = ridged(shape, octaves=octaves, freq=freq, seed=seed + 1)
        base = base * (1 - ridge) + crest * ridge

    z, sed = erode(base, iterations=iterations, **kw)

    def unit(a):
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)

    return unit(z), unit(sed)