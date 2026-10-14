"""Wang frames and a pool of round centres (export_wang.py --frames).

A whole Wang tile repeats everything in it: each of its triangles comes
from one of a few edge patches, so a tuft in one shows at the same spot in
every tile with that edge colour. Here a tile keeps only its frame, the
band along its edges that neighbours must agree on, and the viewer sets a
round centre into every cell, picked from a pool of the material's own
ground and turned. Random placement and turning hide a texture's repeats
(Heitz and Neyret 2018; Mikkelsen 2022, hex-tiling); frame and centre meet
in a wobbled ring where the taller, more opaque splats of either side win
(height-based blending, as in terrain splatting).

How far a centre may turn is measured per material: a texture with a
direction (ripples) keeps it, within the spread its own patches show; one
without (scrub) turns any way.

The centre is a disc, or with a `shape` above 2 a rounded square (the
unit ball of the p-norm), which reaches far closer to the edges and
corners and leaves the frame a thin border; it keeps its shape only under
quarter turns, so it turns in those (few variants, like stones swapped
along a wall's rows, rather than any angle).
"""

import numpy as np

# The strip along each edge that only the frame may draw (as tile_labels'
# edge_margin): neighbours match there.
EDGE_STRIP = 0.06


def norm(x, y, shape=2.0):
    """Distance from the middle as the centre's shape measures it: the
    usual one for a disc (2), the p-norm for a rounded square."""
    x, y = np.abs(x), np.abs(y)
    if shape == 2.0:
        return np.hypot(x, y)
    return (x ** shape + y ** shape) ** (1.0 / shape)


def geometry(size, centre_size=0.6, ring_cm=15.0, shape=2.0):
    """The rings of a frame tile, in metres from its middle (in `norm`):
    `radius` of the centre (`centre_size` of the tile across), `half` the
    ring's half width, `wobble` how far the ring's line wanders, `inner`
    what a frame keeps (r >= inner) and `outer` what a centre keeps
    (r <= outer). `prominence`: heights over the ground (m) where a splat
    starts to win the ring and wins it fully. `centre_size` 0: as large as
    fits inside the edge strip. `reach`: how far the centre
    reaches out in any direction (its corner, for a rounded square)."""
    if shape < 2.0:
        raise ValueError(f"a centre shape of {shape:g}: 2 is a disc, more a rounded square")
    half = ring_cm / 200.0
    wobble = half / 2.0
    # 0 or less: as large as fits inside the edge strip.
    radius = (centre_size * size / 2.0 if centre_size > 0
              else size * (0.5 - EDGE_STRIP) - wobble - half)
    out = {"radius": radius, "half": half, "wobble": wobble,
           "inner": max(radius - wobble - half, 0.0),
           "outer": radius + wobble + half,
           "prominence": [0.02 * size, 0.08 * size], "shape": float(shape)}
    out["reach"] = out["outer"] * 2.0 ** (0.5 - 1.0 / shape)
    if out["outer"] > size * (0.5 - EDGE_STRIP) + 1e-9 or radius <= 0:
        raise ValueError(
            f"a centre {centre_size:g} of the tile across with a {ring_cm:g} cm ring "
            f"reaches {out['outer']:.2f} m out, into the edge strip neighbours match on "
            f"({size * (0.5 - EDGE_STRIP):.2f} m); a smaller centre or ring is needed")
    return out


def _radius(p, up_axis, shape=2.0):
    plane = [i for i in range(3) if i != up_axis]
    xy = p.xyz[:, plane].astype(np.float64)
    return norm(xy[:, 0], xy[:, 1], shape)


def frame_of(tile, inner, up_axis=2, shape=2.0):
    """The frame of a tile: its splats from `inner` out."""
    return tile.subset(np.flatnonzero(_radius(tile, up_axis, shape) >= inner))


def centre_of(patch, outer, up_axis=2, shape=2.0):
    """A centre: a patch's splats within `outer`."""
    return patch.subset(np.flatnonzero(_radius(patch, up_axis, shape) <= outer))


def zone(res, size, r_min=0.0, r_max=np.inf, shape=2.0):
    """Pixels of a res x res top-down image of a size-wide square (as
    graphcut.render_patch draws it) between two radii."""
    c = (np.arange(res) + 0.5) / res * size - size / 2.0
    r = norm(c[None, :], c[:, None], shape)
    return (r >= r_min) & (r <= r_max)


def shown(res, size, inner, diamonds=False, shape=2.0):
    """Pixels of an edge patch's top-down image (as zone) that a frame
    shows: outside `inner` of its middle; with --diamonds (the patch on its
    edge, the tile's middle at two of its tips) its diamond, outside
    `inner` of the tips the tile's middle falls on. Which two depends on
    the axis the patch ends up on (decided later), so either: only what
    both would hide is left out (the part both would show was a fifth of
    the patch, and held no spots on the desert)."""
    if not diamonds:
        return zone(res, size, inner, shape=shape)
    c = (np.arange(res) + 0.5) / res * size - size / 2.0
    x, y = np.abs(c[None, :]), np.abs(c[:, None])
    h = size / 2.0
    return ((x + y <= h) & ((norm(x, h - y, shape) >= inner)
                            | (norm(h - x, y, shape) >= inner)))


def luminance(p, size, up_axis=2, px=0.025):
    """A patch from above as (luminance, coverage), `px` metres a pixel.
    Each splat counts where its centre falls, by how much of it is seen
    (its opacity times what lets light through above it), then spread by a
    pixel, as its footprint would: no per-splat loop, so a whole pool takes
    seconds (the renderer took half a second a patch). Where little covers
    it the luminance is the patch's mean, so a bare spot is not texture."""
    from scipy.ndimage import gaussian_filter
    res = max(int(round(size / px)), 16)
    if not len(p):
        return np.zeros((res, res)), np.zeros((res, res))
    plane = [i for i in range(3) if i != up_axis]
    xy = p.xyz[:, plane].astype(np.float64)
    j = np.floor((xy[:, 0] / size + 0.5) * res).astype(np.int64)
    i = np.floor((0.5 - xy[:, 1] / size) * res).astype(np.int64)    # row 0 north
    ok = (i >= 0) & (i < res) & (j >= 0) & (j < res)
    pix = (i * res + j)[ok]
    a = np.clip(np.asarray(p.opacity, np.float64)[ok], 0.0, 0.999)
    lum = (np.asarray(p.base_rgb, np.float64) @ np.array([0.299, 0.587, 0.114]))[ok]
    order = np.lexsort((-p.xyz[ok, up_axis], pix))                     # top first
    pix, a, lum = pix[order], a[order], lum[order]
    la = np.log1p(-a)
    cs = np.cumsum(la)
    first = np.r_[0, np.flatnonzero(np.diff(pix)) + 1]
    start = np.repeat(first, np.diff(np.r_[first, len(pix)]))
    w = a * np.exp(cs - la - (cs[start] - la[start]))
    n = res * res
    wsum = gaussian_filter(np.bincount(pix, w, n).reshape(res, res), 0.7)
    lsum = gaussian_filter(np.bincount(pix, w * lum, n).reshape(res, res), 0.7)
    cover = 1.0 - np.exp(gaussian_filter(np.bincount(pix, la, n).reshape(res, res), 0.7))
    out = lsum / np.maximum(wsum, 1e-9)
    mean = float(lsum.sum() / max(wsum.sum(), 1e-9))
    return np.where(cover > 0.5, out, mean), cover


def turn_rule(images, min_coherence=0.2, max_spread=30.0, max_skew=0.5):
    """How far the centres of one material may turn, from top-down
    luminance images of its patches ((luminance, coverage) pairs).

    A direction is measured with the structure tensor (the gradients'
    covariance): in a typical patch, gradients lining up (coherence at
    least `min_coherence`) mean ripples or rows. They must keep their way
    only if the patches agree on it, within `max_spread` degrees: that
    spread is then how far a centre may turn. A half turn is allowed
    unless the texture is lopsided along the direction (skewed gradients:
    one face lit, one in shade), which a half turn would show.
    Returns {"any", "half", "spread" (degrees), "coherence", "skew", "why"}."""
    from scipy.ndimage import gaussian_filter
    tensors, grads = [], []
    for lum, w in images:
        g = gaussian_filter(np.asarray(lum, np.float64), 1.0)
        gy, gx = np.gradient(g)
        m = np.asarray(w) > 0.5
        if m.sum() < 16:
            continue
        j = np.array([(gx[m] ** 2).sum(), (gx[m] * gy[m]).sum(), (gy[m] ** 2).sum()])
        tr = j[0] + j[2]
        if tr <= 1e-12:
            continue
        tensors.append(j / tr)                     # each patch counts the same
        grads.append((gx[m], gy[m]))
    if not tensors:
        return {"any": True, "half": True, "spread": 180.0, "coherence": 0.0,
                "skew": 0.0, "why": "any (nothing to measure)"}

    def orient(j):
        jxx, jxy, jyy = j
        coh = float(np.hypot(jxx - jyy, 2 * jxy) / max(jxx + jyy, 1e-12))
        return 0.5 * float(np.arctan2(2 * jxy, jxx - jyy)), coh
    each = [orient(j) for j in tensors]
    coherence = float(np.median([c for _, c in each]))
    if coherence < min_coherence:
        return {"any": True, "half": True, "spread": 180.0, "coherence": coherence,
                "skew": 0.0, "why": f"any (no direction in its texture: coherence "
                                    f"{coherence:.2f}, under {min_coherence:g})"}
    # The patches' own directions, on doubled angles (a direction has no sign).
    z = sum(c * np.exp(2j * t) for t, c in each) / max(sum(c for _, c in each), 1e-12)
    spread = float(np.degrees(np.sqrt(max(-2.0 * np.log(max(abs(z), 1e-6)), 0.0)) / 2.0))
    if spread > max_spread:
        return {"any": True, "half": True, "spread": 180.0, "coherence": coherence,
                "skew": 0.0, "why": f"any (a direction in each patch, coherence "
                                    f"{coherence:.2f}, but they run every way: {spread:.0f} deg "
                                    f"apart)"}
    spread = max(spread, 5.0)
    theta = 0.5 * float(np.angle(z))
    u = np.array([np.cos(theta), np.sin(theta)])
    d = np.concatenate([gx * u[0] + gy * u[1] for gx, gy in grads])
    skew = float((d ** 3).mean() / max((d ** 2).mean(), 1e-12) ** 1.5)
    half = abs(skew) < max_skew
    head = (f"0 or 180 deg, +-{spread:.0f}" if half else f"+-{spread:.0f} deg only")
    why = (f"{head} (a direction in its texture: coherence {coherence:.2f}; "
           f"+-{spread:.0f} is how much its own patches differ"
           + ("" if half else f"; lopsided along it, skew {skew:.2f}: a half turn would show")
           + ")")
    return {"any": False, "half": half, "spread": spread, "coherence": coherence,
            "skew": skew, "why": why}


def salience(images, masks, px=0.025, k=3.0):
    """Per image, the share of its `mask` covered by salient spots: blobs
    some 20 to 50 cm across standing out from the material's usual
    variation (a difference of Gaussians, 5 and 15 cm, past `k` robust
    deviations of the whole set). A frame repeats in every tile with its
    edge colour, so a spot in it repeats at a fixed place."""
    from scipy.ndimage import gaussian_filter
    dogs = [gaussian_filter(np.asarray(lum, np.float64), 0.05 / px)
            - gaussian_filter(np.asarray(lum, np.float64), 0.15 / px) for lum, _ in images]
    pooled = np.concatenate([d[m] for d, m in zip(dogs, masks)] or [np.zeros(1)])
    mid = np.median(pooled)
    t = k * max(1.4826 * float(np.median(np.abs(pooled - mid))), 1e-4)
    return np.array([float((np.abs(d[m] - mid) > t).mean()) if m.any() else 0.0
                     for d, m in zip(dogs, masks)])


def clear_of_frame(c, e, ground, radius, together=0.05):
    """Whether a centre disc (`radius`, at `c`) leaves the frame of an
    edge patch at `e` (a `ground`-wide square) alone: wholly outside its
    square, or on its middle (within `together` of the square's width),
    which that patch's frame never shows; `together` None: only outside
    (a patch on its edge, --diamonds, shows its middle)."""
    dx, dy = abs(c[0] - e[0]), abs(c[1] - e[1])
    if together is not None and np.hypot(dx, dy) <= together * ground:
        return True
    return float(np.hypot(max(dx - ground / 2.0, 0.0), max(dy - ground / 2.0, 0.0))) >= radius


def shared_with_frames(c, edges, ground, geo, size, diamonds=False, n=15):
    """The share of a centre at `c` (its ground cut `ground` wide, as the
    edge patches at `edges`) that some frame also shows: its points inside
    an edge patch's square where that patch's frame draws (outside `inner`
    of its middle; with --diamonds, the patch on its edge, its diamond,
    either way round). That ground would show twice, in a frame at a fixed
    place and in the centre turned elsewhere."""
    k = ground / size
    r = geo["radius"] * k
    t = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    x, y = np.meshgrid(t * r, t * r)
    inside = norm(x, y, geo.get("shape", 2.0)) <= r
    px, py = x[inside] + c[0], y[inside] + c[1]
    h = ground / 2.0
    hit = np.zeros(len(px), bool)
    for e in edges:
        qx, qy = np.abs(px - e[0]), np.abs(py - e[1])
        if diamonds:
            hit |= qx + qy <= h
        else:
            hit |= (np.maximum(qx, qy) <= h) & (norm(qx, qy, geo.get("shape", 2.0))
                                                >= geo["inner"] * k)
    return float(hit.mean()) if len(hit) else 0.0


def block(geo, turns):
    """The manifest's description of frames and their centres."""
    out = {"radius": round(geo["radius"], 4), "ring": round(2 * geo["half"], 4),
           "wobble": round(geo["wobble"], 4), "inner": round(geo["inner"], 4),
           "outer": round(geo["outer"], 4),
           "prominence": [round(v, 4) for v in geo["prominence"]],
           "turns": [{"any": bool(t["any"]), "half": bool(t["half"]),
                      "spread": round(float(t["spread"]), 1)} for t in turns]}
    if geo.get("shape", 2.0) != 2.0:
        out["shape"] = round(geo["shape"], 3)        # a rounded square: quarter turns
    return out
