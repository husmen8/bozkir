"""Decals: distinct pieces of a capture (a shrub and its shadow) baked into
copies of plain tiles, for the viewer to place now and then.

Tiles have to be the plainest ground a capture holds, or every repeat shows;
that throws away what makes the capture look real. A decal goes into a copy
of a finished tile, inside a disc clear of its edges, so the copy meets its
neighbours exactly as the original does and the layout can swap it in at a
few cells (texture bombing, Glanville 2004, with tiles instead of texels).
"""

import numpy as np

from .ply import Splats
from .patches import ground_level

LUMA = np.float32([0.299, 0.587, 0.114])
EDGE_CLEAR = 0.1          # metres of tile edge a decal never touches


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ground_maps(s, up_axis=2, cell=0.05):
    """Brightness and cover of the ground from above, on a `cell` grid.
    Splat statistics, not a render: a whole capture in seconds."""
    plane = [i for i in range(3) if i != up_axis]
    xy = s.xyz[:, plane].astype(np.float64)
    lo = xy.min(axis=0)
    ij = np.floor((xy - lo) / cell).astype(np.int64)
    w, h = ij.max(axis=0) + 1
    flat = ij[:, 1] * w + ij[:, 0]
    sc = np.sort(s.scale, axis=1)
    ink = np.pi * sc[:, 2] * sc[:, 1] * s.opacity
    luma = s.base_rgb @ LUMA
    ink_sum = np.bincount(flat, ink, minlength=w * h).reshape(h, w)
    lum_sum = np.bincount(flat, ink * luma, minlength=w * h).reshape(h, w)
    bright = np.where(ink_sum > 0, lum_sum / np.maximum(ink_sum, 1e-12), 0.0)
    # Cover relative to the capture's own typical paint per cell: an absolute
    # measure rated a capture of thin needles (the desert) as nearly empty.
    typical = np.median(ink_sum[ink_sum > 0]) if (ink_sum > 0).any() else 1.0
    cover = ink_sum / max(typical, 1e-12)
    from .patches import OPPONENT
    col = s.base_rgb.astype(np.float64) @ OPPONENT
    colour = np.stack([np.bincount(flat, ink * col[:, c], minlength=w * h)
                       for c in range(3)], axis=1).reshape(h, w, 3)
    colour = colour / np.maximum(ink_sum, 1e-12)[..., None]
    return bright, cover, lo, cell, colour


def painted(bright, colour, seen):
    """Cells of a painted survey marker: a blue cast no ground, shrub or
    shadow has, or near-black right beside something lit or blue (the
    black square around the cross). Measured on the desert: two markers had
    7 and 13 cells over the blue line and 7 and 9 black-beside-lit cells,
    a shrub none of either."""
    from scipy.ndimage import maximum_filter
    from .patches import OPPONENT
    rgb = colour @ np.linalg.inv(OPPONENT)
    blue = np.where(seen, rgb[..., 2] - np.maximum(rgb[..., 0], rgb[..., 1]), -1.0)
    lit = seen & ((blue > 0.1) | (bright > 0.7))
    beside = maximum_filter(lit.astype(np.uint8), size=3) > 0
    return seen & ((blue > 0.15) | ((bright < 0.05) & beside))


def _disc(radius_cells, inner=0.0, outer=1.0):
    r = int(np.ceil(radius_cells))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    d = np.hypot(x, y) / max(radius_cells, 1e-9)
    return ((d >= inner) & (d <= outer)).astype(np.float64)


def find(s, radius, up_axis=2, count=48, cell=0.05, stride=0.25, tally=None):
    """Candidate decal centres, best first: discs whose middle stands out
    (a shrub, its shadow) while their rim looks like the ground around
    them, so a decal fades into a plain tile without a ring. Thresholds are
    relative to the capture's own ground, not fixed numbers.

    Returns a list of (x, y, score) in the scene's ground plane; `tally`, a
    dict, gets how many positions each test turned away."""
    from scipy.signal import fftconvolve
    from scipy.ndimage import maximum_filter, uniform_filter

    bright, cover, lo, cell, colour = ground_maps(s, up_axis, cell)
    seen = (cover > 0.2).astype(np.float64)
    b = bright * seen
    rc = radius / cell
    marks = fftconvolve(painted(bright, colour, seen > 0).astype(np.float64),
                        _disc(rc), mode="same")

    def mean_std(kernel):
        n = fftconvolve(seen, kernel, mode="same")
        m1 = fftconvolve(b, kernel, mode="same") / np.maximum(n, 1e-9)
        m2 = fftconvolve(b * b, kernel, mode="same") / np.maximum(n, 1e-9)
        return m1, np.sqrt(np.maximum(m2 - m1 * m1, 0.0)), n / kernel.sum()

    m_in, s_in, _ = mean_std(_disc(rc, 0.0, 0.6))
    m_rim, s_rim, _ = mean_std(_disc(rc, 0.75, 1.0))
    _, _, filled = mean_std(_disc(rc))
    ground = float(np.median(bright[seen > 0]))
    q1, q3 = np.percentile(bright[seen > 0], [25, 75])
    spread = float(q3 - q1)
    # Survey markers are white on black: nothing on bare ground is that
    # bright. Smoothed over 3x3 cells first, so one bright grain is not one.
    smooth = uniform_filter(b, 3) / np.maximum(uniform_filter(seen, 3), 1e-9)
    peak = maximum_filter(smooth * seen, footprint=_disc(rc) > 0)

    score = s_in + np.maximum(ground - m_in, 0.0) \
        - 2.0 * np.abs(m_rim - ground) - s_rim
    tests = {"holes in the disc": filled > 0.9,
             "too bright (a marker?)": peak < ground + max(0.25, 4.0 * spread),
             # A shrub turned away costs nothing (there are more); a marker
             # let through repeats on every copy.
             "painted marker": marks < 2.5,
             "rim unlike the ground": np.abs(m_rim - ground) < max(0.04, 0.75 * spread)}
    ok = np.ones_like(score, dtype=bool)
    for v in tests.values():
        ok &= v
    step = max(int(round(stride / cell)), 1)
    edge = int(np.ceil(rc)) + 1
    rows, cols = np.mgrid[edge:score.shape[0] - edge:step,
                          edge:score.shape[1] - edge:step]
    rows, cols = rows.ravel(), cols.ravel()
    if tally is not None:
        tally["positions"] = len(rows)
        for name, v in tests.items():
            tally[name] = int((~v[rows, cols]).sum())
    keep = ok[rows, cols]
    rows, cols = rows[keep], cols[keep]
    order = np.argsort(-score[rows, cols], kind="stable")

    picked = []
    for k in order:
        x = lo[0] + (cols[k] + 0.5) * cell
        y = lo[1] + (rows[k] + 0.5) * cell
        if all((x - px) ** 2 + (y - py) ** 2 >= (2 * radius) ** 2
               for px, py, _ in picked):
            picked.append((float(x), float(y), float(score[rows[k], cols[k]])))
            if len(picked) >= count:
                break
    return picked


def cut(s, centre, radius, up_axis=2, thickness=0.375):
    """The splats of a decal, centred on the origin with the ground around
    it at height zero (as tiles are levelled). The slab is the tiles' own
    below the ground and twice as tall above, so a shrub keeps its top."""
    plane = [i for i in range(3) if i != up_axis]
    d = s.xyz[:, plane] - np.asarray(centre, dtype=np.float32)
    r = np.hypot(d[:, 0], d[:, 1])
    p = s.subset(np.flatnonzero(r <= radius))
    if len(p) < 50:
        raise ValueError(f"nothing to cut at {centre}")
    h = p.xyz[:, up_axis]
    g = ground_level(h, thickness)
    p = p.subset(np.flatnonzero((h >= g - thickness * 0.25) & (h <= g + 2 * thickness)))
    rr = np.hypot(*(p.xyz[:, plane] - np.asarray(centre, dtype=np.float32)).T)
    rim = rr > 0.75 * radius
    off = np.zeros(3, dtype=np.float32)
    off[plane] = centre
    off[up_axis] = np.median(p.xyz[rim, up_axis] if rim.sum() > 20 else p.xyz[:, up_axis])
    out = p.subset(np.arange(len(p)))
    out.xyz = (p.xyz - off).astype(np.float32)
    return out


def room(radius, size):
    """How far from the tile's centre a decal can sit and still leave
    EDGE_CLEAR of every edge alone (each way, a square)."""
    return size / 2.0 - EDGE_CLEAR - radius


def place(tile, decal, radius, size, up_axis=2, feather=0.25, at=(0.0, 0.0)):
    """A copy of `tile` with `decal` set into a disc at `at` (tile frame):
    inside, the tile gives way to the decal over the outer `feather` of the
    radius; outside the disc nothing changes, so the copy's edges are the
    tile's."""
    at = np.asarray(at, dtype=np.float32)
    if max(abs(at[0]), abs(at[1])) > room(radius, size) + 1e-6:
        raise ValueError(f"a decal of radius {radius} m at {tuple(at)} does not "
                         f"fit a {size} m tile with {EDGE_CLEAR} m of edge kept")
    plane = [i for i in range(3) if i != up_axis]
    r_in = radius * (1.0 - feather)

    rt = np.hypot(*(tile.xyz[:, plane] - at).T)
    keep_t = rt >= r_in
    t = tile.subset(np.flatnonzero(keep_t))
    t.opacity = (t.opacity * _smooth((rt[keep_t] - r_in) / (radius - r_in))
                 ).astype(np.float32)

    deg = min(tile.sh_degree, decal.sh_degree)
    dec = decal.truncate_sh(deg) if decal.sh_degree > deg else decal
    t = t.truncate_sh(deg) if t.sh_degree > deg else t
    rd = np.hypot(*dec.xyz[:, plane].T)
    keep_d = rd < radius
    d = dec.subset(np.flatnonzero(keep_d))
    d.opacity = (d.opacity * (1.0 - _smooth((rd[keep_d] - r_in) / (radius - r_in)))
                 ).astype(np.float32)
    d.xyz = d.xyz.copy()
    d.xyz[:, plane] += at

    return Splats(
        xyz=np.concatenate([t.xyz, d.xyz]).astype(np.float32),
        opacity=np.concatenate([t.opacity, d.opacity]).astype(np.float32),
        scale=np.concatenate([t.scale, d.scale]).astype(np.float32),
        rot=np.concatenate([t.rot, d.rot]).astype(np.float32),
        sh_dc=np.concatenate([t.sh_dc, d.sh_dc]).astype(np.float32),
        sh_rest=np.concatenate([t.sh_rest, d.sh_rest]).astype(np.float32),
        sh_degree=deg)
