"""Reading landforms out of a height field.

Hybrid Gaussian Wang Tiles (Zeng, Ma and Sander, SIGGRAPH 2026) selects
which class of tile appears at a world position from a scalar field per
class, the target coverage alpha_c(x). Their formulation takes that field
as input; their tool has a person paint it.

Nobody paints a ten-kilometre map. In procedural terrain the rule has to
come from the terrain: gravel collects where water runs, bare rock shows on
ridges where material is stripped, fines settle in hollows. That is how
every terrain shader in every engine has always worked, and it is the
opposite of painting regions.

So this module computes alpha_c from the height field, and what it computes
is not invented here. Geomorphometry has been classifying landforms from
elevation for decades:

  slope           the gradient magnitude. Steep ground sheds material.
  TPI             a cell's height minus its neighbourhood mean. Positive is
                  a ridge, negative a hollow, near zero a slope or a flat.
                  Weiss (2001); the standard ten-class landform map comes
                  from combining two neighbourhood sizes with slope.
  flow            how much of the terrain drains through a cell. This is
                  the one that produces channels, because it is the thing
                  that makes channels.

None of it needs training or a model. All of it is arithmetic on a grid.

What is not standard is using the result to place captured material, and
that is the part worth being careful about: the maps below are inputs to a
decision, and the decision - which material belongs on a ridge - still has
to come from somewhere. Deriving it by measurement rather than taste is the
open question, not the classification.
"""

import numpy as np

__all__ = ["gradient", "slope", "tpi", "curvature", "flow_accumulation",
           "landform", "coverage", "LANDFORMS",
           "read_dsm", "largest_rectangle", "NODATA_BELOW",
           "height_from_points", "write_height_png",
           "read_height_png", "HEIGHT_ENCODING"]

# The ten-class scheme these produce, in the order the codes run. Named to
# match the geomorphometry literature so a map from here can be compared
# with one from GRASS or ArcGIS without a translation table.
LANDFORMS = ["valley", "hollow", "footslope", "flat", "slope",
             "shoulder", "spur", "ridge", "peak", "pit"]


def gradient(z, spacing=1.0):
    """Partial derivatives of the surface, by central differences.

    Edges use a one-sided difference rather than wrapping. Wrapping would
    join the far side of the terrain to the near side and invent a cliff
    along the seam, which then reads as a ridge and collects a material
    that has no business being there.
    """
    z = np.asarray(z, dtype=np.float64)
    dy, dx = np.gradient(z, spacing)
    return dx, dy


def slope(z, spacing=1.0):
    """Gradient magnitude: rise over run, as a ratio not an angle."""
    dx, dy = gradient(z, spacing)
    return np.hypot(dx, dy)


def _box_mean(z, radius):
    """Mean over a square neighbourhood, by summed-area table.

    Constant time per cell whatever the radius, which matters because the
    standard landform classification wants two very different radii and the
    large one would otherwise dominate the cost.
    """
    r = int(max(1, radius))
    h, w = z.shape
    pad = np.pad(z, r, mode="edge")
    ii = np.zeros((pad.shape[0] + 1, pad.shape[1] + 1))
    np.cumsum(np.cumsum(pad, axis=0), axis=1, out=ii[1:, 1:])
    size = 2 * r + 1
    total = (ii[size:size + h, size:size + w]
             - ii[0:h, size:size + w]
             - ii[size:size + h, 0:w]
             + ii[0:h, 0:w])
    return total / (size * size)


def tpi(z, radius=8):
    """Topographic position index: height above the local mean.

    Positive on ridges and spurs, negative in hollows and valleys, near
    zero on planar slopes and flats - which is why it cannot separate those
    two on its own, and why `landform` below brings slope in as well.

    The radius sets what counts as local, and it is the whole character of
    the result: a small radius finds every boulder, a large one finds the
    shape of the landscape. Neither is more correct.
    """
    z = np.asarray(z, dtype=np.float64)
    return z - _box_mean(z, radius)


def curvature(z, spacing=1.0):
    """The Laplacian: how the surface bends.

    Negative where the ground is convex and shedding, positive where it is
    concave and collecting. Similar in spirit to TPI at a small radius, and
    cheaper, but far noisier on a measured surface - it is a second
    derivative, and a DSM's noise survives one differentiation to become
    the signal in the next.
    """
    z = np.asarray(z, dtype=np.float64)
    p = np.pad(z, 1, mode="edge")
    return ((p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
             - 4.0 * z) / (spacing * spacing))


def flow_accumulation(z, spacing=1.0):
    """How much terrain drains through each cell.

    The single-direction method: every cell sends all of its water to its
    lowest neighbour, and the totals are accumulated from the highest cell
    downwards. Processing in descending height order means a cell's own
    total is complete before it passes anything on, so one pass suffices
    and no iteration is needed.

    This is the map that produces branching channels, and it produces them
    for the right reason - it is tracing where water actually goes on this
    surface, not drawing something channel-shaped. Returns the count of
    cells drained through each cell, so the value is in cells and scales
    with the grid rather than with the terrain.

    Sinks are not filled. A pit keeps whatever drains into it and passes
    nothing on, which is what a pit does; filling them first is the right
    move for hydrology over real catchments and unnecessary here, where the
    interest is in where the channels run rather than where the water ends
    up.
    """
    z = np.asarray(z, dtype=np.float64)
    h, w = z.shape
    flat = z.ravel()
    index = np.arange(h * w).reshape(h, w)

    # Steepest descent to one of the eight neighbours.
    best_drop = np.zeros((h, w))
    best_to = np.full((h, w), -1, dtype=np.int64)
    diag = np.sqrt(2.0) * spacing

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            # The neighbour's height, brought into this cell's position.
            # Cells whose neighbour lies off the grid keep +inf and so are
            # never chosen, which leaves the border draining inwards.
            nz = np.full((h, w), np.inf)
            ni = np.full((h, w), -1, dtype=np.int64)
            src_y = slice(max(0, dy), h + min(0, dy))
            src_x = slice(max(0, dx), w + min(0, dx))
            dst_y = slice(max(0, -dy), h + min(0, -dy))
            dst_x = slice(max(0, -dx), w + min(0, -dx))
            nz[dst_y, dst_x] = z[src_y, src_x]
            ni[dst_y, dst_x] = index[src_y, src_x]

            dist = diag if (dx and dy) else spacing
            drop = np.where(np.isfinite(nz), (z - nz) / dist, -np.inf)
            take = drop > best_drop
            best_drop = np.where(take, drop, best_drop)
            best_to = np.where(take, ni, best_to)

    to = best_to.ravel()
    acc = np.ones(h * w, dtype=np.float64)
    # Highest first, so a cell's own total is complete before it passes
    # anything downhill. One pass, no iteration.
    for i in np.argsort(-flat, kind="stable"):
        j = to[i]
        if j >= 0:
            acc[j] += acc[i]
    return acc.reshape(h, w)


def landform(z, spacing=1.0, small=4, large=16, flat_slope=0.08,
             tpi_sd=1.0):
    """The ten-class landform map, from two TPI radii and slope.

    Weiss's scheme: a small neighbourhood says what the cell is doing
    locally, a large one says where it sits in the landscape, and slope
    separates a planar hillside from a flat. A cell high in both is a peak;
    high locally but low regionally is a spur on a valley wall; near zero in
    both is either a flat or an open slope depending on how steep it is.

    Thresholds are in standard deviations of each TPI, not in metres, so
    the same numbers work on a gentle survey and on a mountain.
    """
    z = np.asarray(z, dtype=np.float64)
    t_small = tpi(z, small)
    t_large = tpi(z, large)
    s = slope(z, spacing)

    def norm(t):
        sd = t.std()
        return t / sd if sd > 1e-12 else np.zeros_like(t)

    a, b = norm(t_small), norm(t_large)
    out = np.full(z.shape, LANDFORMS.index("slope"), dtype=np.int8)
    hi, lo = tpi_sd, -tpi_sd

    def put(name, mask):
        out[mask] = LANDFORMS.index(name)

    put("flat", (a > lo) & (a < hi) & (b > lo) & (b < hi) & (s < flat_slope))
    put("slope", (a > lo) & (a < hi) & (b > lo) & (b < hi) & (s >= flat_slope))
    put("hollow", (a <= lo) & (b > lo) & (b < hi))
    put("spur", (a >= hi) & (b > lo) & (b < hi))
    put("footslope", (a > lo) & (a < hi) & (b <= lo))
    put("shoulder", (a > lo) & (a < hi) & (b >= hi))
    put("valley", (a <= lo) & (b <= lo))
    put("pit", (a <= lo) & (b >= hi))
    put("ridge", (a >= hi) & (b >= hi))
    put("peak", (a >= hi) & (b <= lo))
    return out


def coverage(z, spacing=1.0, rules=None, sharpness=2.0, **kw):
    """Target coverage per class, the alpha_c(x) of Hybrid GSWT Eq. 1.

    `rules` maps a class name to a function of the terrain maps, returning
    an unnormalised weight per cell. The default is deliberately the
    simplest thing that is defensible rather than a tuned one: material
    collects where water runs and where the ground is concave, and is
    stripped where it is steep and convex.

    Returned normalised so the classes sum to one at every cell, because
    that is what a coverage is, and because their selection takes the
    largest and a field that did not sum to one would let one class win
    everywhere by being scaled up.

    `sharpness` is how decisively the rule picks: 1 blends broadly, large
    values approach a hard boundary. It exists because the right answer is
    a judgement about how a landscape looks, not something the terrain can
    settle.
    """
    z = np.asarray(z, dtype=np.float64)
    s = slope(z, spacing)
    t = tpi(z, kw.get("radius", 8))
    f = flow_accumulation(z, spacing)

    # Flow spans orders of magnitude - a main channel drains thousands of
    # cells and a hillside drains one - so it is used as a log, which is
    # also how it is always mapped.
    lf = np.log1p(f)

    def unit(a):
        lo, hi = a.min(), a.max()
        return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)

    maps = {"slope": unit(s), "tpi": unit(t), "flow": unit(lf),
            "curvature": unit(curvature(z, spacing))}

    if rules is None:
        rules = {
            # Where water runs and the ground is concave.
            "collected": lambda m: 0.6 * m["flow"] + 0.4 * (1.0 - m["tpi"]),
            # Steep, convex, no drainage through it.
            "exposed": lambda m: 0.5 * m["slope"] + 0.5 * m["tpi"],
        }

    weights = {k: np.clip(fn(maps), 0, None) ** sharpness
               for k, fn in rules.items()}
    total = sum(weights.values())
    total = np.where(total > 1e-12, total, 1.0)
    return {k: v / total for k, v in weights.items()}


# --- reading a measured surface ----------------------------------------
#
# A digital surface model is the usual source for everything above, so
# loading one belongs beside the analysis rather than inside the script
# that happens to call it first. scripts/heightmap.py is a wrapper over
# these, the same as every other script in that folder.

NODATA_BELOW = -1000.0


def read_dsm(path):
    """Height in metres and a mask of what is real.

    rasterio when it is installed, since it knows the file's own nodata
    value; PIL otherwise, which reads the pixels but not the metadata, so
    the sentinel has to be guessed from the values. Both paths are used -
    the fallback is what runs on a machine that only has Pillow.
    """
    try:
        import rasterio
        with rasterio.open(path) as src:
            a = src.read(1).astype(np.float64)
            nodata = src.nodata
            res = abs(src.transform.a) if src.transform else None
        good = np.isfinite(a)
        if nodata is not None:
            good &= a != nodata
        good &= a > NODATA_BELOW
        return a, good, res
    except ImportError:
        # Imported here rather than at the top: everything else in this
        # module is numpy alone, and a machine doing terrain analysis on an
        # array it already has should not need Pillow installed to do it.
        from PIL import Image
        a = np.asarray(Image.open(path)).astype(np.float64)
        if a.ndim == 3:
            a = a[..., 0]
        good = np.isfinite(a) & (a > NODATA_BELOW)
        return a, good, None


def largest_rectangle(mask):
    """The largest all-True axis-aligned rectangle, as (top, left, h, w).

    Row by row, keeping for each column how many valid rows reach up to
    here; each row is then a histogram whose largest rectangle is the
    classic stack scan. Linear in pixels, which matters because a DSM is
    a few hundred thousand of them and the obvious four-nested-loop
    version is not.
    """
    if not mask.any():
        return (0, 0, 0, 0)
    rows, cols = mask.shape
    heights = np.zeros(cols + 1, dtype=np.int64)
    best = (0, 0, 0, 0)
    best_area = 0

    for r in range(rows):
        heights[:cols] = np.where(mask[r], heights[:cols] + 1, 0)
        stack = []
        for c in range(cols + 1):
            start = c
            while stack and stack[-1][1] >= heights[c]:
                s, h = stack.pop()
                area = h * (c - s)
                if area > best_area:
                    best_area = area
                    best = (r - h + 1, s, h, c - s)
                start = s
            stack.append((start, heights[c]))
    return best


def height_from_points(xyz, up_axis=2, resolution=192, percentile=15.0,
                       fill=True):
    """A height grid rasterised from a point cloud.

    Drone surveys come with a DSM; a phone capture does not. But every
    capture of ground contains the ground, and the patch search already
    asks where it is column by column - this does the same thing across the
    whole scene and keeps the answer.

    `percentile` rather than the minimum: the lowest point in a column is
    as likely to be a floater below the surface as the surface itself, and
    one such point drags a whole cell down into a spike. The fifteenth
    percentile is under the grass and above the noise.

    Empty cells are filled from their neighbours by repeated averaging,
    which is the cheapest thing that produces a continuous surface. It
    invents terrain where there was no capture, so `fill=False` leaves them
    as NaN for a caller that would rather crop than invent.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    axes = [a for a in (0, 1, 2) if a != up_axis]
    u, v, z = xyz[:, axes[0]], xyz[:, axes[1]], xyz[:, up_axis]

    n = int(max(8, resolution))
    lo_u, hi_u = np.percentile(u, [1, 99])
    lo_v, hi_v = np.percentile(v, [1, 99])
    span = max(hi_u - lo_u, hi_v - lo_v, 1e-9)
    # Square cells: a grid stretched to the cloud's bounding box would make
    # slope and flow depend on which way the capture happened to be walked.
    cu = (lo_u + hi_u) / 2, (lo_v + hi_v) / 2
    iu = np.clip(((u - cu[0]) / span + 0.5) * (n - 1), 0, n - 1).astype(int)
    iv = np.clip(((v - cu[1]) / span + 0.5) * (n - 1), 0, n - 1).astype(int)

    grid = np.full((n, n), np.nan)
    flat = iv * n + iu
    order = np.argsort(flat, kind="stable")
    flat_sorted, z_sorted = flat[order], z[order]
    edges = np.searchsorted(flat_sorted, np.arange(n * n + 1))
    for cell in range(n * n):
        a, b = edges[cell], edges[cell + 1]
        if b > a:
            grid.ravel()[cell] = np.percentile(z_sorted[a:b], percentile)

    if not fill:
        return grid

    empty = ~np.isfinite(grid)
    if empty.all():
        return np.zeros_like(grid)
    out = np.where(empty, np.nanmean(grid), grid)
    for _ in range(max(8, n // 8)):
        p = np.pad(out, 1, mode="edge")
        blur = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4.0
        out = np.where(empty, blur, out)
    return out


# --- storing a height field --------------------------------------------
#
# A height field has to reach the browser at more than eight bits. A nine
# metre range in 256 steps is 3.5 cm per step, which terraces visibly on a
# gentle slope - and worse, it manufactures flat plateaus, and flow routing
# treats every flat cell as a sink. So quantisation does not only look bad;
# it breaks the drainage the material rule reads.
#
# A 16-bit greyscale PNG was the obvious answer and does not work. The
# browser decodes an image into a canvas at eight bits per channel whatever
# the file held, and a greyscale image comes back with red, green and blue
# all equal to the high byte. The low byte is simply gone, silently, in
# every browser, and the field arrives at 8 bits while the file on disk
# claims 16.
#
# So the two bytes are written into two channels deliberately: the high
# byte in red, the low in green, in an ordinary 8-bit colour PNG. Nothing
# about that depends on how a browser treats 16-bit images, because there
# is no 16-bit image anywhere. Blue repeats the high byte so the file still
# looks like the terrain when opened in an image viewer, rather than like
# noise.

HEIGHT_ENCODING = "rg16"


def write_height_png(a, path):
    """Write a height field at 16 bits, split across two 8-bit channels.

    Normalises to the full range first, and returns the normalised array
    so the caller can record what it wrote.
    """
    from PIL import Image
    a = np.asarray(a, dtype=np.float64)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    norm = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
    q = np.clip(np.round(norm * 65535), 0, 65535).astype(np.uint32)
    rgb = np.stack([(q >> 8).astype(np.uint8),
                    (q & 0xFF).astype(np.uint8),
                    (q >> 8).astype(np.uint8)], axis=-1)
    Image.fromarray(rgb, mode="RGB").save(path)
    return norm


def read_height_png(path, encoding=HEIGHT_ENCODING):
    """Read a height field back, in 0..1.

    Older files were written as 16-bit greyscale; those are read as they
    are. The two-channel encoding is only assumed when asked for, so an
    ordinary image passed in by mistake is read as greyscale rather than
    being misinterpreted as bytes.
    """
    from PIL import Image
    im = Image.open(path)
    if encoding == HEIGHT_ENCODING and im.mode in ("RGB", "RGBA"):
        px = np.asarray(im.convert("RGB")).astype(np.uint32)
        return (px[..., 0] * 256 + px[..., 1]) / 65535.0
    a = np.asarray(im).astype(np.float64)
    if a.ndim == 3:
        a = a[..., 0]
    top = 65535.0 if a.max() > 255 else 255.0
    return a / top