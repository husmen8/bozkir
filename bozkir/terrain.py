"""Reading landforms out of a height field, and height-field I/O.

Hybrid Gaussian Wang Tiles (Zeng, Ma and Sander, SIGGRAPH 2026) picks a
tile class per position from a coverage field alpha_c(x) that a person
paints. Here it comes from the terrain, with standard geomorphometry:

  slope   gradient magnitude; steep ground sheds material.
  TPI     height minus the neighbourhood mean (Weiss 2001): positive on
          ridges, negative in hollows, ~0 on slopes and flats.
  flow    how much terrain drains through a cell; it traces channels.

The maps are only inputs. Which material belongs on a ridge is still a
choice, and deriving it by measurement is the open question.
"""

import numpy as np

__all__ = ["gradient", "slope", "tpi", "curvature", "flow_accumulation",
           "landform", "coverage", "LANDFORMS",
           "read_dsm", "largest_rectangle", "NODATA_BELOW",
           "height_from_points", "write_height_png",
           "read_height_png", "HEIGHT_ENCODING"]

# The ten-class scheme, named as in the geomorphometry literature so maps
# compare with GRASS or ArcGIS directly.
LANDFORMS = ["valley", "hollow", "footslope", "flat", "slope",
             "shoulder", "spur", "ridge", "peak", "pit"]


def gradient(z, spacing=1.0):
    """Partial derivatives by central differences. One-sided at the edges:
    wrapping would invent a cliff (a fake ridge) along the seam."""
    z = np.asarray(z, dtype=np.float64)
    dy, dx = np.gradient(z, spacing)
    return dx, dy


def slope(z, spacing=1.0):
    """Gradient magnitude: rise over run, as a ratio not an angle."""
    dx, dy = gradient(z, spacing)
    return np.hypot(dx, dy)


def _box_mean(z, radius):
    """Mean over a square neighbourhood by summed-area table: constant cost
    per cell whatever the radius."""
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
    """Topographic position index: height above the local mean. Cannot
    separate slopes from flats on its own (landform adds slope). The radius
    sets the scale: small finds boulders, large the landscape."""
    z = np.asarray(z, dtype=np.float64)
    return z - _box_mean(z, radius)


def curvature(z, spacing=1.0):
    """The Laplacian: negative where convex (shedding), positive where
    concave (collecting). Much noisier than TPI on a measured DSM."""
    z = np.asarray(z, dtype=np.float64)
    p = np.pad(z, 1, mode="edge")
    return ((p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
             - 4.0 * z) / (spacing * spacing))


def flow_accumulation(z, spacing=1.0):
    """Cells drained through each cell (D8).

    Each cell sends its water to its steepest-descent neighbour; processed
    highest first, so one pass suffices. Pits are not filled: the interest
    is where channels run, not where water ends up.
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
            # Neighbour heights shifted into place; off-grid stays +inf, so
            # the border drains inwards.
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
    # Highest first, so a cell's total is complete before it passes it on.
    for i in np.argsort(-flat, kind="stable"):
        j = to[i]
        if j >= 0:
            acc[j] += acc[i]
    return acc.reshape(h, w)


def landform(z, spacing=1.0, small=4, large=16, flat_slope=0.08,
             tpi_sd=1.0):
    """Weiss's ten-class landform map from two TPI radii and slope: the
    small radius says what a cell does locally, the large where it sits in
    the landscape, slope separates a hillside from a flat. Thresholds are
    in standard deviations, so they work at any relief."""
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
    """Soft per-class coverage, alpha_c(x) of Hybrid GSWT Eq. 1, for analysis.

    The decision the viewer draws (altitude, sediment, median filter,
    quantile threshold) is bozkir.landform.classify; use that to measure the
    rule.

    `rules` maps a class name to a function of the maps. Default: material
    collects where water runs and ground is concave, and is stripped where
    steep and convex. Normalised to sum to one per cell. `sharpness` is how
    hard the boundary is (1 blends broadly).
    """
    z = np.asarray(z, dtype=np.float64)
    s = slope(z, spacing)
    t = tpi(z, kw.get("radius", 8))
    f = flow_accumulation(z, spacing)

    # Flow spans orders of magnitude, so it is used as a log.
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

NODATA_BELOW = -1000.0


def read_dsm(path):
    """Height in metres, a validity mask and the pixel size (or None).

    rasterio when installed (it knows the nodata value); otherwise Pillow,
    with nodata guessed from the values. Also reads plain TIFFs with no
    georeferencing, unlike validate.read_geotiff.
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
        # Imported here: the rest of this module needs numpy only.
        from PIL import Image
        a = np.asarray(Image.open(path)).astype(np.float64)
        if a.ndim == 3:
            a = a[..., 0]
        good = np.isfinite(a) & (a > NODATA_BELOW)
        return a, good, None


def largest_rectangle(mask):
    """The largest all-True axis-aligned rectangle, (top, left, h, w).
    Row by row with a column-height histogram and the classic stack scan:
    linear in pixels."""
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
    """A height grid rasterised from a point cloud, for captures without a
    DSM.

    The 15th percentile per column rather than the minimum, which a single
    floater below the surface would drag into a spike. Empty cells are
    filled by repeated neighbour averaging (invented terrain); `fill=False`
    leaves them NaN.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    axes = [a for a in (0, 1, 2) if a != up_axis]
    u, v, z = xyz[:, axes[0]], xyz[:, axes[1]], xyz[:, up_axis]

    n = int(max(8, resolution))
    lo_u, hi_u = np.percentile(u, [1, 99])
    lo_v, hi_v = np.percentile(v, [1, 99])
    span = max(hi_u - lo_u, hi_v - lo_v, 1e-9)
    # Square cells, so slope and flow do not depend on the capture's shape.
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
# Heights must reach the browser at more than 8 bits: 256 steps over 9 m is
# 3.5 cm, which terraces gentle slopes and makes flats that flow routing
# reads as sinks. A 16-bit greyscale PNG does not work - browsers decode it
# to 8 bits per channel. So the two bytes go into red (high) and green (low)
# of an ordinary 8-bit RGB PNG; blue repeats the high byte so the file still
# looks like terrain in an image viewer.

HEIGHT_ENCODING = "rg16"


def write_height_png(a, path):
    """Write a height field at 16 bits across two 8-bit channels.
    Returns the normalised (0..1) array that was written."""
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
    """Read a height field back, in 0..1. Older 16-bit greyscale files are
    read as they are; the two-channel encoding only when asked for."""
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
