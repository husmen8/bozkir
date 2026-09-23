"""Does the rule put material where it really is?

The viewer places two materials by the shape of the ground. That is a claim
about the world, and a drone survey can test it: the DTM gives the shape,
the orthophoto shows where each material actually lies. This module lines
the two up on the viewer's cell grid, runs the same rule the viewer runs
(`bozkir/landform.py`), and scores the agreement.

Three things keep the number honest.

* The share is fixed to the truth. The rule is given the real fraction of
  scrub (its `balance`), so it is only judged on *where* material goes, not
  on how much - otherwise a rule that happened to guess the right amount
  would look good for the wrong reason.

* The chance level is measured, not assumed. Material in a landscape is
  clumped, and two clumpy maps overlap by chance far more than two random
  ones. So the truth map is shifted around a torus - every offset, keeping
  its clumping intact - and the rule is scored against each shifted copy.
  The p-value is how often a shifted, meaningless truth scores as well as
  the real one.

* The mapping is stated before looking. In arid ground, vegetation follows
  water, so the hypothesis is *scrub = collected* (class 0). Both mappings
  are reported, and a result that only appears under the flipped one should
  be read as the rule being wrong for this landscape, not as a success.

Reads GeoTIFFs with rasterio when available and with Pillow's TIFF tags
otherwise, so it runs on a machine with numpy and Pillow alone.
"""

from dataclasses import dataclass

import numpy as np

from .landform import classify
from .terrain import largest_rectangle

__all__ = ["Raster", "read_geotiff", "colour_truth", "height_truth",
           "square_window", "sample_bilinear", "cell_fraction",
           "kappa", "spearman", "torus_null", "score", "evaluate"]

# Class means measured from the desert capture's two tile classes (see
# HANDOFF / export_wang --classes 2). Linear 0..1 RGB.
SAND_RGB = (0.52, 0.47, 0.43)
SCRUB_RGB = (0.19, 0.21, 0.15)


@dataclass
class Raster:
    """A north-up raster and where it sits: pixel (row r, col c) has its
    centre at x = x0 + (c + 0.5) * dx, y = y0 - (r + 0.5) * dy."""
    data: np.ndarray
    valid: np.ndarray
    x0: float
    y0: float
    dx: float
    dy: float

    def world_to_pixel(self, x, y):
        return (self.y0 - y) / self.dy - 0.5, (x - self.x0) / self.dx - 0.5

    @property
    def bounds(self):
        h, w = self.valid.shape
        return (self.x0, self.y0 - h * self.dy, self.x0 + w * self.dx, self.y0)


def read_geotiff(path, nodata_below=-1000.0):
    """Load a single- or multi-band GeoTIFF with its placement.

    Bands come back last (h, w) or (h, w, b). `valid` excludes nodata, NaN,
    ODM's -9999 sentinel and, for RGBA, fully transparent pixels.
    """
    try:
        import rasterio
        with rasterio.open(path) as src:
            a = src.read().astype(np.float64)
            t = src.transform
            nodata = src.nodata
            x0, y0, dx, dy = t.c, t.f, abs(t.a), abs(t.e)
        a = a[0] if a.shape[0] == 1 else np.moveaxis(a, 0, -1)
    except ImportError:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(path)
        tags = im.tag_v2
        scale = tags.get(33550)
        tie = tags.get(33922)
        if scale is None or tie is None:
            raise ValueError(f"{path}: no GeoTIFF placement tags; install "
                             "rasterio or pass a georeferenced file")
        dx, dy = float(scale[0]), float(scale[1])
        x0 = float(tie[3]) - float(tie[0]) * dx
        y0 = float(tie[4]) + float(tie[1]) * dy
        nd = tags.get(42113)
        nodata = float(nd) if nd not in (None, "") else None
        a = np.asarray(im).astype(np.float64)

    if a.ndim == 3:
        valid = np.isfinite(a).all(axis=2)
        if a.shape[2] == 4:
            valid &= a[..., 3] > 0
            a = a[..., :3]
        if a.max() > 1.5:
            a = a / 255.0
    else:
        valid = np.isfinite(a) & (a > nodata_below)
        if nodata is not None:
            valid &= a != nodata
    return Raster(a, valid, x0, y0, dx, dy)


def colour_truth(rgb, scrub=SCRUB_RGB, sand=SAND_RGB):
    """1 where a pixel is nearer the scrub colour than the sand colour.

    Hard shadow is dark too and will read as scrub; captures in soft light
    avoid it, and `height_truth` is the cross-check that is blind to colour.
    """
    d_scrub = ((rgb - np.asarray(scrub)) ** 2).sum(-1)
    d_sand = ((rgb - np.asarray(sand)) ** 2).sum(-1)
    return (d_scrub < d_sand).astype(np.float64)


def height_truth(dsm, dtm, min_height=0.25):
    """1 where the surface stands above the bare ground: dsm - dtm > h.

    Both rasters must already be on the same grid. Independent of colour,
    so shadows cannot fool it; ODM's ground filter can, around the edges of
    large bushes.
    """
    return ((dsm - dtm) > min_height).astype(np.float64)


def sample_bilinear(r, x, y):
    """Sample a single-band raster at world points, clamping at the edges."""
    row, col = r.world_to_pixel(np.asarray(x), np.asarray(y))
    h, w = r.data.shape[:2]
    row = np.clip(row, 0, h - 1)
    col = np.clip(col, 0, w - 1)
    r0 = np.floor(row).astype(int)
    c0 = np.floor(col).astype(int)
    r1 = np.minimum(r0 + 1, h - 1)
    c1 = np.minimum(c0 + 1, w - 1)
    fr, fc = row - r0, col - c0
    a = r.data
    return ((a[r0, c0] * (1 - fc) + a[r0, c1] * fc) * (1 - fr)
            + (a[r1, c0] * (1 - fc) + a[r1, c1] * fc) * fr)


def square_window(dtm, ortho):
    """The largest square, in world coordinates, valid in both rasters.

    Returns (x_min, y_min, side). Computed on the DTM's pixel grid, with the
    orthophoto's validity sampled onto it.
    """
    h, w = dtm.valid.shape
    rows, cols = np.mgrid[0:h, 0:w]
    x = dtm.x0 + (cols + 0.5) * dtm.dx
    y = dtm.y0 - (rows + 0.5) * dtm.dy
    orow, ocol = ortho.world_to_pixel(x, y)
    oh, ow = ortho.valid.shape
    inside = (orow >= 0) & (orow <= oh - 1) & (ocol >= 0) & (ocol <= ow - 1)
    ok = dtm.valid & inside
    ok[inside] &= ortho.valid[np.round(orow[inside]).astype(int),
                              np.round(ocol[inside]).astype(int)]
    top, left, hh, ww = largest_rectangle(ok)
    if hh == 0:
        raise ValueError("the DTM and orthophoto do not overlap")
    side_px = min(hh, ww)
    top += (hh - side_px) // 2
    left += (ww - side_px) // 2
    side = side_px * dtm.dx
    x_min = dtm.x0 + left * dtm.dx
    y_min = dtm.y0 - (top + side_px) * dtm.dy
    return x_min, y_min, side


def cell_fraction(mask_raster, x_min, y_min, side, n):
    """Mean of a 0/1 raster over each of n*n square cells, row-major from
    the north-west corner (the same order as the height samples)."""
    r = mask_raster
    out = np.zeros((n, n))
    cell = side / n
    for j in range(n):
        y_hi = y_min + side - j * cell
        for i in range(n):
            x_lo = x_min + i * cell
            r0, c0 = r.world_to_pixel(x_lo, y_hi)
            r1, c1 = r.world_to_pixel(x_lo + cell, y_hi - cell)
            r0, c0 = int(np.ceil(r0)), int(np.ceil(c0))
            r1, c1 = int(np.floor(r1)) + 1, int(np.floor(c1)) + 1
            blk = r.data[max(r0, 0):r1, max(c0, 0):c1]
            v = r.valid[max(r0, 0):r1, max(c0, 0):c1]
            out[j, i] = blk[v].mean() if v.any() else np.nan
    return out.ravel()


def kappa(a, b):
    """Cohen's kappa for two 0/1 label arrays: agreement beyond chance."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def spearman(a, b):
    """Rank correlation, average ranks for ties."""
    def rank(x):
        x = np.asarray(x, float)
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x))
        r[order] = np.arange(len(x))
        # Average the ranks of tied values.
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=r)
        return sums[inv] / cnt[inv]
    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def torus_null(pred, truth, n, min_shift=2):
    """Kappa of `pred` against every circular shift of `truth`.

    Shifts closer than `min_shift` cells to zero on both axes are left out,
    since those are near copies of the real alignment rather than chance.
    """
    p = np.asarray(pred, bool).reshape(n, n)
    t = np.asarray(truth, bool).reshape(n, n)
    out = []
    for dy in range(n):
        for dx in range(n):
            ny, nx = min(dy, n - dy), min(dx, n - dx)
            if ny < min_shift and nx < min_shift:
                continue
            out.append(kappa(p, np.roll(np.roll(t, dy, 0), dx, 1)))
    return np.array(out)


def score(pred, truth_frac, lead, n):
    """All the numbers for one prediction against one truth."""
    share = float(np.nanmean(truth_frac))
    # Truth labels at the same share the rule was given: the top `share`
    # of cells by scrub fraction. Ties broken by position, deterministically.
    order = np.lexsort((np.arange(n * n), -np.nan_to_num(truth_frac)))
    truth = np.zeros(n * n, bool)
    truth[order[:int(round(share * n * n))]] = True
    pred = np.asarray(pred, bool)
    null = torus_null(pred, truth, n)
    k = kappa(pred, truth)
    inter = (pred & truth).sum()
    union = (pred | truth).sum()
    return {
        "share": share,
        "accuracy": float((pred == truth).mean()),
        "kappa": k,
        "iou": float(inter / union) if union else 0.0,
        "spearman": spearman(lead, np.nan_to_num(truth_frac)),
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "p": float((1 + (null >= k).sum()) / (1 + null.size)),
        "truth": truth,
    }


def evaluate(z, truth_frac, n, spacing, altitude=0.4, coherence=2,
             sharpness=2.0):
    """Score the rule and each of its inputs alone against one truth map.

    `z` is height on an (n*4)^2 grid, `truth_frac` the scrub fraction per
    cell. Returns {name: score dict}. 'rule' is the viewer's rule under the
    stated hypothesis (scrub = collected); 'rule, flipped' the opposite
    mapping; the rest are single cues, to show what the rule leans on.
    """
    share = float(np.nanmean(truth_frac))
    kw = dict(spacing=spacing, balance=share, coherence=coherence,
              sharpness=sharpness)
    out = {}
    r = classify(z, n, 2, altitude=altitude, **kw)
    out["rule"] = score(r["cls"] == 0, truth_frac, r["lead"], n)
    r0 = classify(z, n, 2, altitude=0.0, **kw)
    out["rule, no altitude"] = score(r0["cls"] == 0, truth_frac, r0["lead"], n)
    rf = classify(z, n, 2, altitude=altitude, **{**kw, "balance": 1 - share})
    out["rule, flipped"] = score(rf["cls"] == 1, truth_frac, -rf["lead"], n)

    cues = {
        "low ground": lambda M: 1 - M["elevation"],
        "drainage": lambda M: M["flow"],
        "hollows (low TPI)": lambda M: 1 - M["tpi"],
        "gentle slope": lambda M: 1 - M["slope"],
    }
    for name, fn in cues.items():
        rc = classify(z, n, 2, rules=[fn, lambda M, f=fn: 1 - f(M)], **kw)
        out[name] = score(rc["cls"] == 0, truth_frac, rc["lead"], n)
    out["_maps"] = r["maps"]
    out["_cls"] = r["cls"]
    return out