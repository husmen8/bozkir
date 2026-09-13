"""Measuring popping: how much of a frame changed for reasons other than
the camera moving.

Turn the camera slowly and every pixel changes, because the scene moved
across the screen. That is not popping. Popping is the part that changes
*discontinuously* - a splat, a tile, a whole row of tiles suddenly drawn in
a different order and so suddenly covering something it did not cover in the
previous frame. Differencing two frames directly cannot tell these apart, so
a raw difference rises with camera speed and says nothing about ordering.

The separation is the same one StopThePop makes: estimate where each part of
the image moved to, undo that motion, and whatever difference survives is
the part motion cannot explain. They estimate the motion with RAFT, a
learned optical flow network, and weight the residual with FLIP, a
perceptual image difference. Both are heavy dependencies and neither is
available offline, so the motion here is estimated by block matching and the
weighting is luminance. That is a weaker instrument in two specific ways,
recorded rather than glossed:

  - Block matching finds one translation per block. Rotation and
    perspective inside a block read as residual, so the floor is not zero
    on a moving camera the way an ideal flow's would be. The numbers below
    are therefore comparable *between orderings at the same camera path*,
    which is what an ablation needs, and not comparable to published
    figures.
  - Luminance weighting treats all colour error alike. FLIP would discount
    differences the eye does not resolve.

The design rule this follows: a metric that is honest about being relative
is more useful than one that claims an absolute it cannot support.
"""

import numpy as np

__all__ = ['to_luma', 'motion_field', 'compensate', 'popping', 'sweep']


def to_luma(img):
    """Rec. 709 luminance, float in 0..1, from HxWx3 or HxW input."""
    a = np.asarray(img, dtype=np.float64)
    if a.max() > 1.5:
        a = a / 255.0
    if a.ndim == 2:
        return a
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def motion_field(a, b, block=32, search=12, refine=None):
    """Per-block translation carrying `a` onto `b`, by sum of absolute
    differences over an exhaustive search.


    `refine` is the radius searched per block around the global estimate,
    defaulting to an eighth of the block. Raise it if the scene has strong
    parallax, where near and far move by very different amounts.

    Returns an array of shape (rows, cols, 2) holding (dy, dx) per block.
    """
    a = np.ascontiguousarray(to_luma(a), dtype=np.float32)
    b = np.ascontiguousarray(to_luma(b), dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f'frames differ in size: {a.shape} vs {b.shape}')
    h, w = a.shape

    # Two passes, coarse then fine, rather than one exhaustive search.
    #
    # The exhaustive version scored every block against every offset
    # separately: rows*cols*(2*search+1)^2 small numpy calls, over a million
    # for a 1024-wide frame, where call overhead dominates the arithmetic.
    # Scoring one offset for *all* blocks at once turns that into a few
    # hundred passes over whole arrays.
    #
    # The coarse pass then finds the one translation the whole frame shares
    # - the camera's - on a quarter-size copy, and the fine pass searches a
    # small radius around it per block. Splitting it this way is what makes
    # the search radius affordable, and it is safe for this particular job
    # because the thing being looked for does not move: a tile drawn in the
    # wrong order sits still while everything shifts around it, so it stays
    # well inside `refine` of the global estimate. It would not be safe for
    # tracking objects, which is what the usual warning about coarse-to-fine
    # is about.
    # The coarse pass runs on a quarter-size copy, so its answer is only
    # accurate to `k` pixels. The fine radius has to cover that rounding
    # before it covers anything else, or a motion the coarse pass rounded
    # away is simply unreachable and reads as popping - which is the failure
    # this whole function exists to avoid. A block-sized term on top allows
    # for parts of the frame moving differently from the whole.
    k = 4
    if refine is None:
        refine = k + max(0, block // 8)

    def score(src, dst, blk, dy0, dx0, radius):
        """Best (dy, dx) per block, searched around (dy0, dx0)."""
        hh, ww = dst.shape
        rows = (hh + blk - 1) // blk
        cols = (ww + blk - 1) // blk
        ry = np.arange(0, hh, blk)
        rx = np.arange(0, ww, blk)
        reach = radius + max(abs(dy0), abs(dx0))
        pad = np.pad(src, reach, mode='edge')
        best = np.full((rows, cols), np.inf, dtype=np.float32)
        out = np.zeros((rows, cols, 2), dtype=np.int32)
        for dy in range(dy0 - radius, dy0 + radius + 1):
            for dx in range(dx0 - radius, dx0 + radius + 1):
                win = pad[reach + dy:reach + dy + hh, reach + dx:reach + dx + ww]
                diff = np.abs(win - dst)
                # Block sums in one pass. reduceat handles a last row or
                # column that is narrower than a block, which cropping to a
                # multiple would silently discard.
                sums = np.add.reduceat(np.add.reduceat(diff, ry, axis=0),
                                       rx, axis=1)
                m = sums < best
                if m.any():
                    best[m] = sums[m]
                    out[m] = (dy, dx)
        return out

    # The global translation, by phase correlation rather than by search.
    #
    # A searched coarse pass can only find motion inside whatever radius it
    # was given, and the radius needed is not something the caller can
    # sensibly know: a sweep of 48 frames over a full turn moves the image
    # by an eighth of its width between frames, which is over a hundred
    # pixels. Given a radius of twelve, every block fails to match, every
    # pixel counts as popped, and the metric reports about fifty per cent
    # for any input at all - saturated, and identical for orderings that are
    # visibly different.
    #
    # Phase correlation has no radius. The cross-power spectrum of two
    # images peaks at their offset, so one FFT pair finds a shift of any
    # size in n log n, and `search` then only has to cover how much the
    # scene departs from that one global motion.
    # Negated: phase correlation reports how far the scene moved, while the
    # field records where to read from to undo it.
    gy, gx = _global_shift(a, b)
    return score(a, b, block, -gy, -gx, max(refine, search))


def _global_shift(a, b):
    """The single translation carrying `a` onto `b`, by phase correlation.

    Windowed first: the transform treats both images as periodic, so the
    discontinuity at the frame edge is itself a strong feature and will
    happily correlate with itself at zero shift. A Hann window fades that
    edge out and leaves the scene to decide.
    """
    h, w = a.shape
    wy = np.hanning(h)[:, None]
    wx = np.hanning(w)[None, :]
    fa = np.fft.rfft2(a * wy * wx)
    fb = np.fft.rfft2(b * wy * wx)
    cross = fa.conj() * fb
    mag = np.abs(cross)
    mag[mag == 0] = 1.0
    peak = np.fft.irfft2(cross / mag, s=(h, w))
    dy, dx = np.unravel_index(int(np.argmax(peak)), peak.shape)
    # The result wraps, so the top half of each axis means a negative shift.
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    return int(dy), int(dx)


def compensate(a, field, block=32):
    """Warp `a` by a per-block translation field, giving the prediction of
    the next frame that motion alone accounts for."""
    a = to_luma(a)
    h, w = a.shape
    rows, cols = field.shape[:2]
    out = np.empty_like(a)
    for r in range(rows):
        y0, y1 = r * block, min((r + 1) * block, h)
        for c in range(cols):
            x0, x1 = c * block, min((c + 1) * block, w)
            dy, dx = field[r, c]
            ys = np.clip(np.arange(y0, y1) + dy, 0, h - 1)
            xs = np.clip(np.arange(x0, x1) + dx, 0, w - 1)
            out[y0:y1, x0:x1] = a[np.ix_(ys, xs)]
    return out


def popping(a, b, block=32, search=12, threshold=0.02, margin=None):
    """How much of the change from `a` to `b` motion cannot account for.

    `threshold` is in luminance units, 0..1. A pixel above it is counted as
    popped; below it the difference is resampling noise and the residual
    every block-matched estimate leaves behind.

    `margin` excludes a border from the statistics. Content at the edge of a
    turning frame came from outside it a moment ago, so no previous frame
    can predict it and no compensation can remove it. Counting that as
    popping would make the metric rise with camera speed - the exact failure
    the compensation exists to avoid - and would do it worst at grazing
    angles, where the real artifacts are.

    Left unset, the margin is taken from the motion actually measured, one
    width per axis, because that is exactly how much new content the frame
    took in. A fixed margin cannot do this: a sweep that moves a hundred
    pixels a frame brings in a hundred pixels of unseen scene, and a border
    of twelve leaves nearly all of it counted.

    Returns a dict:
        raw       mean absolute difference with no compensation, for
                  comparison - this is what rises with camera speed
        residual  mean absolute difference after compensation
        fraction  share of pixels whose residual exceeds `threshold`
        p99       99th percentile residual, which catches a small tile
                  popping hard where the mean would not
        mask      the boolean map of popped pixels, for figures
    """
    la, lb = to_luma(a), to_luma(b)
    field = motion_field(la, lb, block=block, search=search)
    pred = compensate(la, field, block=block)
    resid = np.abs(lb - pred)

    if margin is None:
        my = int(abs(np.median(field[..., 0]))) + search
        mx = int(abs(np.median(field[..., 1]))) + search
    else:
        my = mx = int(margin)

    inside = np.zeros(resid.shape, dtype=bool)
    h, w = resid.shape
    if 2 * my < h and 2 * mx < w:
        inside[my:h - my, mx:w - mx] = True
    else:
        # The frame moved by more than its own size; nothing in it was
        # visible a frame ago and there is nothing to compare.
        inside[:] = True

    core = resid[inside]
    return {
        'raw': float(np.abs(lb - la)[inside].mean()),
        'residual': float(core.mean()),
        'fraction': float((core > threshold).mean()),
        'p99': float(np.percentile(core, 99)),
        'mask': (resid > threshold) & inside,
    }


def sweep(frames, **kw):
    """Run `popping` over consecutive pairs of a sequence.

    Reports the peak as well as the mean, because popping is by nature a
    spike: an ordering that is wrong for one frame in forty is not a fortieth
    as bad as one that is wrong throughout, it is a thing the eye catches
    every time it happens.
    """
    out = [popping(frames[i], frames[i + 1], **kw) for i in range(len(frames) - 1)]
    if not out:
        return {'pairs': 0}
    return {
        'pairs': len(out),
        'residual_mean': float(np.mean([r['residual'] for r in out])),
        'residual_peak': float(np.max([r['residual'] for r in out])),
        'fraction_mean': float(np.mean([r['fraction'] for r in out])),
        'fraction_peak': float(np.max([r['fraction'] for r in out])),
        'worst_pair': int(np.argmax([r['fraction'] for r in out])),
        'per_pair': out,
    }