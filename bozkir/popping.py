"""Measuring popping: how much of a frame changed for reasons other than
the camera moving.

A raw frame difference rises with camera speed. As in StopThePop, the
motion is estimated and undone, and what survives is what motion cannot
explain. StopThePop uses RAFT optical flow and FLIP; here it is block
matching and luminance, which is weaker in two ways:

  - One translation per block, so rotation and perspective inside a block
    read as residual. Numbers compare orderings on the same camera path,
    not with published figures.
  - Luminance treats all colour error alike; FLIP would discount what the
    eye does not resolve.

Block matching is integer-accurate, and on splat terrain a one-pixel
registration error is larger than the ordering effect - which is why this
does not yet resolve the ordering difference.
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
    """Per-block translation carrying `a` onto `b` (sum of absolute
    differences), searched around one global shift from phase correlation.

    `refine`: per-block radius around the global shift (raise it for strong
    parallax). Returns (rows, cols, 2) of (dy, dx) per block.
    """
    a = np.ascontiguousarray(to_luma(a), dtype=np.float32)
    b = np.ascontiguousarray(to_luma(b), dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f'frames differ in size: {a.shape} vs {b.shape}')
    h, w = a.shape

    # Each offset is scored for all blocks at once (a few hundred array
    # passes instead of a million small calls). The fine radius covers the
    # global estimate's rounding (`k`) plus a block-sized term for parts of
    # the frame moving differently; a motion it cannot reach would read as
    # popping.
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
                # Block sums in one pass; reduceat keeps a narrower last
                # row or column.
                sums = np.add.reduceat(np.add.reduceat(diff, ry, axis=0),
                                       rx, axis=1)
                m = sums < best
                if m.any():
                    best[m] = sums[m]
                    out[m] = (dy, dx)
        return out

    # The global shift by phase correlation, which has no search radius: a
    # 48-frame full turn moves the image 100+ px per frame, and a searched
    # pass with radius 12 matched nothing and saturated at ~50% for any
    # input. Negated: the field records where to read from to undo it.
    gy, gx = _global_shift(a, b)
    return score(a, b, block, -gy, -gx, max(refine, search))


def _global_shift(a, b):
    """The single translation carrying `a` onto `b`, by phase correlation.
    Hann-windowed, or the frame edge (periodic to the FFT) correlates with
    itself at zero shift."""
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

    `threshold`: luminance (0..1) above which a pixel counts as popped.
    `margin`: border left out, since content entering a turning frame
    cannot be predicted. By default the measured motion per axis (plus
    `search`), i.e. exactly how much new content came in.

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
        # Moved more than its own size: nothing to compare.
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
    """Run `popping` over consecutive pairs. Peaks are reported beside
    means: popping is a spike the eye catches every time."""
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