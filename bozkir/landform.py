"""The material rule, the same one the viewer runs.

A line-for-line port of `web/landform.js`. The browser decides which class
each cell gets; this module makes the same decision in Python, so that the
rule can be measured - against a real survey in `scripts/validate_rule.py`,
and in figures - using the code that is actually drawn rather than a
look-alike.

It is kept separate from `bozkir/terrain.py` on purpose. `terrain.coverage`
is the soft, per-class coverage of Hybrid GSWT's Eq. 1 and is useful for
analysis; `classify` here is the viewer's hard decision, with everything the
viewer adds on top (fine sampling, max-pooled drainage, altitude, the
sediment record, a median filter and a quantile threshold). The two answer
different questions and should not be confused.

`tests/test_pipeline.py` runs both implementations, through tests/bridge.mjs,
on the same height fields
and requires the same class in every cell. If you change one, change the
other, and that test tells you when you have not.

Names and argument order follow the JavaScript, including flat arrays in
row-major order of length n*n, so the two read side by side.
"""

import json
from pathlib import Path

import numpy as np

# The rule's weights, shared with the viewer: web/rule.json is the only
# place they are written.
RULE_PATH = Path(__file__).resolve().parents[1] / "web" / "rule.json"
RULE = json.loads(RULE_PATH.read_text())

__all__ = ["downsample", "slope", "tpi", "flow", "median", "smooth",
           "quantile", "rank_unit", "unit", "classify", "cue_weights", "sample_grid"]


def sample_grid(n, tile_size, height, grid_angle=0.0, over=1):
    """Sample `height(x, y)` onto an (n*over)^2 grid centred on the origin.

    `height` must accept numpy arrays. Mirrors `sampleGrid`.
    """
    k = max(1, int(over))
    m = n * k
    span = (n - 1) * tile_size
    step = span / (m - 1) if m > 1 else 0.0
    half = (m - 1) / 2
    a = np.deg2rad(grid_angle)
    ca, sa = np.cos(a), np.sin(a)
    j, i = np.mgrid[0:m, 0:m].astype(np.float64)
    lx, ly = (i - half) * step, (j - half) * step
    return np.asarray(height(ca * lx - sa * ly, sa * lx + ca * ly),
                      dtype=np.float64).ravel()


def downsample(a, m, n, how="mean"):
    """m*m -> n*n by block mean or block max. Identity when m == n."""
    over = max(1, round(m / n))
    if over == 1:
        return np.asarray(a, dtype=np.float64)
    b = np.asarray(a, dtype=np.float64).reshape(m, m)[:n * over, :n * over]
    b = b.reshape(n, over, n, over)
    if how == "max":
        return b.max(axis=(1, 3)).ravel()
    # Summed in the same order as the JS loop (row of the block, then
    # column), so the float result matches to the last bit.
    acc = np.zeros((n, n))
    for r in range(over):
        for c in range(over):
            acc = acc + b[:, r, :, c]
    return (acc / (over * over)).ravel()


def slope(z, n, spacing=1.0):
    """Gradient magnitude by central differences, edges clamped."""
    g = np.asarray(z, dtype=np.float64).reshape(n, n)
    p = np.pad(g, 1, mode="edge")
    dx = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * spacing)
    dy = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * spacing)
    return np.hypot(dx, dy).ravel()


def _window_sum(g, r):
    """Sum and count over a (2r+1)^2 window, clipped at the edges."""
    n = g.shape[0]
    c = np.zeros((n + 1, n + 1))
    c[1:, 1:] = g.cumsum(0).cumsum(1)
    idx = np.arange(n)
    lo = np.clip(idx - r, 0, n)
    hi = np.clip(idx + r + 1, 0, n)
    s = (c[hi][:, hi] - c[lo][:, hi] - c[hi][:, lo] + c[lo][:, lo])
    cnt = np.outer(hi - lo, hi - lo).astype(np.float64)
    return s, cnt


def tpi(z, n, radius=3):
    """Height minus the mean of the surrounding window."""
    r = max(1, int(radius))
    g = np.asarray(z, dtype=np.float64).reshape(n, n)
    s, cnt = _window_sum(g, r)
    return (g - s / np.maximum(cnt, 1)).ravel()


def flow(z, n):
    """D8 flow accumulation: cells drained through each cell, itself included.

    Each cell passes everything to its steepest-descent neighbour (first in
    scan order on ties, and only if strictly downhill), processed from the
    highest cell down. Ties in height are broken by index, as the stable
    JavaScript sort does.
    """
    z = np.asarray(z, dtype=np.float64)
    g = z.reshape(n, n)
    best = np.zeros((n, n))
    to = np.full((n, n), -1, dtype=np.int64)
    idx = np.arange(n * n).reshape(n, n)
    for b in (-1, 0, 1):
        for a in (-1, 0, 1):
            if a == 0 and b == 0:
                continue
            nz = np.full((n, n), np.nan)
            ni = np.full((n, n), -1, dtype=np.int64)
            ys, yd = slice(max(0, b), n + min(0, b)), slice(max(0, -b), n + min(0, -b))
            xs, xd = slice(max(0, a), n + min(0, a)), slice(max(0, -a), n + min(0, -a))
            nz[yd, xd] = g[ys, xs]
            ni[yd, xd] = idx[ys, xs]
            drop = (g - nz) / (np.sqrt(2.0) if (a and b) else 1.0)
            take = np.nan_to_num(drop, nan=-np.inf) > best
            best = np.where(take, drop, best)
            to = np.where(take, ni, to)
    to = to.ravel()
    order = np.argsort(-z, kind="stable")
    acc = np.ones(n * n)
    for k in order.tolist():
        j = to[k]
        if j >= 0:
            acc[j] += acc[k]
    return acc


def unit(a):
    """Stretch to [0, 1]; a constant field reads 0.5, meaning 'no information'."""
    a = np.asarray(a, dtype=np.float64)
    lo, hi = a.min(), a.max()
    if hi - lo > 1e-12:
        return (a - lo) / (hi - lo)
    return np.full_like(a, 0.5)


def median(a, n, radius):
    """Edge-preserving median over a clipped window.

    Takes element count//2 of the sorted window, as the JS does, which on an
    even count at the border is the upper of the two middle values.
    """
    r = max(0, int(radius))
    a = np.asarray(a, dtype=np.float64)
    if not r:
        return a.copy()
    g = a.reshape(n, n)
    w = 2 * r + 1
    p = np.pad(g, r, mode="constant", constant_values=np.inf)
    win = np.lib.stride_tricks.sliding_window_view(p, (w, w)).reshape(n, n, w * w)
    srt = np.sort(win, axis=2)
    idx = np.arange(n)
    rows = np.minimum(idx + r, n - 1) - np.maximum(idx - r, 0) + 1
    cnt = np.outer(rows, rows)
    return np.take_along_axis(srt, (cnt >> 1)[..., None], axis=2)[..., 0].ravel()


def smooth(a, n, radius):
    """Two passes of a separable box blur, clipped at the edges."""
    r = max(0, int(radius))
    a = np.asarray(a, dtype=np.float64)
    if not r:
        return a
    cur = a.reshape(n, n).copy()
    idx = np.arange(n)
    for _ in range(2):
        # Summed term by term in the same order as the JS inner loop.
        tmp = np.zeros((n, n))
        cnt = np.zeros(n)
        for c in range(-r, r + 1):
            ii = idx + c
            ok = (ii >= 0) & (ii < n)
            tmp[:, ok] = tmp[:, ok] + cur[:, ii[ok]]
            cnt += ok
        tmp = tmp / cnt[None, :]
        out = np.zeros((n, n))
        cnt = np.zeros(n)
        for b in range(-r, r + 1):
            jj = idx + b
            ok = (jj >= 0) & (jj < n)
            out[ok, :] = out[ok, :] + tmp[jj[ok], :]
            cnt += ok
        cur = out / cnt[:, None]
    return cur.ravel()


def rank_unit(a):
    """Each value replaced by where it stands among the others, 0..1, with
    ties sharing a rank. Mirrors `rankUnit` in web/landform.js."""
    a = np.asarray(a, dtype=np.float64)
    if a.size < 2:
        return a
    vals, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    ends = np.cumsum(cnt) - 1
    starts = ends - cnt + 1
    return ((starts + ends) / 2 / (a.size - 1))[inv]


def quantile(values, q):
    """Nearest-rank quantile: element round(q*(len-1)) of the sorted values."""
    s = np.sort(np.asarray(values, dtype=np.float64))
    if not s.size:
        return 0.0
    i = min(s.size - 1, max(0, int(np.floor(q * (s.size - 1) + 0.5))))
    return s[i]


def _tie_hash(count):
    """The per-cell hash landform.js uses on featureless terrain.

    Reproduces JavaScript's number semantics exactly: the multiply happens
    in doubles (not Math.imul), then ToUint32. Only reached when every cell
    ties, so it only matters for a flat field, but there it decides the map.
    """
    def u32(x):
        # ToUint32 on an integral double: truncate, then wrap.
        return np.mod(np.trunc(x), 4294967296.0)

    def i32(x):
        x = u32(x)
        return np.where(x >= 2147483648.0, x - 4294967296.0, x)

    i = np.arange(count, dtype=np.float64)
    h = u32(i * 747796405.0 + 2891336453.0).astype(np.uint64)
    sh = ((h >> np.uint64(28)) + np.uint64(4)) & np.uint64(31)
    x = i32(((h >> sh) ^ h).astype(np.float64))
    h = u32(x * 277803737.0).astype(np.uint64)
    return u32(((h >> np.uint64(22)) ^ h).astype(np.float64)) / 4294967296.0


def cue_weights(altitude=0.4):
    """What each cue is worth at this altitude setting, for the class that
    collects material; the other class mirrors it with slope and ridge.
    Mirrors `cueWeights` in web/landform.js, which the panel prints."""
    a = min(1.0, max(0.0, altitude))
    names = {"drive": "drainage", "hollow": "hollows", "ridge": "ridges",
             "slope": "slope", "flat": "flat ground", "midpos": "mid-slope"}
    terms = RULE["positions"]["collected"]["terms"]
    return [(names.get(k, k), (1 - a) * w) for k, w in terms.items()] + \
        [("height", a)]


def classify(z, n, classes, spacing=1.0, radius=None, sediment=None,
             sharpness=2.0, altitude=0.4, coherence=2, balance=None,
             rules=None):
    """A class per cell, from the shape of the ground. Mirrors `classify`.

    `z` is the height sampled on an (n*over)^2 grid, flat, row-major -
    `sample_grid(..., over=4)` is what the viewer uses. Returns a dict with
    `cls` (int per cell), `strength` (0..1 decision margin), `maps` and
    `source` ('sediment' or 'drainage'), exactly as the JS returns, plus
    `lead`, the signed field the threshold is applied to (Python only).

    `rules`, if given, is a list of functions `f(maps) -> weights` over the
    normalised maps; the default is the viewer's two-class rule.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    m = int(round(np.sqrt(z.size)))
    sp = (spacing or 1) * n / m
    s = unit(downsample(slope(z, m, sp), m, n))
    t = unit(downsample(tpi(z, m, radius or max(2, m >> 4)), m, n))
    f = unit(downsample(np.log1p(flow(z, m)), m, n, "max"))
    e = unit(downsample(z, m, n))
    sed = (unit(downsample(np.asarray(sediment, np.float64).ravel(), m, n, "max"))
           if sediment is not None else None)
    drive = sed if sed is not None else f

    sharp = 2.0 if sharpness is None else sharpness
    alt = 0.4 if altitude is None else altitude
    maps = {"slope": s, "tpi": t, "flow": f, "elevation": e,
            "sediment": sed, "drive": drive}
    if rules is None:
        # From web/rule.json, the same file the viewer imports.
        def midband(v):
            return 1 - 2 * np.abs(v - 0.5)

        terms = {"drive": lambda M: M["drive"],
                 "hollow": lambda M: 1 - M["tpi"],
                 "ridge": lambda M: M["tpi"],
                 "slope": lambda M: M["slope"],
                 "flat": lambda M: 1 - M["slope"],
                 "midpos": lambda M: midband(M["tpi"])}
        heights = {"low": lambda M: 1 - M["elevation"],
                   "high": lambda M: M["elevation"],
                   "middle": lambda M: midband(M["elevation"])}

        def position(name):
            p = RULE["positions"][name]
            items = list(p["terms"].items())
            h = heights[p["height"]]

            def score(M):
                # Summed in the file's order, from zero, as the JS does.
                shape = 0
                for k, wgt in items:
                    shape = shape + wgt * terms[k](M)
                return (1 - alt) * shape + alt * h(M)
            return score

        named = RULE["classes"].get(str(classes)) or \
            RULE["classes"]["4"][:max(2, classes)]
        rules = [position(nm) for nm in named]

    w = []
    for k in range(classes):
        col = np.maximum(0.0, np.asarray(rules[k % len(rules)](maps), np.float64))
        # No exponent here; see web/landform.js. Sharpness is applied to
        # the margin below, where it can actually be seen.
        w.append(smooth(col, n, 1))

    # Three or more classes must be made comparable before they compete;
    # see web/landform.js.
    if classes > 2:
        w = [rank_unit(c) for c in w]

    coh = 2 if coherence is None else coherence
    bias = 0.0
    picked = None
    if isinstance(balance, (list, tuple, np.ndarray)) and classes > 2:
        # Cells handed out strongest preference first; see web/landform.js.
        for k in range(classes):
            w[k] = median(w[k], n, coh)
        share = np.maximum(0.0, np.asarray(balance, float))
        total = share.sum() or 1.0
        quota = np.zeros(classes, int)
        left = n * n
        for k in range(classes):
            quota[k] = left if k == classes - 1 else min(
                left, int(round(n * n * share[k] / total)))
            left -= quota[k]
        stack0 = np.stack(w)
        want = np.argmax(stack0, axis=0)
        part = np.sort(stack0, axis=0)
        conf = part[-1] - part[-2]
        order = np.lexsort((np.arange(n * n), -conf))
        picked = np.full(n * n, -1, dtype=np.int32)
        for i in order.tolist():
            k = int(want[i])
            if quota[k] > 0:
                picked[i] = k
                quota[k] -= 1
        for i in order.tolist():
            if picked[i] >= 0:
                continue
            room = np.flatnonzero(quota > 0)
            if room.size:
                k = int(room[np.argmax(stack0[room, i])])
                picked[i] = k
                quota[k] -= 1
            else:
                picked[i] = int(want[i])
    elif balance is not None and classes == 2:
        lead = median(w[0] - w[1], n, coh)
        if lead.max() - lead.min() < 1e-9:
            lead = _tie_hash(n * n)
        # Exactly the cells asked for, by rank, with the index as the
        # tie-break; see web/landform.js.
        order = np.lexsort((np.arange(n * n), -lead))
        take = int(round(balance * n * n))
        picked = np.ones(n * n, dtype=np.int32)
        picked[order[:take]] = 0
        hi_v = lead[order[take - 1]] if take > 0 else np.inf
        lo_v = lead[order[take]] if take < n * n else -np.inf
        bias = -(hi_v if take == 0 else lo_v if take == n * n
                 else (hi_v + lo_v) / 2)
        w[0] = lead
        w[1] = np.zeros(n * n)

    stack = np.stack(w)
    stack[0] = stack[0] + bias
    # argmax takes the first of equal values, as the JS strict '>' does.
    top = np.argmax(stack, axis=0).astype(np.int32)
    cls = picked if picked is not None else top
    part = np.sort(stack, axis=0)
    margin = part[-1] - part[-2] if classes > 1 else np.zeros(n * n)
    # Which class a cell would take instead; see web/landform.js.
    masked = stack.copy()
    masked[top, np.arange(n * n)] = -np.inf
    runner_up = np.argmax(masked, axis=0).astype(np.int32)
    moved = cls != top
    runner_up = np.where(moved, top, runner_up)
    margin = np.where(moved, 0.0, margin)

    # Sharpness lands here and only here: with a quantile threshold it
    # cannot move the boundary (a quantile is a rank, and a power leaves
    # ranks alone), so what it controls is the width of the undecided band
    # that gets drawn as a per-splat dissolve. See web/landform.js.
    scale = max(quantile(margin, 0.9), 1e-9)
    strength = np.minimum(1.0, np.power(margin / scale, 1 / max(sharp, 1e-6)))
    # Python-only extra: the signed field the decision thresholds, positive
    # where class 0 wins. A ranking of cells, so the rule can be scored
    # without committing to a threshold (scripts/validate_rule.py).
    lead = (stack[0] - stack[1:].max(axis=0)) if classes > 1 else np.zeros(n * n)
    return {"cls": cls, "strength": strength, "lead": lead,
            "runner_up": runner_up,
            "weights": cue_weights(alt),
            "maps": {"slope": s, "tpi": t, "flow": f, "elevation": e,
                     "sediment": sed},
            "source": "sediment" if sed is not None else "drainage"}