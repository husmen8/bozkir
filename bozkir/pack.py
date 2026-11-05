"""Packing a scene into the compact .splat binary the browser reads.

    bytes  0..11   position   3 x float32
    bytes 12..23   scale      3 x float32   (world units, already exp'd)
    bytes 24..27   colour     4 x uint8     r, g, b, opacity
    bytes 28..31   rotation   4 x uint8     w, x, y, z as round(q*128 + 128)

That is antimatter15's .splat layout, so files written here should also
open in other .splat viewers. The compact layout (stride 20, the manifest
says which) keeps colour and rotation and stores position and scale in
16 bits each:

    bytes  0..5    position   3 x uint16    (p + POS_RANGE) / 2 POS_RANGE, 0.12 mm steps
    bytes  6..11   scale      3 x uint16    log2 between LOG2_SCALE, 0.036% steps
    bytes 12..15   colour     4 x uint8
    bytes 16..19   rotation   4 x uint8
"""

import numpy as np

from .ply import SH_C0

STRIDE = 32
COMPACT_STRIDE = 20
POS_RANGE = 4.0                 # metres either side of the tile's origin
LOG2_SCALE = (-30.0, 4.0)       # 1 nanometre to 16 m


def pack(s, sort=False, stride=STRIDE):
    """Splats -> a flat uint8 buffer, `stride` bytes each (32, or 20 compact)."""
    if stride == COMPACT_STRIDE:
        return _pack_compact(s, sort)
    n = len(s)
    buf = np.zeros((n, STRIDE), dtype=np.uint8)

    order = np.arange(n)
    if sort:
        # Largest and most opaque first. Sorting once here means a viewer
        # that streams the file shows the important splats early.
        order = np.argsort(-(s.scale.max(axis=1) * s.opacity))

    xyz = np.ascontiguousarray(s.xyz[order], dtype=np.float32)
    scale = np.ascontiguousarray(s.scale[order], dtype=np.float32)
    buf[:, 0:12] = xyz.view(np.uint8).reshape(n, 12)
    buf[:, 12:24] = scale.view(np.uint8).reshape(n, 12)

    rgb = np.clip(SH_C0 * s.sh_dc[order] + 0.5, 0.0, 1.0)
    buf[:, 24:27] = np.round(rgb * 255.0).astype(np.uint8)
    buf[:, 27] = np.round(np.clip(s.opacity[order], 0, 1) * 255.0).astype(np.uint8)

    q = s.rot[order]
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
    buf[:, 28:32] = np.clip(np.round(q * 128.0 + 128.0), 0, 255).astype(np.uint8)

    return buf.reshape(-1)


def _pack_compact(s, sort=False):
    n = len(s)
    order = np.argsort(-(s.scale.max(axis=1) * s.opacity)) if sort else np.arange(n)
    xyz = np.asarray(s.xyz[order], dtype=np.float64)
    if n and np.abs(xyz).max() > POS_RANGE:
        raise ValueError(f"a splat lies {np.abs(xyz).max():.2f} m out, past the compact "
                         f"layout's {POS_RANGE} m; export without --compact")
    buf = np.zeros((n, COMPACT_STRIDE), dtype=np.uint8)
    pq = np.round((xyz + POS_RANGE) / (2 * POS_RANGE) * 65535).astype("<u2")
    lo, hi = LOG2_SCALE
    ls = np.clip(np.log2(np.maximum(np.asarray(s.scale[order], np.float64), 2.0 ** lo)), lo, hi)
    sq = np.round((ls - lo) / (hi - lo) * 65535).astype("<u2")
    buf[:, 0:6] = pq.view(np.uint8).reshape(n, 6)
    buf[:, 6:12] = sq.view(np.uint8).reshape(n, 6)
    rgb = np.clip(SH_C0 * s.sh_dc[order] + 0.5, 0.0, 1.0)
    buf[:, 12:15] = np.round(rgb * 255.0).astype(np.uint8)
    buf[:, 15] = np.round(np.clip(s.opacity[order], 0, 1) * 255.0).astype(np.uint8)
    q = s.rot[order]
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
    buf[:, 16:20] = np.clip(np.round(q * 128.0 + 128.0), 0, 255).astype(np.uint8)
    return buf.reshape(-1)

def read_splat(splat_path, json_path):
    """A packed tileset back as (manifest, Splats of every splat); the
    inverse of `pack`."""
    import json
    from .ply import Splats, SH_C0
    with open(json_path) as f:
        m = json.load(f)
    raw = np.fromfile(splat_path, dtype=np.uint8)
    stride = int(m.get("stride", STRIDE))
    n = len(raw) // stride
    rec = raw[:n * stride].reshape(n, stride)
    if stride == COMPACT_STRIDE:
        lo, hi = LOG2_SCALE
        p = rec[:, 0:6].copy().view("<u2").reshape(n, 3).astype(np.float64)
        sc = rec[:, 6:12].copy().view("<u2").reshape(n, 3).astype(np.float64)
        f = np.c_[p / 65535 * 2 * POS_RANGE - POS_RANGE, 2.0 ** (lo + sc / 65535 * (hi - lo))]
        rgba = rec[:, 12:16].astype(np.float64) / 255
        q = (rec[:, 16:20].astype(np.float64) - 128) / 128
    else:
        f = rec[:, :24].copy().view(np.float32).reshape(n, 6)
        rgba = rec[:, 24:28].astype(np.float64) / 255
        q = (rec[:, 28:32].astype(np.float64) - 128) / 128
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
    s = Splats(xyz=f[:, :3].astype(np.float64), opacity=rgba[:, 3],
               scale=f[:, 3:6].astype(np.float64), rot=q,
               sh_dc=(rgba[:, :3] - 0.5) / SH_C0,
               sh_rest=np.zeros((n, 0, 3)), sh_degree=0)
    return m, s


def seen_colour(p, up_axis=2, elevation=20.0, res=48):
    """A patch's mean colour as the splats show it from `elevation`
    degrees, over four headings: each splat counts by how much of it is
    seen, its opacity times what lets light through in front of it along
    the view (splat centres binned into res x res pixels). From straight
    above a bush shows its top and the sand between; from low down mostly
    bush. Vectorised: the full tile in milliseconds."""
    if not len(p):
        return None
    plane = [i for i in range(3) if i != up_axis]
    xyz = p.xyz[:, plane + [up_axis]].astype(np.float64)
    a = np.clip(np.asarray(p.opacity, np.float64), 0.0, 0.999)
    rgb = np.asarray(p.base_rgb, np.float64)
    e = np.radians(elevation)
    acc, total = np.zeros(3), 0.0
    for k in range(4):
        h = k * np.pi / 2.0
        x = np.cos(h) * xyz[:, 0] + np.sin(h) * xyz[:, 1]
        y = -np.sin(h) * xyz[:, 0] + np.cos(h) * xyz[:, 1]
        # Looking along +y from low down: screen up mixes height and depth.
        sx, sy = x, y * np.sin(e) + xyz[:, 2] * np.cos(e)
        depth = y * np.cos(e) - xyz[:, 2] * np.sin(e)        # larger is further
        span = max(np.ptp(sx), np.ptp(sy), 1e-9)
        pix = (np.floor((sx - sx.min()) / span * (res - 1e-6)) * res
               + np.floor((sy - sy.min()) / span * (res - 1e-6))).astype(np.int64)
        order = np.lexsort((depth, pix))
        la = np.log1p(-a[order])
        cs = np.cumsum(la)
        first = np.r_[0, np.flatnonzero(np.diff(pix[order])) + 1]
        start = np.repeat(first, np.diff(np.r_[first, len(order)]))
        through = np.exp(cs - la - (cs[start] - la[start]))   # light past those in front
        w = a[order] * through
        acc += (rgb[order] * w[:, None]).sum(axis=0)
        total += float(w.sum())
    return acc / total if total > 0 else None


def tile_atlas(m, s, res=64, elevation=20.0):
    """Every tile rendered straight down, north up, in one image: the far
    field's texture (past a few dozen tiles a splat is under a pixel); with
    frames, the centres after the tiles. Each entry lies over its own mean
    colour where little covers it: a faint or bare spot showed black, or
    one stray splat's colour, in the far field. Returns (rgb image, opaque,
    as uint8 RGBA; layout). The layout's `seen` is each entry's colour from
    `elevation` degrees (seen_colour), what the far field is matched to."""
    from .graphcut import render_patch
    entries = list(m["tiles"]) + list(m.get("centres", []))
    cols = int(np.ceil(np.sqrt(len(entries))))
    rows = int(np.ceil(len(entries) / cols))
    size, up = m["size"], m.get("up_axis", 2)
    img = np.zeros((rows * res, cols * res, 4), dtype=np.uint8)
    seen = []
    for k, t in enumerate(entries):
        a, c = t["levels"][0]
        p = s.subset(np.arange(a, a + c))
        rgb, cov = render_patch(p, size, res, up)
        w = np.clip(cov, 0, 1)[..., None]
        mean = ((rgb * w).reshape(-1, 3).sum(axis=0) / max(float(w.sum()), 1e-9)
                if w.sum() > 0 else np.full(3, 0.5))
        r, q = divmod(k, cols)
        cell = np.clip(rgb * w + mean * (1.0 - w), 0, 1)
        img[r * res:(r + 1) * res, q * res:(q + 1) * res, :3] = \
            np.round(cell * 255).astype(np.uint8)
        v = seen_colour(p, up, elevation)
        seen.append([round(float(x), 4) for x in (mean if v is None else v)])
    img[..., 3] = 255
    return img, {"res": res, "cols": cols, "rows": rows,
                 "count": len(entries), "tiles": len(m["tiles"]),
                 "size": size, "elevation": elevation, "seen": seen}


def merge_level(t, target):
    """About `target` splats from `t`, each the merge of the splats sharing a
    voxel (edge found by bisection on the count). A merged Gaussian keeps its
    members' weight w = opacity x area (two largest axes): the w-weighted
    centre and colour, and the covariance sum w (Sigma_i + d_i d_i^T) / sum w,
    so it covers what they covered. Opacity is the chance light passes them
    all across the new footprint, 1 - prod (1 - a_i)^(A_i / A). Dropping
    three in four splats and growing the rest instead turned fine captures,
    whose thin splats only work stacked, into needles at distance."""
    from .ply import Splats
    from .select import _matrix_to_quat
    n = len(t)
    if target >= n or n < 2:
        return t
    xyz = t.xyz.astype(np.float64)
    lo_corner = xyz.min(axis=0)
    span = float(np.ptp(xyz, axis=0).max()) or 1.0

    def voxels(edge):
        key = np.floor((xyz - lo_corner) / edge).astype(np.int64)
        return np.unique(key, axis=0, return_inverse=True)[1].ravel()

    lo, hi = span * 1e-5, span * 2.0           # one voxel above the span holds all
    for _ in range(30):                        # smallest edge giving <= target voxels
        mid = np.sqrt(lo * hi)
        if voxels(mid).max() + 1 > target:
            lo = mid
        else:
            hi = mid
    inv = voxels(hi)
    m = int(inv.max()) + 1

    sc = np.sort(t.scale.astype(np.float64), axis=1)
    area = sc[:, 2] * sc[:, 1]
    a = np.clip(t.opacity.astype(np.float64), 0.0, 0.999)
    w = a * area + 1e-30
    W = np.bincount(inv, w, m)
    mu = np.stack([np.bincount(inv, w * xyz[:, k], m) for k in range(3)], 1) / W[:, None]
    d = xyz - mu[inv]
    cov = t.covariance().astype(np.float64) + d[:, :, None] * d[:, None, :]
    C = np.zeros((m, 9))
    np.add.at(C, inv, (w[:, None] * cov.reshape(n, 9)))
    C = (C / W[:, None]).reshape(m, 3, 3)
    evals, evecs = np.linalg.eigh(C)
    evecs[np.linalg.det(evecs) < 0, :, 0] *= -1   # proper rotations only
    scale = np.sqrt(np.maximum(evals, (1e-6 * span) ** 2))
    new_sc = np.sort(scale, axis=1)
    new_area = new_sc[:, 2] * new_sc[:, 1]
    # light through all members: sum of (A_i / A) ln(1 - a_i), per merged splat
    passed = np.bincount(inv, area * np.log1p(-a), m) / new_area
    opacity = np.clip(-np.expm1(passed), 0.0, 0.999)
    dc = np.stack([np.bincount(inv, w * t.sh_dc[:, k], m) for k in range(3)], 1) / W[:, None]
    k = t.sh_rest.shape[1]
    rest = np.zeros((m, k, 3), np.float32)
    if k:
        flat = np.zeros((m, k * 3))
        np.add.at(flat, inv, w[:, None] * t.sh_rest.reshape(n, k * 3))
        rest = (flat / W[:, None]).reshape(m, k, 3).astype(np.float32)
    return Splats(xyz=mu.astype(np.float32), opacity=opacity.astype(np.float32),
                  scale=scale.astype(np.float32), rot=_matrix_to_quat(evecs).astype(np.float32),
                  sh_dc=dc.astype(np.float32), sh_rest=rest, sh_degree=t.sh_degree)


def tile_levels(t, lod, size, up_axis=2, compensate=True, tone=False, report=None, merge=False):
    """A tile's levels of detail, finest first: each keeps a quarter of the
    one above, spread over the tile (GSWT retrains each level instead). With
    `compensate` the survivors grow to cover the same total area; area goes
    as size squared, hence the square root. With `merge` each level is
    merged from the one before (merge_level) instead, and nothing grows.
    With `tone` each level's colour
    is matched to the full tile's as seen from low down (seen_colour): the
    survivors are the largest splats, not the ones seen, and a bush's
    levels came out paler, so distant ground was too. `report`, a list,
    gets each level's brightness change before matching (seen, 0..1)."""
    from .patches import part_ink, stratified_keep
    from .ply import SH_C0
    want = seen_colour(t, up_axis) if tone and len(t) else None
    out = []
    prev = t
    for lvl in range(max(lod, 1)):
        budget = max(len(t) // (4 ** lvl), 64)
        if budget >= len(t):
            out.append(t)
            continue
        if merge:
            part = prev = merge_level(prev, budget)
        else:
            ink = part_ink(t)
            keep = stratified_keep(t, budget, size, up_axis)
            part = t.subset(keep)
        if compensate and not merge:
            grow = float(np.sqrt(ink.sum() / max(ink[keep].sum(), 1e-12)))
            grow = min(grow, 4.0)      # a cap, or level 4 turns to soup
            part = part.subset(np.arange(len(part)))
            part.scale = (part.scale * grow).astype(np.float32)
        if want is not None and lvl > 0:
            got = seen_colour(part, up_axis)
            if got is not None and report is not None:
                report.append(float(np.mean(got - want)))
            if got is not None:
                gain = np.clip(want / np.maximum(got, 1e-4), 0.8, 1.25)
                part = part.subset(np.arange(len(part)))
                part.sh_dc = ((np.clip(part.base_rgb * gain, 0, 1) - 0.5)
                              / SH_C0).astype(np.float32)
        out.append(part)
    return out
