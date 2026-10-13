"""Packing a scene into the compact .splat binary the browser reads.

    bytes  0..11   position   3 x float32
    bytes 12..23   scale      3 x float32   (world units, already exp'd)
    bytes 24..27   colour     4 x uint8     r, g, b, opacity
    bytes 28..31   rotation   4 x uint8     w, x, y, z as round(q*128 + 128)

That is antimatter15's .splat layout, so files written here should also
open in other .splat viewers.
"""

import numpy as np

from .ply import SH_C0

STRIDE = 32


def pack(s, sort=False):
    """Splats -> a flat uint8 buffer, 32 bytes each."""
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

def read_splat(splat_path, json_path):
    """A packed tileset back as (manifest, Splats of every splat); the
    inverse of `pack`."""
    import json
    from .ply import Splats, SH_C0
    with open(json_path) as f:
        m = json.load(f)
    raw = np.fromfile(splat_path, dtype=np.uint8)
    n = len(raw) // STRIDE
    rec = raw[:n * STRIDE].reshape(n, STRIDE)
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


def tile_levels(t, lod, size, up_axis=2, compensate=True, tone=False, report=None):
    """A tile's levels of detail, finest first: each keeps a quarter of the
    one above, spread over the tile (GSWT retrains each level instead). With
    `compensate` the survivors grow to cover the same total area; area goes
    as size squared, hence the square root. With `tone` each level's colour
    is matched to the full tile's as seen from low down (seen_colour): the
    survivors are the largest splats, not the ones seen, and a bush's
    levels came out paler, so distant ground was too. `report`, a list,
    gets each level's brightness change before matching (seen, 0..1)."""
    from .patches import part_ink, stratified_keep
    from .ply import SH_C0
    want = seen_colour(t, up_axis) if tone and len(t) else None
    out = []
    for lvl in range(max(lod, 1)):
        budget = max(len(t) // (4 ** lvl), 64)
        if budget >= len(t):
            out.append(t)
            continue
        ink = part_ink(t)
        keep = stratified_keep(t, budget, size, up_axis)
        part = t.subset(keep)
        if compensate:
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
