"""Cutting patches out of a scene, turning and placing copies of them.

Also two CPU renders of a tiled layout, used by the seam experiment
(scripts/experiments/seam_baseline.py): `render_global` sorts every splat
together; `render_tiled` sorts each tile alone and composites tiles in
order, as GSWT does (3.4). Subtracting the two isolates the boundary
artifact - splats of neighbouring tiles that cannot interleave.
"""

import numpy as np

from .ply import Splats
from .render import rasterize_rgba, over, flatten
from .camera import Camera, project_perspective


class PlaneIndex:
    """Splats bucketed by ground-plane cell, built once per scene, so each
    candidate cut in a patch search looks only at nearby cells (on a
    ten-million-splat scene the naive way was ten billion comparisons).
    Same splats, same order, ~5x faster."""

    def __init__(self, s, up_axis=2, cell=0.5):
        self.plane = [i for i in range(3) if i != up_axis]
        xy = np.asarray(s.xyz[:, self.plane], dtype=np.float64)
        self.cell = float(cell)
        self.lo = xy.min(axis=0) if len(xy) else np.zeros(2)
        ij = np.floor((xy - self.lo) / self.cell).astype(np.int64)
        self.shape = (ij.max(axis=0) + 1) if len(xy) else np.array([1, 1])
        key = ij[:, 0] * self.shape[1] + ij[:, 1]
        self.order = np.argsort(key, kind="stable")
        sorted_key = key[self.order]
        n_cells = int(self.shape[0] * self.shape[1])
        self.start = np.searchsorted(sorted_key, np.arange(n_cells + 1))
        self.n = len(xy)

    def candidates(self, centre, size):
        """Indices of every splat in the cells a square touches, ascending."""
        half = size / 2.0
        c = np.asarray(centre, dtype=np.float64)
        a = np.floor((c - half - self.lo) / self.cell).astype(np.int64)
        b = np.floor((c + half - self.lo) / self.cell).astype(np.int64)
        a = np.maximum(a, 0)
        b = np.minimum(b, self.shape - 1)
        if (b < a).any():
            return np.zeros(0, dtype=np.int64)
        parts = []
        for i in range(int(a[0]), int(b[0]) + 1):
            row = i * self.shape[1]
            parts.append(self.order[self.start[row + a[1]]:
                                    self.start[row + b[1] + 1]])
        idx = np.concatenate(parts) if parts else np.zeros(0, np.int64)
        return np.sort(idx)


def extract_patch(s, centre, size, up_axis=2, recentre=True, index=None):
    """Cut a square patch out of the ground plane, by ground-plane position
    only, height unbounded (GSWT 3.2).

    `centre` is a 2-vector in the plane. `recentre` moves the patch to the
    origin. `index`, a PlaneIndex over `s`, makes the cut faster with an
    identical result.
    """
    plane = [i for i in range(3) if i != up_axis]
    centre = np.asarray(centre, dtype=np.float32).reshape(2)
    half = size / 2.0

    if index is not None and index.n == len(s):
        # Only the splats in nearby cells, then the exact test on those.
        cand = index.candidates(centre, size)
        d = s.xyz[cand][:, plane] - centre
        out = s.subset(cand[np.all((d >= -half) & (d <= half), axis=1)])
    else:
        d = s.xyz[:, plane] - centre
        keep = np.all((d >= -half) & (d <= half), axis=1)
        out = s.subset(keep)

    if recentre and len(out):
        offset = np.zeros(3, dtype=np.float32)
        offset[plane] = centre
        out = translate(out, -offset)
    return out


def translate(s, offset):
    """Move a scene. Only positions change - shape and orientation do not."""
    offset = np.asarray(offset, dtype=np.float32).reshape(3)
    return Splats(
        xyz=(s.xyz + offset).astype(np.float32),
        opacity=s.opacity, scale=s.scale, rot=s.rot,
        sh_dc=s.sh_dc, sh_rest=s.sh_rest, sh_degree=s.sh_degree,
    )


def merge(*scenes):
    """Concatenate scenes into one. No blending, no deduplication."""
    scenes = [s for s in scenes if len(s)]
    if not scenes:
        raise ValueError("nothing to merge")
    deg = min(s.sh_degree for s in scenes)
    scenes = [s.truncate_sh(deg) if s.sh_degree > deg else s for s in scenes]
    return Splats(
        xyz=np.concatenate([s.xyz for s in scenes]),
        opacity=np.concatenate([s.opacity for s in scenes]),
        scale=np.concatenate([s.scale for s in scenes]),
        rot=np.concatenate([s.rot for s in scenes]),
        sh_dc=np.concatenate([s.sh_dc for s in scenes]),
        sh_rest=np.concatenate([s.sh_rest for s in scenes]),
        sh_degree=deg,
    )


def grid(patch, nx, ny, size, up_axis=2, gap=0.0):
    """`nx` by `ny` copies of a patch laid edge to edge.

    Returns a list of scenes, one per cell, not a merged scene - the tiled
    renderer needs them kept apart.
    """
    plane = [i for i in range(3) if i != up_axis]
    pitch = size + gap
    tiles = []
    for j in range(ny):
        for i in range(nx):
            offset = np.zeros(3, dtype=np.float32)
            offset[plane[0]] = (i - (nx - 1) / 2.0) * pitch
            offset[plane[1]] = (j - (ny - 1) / 2.0) * pitch
            tiles.append(translate(patch, offset))
    return tiles


def render_global(cam, tiles, sh_degree=None, background=(0, 0, 0)):
    """Every splat sorted together. The reference."""
    scene = merge(*tiles) if isinstance(tiles, (list, tuple)) else tiles
    p = project_perspective(cam, scene, sh_degree=sh_degree)
    return flatten(rasterize_rgba(p), background), p


def render_tiled(cam, tiles, sh_degree=None, background=(0, 0, 0)):
    """Each tile sorted alone, tiles composited by centre depth (GSWT uses a
    topological sort; for a few tiles they agree). Splats in different
    tiles are never sorted against each other."""
    layers = []
    for t in tiles:
        if not len(t):
            continue
        centre = t.xyz.mean(axis=0)
        depth = float((centre - cam.position) @ cam.W[2])   # camera-space z
        p = project_perspective(cam, t, sh_degree=sh_degree)
        if p["kept"] == 0:
            continue
        layers.append((depth, rasterize_rgba(p), p["kept"]))

    if not layers:
        raise ValueError("no tiles produced any visible splats")

    layers.sort(key=lambda L: -L[0])                        # far to near
    acc = layers[0][1]
    for _, rgba, _ in layers[1:]:
        acc = over(acc, rgba)

    return flatten(acc, background), {"kept": sum(L[2] for L in layers),
                                      "layers": len(layers)}


def seam_camera(seam_point, distance, elevation_deg, azimuth_deg,
                up_axis=2, **kw):
    """A camera aimed at a tile boundary. `azimuth_deg` 0 looks along the
    seam (GSWT 3.4's worst case), 90 straight across it."""
    seam_point = np.asarray(seam_point, dtype=np.float32).reshape(3)
    plane = [i for i in range(3) if i != up_axis]
    az = np.radians(azimuth_deg + 90.0)      # 0 -> along +y, the seam axis
    el = np.radians(elevation_deg)

    offset = np.zeros(3, dtype=np.float32)
    offset[plane[0]] = np.cos(el) * np.cos(az)
    offset[plane[1]] = np.cos(el) * np.sin(az)
    offset[up_axis] = np.sin(el)

    up = np.zeros(3, dtype=np.float32)
    up[up_axis] = 1.0
    return Camera(seam_point + offset * distance, seam_point, up=up, **kw)

def rotate_patch(p, quarter_turns, up_axis=2):
    """A patch turned about the vertical by a multiple of 90 degrees.

    New edge colours from the same ground: turned a quarter, a patch reads
    as different ground. Turning a finished tile would break Wang matching;
    turning a patch before tiles are built does not. Positions and
    orientations turn; baked shadows turn with them (why it is optional).
    Higher-order SH is not rotated; .splat keeps only the constant term.
    """
    k = int(quarter_turns) % 4
    if k == 0:
        return p
    a, b = [i for i in range(3) if i != up_axis]
    out = p.subset(np.arange(len(p)))
    xyz = np.array(p.xyz, dtype=np.float64, copy=True)
    x, y = xyz[:, a].copy(), xyz[:, b].copy()
    for _ in range(k):
        x, y = -y, x
    xyz[:, a], xyz[:, b] = x, y
    out.xyz = xyz.astype(p.xyz.dtype)
    # Pre-multiply every orientation by the turn about the up axis.
    half = 0.5 * k * np.pi / 2.0
    axis = np.zeros(3)
    axis[up_axis] = 1.0
    # Right-handed turn from axis a towards axis b is about +up when
    # (a, b, up) is a cyclic order, and about -up otherwise.
    if (a, b, up_axis) not in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        axis = -axis
    w0, v0 = np.cos(half), axis * np.sin(half)
    q = np.asarray(p.rot, dtype=np.float64)
    w1, v1 = q[:, 0], q[:, 1:4]
    w = w0 * w1 - v1 @ v0
    v = w0 * v1 + w1[:, None] * v0 + np.cross(np.broadcast_to(v0, v1.shape), v1)
    out.rot = np.c_[w, v].astype(np.asarray(p.rot).dtype)
    return out


def missing_layer(kept, capture, size, up_axis=2, px=0.025, gap=0.1):
    """What a patch is missing against its capture, seen from above, as a
    layer of flat splats on the patch's own ground.

    Both are drawn straight down over the same square (`kept`, the patch;
    `capture`, every splat of the capture that reaches the square, in the
    same frame, the huge faint ones included). Where the capture covers
    more by `gap` or over, the gap is filled with the capture's own colour
    there, as much as is missing and no more: where the patch is already
    solid nothing is added. Each splat lies just under the lowest splats around it, so the
    layer follows the ground and cannot show above it."""
    from .graphcut import render_patch
    from .ply import SH_C0
    from scipy.ndimage import distance_transform_edt

    res = max(int(round(size / px)), 8)
    cell = size / res
    rgb_t, a_t = render_patch(capture, size, res, up_axis)
    _, a_k = render_patch(kept, size, res, up_axis)
    a_k = np.clip(a_k, 0.0, 0.999)
    want = np.clip((a_t - a_k) / (1.0 - a_k), 0.0, 1.0)
    # Only where the gap shows: under ground that is already nearly solid,
    # a layer would add splats and nothing visible.
    rows, cols = np.nonzero((a_t - a_k) > gap)
    if not len(rows):
        return kept.subset(np.zeros(0, dtype=np.int64))
    plane = [i for i in range(3) if i != up_axis]

    # Ground under each pixel: the low splats of the patch, 10 cm cells,
    # empty cells taking their nearest neighbour's.
    g = max(1, int(round(0.1 / cell)))
    n = -(-res // g)
    kc = np.clip(((kept.xyz[:, plane[0]] + size / 2) / cell).astype(int) // g, 0, n - 1)
    kr = np.clip(((size / 2 - kept.xyz[:, plane[1]]) / cell).astype(int) // g, 0, n - 1)
    low = np.full((n, n), np.nan)
    z = kept.xyz[:, up_axis]
    for key in np.unique(kr * n + kc):
        m = (kr * n + kc) == key
        low[key // n, key % n] = np.percentile(z[m], 10)
    if np.isnan(low).all():
        low[:] = 0.0
    idx = distance_transform_edt(np.isnan(low), return_distances=False, return_indices=True)
    low = low[tuple(idx)]

    xyz = np.zeros((len(rows), 3), dtype=np.float32)
    xyz[:, plane[0]] = -size / 2 + (cols + 0.5) * cell
    xyz[:, plane[1]] = size / 2 - (rows + 0.5) * cell
    xyz[:, up_axis] = low[rows // g, cols // g] - 0.003
    scale = np.full((len(rows), 3), 0.6 * cell, dtype=np.float32)
    scale[:, up_axis] = 0.002
    rest = kept.sh_rest.shape[1]
    layer = Splats(xyz=xyz, opacity=want[rows, cols].astype(np.float32), scale=scale,
                   rot=np.tile(np.float32([1, 0, 0, 0]), (len(rows), 1)),
                   sh_dc=((np.clip(rgb_t[rows, cols], 0, 1) - 0.5) / SH_C0).astype(np.float32),
                   sh_rest=np.zeros((len(rows), rest, 3), np.float32),
                   sh_degree=kept.sh_degree)
    # Neighbours overlap, so a splat's peak is not its pixel's cover: match
    # the layer's own cover to what is missing, in a few steps.
    for _ in range(4):
        _, a_l = render_patch(layer, size, res, up_axis)
        ratio = want[rows, cols] / np.maximum(a_l[rows, cols], 1e-3)
        layer.opacity = np.clip(layer.opacity * np.clip(ratio, 0.5, 2.0),
                                0.0, 0.99).astype(np.float32)
    return layer


def haze_splats(t, size, up_axis=2):
    """Large faint splats standing above the ground: footprint (the two
    largest axes' mean) over twice the tile's median, opacity under its
    median, and more than 2% of the tile above its ground (its lower
    quartile). Training leaves them as soft fill in a canopy; from above
    they sit under the detail, from low down in front of it. On the ground
    they are the ground's cover, and stay in the fill (sparing them there
    starved desert13's sand of hole-free patches)."""
    if not len(t):
        return np.zeros(0, bool)
    s = np.sort(np.asarray(t.scale, np.float64), axis=1)
    foot = np.sqrt(s[:, 1] * s[:, 2])
    op = np.asarray(t.opacity, np.float64)
    z = np.asarray(t.xyz[:, up_axis], np.float64)
    high = z - np.percentile(z, 25) > 0.02 * size
    return (foot > 2.0 * np.median(foot)) & (op < np.median(op)) & high


def fill_opacity(t, size, up_axis=2, target=0.98, max_boost=6.0, px=0.02, spare=None):
    """Make a tile's own splats opaque where its ground is see-through.

    Training only needs each photo pixel right, so the ground can come out
    half transparent with the capture's background behind it (false
    transparency, Noise Guided Splatting, NeurIPS 2025); cut out as a tile,
    the sky shows through. Each splat's opacity becomes 1 - (1 - a)^k, with
    k chosen from the tile's coverage at that spot so the layer reaches
    `target` from above. No splat is added or moved, so nothing new can
    appear from another angle; bare spots with no splat stay bare. `spare`:
    splats left as they are (haze_splats, export --fill-spare-haze): made
    opaque, haze read as smears from low down in the bush."""
    from .graphcut import render_patch
    from scipy.ndimage import gaussian_filter

    if not len(t):
        return t
    res = max(int(round(size / px)), 8)
    _, a = render_patch(t, size, res, up_axis)
    a = np.clip(gaussian_filter(a, 1.0), 0.0, 0.999)
    plane = [i for i in range(3) if i != up_axis]
    cell = size / res
    col = np.clip(((t.xyz[:, plane[0]] + size / 2.0) / cell).astype(int), 0, res - 1)
    row = np.clip(((size / 2.0 - t.xyz[:, plane[1]]) / cell).astype(int), 0, res - 1)
    here = a[row, col]
    k = np.ones(len(t))
    thin = (here > 0.02) & (here < target)
    k[thin] = np.log(1.0 - target) / np.log(1.0 - here[thin])
    k = np.clip(k, 1.0, max_boost)
    if spare is not None:
        k[np.asarray(spare, bool)] = 1.0
    out = t.subset(np.arange(len(t)))
    op = np.clip(t.opacity.astype(np.float64), 0.0, 0.999)
    out.opacity = (1.0 - (1.0 - op) ** k).astype(np.float32)
    return out
