"""Cutting patches out of a scene, placing copies of them, and rendering
the result two different ways.

The two ways are the whole point.

`render_global` sorts every splat in the scene together, which is what a
single-scene 3DGS renderer does. It is the best answer available under the
usual per-splat-depth approximation.

`render_tiled` sorts each tile independently and composites the tiles in
tile order. That is what GSWT does (Section 3.4): tiles are pre-sorted
once and never merged into a common buffer, because a tile set is
instanced too many times to sort globally at runtime. It is also where the
boundary artifact comes from - splats from adjacent tiles that should
interleave in depth cannot, because they live in different layers.

Rendering identical content both ways and subtracting isolates the
artifact from everything else.
"""

import numpy as np

from .ply import Splats
from .render import rasterize_rgba, over, flatten
from .camera import Camera, project_perspective


def extract_patch(s, centre, size, up_axis=2, recentre=True):
    """Cut a square patch out of the ground plane.

    Membership is decided from the ground-plane position only; height is
    unbounded. That is GSWT's patch rule (Section 3.2): they project to the
    plane, cut there, and keep every Gaussian whose projected position
    falls inside, ignoring depth entirely.

    `centre` is a 2-vector in the ground plane. With `recentre`, the patch
    comes back centred on the origin horizontally, which makes copies easy
    to place.
    """
    plane = [i for i in range(3) if i != up_axis]
    centre = np.asarray(centre, dtype=np.float32).reshape(2)
    half = size / 2.0

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
    """Concatenate scenes into one. No blending, no deduplication.

    This is the naive merge that Graph-GSReg characterises: overlapping
    regions end up with redundant density, and nothing reconciles the two
    sets of splats.
    """
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
    """Each tile sorted alone, tiles composited in tile order.

    Tile order here is by the depth of each tile's centre. GSWT does
    something more careful - a topological sort derived from the boundary
    normals between adjacent tiles - but for a small number of tiles the
    two agree, and the artifact under study does not depend on which is
    used. What matters is that splats in different tiles are never sorted
    against each other.
    """
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
    """A camera aimed at a point on a tile boundary.

    `azimuth_deg` is measured from the seam direction: 0 looks along the
    seam, 90 looks straight across it. GSWT (Section 3.4) reports the worst
    artifacts on boundaries aligned with the view direction, which is the
    0 case.
    """
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