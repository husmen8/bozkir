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
    """A packed tileset back as (manifest, Splats of every splat).

    The inverse of `pack` for reading: .splat is 32 bytes a splat -
    position and scale as floats, colour and opacity as bytes, rotation as
    bytes centred on 128.
    """
    import json
    from .ply import Splats, SH_C0
    m = json.loads(open(json_path).read())
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


def tile_atlas(m, s, res=64):
    """Every tile rendered straight down, north up, packed into one image.

    What far-away ground is drawn with: past the distance where a tile's
    splats are smaller than a pixel, sorting and drawing thousands of them
    buys nothing a texture cannot show, and a texture costs one lookup.
    Returns (rgba uint8 image, layout dict). Alpha is coverage.
    """
    from .graphcut import render_patch
    tiles = m["tiles"]
    cols = int(np.ceil(np.sqrt(len(tiles))))
    rows = int(np.ceil(len(tiles) / cols))
    img = np.zeros((rows * res, cols * res, 4), dtype=np.uint8)
    for k, t in enumerate(tiles):
        a, c = t["levels"][0]
        rgb, cov = render_patch(s.subset(np.arange(a, a + c)), m["size"],
                                res, m.get("up_axis", 2))
        r, q = divmod(k, cols)
        cell = np.dstack([np.clip(rgb, 0, 1), np.clip(cov, 0, 1)])
        img[r * res:(r + 1) * res, q * res:(q + 1) * res] = \
            np.round(cell * 255).astype(np.uint8)
    return img, {"res": res, "cols": cols, "rows": rows,
                 "count": len(tiles), "size": m["size"]}