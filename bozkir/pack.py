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