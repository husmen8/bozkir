"""CPU reference renderer for 3D Gaussian scenes.

Slow and correct. The point is to have an image we trust, so that when a
fast GPU renderer disagrees with it we know which one is wrong.

Orthographic only, for now. Orthographic projection is linear, so the 2D
covariance is an exact submatrix of the 3D one - no Jacobian, no affine
approximation. Every error here is a real error, not a projection artifact.
"""

import numpy as np

# 3DGS dilates the 2D covariance so that no splat falls below roughly one
# pixel. Without it, sub-pixel splats alias badly and mostly disappear.
DILATION = 0.3

# Below this alpha a splat contributes less than one 8-bit level.
MIN_ALPHA = 1.0 / 255.0


def project_orthographic(s, view_axis=1, up_sign=-1, resolution=800,
                         bounds_pct=(1.0, 99.0), sh_degree=None, bounds=None):
    """Project a scene to screen space along one world axis.

    view_axis   which world axis the camera looks down (0=x, 1=y, 2=z)
    up_sign     +1 or -1: which end of that axis the camera sits on
    resolution  pixels along the longer screen dimension
    bounds_pct  percentile range used for framing, to ignore floaters
    bounds      ((lo_a, lo_b), (hi_a, hi_b)) to frame explicitly instead.
                Needed when several scenes must land on the same pixel grid.
    sh_degree   evaluate colour at this SH degree (None = whatever the scene has)

    Returns a dict of per-splat screen-space quantities, already culled to
    those that can affect the image.
    """
    ax = [i for i in range(3) if i != view_axis]        # the two screen axes
    scene = s.truncate_sh(sh_degree) if sh_degree is not None else s

    if bounds is not None:
        lo = np.asarray(bounds[0], dtype=np.float64)
        hi = np.asarray(bounds[1], dtype=np.float64)
    else:
        lo = np.percentile(s.xyz[:, ax], bounds_pct[0], axis=0)
        hi = np.percentile(s.xyz[:, ax], bounds_pct[1], axis=0)
    span = np.maximum(hi - lo, 1e-6)

    # Square pixels: one scale for both axes, set by the longer side.
    px_per_unit = resolution / span.max()
    W = int(np.ceil(span[0] * px_per_unit))
    H = int(np.ceil(span[1] * px_per_unit))

    mean2d = (s.xyz[:, ax] - lo) * px_per_unit
    mean2d[:, 1] = H - mean2d[:, 1]                     # screen y points down

    # Orthographic: the 2D covariance is the 2x2 submatrix of the 3D one,
    # scaled to pixels. This is exact.
    cov3 = s.covariance()
    cov2d = cov3[:, ax][:, :, ax] * (px_per_unit ** 2)
    cov2d[:, 0, 0] += DILATION
    cov2d[:, 1, 1] += DILATION

    depth = s.xyz[:, view_axis] * up_sign               # larger = nearer

    view = np.zeros(3, dtype=np.float32)
    view[view_axis] = up_sign
    colour = scene.rgb(view)

    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2
    # 3 sigma covers 99.7% of the Gaussian; beyond it alpha is negligible.
    radius = 3.0 * np.sqrt(np.maximum(
        0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
        + np.sqrt(np.maximum(
            0.25 * (cov2d[:, 0, 0] - cov2d[:, 1, 1]) ** 2 + cov2d[:, 0, 1] ** 2,
            0.0)),
        1e-6))

    keep = (
        (det > 1e-9)
        & (s.opacity > MIN_ALPHA)
        & (mean2d[:, 0] + radius >= 0) & (mean2d[:, 0] - radius < W)
        & (mean2d[:, 1] + radius >= 0) & (mean2d[:, 1] - radius < H)
    )

    return {
        "mean2d": mean2d[keep], "cov2d": cov2d[keep], "depth": depth[keep],
        "colour": colour[keep], "opacity": s.opacity[keep],
        "radius": radius[keep], "W": W, "H": H, "kept": int(keep.sum()),
        "total": len(s), "px_per_unit": float(px_per_unit),
    }


def rasterize(p, background=(1.0, 1.0, 1.0), max_splats=None, progress=None):
    """Blend projected splats back-to-front into an RGB image.

    Back-to-front 'over' compositing:  dst = src*a + dst*(1-a)
    3DGS goes front-to-back so it can stop early; back-to-front is the
    same result and much easier to read.
    """
    W, H = p["W"], p["H"]
    canvas = np.empty((H, W, 3), dtype=np.float32)
    canvas[:] = background

    order = np.argsort(p["depth"])                      # far to near
    if max_splats is not None and len(order) > max_splats:
        # Keep the most visible, then restore depth order among them.
        area = np.pi * p["radius"] ** 2
        ink = area * p["opacity"]
        best = np.argpartition(ink, -max_splats)[-max_splats:]
        order = order[np.isin(order, best)]

    mean, cov, col, op, rad = (p["mean2d"], p["cov2d"], p["colour"],
                               p["opacity"], p["radius"])

    # Inverse of each 2x2 covariance (the "conic"), computed in bulk.
    det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] ** 2
    inv = np.empty((len(cov), 3), dtype=np.float32)     # a, b, c
    inv[:, 0] = cov[:, 1, 1] / det
    inv[:, 1] = -cov[:, 0, 1] / det
    inv[:, 2] = cov[:, 0, 0] / det

    n = len(order)
    for count, i in enumerate(order):
        r = rad[i]
        x0 = max(int(mean[i, 0] - r), 0)
        x1 = min(int(mean[i, 0] + r) + 1, W)
        y0 = max(int(mean[i, 1] - r), 0)
        y1 = min(int(mean[i, 1] + r) + 1, H)
        if x1 <= x0 or y1 <= y0:
            continue

        # Pixel i covers [i, i+1), so its centre is at i + 0.5. Sampling at
        # the corner instead shifts every splat half a pixel, which is
        # invisible in one image and very visible where two tiles meet.
        dx = np.arange(x0, x1, dtype=np.float32) + 0.5 - mean[i, 0]
        dy = np.arange(y0, y1, dtype=np.float32) + 0.5 - mean[i, 1]
        DX = dx[None, :]
        DY = dy[:, None]

        a, b, c = inv[i]
        power = -0.5 * (a * DX * DX + 2.0 * b * DX * DY + c * DY * DY)
        alpha = op[i] * np.exp(np.minimum(power, 0.0))

        m = alpha > MIN_ALPHA
        if not m.any():
            continue
        A = np.where(m, alpha, 0.0)[..., None]
        tile = canvas[y0:y1, x0:x1]
        canvas[y0:y1, x0:x1] = col[i] * A + tile * (1.0 - A)

        if progress is not None and count % 20000 == 0:
            progress(count, n)

    return np.clip(canvas, 0.0, 1.0)


def render_orthographic(s, **kw):
    """Convenience: project and rasterize in one call."""
    ras_kw = {k: kw.pop(k) for k in ("background", "max_splats", "progress")
              if k in kw}
    p = project_orthographic(s, **kw)
    return rasterize(p, **ras_kw), p


def rasterize_rgba(p, max_splats=None):
    """Blend into a premultiplied RGBA buffer over transparency.

    Same maths as `rasterize`, but the result can be composited with other
    buffers afterwards. That is what makes tile-by-tile rendering possible:
    each tile becomes a layer, and the layers are combined in tile order
    rather than every splat being sorted together.

    Premultiplied means the stored colour is already scaled by alpha, so
    'over' is a plain lerp with no division anywhere.
    """
    W, H = p["W"], p["H"]
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    if len(p["depth"]) == 0:
        return rgba

    order = np.argsort(p["depth"])                      # far to near
    if max_splats is not None and len(order) > max_splats:
        ink = np.pi * p["radius"] ** 2 * p["opacity"]
        best = np.argpartition(ink, -max_splats)[-max_splats:]
        order = order[np.isin(order, best)]

    mean, cov, col, op, rad = (p["mean2d"], p["cov2d"], p["colour"],
                               p["opacity"], p["radius"])
    det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] ** 2
    inv = np.empty((len(cov), 3), dtype=np.float32)
    inv[:, 0] = cov[:, 1, 1] / det
    inv[:, 1] = -cov[:, 0, 1] / det
    inv[:, 2] = cov[:, 0, 0] / det

    for i in order:
        r = rad[i]
        x0 = max(int(mean[i, 0] - r), 0)
        x1 = min(int(mean[i, 0] + r) + 1, W)
        y0 = max(int(mean[i, 1] - r), 0)
        y1 = min(int(mean[i, 1] + r) + 1, H)
        if x1 <= x0 or y1 <= y0:
            continue

        dx = np.arange(x0, x1, dtype=np.float32) + 0.5 - mean[i, 0]
        dy = np.arange(y0, y1, dtype=np.float32) + 0.5 - mean[i, 1]
        a, b, c = inv[i]
        DX, DY = dx[None, :], dy[:, None]
        power = -0.5 * (a * DX * DX + 2.0 * b * DX * DY + c * DY * DY)
        alpha = op[i] * np.exp(np.minimum(power, 0.0))
        alpha = np.where(alpha > MIN_ALPHA, alpha, 0.0)[..., None]
        if not alpha.any():
            continue

        dst = rgba[y0:y1, x0:x1]
        dst[..., :3] = col[i] * alpha + dst[..., :3] * (1.0 - alpha)
        dst[..., 3:] = alpha + dst[..., 3:] * (1.0 - alpha)

    return rgba


def over(dst, src):
    """Composite premultiplied `src` over premultiplied `dst`. Src is nearer."""
    a = src[..., 3:]
    out = np.empty_like(dst)
    out[..., :3] = src[..., :3] + dst[..., :3] * (1.0 - a)
    out[..., 3:] = src[..., 3:] + dst[..., 3:] * (1.0 - a)
    return out


def flatten(rgba, background=(0.0, 0.0, 0.0)):
    """Premultiplied RGBA over an opaque background -> RGB."""
    bg = np.asarray(background, dtype=np.float32)
    a = rgba[..., 3:]
    return np.clip(rgba[..., :3] + bg * (1.0 - a), 0.0, 1.0)