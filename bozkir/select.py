"""Cropping a scene and removing noise: floaters (splats in empty space)
and the coarse background shell of a 360-degree capture.

Patch extraction is also a crop, but lives in bozkir/tile.py next to the
plane index that makes it fast.
"""

import numpy as np


def crop_box(s, lo, hi, axes=(0, 1, 2)):
    """Keep splats whose centres lie inside an axis-aligned box; axes left
    out of `axes` are unbounded."""
    lo = np.atleast_1d(np.asarray(lo, dtype=np.float32))
    hi = np.atleast_1d(np.asarray(hi, dtype=np.float32))
    axes = list(axes)
    if len(lo) != len(axes) or len(hi) != len(axes):
        raise ValueError("lo and hi must have one entry per axis in `axes`")

    p = s.xyz[:, axes]
    keep = np.all((p >= lo) & (p <= hi), axis=1)
    return s.subset(keep), keep


def crop_cylinder(s, centre, radius, up_axis=2, height=None):
    """Keep splats within `radius` of a vertical axis through `centre` (a
    box would crop the corners at a different distance than the sides)."""
    centre = np.asarray(centre, dtype=np.float32).reshape(3)
    plane = [i for i in range(3) if i != up_axis]

    d = np.linalg.norm(s.xyz[:, plane] - centre[plane], axis=1)
    keep = d <= radius
    if height is not None:
        h = s.xyz[:, up_axis] - centre[up_axis]
        keep &= (h >= -height / 2.0) & (h <= height / 2.0)
    return s.subset(keep), keep


def remove_large(s, max_extent):
    """Drop splats longer than `max_extent` world units: the background
    shell is a few huge splats with most of the area and no detail."""
    keep = s.scale.max(axis=1) <= max_extent
    return s.subset(keep), keep


def min_size(s, size):
    """Every splat at least `size` wide (metres, one standard deviation).

    Training blurs each splat by about half a photo pixel, so thin splats
    join into solid ground only at the photos' distance; closer, they part
    into needles with gaps (desert v1: 60% cover from above at 1.5 m, 88%
    with 1 cm). s' = sqrt(s^2 + size^2) is Mip-Splatting's 3D filter (Yu et
    al. 2024) applied after training."""
    out = s.subset(np.arange(len(s)))
    out.scale = np.sqrt(s.scale.astype(np.float64) ** 2 + size ** 2).astype(np.float32)
    return out


def _matrix_to_quat(R):
    """Rotation matrices (N, 3, 3) -> unit quaternions (N, 4), (w, x, y, z),
    the inverse of ply.quat_to_matrix (Shepperd's method: divide by the
    largest of the four candidates, so no case loses precision)."""
    R = np.asarray(R, dtype=np.float64)
    q = np.empty((len(R), 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    k = np.stack([tr, R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]], axis=1).argmax(axis=1)
    m = k == 0
    s = np.sqrt(1.0 + tr[m]) * 2.0
    q[m] = np.stack([0.25 * s, (R[m, 2, 1] - R[m, 1, 2]) / s,
                     (R[m, 0, 2] - R[m, 2, 0]) / s, (R[m, 1, 0] - R[m, 0, 1]) / s], 1)
    m = k == 1
    s = np.sqrt(1.0 + R[m, 0, 0] - R[m, 1, 1] - R[m, 2, 2]) * 2.0
    q[m] = np.stack([(R[m, 2, 1] - R[m, 1, 2]) / s, 0.25 * s,
                     (R[m, 0, 1] + R[m, 1, 0]) / s, (R[m, 0, 2] + R[m, 2, 0]) / s], 1)
    m = k == 2
    s = np.sqrt(1.0 + R[m, 1, 1] - R[m, 0, 0] - R[m, 2, 2]) * 2.0
    q[m] = np.stack([(R[m, 0, 2] - R[m, 2, 0]) / s, (R[m, 0, 1] + R[m, 1, 0]) / s,
                     0.25 * s, (R[m, 1, 2] + R[m, 2, 1]) / s], 1)
    m = k == 3
    s = np.sqrt(1.0 + R[m, 2, 2] - R[m, 0, 0] - R[m, 1, 1]) * 2.0
    q[m] = np.stack([(R[m, 1, 0] - R[m, 0, 1]) / s, (R[m, 0, 2] + R[m, 2, 0]) / s,
                     (R[m, 1, 2] + R[m, 2, 1]) / s, 0.25 * s], 1)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def cap_height(s, max_std, up_axis=2):
    """Flatten splats taller than `max_std` (metres, one standard deviation)
    instead of deleting them.

    Photos from above cannot see how tall a splat is, so some grow into
    spikes seen from low angles; deleting them also deletes what they draw
    from above. Sigma' = S Sigma S with S = diag(1, 1, k) squeezes only the
    up axis, so the footprint from straight above is unchanged."""
    up = up_axis
    cov = s.covariance().astype(np.float64)
    tall = cov[:, up, up] > max_std ** 2
    out = s.subset(np.arange(len(s)))
    if not tall.any():
        return out
    c = cov[tall]
    k = np.sqrt(max_std ** 2 / c[:, up, up])
    S = np.ones((len(c), 3))
    S[:, up] = k
    c = c * S[:, :, None] * S[:, None, :]
    vals, vecs = np.linalg.eigh(c)
    flip = np.linalg.det(vecs) < 0
    vecs[flip, :, 2] *= -1.0                    # a rotation, not a reflection
    scale = out.scale.copy()
    rot = out.rot.copy()
    scale[tall] = np.sqrt(np.maximum(vals, 1e-20)).astype(np.float32)
    rot[tall] = _matrix_to_quat(vecs).astype(np.float32)
    out.scale, out.rot = scale, rot
    return out


def remove_floaters(s, k=8, std_ratio=2.0, weight_by_opacity=True):
    """Statistical outlier removal (Rusu et al. 2008) on splat centres: drop
    splats whose mean distance to their k nearest neighbours is more than
    `std_ratio` standard deviations above average. Very opaque splats are
    kept when `weight_by_opacity`."""
    try:
        from scipy.spatial import cKDTree
    except ImportError as e:
        raise ImportError("remove_floaters needs scipy: pip install scipy") from e

    tree = cKDTree(s.xyz)
    d, _ = tree.query(s.xyz, k=k + 1, workers=-1)
    mean_d = d[:, 1:].mean(axis=1)                # column 0 is the splat itself

    thresh = mean_d.mean() + std_ratio * mean_d.std()
    keep = mean_d <= thresh
    if weight_by_opacity:
        keep |= s.opacity > 0.9                   # dense solid geometry stays
    return s.subset(keep), keep