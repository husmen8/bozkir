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