"""Selecting parts of a scene, and throwing away the parts that are noise.

Two jobs that turn out to be the same code.

Cleaning: captured scenes carry floaters (splats hanging in empty space,
supported by no real geometry) and a coarse background shell meant to be
viewed from inside. Both ruin a close view of the subject.

Patch extraction: GSWT (Section 3.2) builds tiles by choosing a region on
the ground plane and keeping every Gaussian whose projected position falls
inside it - height is ignored entirely. That is a crop with the vertical
axis left unbounded.
"""

import numpy as np


def crop_box(s, lo, hi, axes=(0, 1, 2)):
    """Keep splats whose centres lie inside an axis-aligned box.

    `axes` chooses which axes are tested. Leaving one out makes the box
    unbounded along it, which is what patch extraction wants: GSWT decides
    membership from the ground-plane position alone and ignores height.
    """
    lo = np.atleast_1d(np.asarray(lo, dtype=np.float32))
    hi = np.atleast_1d(np.asarray(hi, dtype=np.float32))
    axes = list(axes)
    if len(lo) != len(axes) or len(hi) != len(axes):
        raise ValueError("lo and hi must have one entry per axis in `axes`")

    p = s.xyz[:, axes]
    keep = np.all((p >= lo) & (p <= hi), axis=1)
    return s.subset(keep), keep


def crop_cylinder(s, centre, radius, up_axis=2, height=None):
    """Keep splats within `radius` of a vertical axis through `centre`.

    Better than a box for framing a single subject: a box crops the corners
    at a different distance than the sides, which shows up as straight cuts
    in the background.
    """
    centre = np.asarray(centre, dtype=np.float32).reshape(3)
    plane = [i for i in range(3) if i != up_axis]

    d = np.linalg.norm(s.xyz[:, plane] - centre[plane], axis=1)
    keep = d <= radius
    if height is not None:
        h = s.xyz[:, up_axis] - centre[up_axis]
        keep &= (h >= -height / 2.0) & (h <= height / 2.0)
    return s.subset(keep), keep


def remove_large(s, max_extent):
    """Drop splats whose longest axis exceeds `max_extent` world units.

    The background shell of a 360-degree capture is a small number of very
    large splats. They carry most of the screen area and none of the detail.
    """
    keep = s.scale.max(axis=1) <= max_extent
    return s.subset(keep), keep


def remove_floaters(s, k=8, std_ratio=2.0, weight_by_opacity=True):
    """Statistical outlier removal.

    For each splat, measure the mean distance to its k nearest neighbours.
    Splats sitting in empty space have no close neighbours, so that distance
    is large. Anything beyond `std_ratio` standard deviations above the mean
    is dropped.

    This is the standard point-cloud filter (Rusu et al. 2008), applied to
    Gaussian centres. It does not know about opacity or size, so nearly
    transparent splats are kept unless `weight_by_opacity` also removes them.
    """
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


def auto_clean(s, up_axis=2, radius_pct=60.0, max_extent_pct=99.0,
               floater_std=2.0, verbose=True):
    """A reasonable default cleanup for framing a captured subject.

    Three passes, cheapest first: crop to the dense middle, drop the
    background shell by size, then remove what is left floating.
    """
    n0 = len(s)
    plane = [i for i in range(3) if i != up_axis]

    centre = np.median(s.xyz, axis=0)
    d = np.linalg.norm(s.xyz[:, plane] - centre[plane], axis=1)
    radius = float(np.percentile(d, radius_pct))
    s, _ = crop_cylinder(s, centre, radius, up_axis=up_axis)
    n1 = len(s)

    max_extent = float(np.percentile(s.scale.max(axis=1), max_extent_pct))
    s, _ = remove_large(s, max_extent)
    n2 = len(s)

    s, _ = remove_floaters(s, std_ratio=floater_std)
    n3 = len(s)

    if verbose:
        print(f"  crop to r={radius:.2f}:      {n0:,} -> {n1:,} "
              f"({n1 / n0:.0%})")
        print(f"  drop splats > {max_extent:.3f}:  {n1:,} -> {n2:,} "
              f"({n2 / n1:.0%})")
        print(f"  remove floaters:        {n2:,} -> {n3:,} "
              f"({n3 / n2:.0%})")
        print(f"  kept {n3 / n0:.0%} of the scene")
    return s