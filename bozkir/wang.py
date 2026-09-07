"""Building Wang tiles out of exemplar patches, and laying them out.

The construction is Cohen et al. 2003, "Wang Tiles for Image and Texture
Generation", applied to Gaussians instead of pixels. GSWT builds on the
same idea.

The trick is worth stating plainly. Take a square tile and cut it along
both diagonals into four triangles, one touching each edge. Fill the
triangle touching the north edge from the patch assigned to that edge's
colour, the east triangle from the east colour's patch, and so on.

Because the whole north edge lies inside the north triangle, every tile
sharing a north colour has an identical north edge - not similar, identical,
the same Gaussians in the same places. Two tiles laid side by side with
matching colours therefore agree exactly along their shared boundary. The
match is a property of the construction, not something optimised for
afterwards.

What the construction does not fix is the diagonals, where two different
patches meet inside the tile. Cohen finds a least-visible path there with
graph cut; GSWT (Section 3.2) does the same on orthographic renders and
lifts the cut back into 3D. Here the diagonals are feathered instead, which
is cruder and much simpler, and leaves the graph cut as an obvious upgrade.
"""

import numpy as np

from .ply import Splats
from .tile import merge, translate

# Corners are where all four triangles meet, so a feathered tile is not
# exactly edge-matched within this fraction of the tile size of a corner.
CORNER_NOTE = "edges match exactly except within `blend` of each corner"


def region_weights(xy, size, blend=0.0):
    """Membership of each of the four triangles, as weights summing to one.

    `xy` is (N, 2) in tile-local coordinates, the tile spanning
    [-size/2, size/2] on both axes. Columns of the result are north, east,
    south, west.

    With `blend` at zero this is a hard assignment: exactly one weight is 1.
    Above zero the diagonals become a band of width `blend` where two
    patches mix, which hides the cut at the cost of blurring it.
    """
    u = xy[:, 0] / size
    v = xy[:, 1] / size
    au, av = np.abs(u), np.abs(v)

    # Signed distance to each triangle, positive inside, in tile units.
    d = np.stack([v - au, u - av, -v - au, -u - av], axis=1)

    if blend <= 0.0:
        w = np.zeros_like(d)
        w[np.arange(len(d)), np.argmax(d, axis=1)] = 1.0
        return w

    b = blend / size
    w = np.clip(d / b + 0.5, 0.0, 1.0)
    total = w.sum(axis=1, keepdims=True)
    np.maximum(total, 1e-9, out=total)
    return w / total


def build_tile(patches, size, blend=0.0, up_axis=2, min_weight=0.02,
               labels=None):
    """Assemble one tile from four patches, one per edge.

    `patches` is (north, east, south, west). Each is a Splats already
    centred on the origin in the ground plane, and each contributes the
    Gaussians falling in its own region.

    Without `labels` the regions are the four triangles, feathered across
    the diagonals by `blend`. Feathering blurs the join; it cannot hide a
    change of material.

    With `labels` - a region image from bozkir.graphcut - the boundary has
    already been placed where the patches agree, so the assignment is a
    hard lookup and needs no feathering.
    """
    plane = [i for i in range(3) if i != up_axis]
    parts = []

    if labels is not None:
        from .graphcut import label_at
        for region, p in enumerate(patches):
            if not len(p):
                continue
            keep = label_at(p, labels, size, up_axis) == region
            if keep.any():
                parts.append(p.subset(keep))
    else:
        for region, p in enumerate(patches):
            if not len(p):
                continue
            w = region_weights(p.xyz[:, plane], size, blend)[:, region]
            keep = w > min_weight
            if not keep.any():
                continue
            part = p.subset(keep)
            part.opacity = (part.opacity * w[keep]).astype(np.float32)
            parts.append(part)

    if not parts:
        raise ValueError("no patch contributed any Gaussians")
    return merge(*parts)


def build_tile_set(h_patches, v_patches, size, blend=0.0, up_axis=2,
                   cut=False, resolution=160, band=0.14, verbose=False):
    """Every tile over the given edge colours.

    `h_patches` supplies the colours available on north and south edges,
    `v_patches` those on east and west. With two of each that is sixteen
    tiles, which is the smallest set guaranteeing a tile exists for any
    pair of already-placed neighbours.

    With `cut`, each tile's diagonals are placed by graph cut rather than
    left straight. Every patch is rendered once and the images reused, so
    the cost is one minimum cut per diagonal per tile.

    Returns (tiles, codes) where codes[i] is (n, e, s, w) for tiles[i].
    """
    if len(h_patches) < 1 or len(v_patches) < 1:
        raise ValueError("need at least one patch per axis")

    images = None
    if cut:
        from .graphcut import render_patch, tile_labels
        images = {}
        for tag, group in (("h", h_patches), ("v", v_patches)):
            for i, p in enumerate(group):
                images[(tag, i)] = render_patch(p, size, resolution, up_axis)[0]

    tiles, codes = [], []
    for n in range(len(h_patches)):
        for e in range(len(v_patches)):
            for s in range(len(h_patches)):
                for w in range(len(v_patches)):
                    labels = None
                    if cut:
                        labels = tile_labels(
                            [images[("h", n)], images[("v", e)],
                             images[("h", s)], images[("v", w)]],
                            size=size, band=band)
                    tiles.append(build_tile(
                        (h_patches[n], v_patches[e], h_patches[s], v_patches[w]),
                        size, blend, up_axis, labels=labels))
                    codes.append((n, e, s, w))
                    if verbose:
                        print(f"    tile {len(tiles):>3} ({n}{e}{s}{w}): "
                              f"{len(tiles[-1]):,} splats")
    return tiles, codes


def layout(codes, nx, ny, seed=0):
    """An aperiodic arrangement respecting the edge-matching rule.

    Scanline order. At each cell the west colour is fixed by the cell to the
    left and the south colour by the cell below; north and east stay free.
    With a complete tile set some tile always fits, so this never has to
    backtrack, and the free choices are what stop the result repeating.

    Returns an (ny, nx) array of tile indices.
    """
    rng = np.random.default_rng(seed)
    codes = np.asarray(codes)
    grid = np.zeros((ny, nx), dtype=np.int32)

    for j in range(ny):
        for i in range(nx):
            ok = np.ones(len(codes), dtype=bool)
            if i > 0:
                ok &= codes[:, 3] == codes[grid[j, i - 1], 1]   # w == left.e
            if j > 0:
                ok &= codes[:, 2] == codes[grid[j - 1, i], 0]   # s == below.n
            choices = np.flatnonzero(ok)
            if not len(choices):
                raise ValueError(
                    "tile set is incomplete: no tile fits at "
                    f"({i}, {j}). Every (west, south) pair needs a tile.")
            grid[j, i] = rng.choice(choices)
    return grid


def check_layout(codes, grid):
    """Verify every shared edge in a layout actually matches."""
    codes = np.asarray(codes)
    ny, nx = grid.shape
    bad = 0
    for j in range(ny):
        for i in range(nx):
            if i + 1 < nx and codes[grid[j, i], 1] != codes[grid[j, i + 1], 3]:
                bad += 1
            if j + 1 < ny and codes[grid[j, i], 0] != codes[grid[j + 1, i], 2]:
                bad += 1
    return bad


def edge_gaussians(t, size, up_axis=2, edge="n", band=0.02, margin=0.03):
    """Gaussians along one edge, excluding the corners.

    Two tiles sharing an edge colour must return the same set here. That is
    the property the whole construction exists to provide, so it is worth
    being able to check directly.

    The corners are excluded deliberately. The diagonals reach the edge at
    each corner, so the last sliver before a corner belongs to the
    neighbouring triangle and is drawn from a different patch. Corners are
    shared by four tiles rather than two, and Cohen's original construction
    has the same gap; the known fix is corner tiles (Lagae and Dutre 2006).
    `margin` is how far back from each corner to stop looking, as a
    fraction of the tile.
    """
    plane = [i for i in range(3) if i != up_axis]
    x = t.xyz[:, plane[0]] / size
    y = t.xyz[:, plane[1]] / size
    b = band
    m = margin
    sel = {
        "n": (y > 0.5 - b) & (y > np.abs(x) + m),
        "s": (y < -0.5 + b) & (-y > np.abs(x) + m),
        "e": (x > 0.5 - b) & (x > np.abs(y) + m),
        "w": (x < -0.5 + b) & (-x > np.abs(y) + m),
    }[edge]
    pts = t.xyz[sel]
    return pts[np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))]