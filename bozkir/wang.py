"""Building Wang tiles out of exemplar patches, and laying them out.

Cohen et al. 2003 ("Wang Tiles for Image and Texture Generation") applied
to Gaussians, as in GSWT. A square tile is cut along both diagonals into
four triangles, each filled from the patch of its edge's colour. The whole
north edge lies in the north triangle, so every tile with the same north
colour has an identical north edge - the same Gaussians in the same places -
and matching neighbours agree exactly by construction.

Inside the tile two patches meet along the diagonals. By default the seam
is placed by graph cut on top-down renders (bozkir/graphcut.py, as in
Cohen and GSWT 3.2); feathered straight diagonals are the fallback.
"""

import numpy as np

from .tile import merge

# Corners are where all four triangles meet, so a feathered tile is not
# exactly edge-matched within this fraction of the tile size of a corner.
CORNER_NOTE = "edges match exactly except within `blend` of each corner"


def region_weights(xy, size, blend=0.0):
    """Membership of each of the four triangles (columns N, E, S, W),
    summing to one. `xy` is tile-local, spanning [-size/2, size/2].
    `blend` 0 is a hard assignment; above 0 the diagonals become a mixing
    band of that width."""
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
    """Assemble one tile from four patches (north, east, south, west), each
    centred on the origin, each contributing the Gaussians in its region.

    Without `labels` the regions are the four triangles, feathered by
    `blend` (which blurs a join but cannot hide a change of material). With
    `labels`, a region image from bozkir.graphcut, the assignment is a hard
    lookup along a seam already placed where the patches agree.
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


def minimal_codes(kh, kv):
    """Two tiles for every (west, south) pair: 2 k^2 instead of k^4.

    Cohen et al. 2003: a stochastic tiling only needs two choices at every
    step. The repetition the eye catches is inside the tiles (each south
    triangle is always part of one of k patches), so more colours is the
    fix, and this makes three colours cost 18 tiles instead of 81.

    The free north and east colours are spread evenly over every side, or
    the tiling gets a grain. Returns a list of (n, e, s, w) codes.
    """
    if kh < 2 or kv < 2:
        raise ValueError("the minimal set needs at least two colours per axis")
    codes = []
    for w in range(kv):
        for s in range(kh):
            for extra in (0, 1):
                n = (w + s + extra) % kh
                e = (2 * w + s + 2 * extra) % kv
                if extra and (n, e) == ((w + s) % kh, (2 * w + s) % kv):
                    e = (e + 1) % kv
                codes.append((n, e, s, w))
    return codes


def build_tile_set(h_patches, v_patches, size, blend=0.0, up_axis=2,
                   cut=False, resolution=160, band=0.14, verbose=False,
                   codes=None):
    """Build tiles over the given edge colours: `h_patches` for north and
    south, `v_patches` for east and west. All k^4 by default, or only
    `codes` (see minimal_codes). With `cut`, seams by graph cut; each patch
    is rendered once and reused.

    Returns (tiles, codes), codes[i] being (n, e, s, w) for tiles[i].
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

    if codes is None:
        wanted = [(n, e, s, w)
                  for n in range(len(h_patches)) for e in range(len(v_patches))
                  for s in range(len(h_patches)) for w in range(len(v_patches))]
    else:
        wanted = [tuple(int(x) for x in c) for c in codes]
    tiles, codes = [], []
    for (n, e, s, w) in wanted:
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
    """An aperiodic, edge-matched arrangement, in scanline order: west is
    fixed by the left cell, south by the one below, north and east are
    free. A complete set never needs backtracking.

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
    """Gaussians along one edge, sorted, for checking that tiles sharing an
    edge colour really share the edge.

    Corners are excluded (`margin`, a fraction of the tile): there the
    diagonals meet the edge and four tiles share the point. Cohen's
    construction has the same gap; corner tiles (Lagae and Dutre 2006)
    are the fix.
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