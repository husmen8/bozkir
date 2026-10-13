"""Building Wang tiles out of exemplar patches, and laying them out.

Cohen et al. 2003 ("Wang Tiles for Image and Texture Generation") applied
to Gaussians, as in GSWT. A square tile is cut along both diagonals into
four triangles, each filled from the patch of its edge's colour. The whole
north edge lies in the north triangle, so every tile with the same north
colour has an identical north edge - the same Gaussians in the same places.
For the ground to run on into the neighbour, the patch must sit on the
edge, as Cohen's diamond does (on_edges, export --diamonds); centred on the
tile, as before, the two sides of an edge are ground a tile apart.

Inside the tile two patches meet along the diagonals. By default the seam
is placed by graph cut on top-down renders (bozkir/graphcut.py, as in
Cohen and GSWT 3.2); feathered straight diagonals are the fallback.
"""

import numpy as np

from .tile import merge, translate

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


def seam_membrane(diff, weight, inside, iters=500):
    """An offset field over `inside` equal to `diff` along its edge and as
    smooth as possible within (the membrane of Poisson and mean-value
    cloning: Perez et al. 2003, Farbman et al. 2009). `weight` says how
    much each edge pixel's difference can be trusted (coverage); the edge
    values are smoothed along the edge first, so one odd stone does not
    tint a whole tile."""
    from scipy.ndimage import binary_erosion, gaussian_filter
    edge = inside & ~binary_erosion(inside)
    w = weight * edge
    smooth = np.stack([gaussian_filter(diff[..., c] * w, 6.0) for c in range(3)], -1)
    norm = gaussian_filter(w, 6.0)[..., None]
    known = smooth / np.maximum(norm, 1e-6)
    f = np.where(edge[..., None], known, 0.0)
    free = inside & ~edge
    for _ in range(iters):
        g = np.pad(f, ((1, 1), (1, 1), (0, 0)), mode="edge")
        avg = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]) / 4.0
        f = np.where(free[..., None], avg, f)
    return np.where(inside[..., None], f, 0.0)


def with_centre(tile, centre, size, up_axis=2, resolution=160,
                frame=0.06, band=0.10):
    """`tile` with its interior from `centre`, a patch of its own: the
    frame along the edges stays (edges still match), a closed seam by
    graph cut in the band inside it, and the centre's colour corrected by
    a membrane so it meets the frame along the whole seam (gradient-domain
    blending, as the Poisson smoothing of omega-tiles), its detail kept."""
    from .graphcut import render_patch, centre_labels
    from .ply import SH_C0
    rgb_t, a_t = render_patch(tile, size, resolution, up_axis)
    rgb_c, a_c = render_patch(centre, size, resolution, up_axis)
    use_c = centre_labels(rgb_t, rgb_c, frame, band)
    offset = np.clip(seam_membrane(rgb_t - rgb_c, np.minimum(a_t, a_c), use_c),
                     -0.2, 0.2)

    def lookup(p):
        plane = [i for i in range(3) if i != up_axis]
        xy = p.xyz[:, plane] / size
        inside = np.all(np.abs(xy) <= 0.5, axis=1)
        j = np.clip(((xy[:, 0] + 0.5) * resolution).astype(int), 0, resolution - 1)
        i = np.clip(((0.5 - xy[:, 1]) * resolution).astype(int), 0, resolution - 1)
        return inside, use_c[i, j], (i, j)

    inside, at, _ = lookup(tile)
    keep_t = ~inside | ~at
    inside, at, (i, j) = lookup(centre)
    keep_c = inside & at
    c = centre.subset(np.flatnonzero(keep_c))
    shift = offset[i[keep_c], j[keep_c]]
    c.sh_dc = ((np.clip(c.base_rgb + shift, 0, 1) - 0.5) / SH_C0).astype(np.float32)
    return merge(tile.subset(np.flatnonzero(keep_t)), c)


def on_edges(h_patches, v_patches, size, up_axis=2):
    """Each patch moved onto the edge it colours (Cohen et al. 2003: the
    sample diamond is centred on the edge). The tile below an edge then
    takes the patch's lower half and the tile above its upper half, so the
    ground runs on across the edge. A patch left on the tile's middle shows
    its top border below the edge and its bottom border above it: ground a
    tile apart, a cut along every edge. Returns the patches for the north,
    east, south and west triangles; what lies more than a quarter tile past
    the tile is dropped (the export cuts at the square anyway, and half of
    each moved patch lies outside it).
    """
    plane = [i for i in range(3) if i != up_axis]

    def moved(p, dx, dy):
        off = np.zeros(3)
        off[plane[0]], off[plane[1]] = dx, dy
        q = translate(p, off)
        return q.subset(np.flatnonzero(np.abs(q.xyz[:, plane]).max(axis=1) <= 0.75 * size))
    h = size / 2.0
    return ([moved(p, 0, h) for p in h_patches], [moved(p, h, 0) for p in v_patches],
            [moved(p, 0, -h) for p in h_patches], [moved(p, -h, 0) for p in v_patches])


def build_tile_set(h_patches, v_patches, size, blend=0.0, up_axis=2,
                   cut=False, resolution=160, band=0.14, verbose=False,
                   codes=None, centres=None, diamonds=False):
    """Build tiles over the given edge colours: `h_patches` for north and
    south, `v_patches` for east and west. All k^4 by default, or only
    `codes` (see minimal_codes). With `cut`, seams by graph cut; each patch
    is rendered once and reused. With `centres`, one patch per tile (in
    code order) gives that tile its interior (with_centre). With
    `diamonds`, each patch sits on its edge (on_edges) and the ground runs
    on into the neighbour.

    Returns (tiles, codes), codes[i] being (n, e, s, w) for tiles[i].
    """
    if len(h_patches) < 1 or len(v_patches) < 1:
        raise ValueError("need at least one patch per axis")

    sides = (on_edges(h_patches, v_patches, size, up_axis) if diamonds
             else (h_patches, v_patches, h_patches, v_patches))
    images, cover = None, None
    if cut:
        from .graphcut import render_patch, tile_labels, tile_uv
        images = {}
        for k, group in enumerate(sides):
            for i, p in enumerate(group):
                twin = k - 2 if k >= 2 and sides[k - 2] is group else None
                images[(k, i)] = (images[(twin, i)] if twin is not None
                                  else render_patch(p, size, resolution, up_axis)[0])
        if diamonds:
            u, v = tile_uv(resolution)
            cover = [v > 0, u > 0, v < 0, u < 0]

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
                [images[(0, n)], images[(1, e)], images[(2, s)], images[(3, w)]],
                size=size, band=band, cover=cover)
        t = build_tile((sides[0][n], sides[1][e], sides[2][s], sides[3][w]),
                       size, blend, up_axis, labels=labels)
        if centres is not None:
            t = with_centre(t, centres[len(tiles) % len(centres)], size,
                            up_axis, resolution)
        tiles.append(t)
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