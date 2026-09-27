"""Finding the least visible place to cut between two patches.

Four patches meet along a tile's diagonals, and a straight diagonal slices
through whatever is there (the visible X). Instead the seam wanders to where
the patches already agree - around a stone, not through it - found as a
minimum cut on a pixel grid (Kwatra et al. 2003, Cohen et al. 2003, GSWT
3.2), then lifted back to the Gaussians. The cost of separating neighbours
p and q is

    |A(p) - B(p)| + |A(q) - B(q)|
"""

import numpy as np

from .render import project_orthographic, rasterize_rgba

# Max-flow wants integer capacities; costs are scaled by this first.
COST_SCALE = 1000

# "Infinity" for edges that must never be cut, safely below overflow.
LOCKED = 1 << 28


def render_patch(patch, size, resolution=192, up_axis=2, sh_degree=0):
    """Render a patch straight down onto a fixed square grid (the same for
    every patch, so images compare). Returns (rgb, coverage) in [0, 1]."""
    half = size / 2.0
    p = project_orthographic(
        patch, view_axis=up_axis, up_sign=+1, resolution=resolution,
        bounds=((-half, -half), (half, half)), sh_degree=sh_degree)
    rgba = rasterize_rgba(p)
    alpha = rgba[..., 3]
    # Un-premultiply so colour is comparable where coverage differs.
    rgb = rgba[..., :3] / np.maximum(alpha[..., None], 1e-6)
    return np.clip(rgb, 0.0, 1.0), alpha


def seam_cost(a, b):
    """Per-pixel disagreement between two images, summed over channels."""
    return np.abs(a.astype(np.float64) - b.astype(np.float64)).sum(axis=-1)


def two_label_cut(a, b, take_a, take_b, free=None):
    """Split the image between patches A and B along the cheapest path.
    take_a / take_b: pixels forced to that patch. `free`: where the
    boundary may move. Returns True where B is used."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import maximum_flow

    h, w = a.shape[:2]
    n = h * w
    if free is None:
        free = ~(take_a | take_b)

    d = seam_cost(a, b)
    idx = np.arange(n).reshape(h, w)

    rows, cols, caps = [], [], []

    def add(u, v, c):
        rows.append(u); cols.append(v); caps.append(c)

    # Neighbour edges, both directions, cost = how visible a cut here is.
    for (di, dj) in ((0, 1), (1, 0)):
        p = idx[:h - di, :w - dj].ravel()
        q = idx[di:, dj:].ravel()
        cost = (d[:h - di, :w - dj] + d[di:, dj:]).ravel()
        c = np.maximum(np.rint(cost * COST_SCALE), 1).astype(np.int64)
        # A boundary is only allowed to pass between two free pixels.
        movable = (free[:h - di, :w - dj] & free[di:, dj:]).ravel()
        c = np.where(movable, c, LOCKED)
        rows.append(p); cols.append(q); caps.append(c)
        rows.append(q); cols.append(p); caps.append(c)

    # Forced pixels are tied to source (A) or sink (B) with uncuttable edges.
    src, snk = n, n + 1
    fa = idx[take_a].ravel()
    fb = idx[take_b].ravel()
    if len(fa):
        rows.append(np.full(len(fa), src)); cols.append(fa)
        caps.append(np.full(len(fa), LOCKED, dtype=np.int64))
    if len(fb):
        rows.append(fb); cols.append(np.full(len(fb), snk))
        caps.append(np.full(len(fb), LOCKED, dtype=np.int64))

    rows = np.concatenate([np.atleast_1d(r) for r in rows])
    cols = np.concatenate([np.atleast_1d(c) for c in cols])
    caps = np.concatenate([np.atleast_1d(c) for c in caps]).astype(np.int32)

    g = coo_matrix((caps, (rows, cols)), shape=(n + 2, n + 2)).tocsr()
    res = maximum_flow(g, src, snk)

    # Reachable from the source in the residual graph: A. The rest: B.
    residual = g - res.flow
    reach = np.zeros(n + 2, dtype=bool)
    reach[src] = True
    stack = [src]
    indptr, indices, data = residual.indptr, residual.indices, residual.data
    while stack:
        u = stack.pop()
        for k in range(indptr[u], indptr[u + 1]):
            v = indices[k]
            if data[k] > 0 and not reach[v]:
                reach[v] = True
                stack.append(v)
    return ~reach[:n].reshape(h, w)


def cut_cost(a, b, use_b):
    """Total visible cost of a given split, for comparing two of them."""
    d = seam_cost(a, b)
    total = 0.0
    for (di, dj) in ((0, 1), (1, 0)):
        differs = use_b[:use_b.shape[0] - di, :use_b.shape[1] - dj] != use_b[di:, dj:]
        total += float((d[:d.shape[0] - di, :d.shape[1] - dj] + d[di:, dj:])[differs].sum())
    return total


def tile_uv(resolution):
    """Tile-local coordinates per pixel, in [-0.5, 0.5]. Row 0 is north, so
    v decreases down the rows (as render_patch renders)."""
    i, j = np.mgrid[0:resolution, 0:resolution]
    u = (j + 0.5) / resolution - 0.5
    v = 0.5 - (i + 0.5) / resolution
    return u, v


def hard_regions(u, v):
    """The four triangles, as a label image. 0=N, 1=E, 2=S, 3=W."""
    d = np.stack([v - np.abs(u), u - np.abs(v),
                  -v - np.abs(u), -u - np.abs(v)], axis=0)
    return np.argmax(d, axis=0).astype(np.int8)


def tile_labels(images, size=1.0, band=0.12, edge_margin=0.06,
                centre_hold=0.05, verbose=False):
    """Which patch each pixel of a tile comes from.

    From the four hard triangles, each diagonal in turn moves to where its
    two patches agree. Fixed: a strip along every edge (so edges stay one
    patch and still match), a disc at the centre (where all four meet), and
    everything outside a band round the diagonal (so cuts cannot collide).

    `images`: four RGB arrays (north, east, south, west) on a common grid.
    """
    imgs = [np.asarray(im, dtype=np.float64) for im in images]
    r = imgs[0].shape[0]
    u, v = tile_uv(r)
    labels = hard_regions(u, v)

    inner = (np.maximum(np.abs(u), np.abs(v)) < 0.5 - edge_margin)
    inner &= (np.hypot(u, v) > centre_hold)

    # The four diagonal rays, each between two regions.
    rays = [
        (0, 1, v - u, u > 0),      # north | east
        (1, 2, v + u, u > 0),      # east  | south
        (2, 3, v - u, u < 0),      # south | west
        (3, 0, v + u, u < 0),      # west  | north
    ]

    for la, lb, signed, side in rays:
        near = (np.abs(signed) / np.sqrt(2.0) < band) & side & inner
        if near.sum() < 16:
            continue
        pair = (labels == la) | (labels == lb)
        free = near & pair
        take_a = ((labels == la) & ~free) | ~pair
        take_b = (labels == lb) & ~free
        if not take_a.any() or not take_b.any():
            continue
        use_b = two_label_cut(imgs[la], imgs[lb], take_a, take_b, free=free)
        labels = np.where(free, np.where(use_b, lb, la), labels).astype(np.int8)
        if verbose:
            print(f"    diagonal {la}-{lb}: {free.sum():,} pixels free")
    return labels


def label_at(patch, labels, size, up_axis=2):
    """Which region each Gaussian falls in: the label image looked up at its
    projected position (GSWT 3.2's lift from 2D to 3D).

    Gaussians just outside the square (levelling rotates patches) still
    cover ground inside, so they are kept, and take the plain triangle their
    angle gives. Clamping to the border pixel would not work: that label
    differs between tiles and would break edge matching.
    """
    plane = [i for i in range(3) if i != up_axis]
    r = labels.shape[0]
    xy = patch.xyz[:, plane] / max(size, 1e-9)
    u, v = xy[:, 0], xy[:, 1]

    inside = (np.abs(u) <= 0.5) & (np.abs(v) <= 0.5)
    j = np.clip(((u + 0.5) * r).astype(int), 0, r - 1)
    i = np.clip(((0.5 - v) * r).astype(int), 0, r - 1)

    outer = np.argmax(np.stack([v - np.abs(u), u - np.abs(v),
                                -v - np.abs(u), -u - np.abs(v)]), axis=0)
    return np.where(inside, labels[i, j], outer).astype(np.int8)


def edge_purity(labels, edge_margin=0.06):
    """Fraction of each edge strip from the right patch; below 1.0 the cut
    leaked into an edge and matching tiles would no longer match."""
    r = labels.shape[0]
    u, v = tile_uv(r)
    out = {}
    for name, lbl, sel in (("n", 0, v > 0.5 - edge_margin),
                           ("e", 1, u > 0.5 - edge_margin),
                           ("s", 2, v < -0.5 + edge_margin),
                           ("w", 3, u < -0.5 + edge_margin)):
        # Only the part of the strip inside this region (corners belong to
        # the neighbouring triangles).
        own = sel & (hard_regions(u, v) == lbl)
        out[name] = float((labels[own] == lbl).mean()) if own.any() else 1.0
    return out