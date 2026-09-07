"""Finding the least visible place to cut between two patches.

A tile is assembled from four patches meeting along the two diagonals of the
square. Cutting on the diagonal itself is arbitrary: it slices straight
through whatever happens to be there, and where the two patches differ the
join shows.

The alternative is to let the boundary wander. Between two patches there is
usually a path along which they already look nearly the same - around a
stone rather than through it - and a cut placed there is close to invisible.
Finding that path is a minimum cut on a grid of pixels, which is the method
in Kwatra et al. 2003 (Graphcut Textures) and Cohen et al. 2003, and what
GSWT does in Section 3.2 before lifting the result back into 3D.

The cost of separating two neighbouring pixels p and q is

    |A(p) - B(p)| + |A(q) - B(q)|

so the cut is cheap exactly where the two patches agree.
"""

import numpy as np

from .render import project_orthographic, rasterize_rgba
from .tile import extract_patch  # noqa: F401  (re-exported for convenience)

# Capacities have to be integers for the max-flow solver, so costs are
# scaled by this before rounding. Large enough that rounding does not
# change which cut wins.
COST_SCALE = 1000

# Stands in for infinity on edges that must never be cut. Well below the
# range where summing capacities could overflow.
LOCKED = 1 << 28


def render_patch(patch, size, resolution=192, up_axis=2, sh_degree=0):
    """Render a patch straight down onto a fixed square grid.

    Every patch in a tile set has to land on the same pixel grid or their
    images cannot be compared, so the framing is given explicitly rather
    than fitted to the content.

    Returns (rgb, coverage), both resolution x resolution, values in [0, 1].
    """
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
    """Split the image between patch A and patch B along the cheapest path.

    take_a / take_b are boolean masks of pixels that must come from that
    patch. `free` limits where the boundary is allowed to move; pixels
    outside it keep whichever side they start on.

    Returns a boolean array, True where patch B should be used.
    """
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

    # Source and sink. Anything forced to A is tied to the source, anything
    # forced to B to the sink, with a capacity no cut can afford.
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

    # Everything still reachable from the source after the flow keeps
    # label A; the rest takes B.
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
    """Tile-local coordinates for each pixel, in [-0.5, 0.5].

    Row 0 is the top of the image, which is the north edge, so v decreases
    down the rows. Matches the orientation render_patch produces.
    """
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
    """Decide which patch each pixel of a tile comes from.

    Starts from the four hard triangles and lets each diagonal move, one at
    a time, to wherever the two patches either side of it already agree.

    Three regions stay fixed, and each for a reason:

    - a strip along every edge, so the edge stays entirely one patch. That
      is what makes two tiles with the same edge colour identical along
      their shared boundary, and it is the whole point of the construction.
    - a small disc at the centre, where all four regions meet and a moving
      boundary would have three neighbours to argue with.
    - everything outside a band around the diagonal being relaxed, so the
      four cuts cannot reach each other.

    `images` is four RGB arrays (north, east, south, west) on a common grid.
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
            moved = int((free & (use_b != (hard_regions(u, v)[free.nonzero()]
                                           == lb).reshape(-1).any())).sum())
            print(f"    diagonal {la}-{lb}: {free.sum():,} pixels free")
    return labels


def label_at(patch, labels, size, up_axis=2):
    """Which region each Gaussian of a patch falls in, or -1 if it is outside.

    The cut is found on a 2D image and applied in 3D by looking up where a
    Gaussian projects to - the same lift GSWT describes in Section 3.2.

    Patches overflow the tile square routinely, because levelling rotates
    them, and those Gaussians matter: one whose centre sits just outside
    still covers ground just inside, so dropping them opens a gap along
    every edge.

    They cannot be clamped to the nearest border pixel either - that pixel's
    label differs from tile to tile, so the same Gaussian would be kept in
    one tile and dropped in another, and edge matching would break.

    Instead they take the plain triangle their angle puts them in. That is a
    function of position alone, so every tile agrees on it, and the cut only
    ever moves boundaries inside the square anyway.
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
    """Fraction of each edge strip that comes from the right patch.

    Anything below 1.0 means the cut leaked into an edge and two tiles
    sharing that colour would no longer match.
    """
    r = labels.shape[0]
    u, v = tile_uv(r)
    out = {}
    for name, lbl, sel in (("n", 0, v > 0.5 - edge_margin),
                           ("e", 1, u > 0.5 - edge_margin),
                           ("s", 2, v < -0.5 + edge_margin),
                           ("w", 3, u < -0.5 + edge_margin)):
        # Corners belong to the neighbouring triangles by construction, so
        # only the part of the strip inside this region is checked.
        own = sel & (hard_regions(u, v) == lbl)
        out[name] = float((labels[own] == lbl).mean()) if own.any() else 1.0
    return out