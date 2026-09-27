"""How much a tiled ground repeats, as a number.

    python scripts/repetition.py desert
    python scripts/repetition.py desert --dir other/data --label before

The layout never repeats, but the tiles do: each triangle is always the
same part of a few patches, at the same place in the cell. So the ground
correlates with itself at whole-tile shifts and hardly at half-tile shifts:

    lattice   = mean autocorrelation at shifts of 1, 2, 3 tiles (x and y)
    off       = the same at 1.5, 2.5, 3.5 tiles
    excess    = lattice - off

Measured on brightness ("texture") and on a map of what stands out
("landmarks"). Near zero: no grid to find. Each material separately, laid
out by Wang edge matching (not the viewer's exact sequence), over seeds.

Plus busyness: how different a tile's four triangles are (busyness()).
Neither repetition number matched my eye on the desert tilesets; busyness
did, on the two runs left to measure.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bozkir.pack import read_splat, tile_atlas  # noqa: E402


def layout(tiles, cls, n, rng):
    """Wang layout of one class, south to north, west to east."""
    idx = [k for k, t in enumerate(tiles) if t["class"] == cls]
    g = np.zeros((n, n), int)
    for j in range(n):
        for i in range(n):
            ok = [k for k in idx
                  if (i == 0 or tiles[k]["w"] == tiles[g[j, i - 1]]["e"])
                  and (j == 0 or tiles[k]["s"] == tiles[g[j - 1, i]]["n"])]
            g[j, i] = ok[int(rng.integers(len(ok)))]
    return g


def mosaic(g, cells, res):
    n = g.shape[0]
    out = np.zeros((n * res, n * res))
    for j in range(n):
        for i in range(n):
            r0 = (n - 1 - j) * res
            out[r0:r0 + res, i * res:(i + 1) * res] = cells[g[j, i]]
    return out


def autocorr(img):
    """Normalised circular autocorrelation, lag (0, 0) = 1."""
    a = img - img.mean()
    f = np.fft.rfft2(a)
    r = np.fft.irfft2(f * np.conj(f), s=a.shape)
    return r / max(r[0, 0], 1e-12)


def _box(a, r):
    """Mean over a (2r+1) square window, wrapping at the borders."""
    c = np.cumsum(np.cumsum(np.pad(a, r, mode="wrap"), 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    k = 2 * r + 1
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def landmarks(img, res, radius=None, floor=2.0):
    """What stands out: each pixel's distance from its neighbourhood mean in
    units of the local spread, minus `floor`. Grain and small bushes zero
    out; a pale opening or a dark stone stays."""
    r = radius or max(2, res // 3)
    m = _box(img, r)
    sd = np.sqrt(np.maximum(_box(img * img, r) - m * m, 1e-8))
    z = np.abs(img - m) / sd
    return np.maximum(z - floor, 0.0)


def lattice_excess(img, res):
    """Correlation at whole-tile shifts minus that at half-tile shifts."""
    r = autocorr(img)

    def at(k):
        d = int(round(k * res))
        return 0.5 * (r[d, 0] + r[0, d])
    lat = float(np.mean([at(k) for k in (1, 2, 3)]))
    off = float(np.mean([at(k) for k in (1.5, 2.5, 3.5)]))
    return lat, off, lat - off


def busyness(cell):
    """How much a tile reads as four pieces rather than one ground (an X of
    different grounds shows the construction even without repetition).

    Share of the tile's brightness variance explained by which triangle a
    pixel is in (eta squared): 0 = four alike, 1 = four flat blocks. Grain
    inside a triangle counts against it, so texture alone is not busy.
    """
    res = cell.shape[0]
    y, x = (np.mgrid[0:res, 0:res] + 0.5) / res      # row 0 is north
    a, b = y < x, y < 1 - x
    parts = (a & b, a & ~b, ~a & ~b, ~a & b)          # N, E, S, W
    total = cell.var()
    if total < 1e-12:
        return 0.0
    mu = cell.mean()
    between = sum(p.mean() * (cell[p].mean() - mu) ** 2 for p in parts)
    return float(between / total)


def interior_variants(cells, tiles, k, rng, radius=0.32):
    """Predict interior variants before building them: k extra versions of
    each tile whose middle (a feathered disc touching no edge) comes from
    another tile of the same material, so edges still match. (Predicted
    useless: they recycle the same patches.) Returns cells and tiles with
    the variants appended."""
    res = cells[0].shape[0]
    y, x = np.mgrid[0:res, 0:res] + 0.5
    d = np.hypot(x - res / 2, y - res / 2) / res
    w = np.clip((radius - d) / 0.08, 0, 1)          # 1 inside, soft edge
    out_cells, out_tiles = list(cells), list(tiles)
    for a, t in enumerate(tiles):
        same = [b for b, u in enumerate(tiles)
                if u["class"] == t["class"] and b != a]
        for _ in range(k):
            b = same[int(rng.integers(len(same)))]
            out_cells.append(cells[a] * (1 - w) + cells[b] * w)
            out_tiles.append(t)
    return out_cells, out_tiles


def measure(splat_dir, name, n=32, res=48, seeds=5, variants=0):
    m, s = read_splat(splat_dir / f"{name}.splat", splat_dir / f"{name}.json")
    atlas, lay = tile_atlas(m, s, res)
    lum = (atlas[..., :3].astype(np.float64) / 255) @ [0.299, 0.587, 0.114]
    cells = []
    for k in range(lay["count"]):
        r, q = divmod(k, lay["cols"])
        cells.append(lum[r * res:(r + 1) * res, q * res:(q + 1) * res])
    tiles = m["tiles"]
    if variants:
        cells, tiles = interior_variants(cells, tiles, variants,
                                         np.random.default_rng(99))
    out = {}
    for cls in sorted({t["class"] for t in tiles}):
        vals = []
        for seed in range(seeds):
            g = layout(tiles, cls, n, np.random.default_rng(seed))
            img = mosaic(g, cells, res)
            vals.append(lattice_excess(img, res)
                        + lattice_excess(landmarks(img, res), res))
        v = np.array(vals)
        busy = [busyness(cells[k]) for k, t in enumerate(tiles)
                if t["class"] == cls]
        out[cls] = {"lattice": v[:, 0].mean(), "off": v[:, 1].mean(),
                    "excess": v[:, 2].mean(), "sd": v[:, 2].std(),
                    "landmark": v[:, 5].mean(), "landmark_sd": v[:, 5].std(),
                    "busyness": float(np.mean(busy)),
                    "busyness_sd": float(np.std(busy)),
                    "tiles": sum(t["class"] == cls for t in tiles)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--dir", type=Path, default=ROOT / "web" / "data")
    ap.add_argument("--label", default="")
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--variants", type=int, default=0,
                    help="predict interior variants: this many extra "
                         "versions of each tile with a middle from another")
    args = ap.parse_args(argv)
    res = measure(args.dir, args.name, args.grid, seeds=args.seeds,
                  variants=args.variants)
    print(f"  {args.label or args.name}: repetition at whole-tile shifts, "
          f"{args.grid}x{args.grid} tiles, {args.seeds} seeds")
    for cls, r in res.items():
        print(f"    class {cls} ({r['tiles']} tiles): texture {r['excess']:.3f}"
              f" ± {r['sd']:.3f}   landmarks {r['landmark']:.3f}"
              f" ± {r['landmark_sd']:.3f}   busyness {r['busyness']:.3f}"
              f" ± {r['busyness_sd']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())