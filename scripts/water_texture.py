"""A tileable water texture from a straight-down drone photo of a river, for
the viewer's water shading (web/data/water.png and water.json).

    python scripts/water_texture.py D:/bozkir-data/gravel/L2/images/DJI_0334.JPG

Ground size per pixel from the photo's own metadata: height above the take-off
point (DJI RelativeAltitude) x sensor width / (focal length x image width);
--sensor-mm is the sensor width (13.2 mm, the Phantom 4 Pro's FC6310).
Water is where the photo has the river's cyan cast (green and blue over red,
split from the neutral gravel by Otsu's threshold). The
largest square inside the water is cut, made tileable by cross-fading it with
itself shifted half a tile (the shifted copy's middle covers the edges), and
written at --px pixels with its width in metres and mean colour.

--deep PHOTO cuts a second tile from a photo of the deep channel (dark and
rippled, where the first is the milky shallow water near the banks).

--layers PHOTO ... measures how the bed fades into the shallow water with
distance from the waterline, over the given photos: each pixel's signed
distance to the water's edge in metres, and per distance band the stone-scale
contrast against the photo's dry gravel. The water's colour changes from photo
to photo (depth, light), its contrast much less, so the bed's share T comes
from contrast (see layers()). T = T0 exp(-d / L) is written as "layers".
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def ground_mm_per_px(path, sensor_mm):
    """Millimetres of ground per pixel of a straight-down photo, from EXIF
    focal length and DJI's height above take-off."""
    im = Image.open(path)
    focal = float((im.getexif().get_ifd(0x8769) or {}).get(37386) or 0)
    head = Path(path).read_bytes()[:400_000].decode("latin1")
    alt = re.search(r'RelativeAltitude="([-+0-9.]+)"', head)
    if not focal or not alt:
        raise SystemExit(f"{path}: no focal length or height in its metadata")
    return float(alt.group(1)) * 1000 * sensor_mm / (focal * im.width), im


def otsu(values):
    """The threshold that best splits `values` in two (Otsu 1979)."""
    hist, edges = np.histogram(values, 256)
    w = np.cumsum(hist)
    mu = np.cumsum(hist * edges[:-1])
    between = (mu[-1] * w - mu * w[-1]) ** 2 / np.maximum(w * (w[-1] - w), 1)
    # Every cut across an empty gap scores the same: the middle of them.
    best = np.flatnonzero(between >= between.max() * (1 - 1e-9))
    return edges[int(best[len(best) // 2]) + 1]


def water_mask(rgb, mm_px):
    """True where the photo is water: its cyan cast (green and blue over
    red, averaged over ~5 cm) above Otsu's split. Gravel is neutral grey and
    white; river water, deep or shallow, rippled or calm, carries the cast."""
    from scipy import ndimage
    cyan = (rgb[..., 1] + rgb[..., 2]) / 2 - rgb[..., 0]
    cyan = ndimage.uniform_filter(cyan, max(3, int(50 / mm_px)))
    m = cyan > otsu(cyan[::4, ::4].ravel())
    m = ndimage.binary_opening(m, iterations=4)
    m = ndimage.binary_closing(m, iterations=4)
    lab, n = ndimage.label(m)
    if n == 0:
        raise SystemExit("no water found")
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return lab == 1 + int(np.argmax(sizes))


def largest_square(mask):
    """Centre and side of the largest square inside `mask` (chessboard
    distance to the outside)."""
    from scipy import ndimage
    d = ndimage.distance_transform_cdt(mask, metric="chessboard")
    j, i = np.unravel_index(int(np.argmax(d)), d.shape)
    return i, j, 2 * int(d[j, i]) - 1


def tileable(a):
    """`a` cross-faded with itself shifted half a tile: weight 1 in the
    middle, 0 at the border, so the shifted copy (whose middle is the
    original's edges, already continuous) fills the border."""
    n = a.shape[0]
    t = np.minimum(np.arange(n), np.arange(n)[::-1]) / (n / 2)
    w = np.clip(np.minimum.outer(t, t) * 1.6, 0, 1)[..., None]
    shifted = np.roll(a, (n // 2, n // 2), axis=(0, 1))
    return w * a + (1 - w) * shifted


LAYER_BANDS_M = (0.0, 0.04, 0.08, 0.15, 0.3, 0.6, 1.0)
DRY_BAND_M = (-0.6, -0.15)
STONE_CM = (1.0, 4.0)


def stone_contrast(lum, mm_px):
    """Brightness detail at stone scale: the difference of two blurs
    (STONE_CM), whose spread is the contrast compared."""
    from scipy import ndimage
    s1, s2 = (10 * c / mm_px for c in STONE_CM)
    return ndimage.gaussian_filter(lum, s1) - ndimage.gaussian_filter(lum, s2)


def layers(photos, sensor_mm, tile, tile_mm_px, reduce=4):
    """How much of the bed the water lets through against distance from the
    waterline, T = T0 exp(-d / L). Per photo, the stone-scale contrast in
    each distance band over that of its dry gravel; T is the bed's share
    in bed x T + tile x (1 - T) that gives it, the two textures adding in
    quadrature: c^2 = T^2 + (1 - T)^2 w^2, w the tile's own contrast over
    the dry gravel's. Fitted on log T, weighted by pixels."""
    from scipy import ndimage
    bands = np.array(LAYER_BANDS_M)
    rel_sum = np.zeros(len(bands) - 1)
    counts = np.zeros(len(bands) - 1)
    dry_c = []
    for p in photos:
        mm_px, im = ground_mm_per_px(p, sensor_mm)
        rgb = np.asarray(im.reduce(reduce).convert("RGB"), np.float64) / 255
        mm_q = mm_px * reduce
        m = water_mask(rgb, mm_q)
        d = (ndimage.distance_transform_edt(m) - ndimage.distance_transform_edt(~m)) * mm_q / 1000
        c = stone_contrast(rgb.mean(2), mm_q)
        dry = (d >= DRY_BAND_M[0]) & (d < DRY_BAND_M[1])
        if dry.sum() < 300:
            continue
        c_dry = c[dry].std()
        dry_c.append(c_dry)
        for b in range(len(bands) - 1):
            s = (d >= bands[b]) & (d < bands[b + 1])
            if s.sum() > 300:
                rel_sum[b] += c[s].std() / c_dry * s.sum()
                counts[b] += s.sum()
    if not dry_c or counts.sum() == 0:
        raise SystemExit("no waterline in the layer photos")
    w = stone_contrast(tile.mean(2), tile_mm_px).std() / float(np.mean(dry_c))
    keep = counts > 0
    rel = rel_sum[keep] / counts[keep]
    # c^2 = T^2 + (1 - T)^2 w^2 for T, the larger root; below the curve's
    # minimum the bed is gone.
    a, bq, cq = 1 + w * w, -2 * w * w, w * w - rel ** 2
    disc = np.maximum(bq * bq - 4 * a * cq, 0)
    t = np.clip((-bq + np.sqrt(disc)) / (2 * a), 1e-3, 1.0)
    mid = 0.5 * (bands[:-1] + bands[1:])[keep]
    A = np.stack([np.ones(keep.sum()), -mid], 1)
    sw = np.sqrt(counts[keep])
    (ln_t0, inv_l), *_ = np.linalg.lstsq(A * sw[:, None], np.log(t) * sw, rcond=None)
    return {"t0": round(float(min(np.exp(ln_t0), 1.0)), 3), "length_m": round(1 / max(float(inv_l), 1e-6), 3),
            "tile_contrast": round(float(w), 3),
            "bands": [{"from_m": float(lo), "to_m": float(hi), "pixels": int(n), "contrast": round(float(r), 3),
                       "t": round(float(tt), 3)}
                      for lo, hi, n, r, tt in zip(bands[:-1][keep], bands[1:][keep], counts[keep], rel, t)],
            "photos": [Path(p).name for p in photos]}


def water_tile(path, metres, px, sensor_mm):
    """The largest square of water in a photo, at most `metres` across, made
    tileable and resized to `px`. Returns the image, its metadata and the
    crop (full resolution) with its ground size per pixel."""
    mm_px, im = ground_mm_per_px(path, sensor_mm)
    rgb = np.asarray(im.convert("RGB"), np.float64) / 255
    mask = water_mask(rgb, mm_px)
    cx, cy, side = largest_square(mask)
    side = min(side, int(round(metres * 1000 / mm_px)))
    if side < 64:
        raise SystemExit(f"{path}: the largest water square is only {side} px")
    h = side // 2
    crop = rgb[cy - h:cy - h + side, cx - h:cx - h + side]
    out = Image.fromarray((np.clip(tileable(crop), 0, 1) * 255).astype(np.uint8)).resize((px, px), Image.LANCZOS)
    meta = {"metres": round(side * mm_px / 1000, 3), "mean": [round(float(v), 4) for v in crop.reshape(-1, 3).mean(0)],
            "source": Path(path).name, "mm_per_px": round(mm_px, 3), "water_share": round(float(mask.mean()), 3),
            "at": [int(cx), int(cy)]}
    return out, meta, crop, mm_px


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo", type=Path)
    ap.add_argument("--metres", type=float, default=2.5, help="tile width on the ground (at most what fits)")
    ap.add_argument("--px", type=int, default=512, help="texture width in pixels")
    ap.add_argument("--sensor-mm", type=float, default=13.2)
    ap.add_argument("--deep", type=Path, default=None,
                    help="a photo of the deep channel: its water tile is written beside the first as _deep.png")
    ap.add_argument("--layers", type=Path, nargs="+", default=[],
                    help="photos with a waterline: measure how the bed fades into the water")
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "web" / "data" / "water")
    args = ap.parse_args(argv)

    out, meta, crop, mm_px = water_tile(args.photo, args.metres, args.px, args.sensor_mm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out.with_suffix(".png"))
    print(f"{args.photo.name}: {mm_px:.2f} mm/px, water {meta['water_share']:.0%} of the photo; "
          f"tile {meta['metres']} m at {tuple(meta['at'])}, mean rgb {meta['mean']} -> {args.out.with_suffix('.png')}")
    if args.deep:
        d_out, d_meta, _, _ = water_tile(args.deep, args.metres, args.px, args.sensor_mm)
        d_path = args.out.with_name(args.out.name + "_deep.png")
        d_out.save(d_path)
        meta["deep"] = d_meta
        print(f"deep: {args.deep.name}, tile {d_meta['metres']} m, mean rgb {d_meta['mean']} -> {d_path}")
    if args.layers:
        meta["layers"] = layers(args.layers, args.sensor_mm, crop, mm_px)
        L = meta["layers"]
        print(f"layers over {len(args.layers)} photos: T = {L['t0']} exp(-d / {L['length_m']} m), "
              f"tile contrast {L['tile_contrast']} of dry gravel")
        for b in L["bands"]:
            print(f"  {b['from_m']:.2f}-{b['to_m']:.2f} m  contrast {b['contrast']:.2f}  T {b['t']:.2f}  {b['pixels']:,} px")
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
