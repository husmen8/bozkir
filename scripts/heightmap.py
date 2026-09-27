"""Turn a drone DSM (or any PLY) into a height field for the viewer.

    python scripts/heightmap.py C:/odm/copr/odm_dem/dsm.tif --name desert

Real ground in metres, a by-product of the same photogrammetry that gives
the tiles. The survey is a polygon in a rectangular file, so the largest
rectangle inside the valid region is used (interpolating over nodata would
invent terrain). Heights are stored 0..1; the metre range goes in the sidecar.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bozkir.terrain import (HEIGHT_ENCODING, height_from_points,  # noqa: E402
                            largest_rectangle, read_dsm, write_height_png)

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path",
                    help="a DSM (odm_dem/dsm.tif) or a .ply capture. A "
                         "drone survey has a DSM; a phone capture does not, "
                         "but it still contains the ground, so the surface "
                         "is rasterised out of the points instead")
    ap.add_argument("--name", help="output name; defaults to the file's stem")
    ap.add_argument("--out", default="web/data", help="where to write")
    ap.add_argument("--size", type=int, default=512,
                    help="output resolution; the field is resampled to at "
                         "most this on its long edge (default 512)")
    ap.add_argument("--smooth", type=int, default=0,
                    help="box-blur passes. A DSM carries the vegetation and "
                         "the survey's own noise, and a tile laid on a bush "
                         "sits on a spike. 1 or 2 takes that off without "
                         "flattening the landforms")
    args = ap.parse_args(argv)

    if str(args.path).lower().endswith(".ply"):
        # The scene has to be aligned first, or "up" is whatever direction
        # the camera happened to face and the height field is a shear.
        from bozkir.scene import prepare
        s_ = prepare(args.path, verbose=True)
        a = height_from_points(s_.xyz, up_axis=2, resolution=args.size)
        good = np.isfinite(a)
        res = None
        print(f"  rasterised {len(s_):,} splats to {a.shape[1]}x{a.shape[0]}")
    else:
        a, good, res = read_dsm(args.path)
    print(f"{Path(args.path).name}: {a.shape[1]}x{a.shape[0]}, "
          f"{100 * good.mean():.1f}% valid")

    top, left, h, w = largest_rectangle(good)
    if h < 8 or w < 8:
        raise SystemExit("  no usable rectangle inside the valid region")
    a = a[top:top + h, left:left + w]
    print(f"  largest solid rectangle: {w}x{h} at ({left},{top}), "
          f"{100.0 * h * w / good.size:.0f}% of the file")

    lo, hi = float(a.min()), float(a.max())
    span = hi - lo
    print(f"  height {lo:.2f} to {hi:.2f} m  (range {span:.2f} m)")
    if res:
        print(f"  ground sample distance {res * 100:.1f} cm, "
              f"so {w * res:.0f} x {h * res:.0f} m")

    # Resample before smoothing: a box blur on the output grid is cheaper
    # and the result is the same shape of hill.
    long_edge = max(h, w)
    if long_edge > args.size:
        k = args.size / long_edge
        out_w, out_h = max(8, int(w * k)), max(8, int(h * k))
        img = Image.fromarray(a.astype(np.float32)).resize(
            (out_w, out_h), Image.BILINEAR)
        a = np.asarray(img).astype(np.float64)
        print(f"  resampled to {out_w}x{out_h}")

    for _ in range(max(0, args.smooth)):
        p = np.pad(a, 1, mode="edge")
        a = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
             + a * 4) / 8.0

    # 0..1 for storage; the metre range travels in the sidecar.
    lo, hi = float(a.min()), float(a.max())
    norm = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)

    name = args.name or Path(args.path).stem
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{name}.height.png"
    # Split across two channels rather than written as 16-bit greyscale,
    # which browsers silently decode to eight bits. See write_height_png.
    write_height_png(norm, png)

    meta = {
        "width": int(norm.shape[1]),
        "height": int(norm.shape[0]),
        "metres_low": lo,
        "metres_high": hi,
        "metres_range": hi - lo,
        "sample_metres": (res * max(h, w) / max(norm.shape)) if res else None,
        "source": Path(args.path).name,
        "encoding": HEIGHT_ENCODING,
    }
    side = out / f"{name}.height.json"
    with open(side, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  wrote {png}")
    print(f"  wrote {side}")
    if meta["sample_metres"]:
        print(f"  each sample is {meta['sample_metres']:.2f} m of ground")
    return 0


if __name__ == "__main__":
    sys.exit(main())