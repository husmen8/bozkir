"""Export a scene to the .splat binary format for the browser renderer.

A degree-3 PLY is 248 bytes per splat, most of it spherical harmonics the
web renderer does not use. This packs each splat into 32 bytes:

    bytes  0..11   position   3 x float32
    bytes 12..23   scale      3 x float32   (world units, already exp'd)
    bytes 24..27   colour     4 x uint8     r, g, b, opacity
    bytes 28..31   rotation   4 x uint8     w, x, y, z as round(q*128 + 128)

That is antimatter15's .splat layout, so files written here should also
open in other .splat viewers - useful for sanity checking, and for handing
someone a file without handing them this repo.

Garden at 730,850 splats comes out around 23 MB instead of 181.

    python scripts/export_splat.py data/aligned/garden.ply --clean
    python scripts/export_splat.py data/raw/garden.ply --clean --sort
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bozkir.scene import add_scene_args, scene_from_args  # noqa: E402
from bozkir.ply import SH_C0  # noqa: E402

STRIDE = 32


def pack(s, sort=False):
    """Splats -> a flat uint8 buffer, 32 bytes each."""
    n = len(s)
    buf = np.zeros((n, STRIDE), dtype=np.uint8)

    order = np.arange(n)
    if sort:
        # Largest and most opaque first. Sorting once here means a viewer
        # that streams the file shows the important splats early.
        order = np.argsort(-(s.scale.max(axis=1) * s.opacity))

    xyz = np.ascontiguousarray(s.xyz[order], dtype=np.float32)
    scale = np.ascontiguousarray(s.scale[order], dtype=np.float32)
    buf[:, 0:12] = xyz.view(np.uint8).reshape(n, 12)
    buf[:, 12:24] = scale.view(np.uint8).reshape(n, 12)

    rgb = np.clip(SH_C0 * s.sh_dc[order] + 0.5, 0.0, 1.0)
    buf[:, 24:27] = np.round(rgb * 255.0).astype(np.uint8)
    buf[:, 27] = np.round(np.clip(s.opacity[order], 0, 1) * 255.0).astype(np.uint8)

    q = s.rot[order]
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
    buf[:, 28:32] = np.clip(np.round(q * 128.0 + 128.0), 0, 255).astype(np.uint8)

    return buf.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output path (default: web/data/<stem>.splat)")
    ap.add_argument("--sort", action="store_true",
                    help="write largest and most opaque splats first")
    ap.add_argument("--max-splats", type=int, default=None,
                    help="keep only this many, most visible first")
    add_scene_args(ap)
    args = ap.parse_args()

    s = scene_from_args(args)

    if args.max_splats is not None and len(s) > args.max_splats:
        ink = s.scale.max(axis=1) * s.opacity
        keep = np.argpartition(ink, -args.max_splats)[-args.max_splats:]
        print(f"  reduced {len(s):,} -> {args.max_splats:,} splats")
        s = s.subset(np.sort(keep))

    buf = pack(s, sort=args.sort)

    dest = args.out or Path("web/data") / f"{args.path.stem}.splat"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(buf.tobytes())

    lo = np.percentile(s.xyz, 1, axis=0)
    hi = np.percentile(s.xyz, 99, axis=0)
    print(f"\n  {len(s):,} splats x {STRIDE} bytes = "
          f"{len(buf) / 1e6:.1f} MB")
    print(f"  extent (1-99 pct)  {np.round(hi - lo, 2)}")
    print(f"  centre             {np.round((hi + lo) / 2, 2)}")
    print(f"  wrote {dest}")


if __name__ == "__main__":
    main()