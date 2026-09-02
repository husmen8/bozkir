"""Render orthographic views of a 3DGS scene with the CPU reference renderer.

Run from the repo root:

    python scripts/render_ortho.py data/raw/garden.ply
    python scripts/render_ortho.py data/raw/garden.ply --res 1200 --axis 1

Writes out/render_<name>_<axis>.png. Slow by design; --max-splats trades
accuracy for time while you are still finding the right framing.
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bozkir.scene import add_scene_args, scene_from_args  # noqa: E402
from bozkir.render import project_orthographic, rasterize  # noqa: E402

AXES = "xyz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--axis", type=int, default=None,
                    help="world axis to look down (default: auto-detect up)")
    ap.add_argument("--sign", type=int, default=-1, choices=(-1, 1),
                    help="which end of that axis the camera sits on")
    ap.add_argument("--res", type=int, default=800)
    ap.add_argument("--max-splats", type=int, default=None)
    ap.add_argument("--bg", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    ap.add_argument("--out", type=Path, default=Path("out"))
    add_scene_args(ap)
    args = ap.parse_args()

    s = scene_from_args(args)

    axis = args.axis
    if axis is None:
        # Same trick as probe 1: the dense part of the scene is mostly
        # ground, so its thinnest principal direction is 'up'.
        centre = np.median(s.xyz, axis=0)
        r = np.linalg.norm(s.xyz - centre, axis=1)
        core = s.xyz[r <= np.percentile(r, 90)]
        d = core - core.mean(axis=0)
        _, evec = np.linalg.eigh((d.T @ d) / len(d))
        axis = int(np.argmax(np.abs(evec[:, 0])))
        print(f"  auto-detected up axis: {AXES[axis]}")

    t0 = time.time()
    p = project_orthographic(s, view_axis=axis, up_sign=args.sign,
                             resolution=args.res)
    print(f"  projected {p['kept']:,} / {p['total']:,} splats "
          f"({p['kept'] / p['total']:.1%}) to {p['W']}x{p['H']} "
          f"in {time.time() - t0:.1f}s")

    n = p["kept"] if args.max_splats is None else min(p["kept"], args.max_splats)
    t0 = time.time()

    def tick(done, total):
        el = time.time() - t0
        eta = el / max(done, 1) * (total - done)
        print(f"\r  blending {done:,}/{total:,}  {el:5.1f}s elapsed, "
              f"{eta:5.1f}s left", end="", flush=True)

    img = rasterize(p, background=tuple(args.bg),
                    max_splats=args.max_splats, progress=tick)
    print(f"\r  blended {n:,} splats in {time.time() - t0:.1f}s" + " " * 30)

    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f"render_{args.path.stem}_{AXES[axis]}{'-' if args.sign < 0 else '+'}.png"
    plt.imsave(dest, img)
    print(f"  wrote {dest}")

    covered = float((np.abs(img - np.array(args.bg)).sum(axis=2) > 0.01).mean())
    print(f"  {covered:.1%} of the frame has content")


if __name__ == "__main__":
    main()