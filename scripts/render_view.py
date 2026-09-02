"""Render a 3DGS scene through a perspective camera.

Run from the repo root, on the ALIGNED scene:

    python scripts/render_view.py data/aligned/garden.ply
    python scripts/render_view.py data/aligned/garden.ply --elev 5 --azim 120
    python scripts/render_view.py data/aligned/garden.ply --turntable 8

Elevation 0 looks along the ground (grazing), 90 looks straight down.
Assumes z is up, which is what scripts/align.py produces.
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
from bozkir.camera import orbit_camera, project_perspective  # noqa: E402
from bozkir.render import rasterize  # noqa: E402


def frame(s, up_axis=2):
    """Pick a target point and distance that put the dense part on screen.

    Uses the interquartile range rather than the full extent: floaters are
    a few percent of the splats but most of the bounding box, and framing
    on them puts the actual subject in the middle distance.
    """
    q1 = np.percentile(s.xyz, 25, axis=0)
    q3 = np.percentile(s.xyz, 75, axis=0)
    target = (q1 + q3) / 2.0
    plane = [i for i in range(3) if i != up_axis]
    span = float(np.max((q3 - q1)[plane]))
    return target.astype(np.float32), max(span * 2.5, 1e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--azim", type=float, default=45.0)
    ap.add_argument("--elev", type=float, default=25.0)
    ap.add_argument("--dist", type=float, default=None,
                    help="camera distance (default: auto from scene size)")
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--bg", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--max-splats", type=int, default=None)
    ap.add_argument("--turntable", type=int, default=0,
                    help="render N frames evenly spaced in azimuth")
    ap.add_argument("--out", type=Path, default=Path("out"))
    add_scene_args(ap)
    args = ap.parse_args()

    s = scene_from_args(args)
    target, auto_dist = frame(s, args.up_axis)
    dist = args.dist if args.dist is not None else auto_dist

    print(f"  target {np.round(target, 2)}   distance {dist:.2f}   fov {args.fov:.0f}")

    args.out.mkdir(parents=True, exist_ok=True)
    azims = ([args.azim] if args.turntable <= 0
             else list(np.linspace(0, 360, args.turntable, endpoint=False)))

    for i, az in enumerate(azims):
        cam = orbit_camera(target, dist, az, args.elev, up_axis=args.up_axis,
                           fov_deg=args.fov, width=args.width, height=args.height)

        t0 = time.time()
        p = project_perspective(cam, s)
        t_proj = time.time() - t0

        n = p["kept"] if args.max_splats is None else min(p["kept"], args.max_splats)
        t0 = time.time()

        def tick(done, total):
            el = time.time() - t0
            print(f"\r    blending {done:,}/{total:,}  {el:5.1f}s elapsed, "
                  f"{el / max(done, 1) * (total - done):5.1f}s left",
                  end="", flush=True)

        img = rasterize(p, background=tuple(args.bg),
                        max_splats=args.max_splats, progress=tick)
        t_blend = time.time() - t0

        stem = (f"view_{args.path.stem}_az{int(az):03d}_el{int(args.elev):02d}"
                if args.turntable <= 0 else
                f"turn_{args.path.stem}_{i:03d}")
        dest = args.out / f"{stem}.png"
        plt.imsave(dest, img)

        covered = float((np.abs(img - np.array(args.bg)).sum(axis=2) > 0.01).mean())
        print(f"\r  az {az:5.1f} el {args.elev:4.1f}: "
              f"{p['kept']:,} visible, {p['behind']:,} behind camera, "
              f"{covered:.0%} of frame covered   "
              f"({t_proj:.1f}s + {t_blend:.0f}s)")
        print(f"    -> {dest}")


if __name__ == "__main__":
    main()