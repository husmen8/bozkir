"""Straighten a 3DGS scene so the ground lies in the XY plane.

Run from the repo root:

    python scripts/align.py data/raw/garden.ply
    python scripts/align.py data/raw/garden.ply --render

Writes data/aligned/<name>.ply. Everything downstream should use that file,
not the raw one.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bozkir.presets import add_preset_args, apply as apply_preset  # noqa: E402
from bozkir.ply import load_ply, save_ply  # noqa: E402
from bozkir.transform import align_to_ground, recentre  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--axis", type=int, default=2, choices=(0, 1, 2),
                    help="which axis the ground normal should end up on")
    ap.add_argument("--no-recentre", action="store_true")
    ap.add_argument("--render", action="store_true",
                    help="also render before/after top-down views")
    ap.add_argument("--res", type=int, default=600)
    ap.add_argument("--out", type=Path, default=Path("data/aligned"))
    add_preset_args(ap)
    args = apply_preset(ap)

    s = load_ply(args.path)
    print(f"{args.path.name}: {len(s):,} splats")

    t0 = time.time()
    out, info = align_to_ground(s, target_axis=args.axis)

    print(f"\n  ground normal before  ({info['normal_before'][0]:+.3f}, "
          f"{info['normal_before'][1]:+.3f}, {info['normal_before'][2]:+.3f})")
    print(f"  ground normal after   ({info['normal_after'][0]:+.3f}, "
          f"{info['normal_after'][1]:+.3f}, {info['normal_after'][2]:+.3f})")
    print(f"  tilt   {info['tilt_before_deg']:.2f} deg  ->  "
          f"{info['tilt_after_deg']:.3f} deg")
    print(f"  planarity {info['planarity']:.5f}  ->  {info['planarity_after']:.5f}")
    q = info["quaternion"]
    print(f"  rotation quaternion  ({q[0]:+.4f}, {q[1]:+.4f}, "
          f"{q[2]:+.4f}, {q[3]:+.4f})")

    if not args.no_recentre:
        out, offset = recentre(out, "ground")
        print(f"  recentred by  {np.round(offset, 3)}")

    lo = np.percentile(out.xyz, 1, axis=0)
    hi = np.percentile(out.xyz, 99, axis=0)
    print(f"  extent (1-99 pct)  {np.round(hi - lo, 2)}")
    print(f"  aligned in {time.time() - t0:.1f}s")

    # Sanity: rotation must not change any splat's shape, only its
    # orientation. Comparing eigenvalues to themselves is ill-conditioned -
    # the thinnest splats have eigenvalues near the floor of float64 and
    # some round below zero. Comparing the covariance directly against
    # R Sigma R^T, scaled by the largest value present, is well behaved.
    from bozkir.ply import quat_to_matrix
    R = quat_to_matrix(info["quaternion"])[0].astype(np.float64)
    idx = np.random.default_rng(0).choice(
        len(s), size=min(50_000, len(s)), replace=False)
    C0 = s.covariance()[idx].astype(np.float64)
    C1 = out.covariance()[idx].astype(np.float64)
    err = np.abs(C1 - R @ C0 @ R.T).max() / max(np.abs(C0).max(), 1e-30)
    print(f"  shape check: covariance error {err:.1e} "
          f"({'ok' if err < 1e-4 else 'FAIL - orientations are wrong'})")

    tr0 = np.trace(C0, axis1=1, axis2=2)
    tr1 = np.trace(C1, axis1=1, axis2=2)
    tr = (np.abs(tr1 - tr0) / np.maximum(tr0, 1e-30)).max()
    print(f"  size check:  trace drift {tr:.1e} "
          f"({'ok' if tr < 1e-4 else 'FAIL - splats changed size'})")

    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / args.path.name
    save_ply(out, dest)
    print(f"\n  wrote {dest}")

    if args.render:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from bozkir.render import render_orthographic

        fig = Path("out")
        fig.mkdir(parents=True, exist_ok=True)
        # After alignment the ground is at z ~ 0 with content above it, so
        # the camera belongs at +z looking down: nearer means larger z,
        # which is up_sign = +1. Getting this backwards renders the scene
        # from underground, where the ground hides everything.
        for label, scene, axis, sign in (("before", s, 1, -1),
                                         ("after", out, args.axis, +1)):
            t = time.time()
            img, p = render_orthographic(scene, view_axis=axis, up_sign=sign,
                                         resolution=args.res, sh_degree=0,
                                         background=(0, 0, 0))
            path = fig / f"align_{args.path.stem}_{label}.png"
            plt.imsave(path, img)
            print(f"  rendered {label} ({p['W']}x{p['H']}) "
                  f"in {time.time() - t:.0f}s -> {path}")


if __name__ == "__main__":
    main()