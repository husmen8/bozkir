"""Where does a patch lose its ground? Follows one patch from the raw
capture through each preparation step and renders it straight down after
every step, holes in magenta.

    python scripts/probe_patch.py data/raw/desert_v1.ply --preset desert3
    python scripts/probe_patch.py data/raw/desert_v1.ply --preset desert3 --at 5.47,5.33

--at is a centre from the preview sheet (same frame: alignment and
recentring come before cleaning). Panels, left to right:

  1 raw, colour as seen from above (full SH): closest to Brush top-down
  2 raw, base colour only (what the exporter keeps)
  3 after the size cut (max_extent_pct)
  4 after floater removal (floater_std)
  5 after the ground slab (thickness)
  6 centres outside the square dropped: what the preview sheet shows

Read-only: nothing is cached or written except out/probe_<name>_<x>_<y>.png.
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
from bozkir.ply import Splats, SH_C0, load_ply, quat_to_matrix  # noqa: E402
from bozkir.presets import add_preset_args, apply as apply_preset  # noqa: E402
from bozkir.render import project_orthographic, rasterize_rgba  # noqa: E402
from bozkir.scene import add_scene_args  # noqa: E402
from bozkir.select import cap_height, min_size, remove_floaters  # noqa: E402
from bozkir.transform import align_to_ground, ground_normal, recentre  # noqa: E402
from bozkir.patches import ground_level  # noqa: E402

HOLE = np.float32([1.0, 0.0, 1.0])


def ink(s, up_axis):
    """Opacity times area seen from above, per splat: how much ground each
    one paints."""
    plane = [i for i in range(3) if i != up_axis]
    c = s.covariance()[:, plane][:, :, plane]
    det = c[:, 0, 0] * c[:, 1, 1] - c[:, 0, 1] ** 2
    return np.pi * np.sqrt(np.maximum(det, 0.0)) * s.opacity


def render(s, size, res, up_axis):
    """Top-down render of the square, base colour; returns (rgb with holes
    magenta, mean coverage)."""
    half = size / 2.0
    p = project_orthographic(s, view_axis=up_axis, up_sign=+1,
                             resolution=res, sh_degree=0,
                             bounds=((-half, -half), (half, half)))
    rgba = rasterize_rgba(p)
    a = rgba[..., 3:]
    rgb = np.clip(rgba[..., :3] + HOLE * (1.0 - a), 0.0, 1.0)
    return rgb, float(a.mean())


def colour_from_above(s, q, up_axis):
    """Full-SH colour for a camera straight above. SH stay in the capture's
    original frame when the scene is rotated, so the view direction goes
    back through the alignment. 3DGS convention: direction from camera to
    splat, i.e. down."""
    down = np.zeros(3, dtype=np.float32)
    down[up_axis] = -1.0
    R = quat_to_matrix(np.asarray(q, dtype=np.float32).reshape(4))[0]
    return s.rgb(R.T @ down)


def with_colour(s, rgb):
    """A copy whose DC term reproduces `rgb`, so the DC renderer draws it."""
    return Splats(xyz=s.xyz, opacity=s.opacity, scale=s.scale, rot=s.rot,
                  sh_dc=((rgb - 0.5) / SH_C0).astype(np.float32),
                  sh_rest=np.zeros((len(s), 0, 3), dtype=np.float32),
                  sh_degree=0)


def probe(s, q, args, centre):
    """Masks over `s` (aligned, recentred) for each step, the pipeline's
    own thresholds computed over the whole scene as prepare() does."""
    up = args.up_axis
    plane = [i for i in range(3) if i != up]
    n = len(s)

    # 1. the crop (radius_pct), as in scene.prepare
    mid = np.median(s.xyz, axis=0)
    d = np.linalg.norm(s.xyz[:, plane] - mid[plane], axis=1)
    crop = d <= float(np.percentile(d, args.radius_pct))

    # 2. size cut, threshold from the cropped scene
    ext = s.scale.max(axis=1)
    limit = float(np.percentile(ext[crop], args.max_extent_pct))
    big = ext > limit
    print(f"  size cut: splats longer than {limit * 100:.1f} cm (scale axis) "
          f"are dropped, {big[crop].sum():,} in the scene")
    after_size = crop & ~big

    # 3. floaters, over everything that survived the size cut
    after_float = after_size.copy()
    if args.floater_std > 0:
        idx = np.flatnonzero(after_size)
        _, keep = remove_floaters(s.subset(idx), std_ratio=args.floater_std)
        after_float[idx[~keep]] = False

    # The column: generous margin so splats centred outside but painting
    # inside are counted, as Brush draws them.
    rel = s.xyz[:, plane] - np.asarray(centre, dtype=np.float32)
    half = args.size / 2.0
    col = np.all(np.abs(rel) <= half + args.margin, axis=1)
    square = np.all(np.abs(rel) <= half, axis=1)

    def cut(mask):
        idx = np.flatnonzero(mask)
        sub = s.subset(idx)
        sub.xyz = sub.xyz.copy()
        sub.xyz[:, plane] -= np.asarray(centre, dtype=np.float32)
        return sub, idx

    raw, raw_idx = cut(col)
    if len(raw):
        e = raw.scale.max(axis=1)
        print(f"  this column: median splat {np.median(e) * 100:.1f} cm, "
              f"largest {e.max() * 100:.0f} cm")
    steps = [("raw", col)]
    steps.append(("size cut", col & after_size))
    steps.append(("floaters", col & after_float))

    # 4. slab, on what is left in the column, as pick_patches does
    p4, idx4 = cut(col & after_float)
    if len(p4):
        h = p4.xyz[:, up]
        g = ground_level(h, args.thickness)
        slab_keep = (h >= g - args.thickness * 0.25) & (h <= g + args.thickness)
        slab = np.zeros(n, dtype=bool)
        slab[idx4[slab_keep]] = True
    else:
        slab = np.zeros(n, dtype=bool)
    steps.append(("slab", slab))
    steps.append(("square", slab & square))

    above = colour_from_above(raw, q, up) if s.sh_degree > 0 else raw.base_rgb
    return raw, raw_idx, above, steps, crop


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    ap.add_argument("--at", default="1.87,10.13",
                    help="patch centre x,y from the preview sheet")
    ap.add_argument("--size", type=float, default=1.5)
    ap.add_argument("--thickness", type=float, default=None,
                    help="ground slab (default: size / 4)")
    ap.add_argument("--margin", type=float, default=0.5,
                    help="metres around the square whose splats are counted")
    ap.add_argument("--res", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("out"))
    add_scene_args(ap)
    add_preset_args(ap)
    args = apply_preset(ap)
    if args.thickness is None:
        args.thickness = args.size * 0.25
    centre = [float(v) for v in args.at.split(",")]

    t0 = time.time()
    s = load_ply(args.path)
    print(f"{args.path.name}: {len(s):,} splats, SH degree {s.sh_degree}")
    hint = -ground_normal(s)[0] if args.flip else None
    s, info = align_to_ground(s, target_axis=args.up_axis, up_hint=hint)
    s, _ = recentre(s, "ground" if args.up_axis == 2 else "median")

    raw, raw_idx, above, steps, crop = probe(s, info["quaternion"], args, centre)
    # After the masks: the pipeline cuts on the original sizes too.
    if args.max_height > 0:
        raw = cap_height(raw, args.max_height / 100.0, args.up_axis)
        print(f"  splats flattened to at most {args.max_height:g} cm tall")
    if args.min_splat > 0:
        raw = min_size(raw, args.min_splat / 100.0)
        print(f"  splats widened to at least {args.min_splat:g} cm")
    if not len(raw):
        raise SystemExit(f"no splats near {args.at}: is it a centre from "
                         "this capture's preview sheet?")
    if not crop[raw_idx].all():
        print(f"  note: {np.mean(~crop[raw_idx]):.0%} of this column is "
              "outside the radius crop")

    up = args.up_axis
    w_all = ink(raw, up)

    panels = []
    img, cov = render(with_colour(raw, above), args.size, args.res, up)
    panels.append(("1 raw, colour from above", img, len(raw), cov))
    print(f"\n  {'step':<28}{'splats':>9}{'ink kept':>10}{'cover':>8}")
    print(f"  {'1 raw (full SH)':<28}{len(raw):>9,}{'100%':>10}{cov:>8.0%}")

    names = {"raw": "2 raw, base colour", "size cut": "3 after size cut",
             "floaters": "4 after floaters", "slab": "5 after slab",
             "square": "6 square (sheet)"}
    for name, mask in steps:
        # Every step is a subset of the column, so positions in raw_idx.
        keep = np.searchsorted(raw_idx, np.flatnonzero(mask))
        sub = raw.subset(keep)
        img, cov = render(sub, args.size, args.res, up)
        # Ink is over the whole column, so it means nothing once the
        # column is cut down to the square.
        share = ("-" if name == "square" else
                 f"{w_all[keep].sum() / max(w_all.sum(), 1e-12):.0%}")
        panels.append((names[name], img, len(sub), cov))
        print(f"  {names[name]:<28}{len(sub):>9,}{share:>10}{cov:>8.0%}")

    drops = [(panels[i][3] - panels[i + 1][3], panels[i + 1][0])
             for i in range(1, len(panels) - 1)]
    loss, label = max(drops)
    if loss > 0:
        print(f"\n  biggest loss of cover: '{label}' (-{loss:.0%})")

    fig, ax = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.7))
    for a, (label, img, cnt, cov) in zip(ax, panels):
        a.imshow(img, interpolation="nearest")
        a.set_title(f"{label}\n{cnt:,} splats, cover {cov:.0%}", fontsize=8)
        a.axis("off")
    fig.suptitle(f"{args.path.stem} at ({centre[0]:.2f}, {centre[1]:.2f}), "
                 f"{args.size} m, magenta = no ground", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f"probe_{args.path.stem}_{centre[0]:.2f}_{centre[1]:.2f}.png"
    fig.savefig(dest, dpi=130)
    print(f"\n  wrote {dest}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
