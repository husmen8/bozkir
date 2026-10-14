"""Chosen candidates cut, levelled and centred into tile-ready patches."""

import numpy as np

from .patches import level_patch, stratified_keep
from .tile import extract_patch, merge, missing_layer


def even_mean(p):
    """A patch's mean colour from above (opacity-weighted base colour)."""
    w = np.asarray(p.opacity, dtype=np.float64)[:, None]
    return (p.base_rgb * w).sum(axis=0) / max(float(w.sum()), 1e-9)


def even_out(p, level, most):
    """`p` with its mean colour moved to `level`, by at most `most` per
    channel; detail and its own light and dark spots kept. Patches of one
    material that differ in tone show as the triangles and frames of the
    tiles built from them; every tile uses the same corrected patch, so
    edges still match."""
    from bozkir.ply import SH_C0
    shift = np.clip(np.asarray(level) - even_mean(p), -most, most)
    q = p.subset(np.arange(len(p)))
    q.sh_dc = ((np.clip(p.base_rgb + shift, 0, 1) - 0.5) / SH_C0).astype(np.float32)
    return q


def with_missing(wide, raw, centre, args, up):
    """`wide` (a candidate's wider cut, centred on `centre`) with what it
    is missing against the uncleaned capture from above (--missing-layer)."""
    plane = [j for j in range(3) if j != up]
    w = args.size * (1.0 + args.extract_margin)
    d = np.abs(raw.xyz[:, plane] - np.float32(centre)).max(axis=1)
    # Splats centred far off still paint here if they are big enough; ones
    # high above the ground (floaters, sky) do not belong to it.
    near = np.flatnonzero((d <= w / 2 + 3.0 * raw.scale.max(axis=1))
                          & (raw.xyz[:, up] <= wide.xyz[:, up].max() + 0.5))
    cap = raw.subset(near)
    cap.xyz = cap.xyz.copy()
    cap.xyz[:, plane] -= np.float32(centre)
    layer = missing_layer(wide, cap, w, up)
    print(f"    ({centre[0]:.2f}, {centre[1]:.2f}): {len(layer):,} splats "
          f"fill what the capture shows and the patch is missing")
    return merge(wide, layer) if len(layer) else wide


def prepare_patches(chosen, need, args, up, raw=None, scene=None, scale=1.0):
    """Cut, level and centre chosen candidates into tile-ready patches
    (run once per class). With `raw`, each gets its missing layer first.
    With `scale` under 1, each is cut from a square 1/scale as wide in
    `scene` and shrunk to the tile, splats and heights with it (see
    --class-scale)."""
    patches = []
    print(f"\n  {'#':>2} {'centre':>16} {'splats':>9} {'relief':>7} "
          f"{'tilt':>7} {'rim':>6} {'mid':>6} {'mean rgb':>18}")
    for i, (sc, (x, y), p, info) in enumerate(chosen[:need]):
        # Level the wider cut so rotation has material to draw in from,
        # then take the tile-sized middle of the result.
        wide = info.get("wide", p)
        if scale != 1.0 and scene is not None:
            # The slab around the ground's own plane, not a level one: a
            # 2-3 m cut on a slope lost its uphill and downhill sides to a
            # level slab, which showed as gaps.
            th = args.thickness or args.size * 0.25
            cut = info.get("cut", args.size / scale)
            raw_w = extract_patch(scene, [x, y], cut * (1.0 + args.extract_margin), up_axis=up)
            plane = [j for j in range(3) if j != up]
            ref = wide.xyz.astype(np.float64)
            # The ground's plane, not the canopy's: fitted to all, then again
            # to the lowest third. Through the bush top, the slab cut away
            # the ground around and under it: holes in every bush tile.
            A = np.c_[ref[:, plane], np.ones(len(ref))]
            coef = np.linalg.lstsq(A, ref[:, up], rcond=None)[0]
            low = ref[:, up] - A @ coef <= np.percentile(ref[:, up] - A @ coef, 33)
            if low.sum() >= 50:
                coef = np.linalg.lstsq(A[low], ref[low, up], rcond=None)[0]
            res = raw_w.xyz[:, up] - (raw_w.xyz[:, plane] @ coef[:2] + coef[2])
            wide = raw_w.subset((res >= -0.25 * th) & (res <= th))
        if raw is not None:
            wide = with_missing(wide, raw, (x, y), args, up)
        p, tilt = level_patch(wide, up)
        # Centre on the origin so every tile is assembled in the same frame.
        plane = [j for j in range(3) if j != up]
        off = np.zeros(3, dtype=np.float32)
        off[plane] = p.xyz[:, plane].mean(axis=0)
        q = p.subset(np.arange(len(p)))
        q.xyz = (q.xyz - off).astype(np.float32)
        if scale != 1.0:
            q.xyz = (q.xyz * scale).astype(np.float32)
            q.scale = (q.scale * scale).astype(np.float32)

        # Cap after levelling (which reads the wide cut), spread evenly.
        if args.max_per_tile and len(q) > args.max_per_tile:
            print(f"    patch {i} at ({x:.2f}, {y:.2f}): {len(q):,} splats "
                  f"capped to {args.max_per_tile:,}")
            q = q.subset(stratified_keep(q, args.max_per_tile,
                                         args.size * (1.0 + args.extract_margin),
                                         up))
        # Overhang kept: a Gaussian just outside still covers ground inside.
        plane2 = [j for j in range(3) if j != up]
        over = float((np.abs(q.xyz[:, plane2]).max(axis=1)
                      > args.size / 2).mean())
        patches.append(q)
        rgb = q.base_rgb.mean(axis=0)
        print(f"  {i:>2} ({x:6.2f},{y:6.2f}) {len(q):>9,} "
              f"{info['relief']:>7.2f} {tilt:>6.1f}° "
              f"{info.get('edge_relief', 0):>6.2f} "
              f"{info.get('interior_relief', 0):>6.2f}  "
              f"{rgb[0]:.2f} {rgb[1]:.2f} {rgb[2]:.2f}"
              f"  overhang {over:>4.0%}")
    return patches
