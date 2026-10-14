"""Materials in a capture: how many get tiles of their own, the share of
the map each gets, and how strictly they follow the ground's height."""

import numpy as np

from .patches import (appearance, cached_search, choose, class_shares, search_kwargs,
                      seen_points, split_materials, thickness)


def haze(c, up=2):
    """A candidate's splat layer thickness (m), measured once."""
    if "haze" not in c[3]:
        c[3]["haze"] = thickness(c[2], up)
    return c[3]["haze"]


def weigh_shares(shares, thick, weight):
    """The share each material gets in the viewer: its share of the ground
    times (thinnest layer / its layer) ** weight, so a material the capture
    holds only as haze shows up less without being dropped. 0 keeps the
    ground's shares."""
    ref = max(min(thick), 1e-4)
    w = [sh * (ref / max(t, 1e-4)) ** weight for sh, t in zip(shares, thick)]
    return [v / sum(w) for v in w]


def class_reasons(materials, shares, size, need, min_share, gap=0.04):
    """Why each material cannot be a class of its own (empty when it can):
    it must be a real part of the ground (`min_share` of it), have its own
    colour (mean rgb `gap` from the nearest other) and give `need` patches
    on ground of their own."""
    rgb = [np.mean([appearance(c[2])[:3] for c in g], axis=0) for g in materials]
    out = []
    for i, g in enumerate(materials):
        why = []
        if shares[i] < min_share:
            why.append(f"only {shares[i]:.0%} of the ground")
        near = min((float(np.abs(rgb[i] - rgb[j]).max()) for j in range(len(rgb)) if j != i),
                   default=1.0)
        if near < gap:
            why.append(f"colour only {near:.2f} from another")
        room = []
        for c in g:
            if all(abs(c[1][0] - x) >= size or abs(c[1][1] - y) >= size for x, y in room):
                room.append(c[1])
        if len(room) < need:
            why.append(f"only {len(room)} patches apart, {need} needed")
        out.append(why)
    return out


def choose_classes(s, args, up, thickness_cm, need):
    """How many materials to make tiles for, and why: the most (2 to 4)
    for which every one passes class_reasons. 1 when none does."""
    every = search_kwargs(args, thickness_cm)
    sep = min(every.pop("min_separation", 1.0), 0.25)
    found, st, _ = cached_search(s, args.size, up, every, cache=args.search_cache,
                                 where=True)
    seen = seen_points(found, st) if args.seen_shares else None
    pos = choose(found, len(found), args.size, sep)
    least = args.min_share or 0.1
    print(f"\n  how many materials? each must be {least:.0%} of the ground or more, "
          f"have its own colour and give {need} patches apart:")
    best = 1
    for k in (2, 3, 4):
        mats = split_materials(pos, k)
        sh = class_shares(s, mats, args.size, up, seen=seen)
        why = class_reasons(mats, sh, args.size, need, least)
        bad = [f"material {i} ({sh[i]:.0%}): {', '.join(w)}" for i, w in enumerate(why) if w]
        print(f"    {k}: " + (" / ".join(f"{v:.0%}" for v in sorted(sh, reverse=True))
                              + "  ok" if not bad else "no - " + "; ".join(bad)))
        if not bad:
            best = k
    print(f"  -> {best} material{'s' if best > 1 else ''}")
    return best


def height_order(groups, up=2, radius=3.0):
    """How strictly the materials, in order (hollows first), follow the
    ground's height in the capture. Each position's ground (its splats'
    lower quartile) against the mean of the positions within `radius`;
    for each two materials in a row, the share of position pairs the wrong
    way round (Mann-Whitney's U over all pairs, ties half). 0: strictly by
    height; 0.5: no relation. None with nothing to compare."""
    from scipy.spatial import cKDTree
    from scipy.stats import rankdata
    xy = np.array([c[1][:2] for g in groups for c in g], np.float64)
    z = np.array([np.percentile(c[2].xyz[:, up], 25) if len(c[2]) else np.nan
                  for g in groups for c in g])
    lab = np.concatenate([[k] * len(g) for k, g in enumerate(groups)]) if len(xy) else []
    ok = np.isfinite(z)
    if ok.sum() < 4:
        return None
    tree = cKDTree(xy[ok])
    rel = np.full(len(z), np.nan)
    rel[ok] = z[ok] - np.array([z[ok][nb].mean() for nb in
                                tree.query_ball_point(xy[ok], radius)])
    wrong = total = 0.0
    for k in range(len(groups) - 1):
        a, b = rel[(lab == k) & ok], rel[(lab == k + 1) & ok]
        if not len(a) or not len(b):
            continue
        r = rankdata(np.concatenate([a, b]))[:len(a)]
        wrong += r.sum() - len(a) * (len(a) + 1) / 2.0        # pairs with a above b
        total += len(a) * len(b)
    return float(wrong / total) if total else None
