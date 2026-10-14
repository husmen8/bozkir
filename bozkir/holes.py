"""See-through holes from above, in a candidate or a finished tile, and
the opacity fill that closes the faint ones."""

import json
from pathlib import Path

import numpy as np

from . import frames as frames_mod
from .tile import fill_opacity, haze_splats, merge


def holes(p, size, up=2, res=60, min_cm=8.0, ring=None, shape=2.0):
    """See-through holes in a patch or tile from above: spots under half
    covered, at least `min_cm` across, within `ring` (r_min, r_max) of its
    middle if given (a frame, a centre). Returns (count, largest in cm^2,
    mask of the holes)."""
    from bozkir.graphcut import render_patch
    from scipy.ndimage import label
    if not len(p):
        return 1, size * size * 1e4, np.ones((res, res), bool)
    a = render_patch(p, size, res, up)[1]
    see = a < 0.5
    if ring is not None:
        see &= frames_mod.zone(res, size, *ring, shape=shape)
    lab, n = label(see)
    px = (size * 100.0 / res) ** 2
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:] * px
    big = [k + 1 for k, v in enumerate(sizes) if v >= min_cm ** 2]
    mask = np.isin(lab, big)
    return len(big), float(max((sizes[k - 1] for k in big), default=0.0)), mask


# hole_free's answers, kept in data/cache between runs: --look and the
# build after it ask about the same patches, and each answer is two renders.
_HOLED = {"path": None, "data": {}, "new": 0}


def holed_cache(s, thickness, margin, use=True):
    """Load the answers for this prepared capture and cut (none if `use` is off)."""
    import hashlib
    from bozkir.patches import scene_fingerprint
    _HOLED.update(path=None, data={}, new=0)
    if not use:
        return
    key = hashlib.sha1(f"v1|{scene_fingerprint(s)}|{thickness}|{margin}".encode()).hexdigest()
    _HOLED["path"] = Path("data/cache") / f"holes_{key[:16]}.json"
    try:
        _HOLED["data"] = json.loads(_HOLED["path"].read_text())
    except (OSError, ValueError):
        pass


def save_holed():
    if _HOLED["path"] is not None and _HOLED["new"]:
        try:
            _HOLED["path"].parent.mkdir(parents=True, exist_ok=True)
            _HOLED["path"].write_text(json.dumps(_HOLED["data"]))
            _HOLED["new"] = 0
        except OSError:
            pass


def filled(t, size, up=2, target=0.98, spare_haze=False):
    """The opacity fill (tile.fill_opacity), sparing large faint splats
    above the ground with `spare_haze` (--fill-spare-haze)."""
    return fill_opacity(t, size, up, target,
                        spare=haze_splats(t, size, up) if spare_haze else None)


def hole_free(c, size, up=2, fill=0.0, spare_haze=False):
    """True when a candidate has no see-through hole from above once its
    opacity is filled as its tile's will be (filled). Holes passed by
    the search (faint, gappy ground) showed in every tile built from the
    patch: on the desert, 16 of 18 bush tiles and 11 mid-sand ones. Measured once, and
    remembered between runs (holed_cache)."""
    if "holed" not in c[3]:
        key = (f"{c[1][0]:.4f},{c[1][1]:.4f},{size:.4f},{fill:g},{len(c[2])},{up}"
               + (",haze" if spare_haze else ""))
        if _HOLED["path"] is None:
            _HOLED["data"].pop(key, None)            # no capture to remember it by
        if key not in _HOLED["data"]:
            p = c[2]
            if fill > 0:
                p = filled(p, size, up, fill, spare_haze)
            _HOLED["data"][key] = bool(holes(p, size, up)[0] > 0)
            _HOLED["new"] += 1
        c[3]["holed"] = _HOLED["data"][key]
    return not c[3]["holed"]


def with_neighbours(tiles, codes, tile_class, size, up=2, band=0.1):
    """Each tile with a `band` of its neighbours' ground past each edge (from
    any tile of its material with that edge colour on the other side; they
    all share that edge): a tile seen alone is thin along its edges, since
    the splats just past them belong to the neighbour, and a faint spot
    there read as a hole that the world does not have."""
    from bozkir.tile import translate
    plane = [i for i in range(3) if i != up]
    h = size / 2.0
    # (edge, its code column, the other side's column, which way, along which axis)
    sides = (("n", 0, 2, 1, 1), ("e", 1, 3, 1, 0), ("s", 2, 0, -1, 1), ("w", 3, 1, -1, 0))
    out = []
    for t, c, k in zip(tiles, codes, tile_class):
        parts = [t]
        for _, mine, theirs, sign, axis in sides:
            other = next((u for u, d, q in zip(tiles, codes, tile_class)
                          if q == k and d[theirs] == c[mine]), None)
            if other is None or not len(other):
                continue
            a = other.xyz[:, plane[axis]] * -sign            # how far in from its edge
            near = np.flatnonzero(a >= h - band)
            if len(near):
                off = np.zeros(3)
                off[plane[axis]] = sign * size
                parts.append(translate(other.subset(near), off))
        out.append(merge(*parts) if len(parts) > 1 else t)
    return out


def hole_report(tiles, codes, tile_class, size, up, dest, ring=None, shape=2.0):
    """Which finished tiles have see-through holes, per material, and a
    picture of them (holes in red). `ring`, one per tile: only there (a
    frame's band; None for a whole tile). Each tile is seen with its
    neighbours' ground just past its edges (with_neighbours)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from bozkir.graphcut import render_patch
    seen = with_neighbours(tiles, codes, tile_class, size, up)
    found = [holes(t, size, up, ring=r, shape=shape)
             for t, r in zip(seen, ring or [None] * len(tiles))]
    print("\n  holes from above (8 cm or more across, under half covered):")
    for k in sorted(set(tile_class)):
        ids = [i for i, c in enumerate(tile_class) if c == k]
        bad = [i for i in ids if found[i][0]]
        worst = max((found[i][1] for i in bad), default=0.0)
        print(f"    class {k}: {len(bad)} of {len(ids)} tiles"
              + (f", largest {worst:.0f} cm2" if bad else ""))
    bad = [i for i, f in enumerate(found) if f[0]]
    if not bad:
        return
    cols = 9
    rows = -(-len(bad) // cols)
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.8))
    ax = np.atleast_2d(ax)
    for a in ax.flat:
        a.axis("off")
    for j, i in enumerate(bad):
        rgb, al = render_patch(seen[i], size, 60, up)
        img = np.clip(rgb * al[..., None], 0, 1)
        img[found[i][2]] = [1.0, 0.1, 0.1]
        ax.flat[j].imshow(img)
        ax.flat[j].set_title(f"class {tile_class[i]} tile {i} {''.join(map(str, codes[i]))}",
                             fontsize=6)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=110)
    plt.close(fig)
    print(f"  the tiles with holes: {dest}")
