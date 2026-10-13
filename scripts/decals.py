"""Decals for a finished tileset: distinct pieces of the capture (a shrub
and its shadow) baked into copies of the tileset's plain tiles.

    python scripts/decals.py bake web/data/desert_base -o web/data/desert_final
    python scripts/decals.py bake web/data/desert_base --preset desert4 -o web/data/desert_final
    python scripts/decals.py sheet web/data/desert_base
    python scripts/bake_atlas.py desert_final

Without --pick, bake takes the best --decals candidates itself and draws
them to out/decals_<new name>.png to look over. `sheet` shows every
candidate, numbered, for picking by hand (--pick 3,7,12). Flags can come
from a preset (decal_radius, decal_auto, ...). scripts/build.py runs this
after the export.

Both steps read the capture and its preparation from the tileset's own
settings, so the sheet's numbers always mean the same discs in the bake;
decals skip the tiles' --max-height flattening, which in a shrub would
flatten the shrub. Every tile of --class gets --copies copies, each with a
different decal at a random spot clear of the edges; the viewer swaps
copies in now and then ("decals" under materials). The tileset given is not
changed.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bozkir import decals as dec  # noqa: E402
from bozkir.catalogue import rebuild as reindex  # noqa: E402
from bozkir.pack import pack, read_splat, tile_levels  # noqa: E402
from bozkir.presets import add_preset_args, apply as apply_preset  # noqa: E402
from bozkir.scene import scene_from_args  # noqa: E402
from bozkir.tile import fill_opacity as _fill_opacity, haze_splats  # noqa: E402


def capture_of(manifest, override=None):
    """The prepared capture a tileset was made from, as its settings say,
    but not flattened: --max-height is for bare ground's spikes, and a
    shrub's upright splats are the shrub."""
    st = manifest.get("settings", {})
    path = Path(override or st.get("path", ""))
    if not path.exists():
        raise SystemExit(f"capture {path} not found; pass --capture")
    ns = SimpleNamespace(
        path=path, raw=st.get("raw", False), clean=st.get("clean", False),
        radius_pct=st.get("radius_pct", 60.0),
        max_extent_pct=st.get("max_extent_pct", 99.0),
        floater_std=st.get("floater_std", 2.0), sh=st.get("sh"),
        up_axis=st.get("up_axis", 2), flip=st.get("flip", False),
        no_cache=False, max_height=0.0,
        min_splat=st.get("min_splat", 0.0))
    return scene_from_args(ns)


def find_or_explain(s, radius, up, count):
    """The candidates, or why there are none."""
    tally = {}
    found = dec.find(s, radius, up, count=count, tally=tally)
    if not found:
        print(f"  no candidates among {tally.get('positions', 0):,} positions; "
              f"turned away by each test:")
        for k, v in tally.items():
            if k != "positions":
                print(f"    {k:<26} {v:,}")
        raise SystemExit("  a smaller --radius finds more; so does a capture "
                         "with more bare ground between features")
    return found


def draw(s, found, labels, radius, up, title, dest):
    """Discs from above, one per entry of `found`, titled by `labels`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from bozkir.graphcut import render_patch

    plane = [i for i in range(3) if i != up]
    side = 2.4 * radius
    cols = 6
    rows = max(1, -(-len(found) // cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    ax = np.atleast_2d(ax)
    for a in ax.flat:
        a.axis("off")
    for k, (x, y, sc) in enumerate(found):
        d = s.xyz[:, plane] - np.float32([x, y])
        near = np.flatnonzero(np.all(np.abs(d) <= side / 2 + 0.2, axis=1))
        p = s.subset(near)
        p.xyz = p.xyz.copy()
        p.xyz[:, plane] -= np.float32([x, y])
        rgb, al = render_patch(p, side, 120, up)
        a = ax.flat[k]
        a.imshow(np.clip(rgb * al[..., None], 0, 1))
        a.add_patch(plt.Circle((60, 60), 60 * radius / (side / 2),
                               fill=False, color="w", lw=0.8, ls="--"))
        a.set_title(f"{labels[k]}  ({x:.1f}, {y:.1f})", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=110)
    plt.close(fig)


def sheet(args, m):
    s = capture_of(m, args.capture)
    up = m.get("up_axis", 2)
    found = find_or_explain(s, args.decal_radius, up, args.decal_count)
    print(f"  {len(found)} candidate decals of radius {args.decal_radius} m")
    for k, (x, y, sc) in enumerate(found):
        print(f"  {k:>3} ({x:6.2f}, {y:6.2f})  score {sc:.3f}")
    stem = args.tileset.name
    dest = args.out_dir / f"decals_{stem}.png"
    draw(s, found, list(range(len(found))), args.decal_radius, up,
         f"{stem}: candidate decals, radius {args.decal_radius} m "
         f"(dashed: what is kept)", dest)
    print(f"\n  wrote {dest}\n  bake picks the best {args.decal_auto} itself; "
          f"to choose by hand: --pick 3,7,12")


def fill_opacity(v, size, up, fill, st):
    """The export's fill, sparing haze when its settings did
    (--fill-spare-haze)."""
    return _fill_opacity(v, size, up, fill,
                         spare=haze_splats(v, size, up) if st.get("fill_spare_haze") else None)


def into_centres(args, m, all_splats, pieces, parts, cursor, fill, lod, compensate):
    """Frames: a decal goes into a copy of a pool centre, in its middle (the
    frames, and so the matching, are untouched). Returns (the centres with
    the copies after them, cursor, copies made, centres copied)."""
    size, up = m["size"], m.get("up_axis", 2)
    st = m.get("settings", {})
    if args.decal_radius > m["frames"]["radius"] + 1e-6:
        print(f"  ! a {args.decal_radius:g} m decal is wider than the centres "
              f"({m['frames']['radius']:g} m): its rim blends into the frame")
    centres = list(m["centres"])
    plain = [i for i, c in enumerate(m["centres"])
             if c.get("base") is None and c.get("class", 0) == args.decal_class]
    if not plain:
        raise SystemExit(f"no centres of class {args.decal_class} in {args.tileset.name}")
    made = 0
    for n, i in enumerate(plain):
        a, c = m["centres"][i]["levels"][0]
        base = all_splats.subset(np.arange(a, a + c))
        for k in range(args.decal_copies):
            j = (n * args.decal_copies + k) % len(pieces)
            v = dec.place(base, pieces[j], args.decal_radius, size, up)
            if fill > 0:
                # The decal's own faint parts filled; the centre's rim as it
                # was (a fill there would thicken it).
                full = fill_opacity(v, size, up, fill, st)
                plane = [q for q in range(3) if q != up]
                inside = np.hypot(*v.xyz[:, plane].T) < args.decal_radius
                v.opacity = np.where(inside, full.opacity, v.opacity).astype(np.float32)
            levels = []
            for part in tile_levels(v, lod, size, up, compensate,
                                    bool(st.get("level_tone"))):
                parts.append(pack(part).reshape(-1))
                levels.append([cursor, len(part)])
                cursor += len(part)
            centres.append({"start": levels[0][0], "count": levels[0][1], "levels": levels,
                            "class": m["centres"][i].get("class", 0), "base": i, "decal": j})
            made += 1
    return centres, cursor, made, len(plain)


def bake(args, m, raw):
    if not args.out:
        raise SystemExit("-o is needed: a new tileset name, so the given one stays")
    s = capture_of(m, args.capture)
    up = m.get("up_axis", 2)
    size = m["size"]
    st = m.get("settings", {})
    thickness = m.get("thickness") or size * 0.25
    found = find_or_explain(s, args.decal_radius, up, args.decal_count)
    picks = [int(v) for v in str(args.decal_pick).replace(" ", "").split(",") if v]
    if not picks:
        # Best first, and already apart (find keeps two radii between them).
        # Past a quarter of the best score a disc is mostly plain ground.
        picks = [k for k in range(min(args.decal_auto, len(found)))
                 if found[k][2] >= 0.25 * found[0][2]]
        print(f"  picked the best {len(picks)} of {len(found)} candidates"
              + (f" (the rest are mostly plain ground)" if len(picks) < args.decal_auto else ""))
    if max(picks) >= len(found):
        raise SystemExit(f"--pick {max(picks)} is past the {len(found)} on the sheet")
    pieces = []
    for k in picks:
        x, y, _ = found[k]
        pieces.append(dec.cut(s, (x, y), args.decal_radius, up, thickness))
        print(f"  decal {k} at ({x:.2f}, {y:.2f}): {len(pieces[-1]):,} splats")
    # Numbered in colour order, greenest first, so near numbers look alike
    # and the viewer can group similar shrubs without repeating one.
    green = [float(np.average(p.base_rgb[:, 1] - p.base_rgb[:, 0], weights=p.opacity))
             for p in pieces]
    order = sorted(range(len(picks)), key=lambda q: -green[q])
    picks = [picks[q] for q in order]
    pieces = [pieces[q] for q in order]

    _, all_splats = read_splat(args.tileset.with_suffix(".splat"),
                               args.tileset.with_suffix(".json"))
    if args.decal_class < 0:
        # The material with the largest share: shrubs stand on open ground.
        # Class 0 was that until desert10 ordered materials by tone, and the
        # shrubs went into the bush where nobody saw them.
        sh = m.get("shares") or [1.0]
        args.decal_class = int(np.argmax(sh))
        print(f"  decals go on class {args.decal_class}, the largest share ({sh[args.decal_class]:.0%})")
    fill = float(st.get("fill_opacity") or 0.0)
    lod = max(int(m.get("lod", 1)), 1)
    compensate = bool(st.get("lod_compensate", True))
    parts = [raw]
    cursor = int(m["total"])
    tiles, centres, plain = list(m["tiles"]), None, []
    # Frames: into the class's centres; a class that kept whole tiles (its
    # texture runs one way) takes them in its tiles, as without frames.
    if any(c.get("base") is None and c.get("class", 0) == args.decal_class
           for c in m.get("centres", [])):
        centres, cursor, made, copied = into_centres(args, m, all_splats, pieces, parts,
                                                     cursor, fill, lod, compensate)
        print(f"  {made} copies of {copied} class {args.decal_class} centres, "
              f"{len(picks)} decals, each in the middle of its centre (frames: the "
              f"viewer turns it and puts it in a cell now and then)")
    else:
        plain = [i for i, t in enumerate(m["tiles"])
                 if t.get("base") is None and t.get("class", 0) == args.decal_class]
        if not plain:
            raise SystemExit(f"no plain tiles of class {args.decal_class} in {args.tileset.name}")
    rng = np.random.default_rng(args.decal_seed)
    reach = dec.room(args.decal_radius, size)
    made = 0 if plain else made
    for n, i in enumerate(plain):
        t = m["tiles"][i]
        a, c = t["levels"][0]
        base = all_splats.subset(np.arange(a, a + c))
        for k in range(args.decal_copies):
            # Consecutive picks, so a tile's copies carry different decals.
            j = (n * args.decal_copies + k) % len(picks)
            at = rng.uniform(-reach, reach, 2) if reach > 0 else (0.0, 0.0)
            v = dec.place(base, pieces[j], args.decal_radius, size, up, at=at)
            if fill > 0:
                v = fill_opacity(v, size, up, fill, st)
            levels = []
            for part in tile_levels(v, lod, size, up, compensate,
                                    bool(st.get("level_tone"))):
                parts.append(pack(part).reshape(-1))
                levels.append([cursor, len(part)])
                cursor += len(part)
            tiles.append({"start": levels[0][0], "count": levels[0][1],
                          "levels": levels, "n": t["n"], "e": t["e"],
                          "s": t["s"], "w": t["w"], "class": t.get("class", 0),
                          "base": i, "decal": j,
                          "at": [round(float(at[0]), 3), round(float(at[1]), 3)]})
            made += 1
    if plain:
        print(f"  {made} copies of {len(plain)} class {args.decal_class} tiles, "
              f"{len(picks)} decals, set up to {reach:.2f} m off centre")

    out = Path(args.out)
    out = out.with_suffix("") if out.suffix in (".splat", ".json") else out
    out.parent.mkdir(parents=True, exist_ok=True)
    buf = np.concatenate(parts)
    out.with_suffix(".splat").write_bytes(buf.tobytes())
    man = dict(m)
    man["tiles"] = tiles
    if centres is not None:
        man["centres"] = centres
    man["total"] = cursor
    man["decals"] = {"radius": args.decal_radius, "class": args.decal_class,
                     "copies": args.decal_copies, "seed": args.decal_seed,
                     "from": args.tileset.name,
                     "centres": [list(found[k][:2]) for k in picks],
                     "picks": picks}
    out.with_suffix(".json").write_text(json.dumps(man, indent=2))
    print(f"  {cursor:,} splats total, {len(buf) / 1e6:.1f} MB")
    print(f"  wrote {out.with_suffix('.splat')} and .json")
    reindex(out.parent, quiet=True)
    dest = args.out_dir / f"decals_{out.name}.png"
    draw(s, [found[k] for k in picks], [f"sheet {k}" for k in picks],
         args.decal_radius, up, f"{out.name}: the decals used, greenest first", dest)
    print(f"  the decals used: {dest}")
    print(f"  then: python scripts/bake_atlas.py {out.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["sheet", "bake"])
    ap.add_argument("tileset", type=Path, help="a tileset without extension, "
                    "e.g. web/data/desert_solid")
    # Named decal_* so one preset can hold these beside the export's own
    # --count and --seed.
    ap.add_argument("--radius", "--decal-radius", dest="decal_radius",
                    type=float, default=0.45,
                    help="metres; a decal must fit inside a tile with 10 cm "
                         "of edge left; what is left over is where it can "
                         "sit (0.45 on 1.5 m tiles: up to 0.2 m off centre)")
    ap.add_argument("--copies", "--decal-copies", dest="decal_copies",
                    type=int, default=2,
                    help="bake: copies per tile, each a different decal")
    ap.add_argument("--seed", "--decal-seed", dest="decal_seed", type=int,
                    default=0, help="bake: where in its tile each decal sits")
    ap.add_argument("--count", "--decal-count", dest="decal_count", type=int,
                    default=48, help="candidates on the sheet")
    ap.add_argument("--decals", dest="decal_auto", type=int, default=12,
                    help="bake without --pick: how many of the best to use")
    ap.add_argument("--pick", "--decal-pick", dest="decal_pick", default="",
                    help="bake: sheet numbers, e.g. 3,7,12 (default: the "
                         "best --decals)")
    ap.add_argument("--class", "--decal-class", dest="decal_class", type=int,
                    default=0, help="bake: which material's tiles get decals; "
                                    "-1 = the one with the largest share")
    ap.add_argument("--capture", default=None,
                    help="the capture, if not where the tileset's settings say")
    ap.add_argument("-o", "--out", default=None, help="bake: new tileset name")
    ap.add_argument("--out-dir", type=Path, default=Path("out"),
                    help="where the images go")
    add_preset_args(ap)
    args = apply_preset(ap)

    tileset = args.tileset.with_suffix("") if args.tileset.suffix in (".splat", ".json") \
        else args.tileset
    args.tileset = tileset
    m = json.loads(tileset.with_suffix(".json").read_text())
    if m.get("decals"):
        raise SystemExit(f"{tileset.name} already has decals; start from "
                         f"{m['decals']['from']}")
    if args.decal_radius > m["size"] / 2 - dec.EDGE_CLEAR:
        raise SystemExit(f"--radius {args.decal_radius} does not fit {m['size']} m tiles; "
                         f"at most {m['size'] / 2 - dec.EDGE_CLEAR:.2f}")
    if args.mode == "sheet":
        sheet(args, m)
    else:
        raw = np.fromfile(tileset.with_suffix(".splat"), dtype=np.uint8)
        bake(args, m, raw)


if __name__ == "__main__":
    main()
