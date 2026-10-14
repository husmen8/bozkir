"""A finished tileset from a capture in one command: export, decals, far
view, all from one preset.

    python scripts/build.py data/raw/scene.ply --preset ground -o web/data/scene --look
    python scripts/build.py data/raw/scene.ply --preset ground -o web/data/scene
    python scripts/build.py data/raw/scene.ply --preset ground -o web/data/test --odd-spots 30

Each step is its own script (export_wang, decals, bake_atlas) reading the
same preset; flags after the preset go to the step they belong to
(--decal-* and --decals to decals, the rest to the export) and win over the
preset. Decals are added when the preset or a flag asks for them
(decal_auto above 0, or decal_pick). Files already at the output names are
moved to web/data/backup/<time>/ first, never overwritten. --look builds
nothing: it draws the materials and the patches that would be used, to
out/classes_<name>.png, and leaves every file where it is.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bozkir.presets import load_all, resolve, _names  # noqa: E402

DECAL_FLAGS = ("--decal", "--decals", "--pick", "--copies")
SUFFIXES = (".splat", ".json", ".atlas.png", ".atlas.json")


def split_flags(rest):
    """Extra flags, each with its values, sorted by the step they are for."""
    groups, cur = [], None
    for tok in rest:
        if tok.startswith("--"):
            cur = [tok]
            groups.append(cur)
        elif cur is not None:
            cur.append(tok)
        else:
            raise SystemExit(f"{tok!r} is not a flag")
    export, decals = [], []
    for g in groups:
        name = g[0].split("=")[0]
        (decals if name.startswith(DECAL_FLAGS) else export).extend(g)
    return export, decals


def back_up(stem):
    """Move whatever is at `stem`.* into backup/<time>/; returns the folder."""
    found = [stem.parent / (stem.name + s) for s in SUFFIXES
             if (stem.parent / (stem.name + s)).exists()]
    if not found:
        return None
    dest = stem.parent / "backup" / time.strftime("%Y%m%d-%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    for f in found:
        shutil.move(str(f), str(dest / f.name))
    return dest


def run(step, cmd):
    print(f"\n== {step}: {' '.join(str(c) for c in cmd[1:])}", flush=True)
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        raise SystemExit(f"\n{step} stopped; nothing after it was run")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="the capture (.ply)")
    ap.add_argument("-o", "--out", required=True, type=Path,
                    help="the finished tileset, e.g. web/data/scene")
    ap.add_argument("--preset", required=True,
                    help="settings for every step; several stack, as a,b")
    args, rest = ap.parse_known_args(argv)
    from bozkir import BUILD
    print(f"bozkir {BUILD}")
    export_flags, decal_flags = split_flags(rest)

    settings = resolve(load_all(), _names(args.preset))
    auto, pick = settings.get("decal_auto", 0), settings.get("decal_pick", "")
    for i, f in enumerate(decal_flags):
        name, _, val = f.partition("=")
        val = val or (decal_flags[i + 1] if i + 1 < len(decal_flags) else "")
        if name == "--decals":
            auto = int(val)
        elif name in ("--pick", "--decal-pick"):
            pick = val
    with_decals = auto > 0 or bool(pick)
    out = args.out.with_suffix("") if args.out.suffix in (".splat", ".json") else args.out
    out = out if out.is_absolute() else (Path.cwd() / out)
    if out.name == "desert":
        print("  note: web/data/desert is the public tileset; the current one "
              "goes to backup/ first")
    py = sys.executable
    if "--look" in export_flags:
        run("look", [py, "scripts/export_wang.py", str(args.path.resolve()),
                     "--preset", args.preset, *export_flags,
                     "-o", str(out.with_suffix(".splat"))])
        return
    base = out.parent / (out.name + "_base") if with_decals else out
    for stem in {out, base}:
        moved = back_up(stem)
        if moved:
            print(f"  moved the old {stem.name} files to {moved}")

    run("export", [py, "scripts/export_wang.py", str(args.path.resolve()),
                   "--preset", args.preset, *export_flags,
                   "-o", str(base.with_suffix(".splat"))])
    if with_decals:
        run("decals", [py, "scripts/decals.py", "bake", str(base),
                       "--preset", args.preset, *decal_flags, "-o", str(out)])
    run("far view", [py, "scripts/bake_atlas.py", out.name, "--dir", str(out.parent)])

    m = json.loads(out.with_suffix(".json").read_text())
    print(f"\n  done: {out.name}, {len(m['tiles'])} tiles, "
          f"{out.with_suffix('.splat').stat().st_size / 1e6:.1f} MB")
    if with_decals:
        print(f"  the decals used: out/decals_{out.name}.png")
    print(f"  open {out.name} in the viewer")


if __name__ == "__main__":
    main()
