"""What is in web/data, written down so the viewer can offer it.

A static server cannot be asked what files a folder holds, so until now the
viewer could only show what somebody typed into the URL, and a tileset
could only find a height field that happened to share its name. That is why
putting the desert tiles on mesa terrain meant copying a 17 MB .splat under
a second name.

This writes `web/data/index.json`: the tilesets and the height fields that
are actually there, each with the few facts a menu needs. The viewer reads
it at startup, lists both, and lets one be chosen independently of the
other - so a terrain is something you pick, not something a filename
forces on you.

Rebuilt by the tools that write into the folder (`terrain_gen.py`,
`export_wang.py`), and by `scripts/index_data.py` by hand. It describes
files rather than replacing them: delete the index and everything still
works, minus the menus.
"""

import json
from pathlib import Path

__all__ = ["scan", "rebuild", "INDEX_NAME"]

INDEX_NAME = "index.json"


def _json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def scan(folder):
    """Look at a data folder and describe what a viewer could load from it."""
    folder = Path(folder)
    scenes, terrains = [], []

    for splat in sorted(folder.glob("*.splat")):
        name = splat.stem
        meta = _json(folder / f"{name}.json") or {}
        tiles = meta.get("tiles")
        scenes.append({
            "name": name,
            "megabytes": round(splat.stat().st_size / 1e6, 1),
            "tiles": len(tiles) if isinstance(tiles, list) else meta.get("count"),
            "classes": meta.get("classes", 1),
            "tile_size": meta.get("tile_size"),
            "has_manifest": bool(meta),
        })

    for sidecar in sorted(folder.glob("*.height.json")):
        name = sidecar.name[: -len(".height.json")]
        meta = _json(sidecar) or {}
        settings = meta.get("settings") or {}
        terrains.append({
            "name": name,
            "width": meta.get("width"),
            "height": meta.get("height"),
            "source": meta.get("source", "unknown"),
            "profile": settings.get("profile"),
            "seed": settings.get("seed"),
            "metres": meta.get("metres_range"),
            "sediment": bool(meta.get("sediment")),
        })

    return {"scenes": scenes, "terrains": terrains}


def rebuild(folder="web/data", quiet=False):
    """Write `index.json` for a data folder. Returns what it wrote."""
    folder = Path(folder)
    if not folder.is_dir():
        return None
    index = scan(folder)
    (folder / INDEX_NAME).write_text(json.dumps(index, indent=2))
    if not quiet:
        print(f"  indexed {len(index['scenes'])} tileset(s) and "
              f"{len(index['terrains'])} terrain(s) in {folder}")
    return index