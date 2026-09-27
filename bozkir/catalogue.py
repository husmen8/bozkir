"""Writes web/data/index.json: the tilesets and height fields in the
folder, with what the viewer's menus show. A static server cannot list a
folder, so the folder lists itself, and tileset and terrain can be chosen
independently.

Rebuilt by export_wang.py and terrain_gen.py, or by hand with
scripts/index_data.py. It stays in the working copy (not committed); the
public viewer finds the desert tileset without it.
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