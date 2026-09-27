"""Bake a tileset's far-field atlas: every tile seen from above, one image.

    python scripts/bake_atlas.py desert
    python scripts/bake_atlas.py desert --res 96

Writes web/data/<name>.atlas.png and .atlas.json, which the viewer's far
field draws distant ground with. Re-bake after every export: an atlas with
a different tile count is ignored.
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bozkir.pack import read_splat, tile_atlas  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="tileset name in the data folder")
    ap.add_argument("--dir", type=Path, default=ROOT / "web" / "data")
    ap.add_argument("--res", type=int, default=64,
                    help="pixels per tile side (default 64)")
    args = ap.parse_args(argv)
    splat = args.dir / f"{args.name}.splat"
    manifest = args.dir / f"{args.name}.json"
    if not splat.exists() or not manifest.exists():
        raise SystemExit(f"need {splat} and {manifest}")
    m, s = read_splat(splat, manifest)
    img, layout = tile_atlas(m, s, args.res)
    Image.fromarray(img, "RGBA").save(args.dir / f"{args.name}.atlas.png")
    (args.dir / f"{args.name}.atlas.json").write_text(json.dumps(layout))
    cover = img[..., 3].mean() / 255
    print(f"  {layout['count']} tiles at {args.res}px, "
          f"{layout['cols']}x{layout['rows']} atlas, mean coverage {cover:.0%}")
    print(f"  wrote {args.name}.atlas.png and .atlas.json in {args.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())