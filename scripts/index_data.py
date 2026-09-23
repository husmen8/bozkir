"""List what the viewer can load, into web/data/index.json.

    python scripts/index_data.py
    python scripts/index_data.py --dir web/data

A static server cannot be asked what is in a folder, so the viewer reads
this file instead and offers the tilesets and terrains it names. The tools
that write into the folder rebuild it themselves; run this after moving,
renaming or deleting files by hand.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bozkir.catalogue import rebuild, scan  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="web/data", help="the data folder")
    ap.add_argument("--list", action="store_true",
                    help="print what is there without writing the index")
    args = ap.parse_args(argv)

    folder = Path(args.dir)
    if not folder.is_dir():
        print(f"no such folder: {folder}")
        return 1
    index = scan(folder) if args.list else rebuild(folder)
    for s in index["scenes"]:
        print(f"  tileset  {s['name']:<16} {s['classes']} class(es), "
              f"{s['megabytes']} MB")
    for t in index["terrains"]:
        print(f"  terrain  {t['name']:<16} {t['width']}x{t['height']}, "
              f"{t['source']}")
    if not index["scenes"] and not index["terrains"]:
        print("  nothing there yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())