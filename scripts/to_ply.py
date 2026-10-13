"""Write a piece of a capture, or one finished tile, as a PLY to open in
Brush, so both can be judged in the same viewer.

    python scripts/to_ply.py raw data/raw/desert_v1.ply --at 1.87,10.13 -o out/raw_piece.ply
    python scripts/to_ply.py tile web/data/desert_ms1 --index 0 -o out/tile0.ply

raw:  the splats of the capture inside a --size square at --at (preview
      sheet coordinates), untouched: no cleaning, no slab, full SH, in the
      capture's own frame, so Brush shows them exactly as in the whole scene.
tile: level 0 of tile --index from an exported tileset (.splat + .json).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bozkir.ply import load_ply, save_ply  # noqa: E402
from bozkir.pack import read_splat  # noqa: E402
from bozkir.transform import align_to_ground, ground_normal, recentre  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["raw", "tile"])
    ap.add_argument("path", type=Path,
                    help="raw: the capture .ply; tile: the tileset without extension")
    ap.add_argument("--at", default="1.87,10.13", help="raw: centre x,y")
    ap.add_argument("--size", type=float, default=1.5, help="raw: square, metres")
    ap.add_argument("--flip", action="store_true", help="raw: as the preset's flip")
    ap.add_argument("--index", type=int, default=0, help="tile: which tile")
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()

    if args.mode == "raw":
        s = load_ply(args.path)
        # Find the square in the aligned frame (where the sheet's numbers
        # are), but write the original splats: SH are not turned by the
        # alignment, so only the original frame shows true colour.
        hint = -ground_normal(s)[0] if args.flip else None
        a, _ = align_to_ground(s, target_axis=2, up_hint=hint)
        a, _ = recentre(a, "ground")
        cx, cy = (float(v) for v in args.at.split(","))
        half = args.size / 2.0
        keep = ((np.abs(a.xyz[:, 0] - cx) <= half)
                & (np.abs(a.xyz[:, 1] - cy) <= half))
        piece = s.subset(np.flatnonzero(keep))
        print(f"{args.path.name}: {keep.sum():,} of {len(s):,} splats in the "
              f"{args.size} m square at ({cx}, {cy}), SH degree {s.sh_degree}")
    else:
        m, s = read_splat(args.path.with_suffix(".splat"),
                          args.path.with_suffix(".json"))
        a, c = m["tiles"][args.index]["levels"][0]
        piece = s.subset(np.arange(a, a + c))
        print(f"tile {args.index} of {len(m['tiles'])}: {c:,} splats")

    save_ply(piece, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
