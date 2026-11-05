"""Put several captures next to each other along x in one PLY, so one export
finds the materials of all of them (gravel and the lawn of its banks).

    python scripts/side_by_side.py OUT.ply GRAVEL.ply LAWN.ply
    python scripts/side_by_side.py OUT.ply GRAVEL.ply,core=85,photos=D:/L2/images LAWN.ply,photos=D:/bicycle/images_2

Each input is a prepared capture: metres, +z up, already cleaned on its own.
A merged file has to skip the export's own clean (--raw --max-extent-pct 100
--floater-std 0): it judges floaters and splat size against the whole scene,
and a sparser capture with larger splats next to a dense one is dropped.

Per input, comma-separated:
    core=P     keep the disc holding P% of it around its median
    photos=DIR white balance from the capture's own photos: the mean of their
               near-grey pixels (saturation under 0.12) is made grey. Two
               cameras then agree on what grey is (the L2 drone shoots cool,
               the bicycle DSLR warm); brightness and contrast stay as
               captured, they are the material's own.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bozkir.ply import SH_C0, Splats, load_ply, save_ply  # noqa: E402

PHOTO_TYPES = (".jpg", ".jpeg", ".png")


def grey_gains(folder, every=5, reduce=8):
    """Per-channel gains that make the mean of the photos' near-grey pixels
    grey, green kept. Every `every`th photo, shrunk `reduce` times."""
    from PIL import Image
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in PHOTO_TYPES)
    if not files:
        raise SystemExit(f"no photos in {folder}")
    total, n = np.zeros(3), 0
    for f in files[::every]:
        m = np.asarray(Image.open(f).convert("RGB").reduce(reduce), np.float64).reshape(-1, 3) / 255
        hi, lo = m.max(1), m.min(1)
        grey = ((hi - lo) / np.maximum(hi, 1e-6) < 0.12) & (hi > 0.15) & (hi < 0.85)
        total += m[grey].sum(0)
        n += int(grey.sum())
    if n == 0:
        raise SystemExit(f"no near-grey pixels in {folder}")
    mean = total / n
    return mean[1] / mean, n


def white_balance(s, gains):
    """Scale each colour channel of the splats by `gains`."""
    rgb = SH_C0 * s.sh_dc + 0.5
    s.sh_dc = ((rgb * gains - 0.5) / SH_C0).astype(s.sh_dc.dtype)
    if s.sh_rest.size:
        s.sh_rest = (s.sh_rest * gains).astype(s.sh_rest.dtype)
    return s


def parse(spec):
    path, *opts = spec.split(",")
    o = dict(kv.split("=", 1) for kv in opts)
    unknown = set(o) - {"core", "photos"}
    if unknown:
        raise SystemExit(f"unknown option {sorted(unknown)} in {spec}")
    return Path(path), float(o["core"]) if "core" in o else None, o.get("photos")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("inputs", nargs="+", metavar="IN.ply[,core=P][,photos=DIR]")
    ap.add_argument("--gap", type=float, default=1.0, help="metres between captures")
    args = ap.parse_args(argv)

    parts, x = [], 0.0
    for spec in args.inputs:
        path, core, photos = parse(spec)
        s = load_ply(path)
        if core is not None:
            c = np.median(s.xyz[:, :2], axis=0)
            r = np.hypot(*(s.xyz[:, :2] - c).T)
            s = s.subset(np.flatnonzero(r <= np.percentile(r, core)))
        note = ""
        if photos:
            gains, n = grey_gains(photos)
            s = white_balance(s, gains)
            note = f", white balance r x{gains[0]:.3f} b x{gains[2]:.3f} ({n:,} grey pixels)"
        lo, hi = np.percentile(s.xyz[:, 0], [0.5, 99.5])
        s.xyz = s.xyz.copy()
        s.xyz[:, 0] += x - lo
        s.xyz[:, 1] -= np.median(s.xyz[:, 1])
        s.xyz[:, 2] -= np.median(s.xyz[:, 2])
        parts.append(s)
        print(f"{path.name}: {len(s):,} splats, x {x:.2f} to {x + hi - lo:.2f} m{note}")
        x += (hi - lo) + args.gap

    deg = min(q.sh_degree for q in parts)
    parts = [q.truncate_sh(deg) if q.sh_degree > deg else q for q in parts]
    cat = lambda k: np.concatenate([getattr(q, k) for q in parts])  # noqa: E731
    save_ply(Splats(xyz=cat("xyz"), opacity=cat("opacity"), scale=cat("scale"), rot=cat("rot"),
                    sh_dc=cat("sh_dc"), sh_rest=cat("sh_rest"), sh_degree=deg), args.out)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
