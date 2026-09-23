"""Generate terrain for the tiles to lie on.

    python scripts/terrain_gen.py --name bigsur --size 512
    python scripts/terrain_gen.py --name desert --ridge 0.8 --iterations 80

Writes the same pair scripts/heightmap.py does - a 16-bit height PNG and a
sidecar - so the viewer loads it with no change at all. A generated field
and a drone survey are interchangeable from the renderer's side, which is
the point of going through a file.

Why generate rather than capture: a survey covers the few dozen metres
somebody flew over. The desert DTM is sixteen metres across. A grid of 64
tiles at 1.5 m is ninety-six, so the survey is already being stretched six
times past its real scale, and a game map is kilometres. There is no
capture of the terrain a game needs, and there does not need to be - the
capture supplies the *material*, which cannot be generated, and the terrain
supplies the *shape*, which can.

It also writes a sediment map. The rule that places material currently
infers where loose ground would collect from the shape of the surface.
Here that is not an inference: the erosion moved material and recorded
where it settled.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bozkir.catalogue import rebuild as reindex  # noqa: E402
from bozkir.erosion import PROFILES, generate  # noqa: E402
from bozkir.terrain import (HEIGHT_ENCODING, flow_accumulation,  # noqa: E402
                            slope, write_height_png)


def save_field(a, path):
    """Two channels, not 16-bit greyscale. See write_height_png."""
    write_height_png(a, path)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="generated",
                    help="written as web/data/<name>.height.png, so a "
                         "tileset of the same name picks it up")
    ap.add_argument("--out", default="web/data")
    ap.add_argument("--profile", choices=sorted(PROFILES),
                    help="a named terrain: " + ", ".join(sorted(PROFILES))
                         + ". Fills in the settings below; anything passed "
                           "explicitly still wins")
    ap.add_argument("--size", type=int, default=512,
                    help="grid resolution (default 512)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ridge", type=float, default=0.5,
                    help="0 rolling hills, 1 a mountain range. The mixture "
                         "rather than either alone: pure ridged noise is "
                         "all crest and no basin, pure fractal noise has "
                         "nothing for the water to cut between")
    ap.add_argument("--octaves", type=int, default=6)
    ap.add_argument("--freq", type=float, default=4,
                    help="features per side at the coarsest octave; higher "
                         "is smaller landforms")
    ap.add_argument("--iterations", type=int, default=40,
                    help="erosion steps. More cuts deeper valleys; the "
                         "network stops changing shape well before it "
                         "stops deepening")
    ap.add_argument("--incision", type=float, default=0.3,
                    help="K in the stream power law: how hard rivers cut")
    ap.add_argument("--uplift", type=float, default=0.0,
                    help="U: relief grown per step from the noise as an "
                         "uplift map, instead of only cut into it")
    ap.add_argument("--diffusion", type=float, default=0.1,
                    help="how fast hillsides creep. Against --incision this "
                         "sets how far apart the valleys sit, which is the "
                         "most recognisable thing about a landscape")
    ap.add_argument("--metres", type=float, default=0.0,
                    help="vertical range to record in the sidecar. Does not "
                         "change the field, only what it claims to be")
    args = ap.parse_args(argv)

    print(f"generating {args.size}x{args.size}, seed {args.seed}")
    # Only what was actually typed overrides the profile, so `--profile
    # canyon --seed 3` keeps the canyon and changes the seed.
    given = {k: v for k, v in vars(args).items()
             if f"--{k}" in (argv if argv is not None else sys.argv[1:])}
    settings = dict(PROFILES[args.profile]) if args.profile else {}
    for k in ("octaves", "freq", "ridge", "iterations", "incision",
              "diffusion", "uplift"):
        if k in given or k not in settings:
            settings[k] = getattr(args, k)
    if args.profile:
        print(f"profile {args.profile}: "
              + ", ".join(f"{k} {v}" for k, v in sorted(settings.items())))
    z, sed = generate(size=args.size, seed=args.seed, **settings)

    # What the erosion produced, in the terms the material rule reads.
    a = flow_accumulation(z)
    lf = np.log1p(a)
    network = float((lf > 2.5).mean())
    s = slope(z)
    print(f"  channel network covers {network:.0%} of the field")
    print(f"  slope: median {np.median(s):.3f}, steepest {s.max():.3f}")
    print(f"  valleys drain {np.exp(lf.max()) - 1:.0f} cells at the outlet")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{args.name}.height.png"
    save_field(z, png)
    sedpng = out / f"{args.name}.sediment.png"
    save_field(sed, sedpng)

    meta = {
        "width": int(z.shape[1]), "height": int(z.shape[0]),
        "metres_low": 0.0,
        "metres_high": float(args.metres) if args.metres else None,
        "metres_range": float(args.metres) if args.metres else None,
        "sample_metres": None,
        "source": (f"generated {args.profile} seed {args.seed}"
                   if args.profile else f"generated seed {args.seed}"),
        "sediment": f"{args.name}.sediment.png",
        "encoding": HEIGHT_ENCODING,
        "settings": {
            "size": args.size, "seed": args.seed,
            "profile": args.profile, **settings,
        },
    }
    with open(out / f"{args.name}.height.json", "w") as f:
        json.dump(meta, f, indent=2)
    # So the viewer's terrain menu knows this exists without anyone
    # editing a list by hand.
    reindex(out, quiet=True)

    print(f"  wrote {png}")
    print(f"  wrote {sedpng}")
    print(f"  wrote {out / f'{args.name}.height.json'}")
    print(f"\n  pick '{args.name}' in the viewer's terrain menu, or open "
          f"?scene=<tileset>&height={args.name}")
    if args.profile:
        print(f"  the same ground without a file: "
              f"?gen={args.profile}&seed={args.seed}&size={args.size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())