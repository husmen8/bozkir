"""One path from a file on disk to a scene ready to use.

Every script was repeating load -> align -> clean with slightly different
defaults, which meant results from one script could not be compared with
results from another. This is that sequence, once.

Results are cached, keyed by the settings that produced them. Alignment is
fast but floater removal builds a KD-tree over every splat, which is slow
enough to be worth not repeating.
"""

import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .ply import load_ply, save_ply
from .transform import align_to_ground, ground_normal, recentre
from .select import crop_cylinder, remove_large, remove_floaters

CACHE_DIR = Path("data/cache")


@dataclass
class SceneConfig:
    """Everything that changes what a prepared scene contains.

    Anything not in here must not affect the output, or the cache will
    hand back the wrong scene.
    """

    align: bool = True
    recentre: bool = True
    up_axis: int = 2
    flip: bool = False              # invert the detected up direction

    clean: bool = False
    radius_pct: float = 60.0        # keep this percentile of horizontal distance
    max_extent_pct: float = 99.0    # drop splats larger than this percentile
    floater_std: float = 2.0        # lower removes more floaters

    sh_degree: int = None           # truncate colour; None keeps the file's

    def key(self):
        """Short stable hash of the settings, for cache filenames."""
        parts = sorted(f"{k}={v}" for k, v in asdict(self).items())
        return hashlib.sha1("|".join(parts).encode()).hexdigest()[:10]

    def describe(self):
        bits = []
        if self.align:
            bits.append(f"align->{'xyz'[self.up_axis]}" + ("(flipped)" if self.flip else ""))
        if self.recentre:
            bits.append("recentre")
        if self.clean:
            bits.append(f"clean(r{self.radius_pct:g},f{self.floater_std:g})")
        if self.sh_degree is not None:
            bits.append(f"sh{self.sh_degree}")
        return " + ".join(bits) if bits else "raw"


def prepare(path, cfg=None, cache=True, cache_dir=CACHE_DIR, verbose=True):
    """Load a PLY and apply the configured preparation.

    Returns a Splats. With `cache`, the result is written to
    data/cache/<stem>_<key>.ply and reused on the next call with the same
    settings.
    """
    path = Path(path)
    cfg = cfg or SceneConfig()

    cache_path = Path(cache_dir) / f"{path.stem}_{cfg.key()}.ply"
    if cache and cache_path.exists():
        s = load_ply(cache_path)
        if verbose:
            print(f"{path.name}: {len(s):,} splats "
                  f"[cached: {cfg.describe()}]")
        return s

    s = load_ply(path)
    if verbose:
        print(f"{path.name}: {len(s):,} splats, SH degree {s.sh_degree}")
        print(f"  preparing: {cfg.describe()}")

    if cfg.align:
        hint = None
        if cfg.flip:
            # Ask for the opposite of whatever the detector chose.
            n, _ = ground_normal(s)
            hint = -n
        s, info = align_to_ground(s, target_axis=cfg.up_axis, up_hint=hint)
        if verbose:
            print(f"  tilt {info['tilt_before_deg']:.2f} -> "
                  f"{info['tilt_after_deg']:.2f} deg")

    if cfg.recentre:
        s, _ = recentre(s, "ground" if cfg.up_axis == 2 else "median")

    if cfg.clean:
        n0 = len(s)
        plane = [i for i in range(3) if i != cfg.up_axis]
        centre = np.median(s.xyz, axis=0)
        d = np.linalg.norm(s.xyz[:, plane] - centre[plane], axis=1)
        s, _ = crop_cylinder(s, centre, float(np.percentile(d, cfg.radius_pct)),
                             up_axis=cfg.up_axis)
        s, _ = remove_large(
            s, float(np.percentile(s.scale.max(axis=1), cfg.max_extent_pct)))
        s, _ = remove_floaters(s, std_ratio=cfg.floater_std)
        if verbose:
            print(f"  cleaned {n0:,} -> {len(s):,} ({len(s) / n0:.0%})")

    if cfg.sh_degree is not None and s.sh_degree > cfg.sh_degree:
        s = s.truncate_sh(cfg.sh_degree)

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_ply(s, cache_path)
        if verbose:
            print(f"  cached -> {cache_path}")

    return s


def add_scene_args(parser):
    """Attach the standard preparation flags to an argparse parser.

    Every script takes the same flags and means the same thing by them.
    """
    g = parser.add_argument_group("scene preparation")
    g.add_argument("--raw", action="store_true",
                   help="skip alignment and recentring")
    g.add_argument("--clean", action="store_true",
                   help="crop to the subject, drop the background shell, "
                        "remove floaters")
    g.add_argument("--radius-pct", type=float, default=60.0)
    g.add_argument("--floater-std", type=float, default=2.0)
    g.add_argument("--max-extent-pct", type=float, default=99.0)
    g.add_argument("--sh", type=int, default=None,
                   help="truncate spherical harmonics to this degree")
    g.add_argument("--up-axis", type=int, default=2, choices=(0, 1, 2))
    g.add_argument("--flip", action="store_true",
                   help="invert the detected up direction; use when a scene "
                        "comes out upside down")
    g.add_argument("--no-cache", action="store_true")
    return parser


def config_from_args(args):
    return SceneConfig(
        align=not args.raw,
        recentre=not args.raw,
        up_axis=args.up_axis,
        flip=args.flip,
        clean=args.clean,
        radius_pct=args.radius_pct,
        max_extent_pct=args.max_extent_pct,
        floater_std=args.floater_std,
        sh_degree=args.sh,
    )


def scene_from_args(args, verbose=True):
    """The one line every script needs."""
    return prepare(args.path, config_from_args(args),
                   cache=not args.no_cache, verbose=verbose)