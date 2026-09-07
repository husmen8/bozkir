"""Named settings, so a working command line survives being closed.

Getting a good export out of a scene takes a dozen flags found by trial and
error, and a flag list is a bad place to keep a result. A preset records
what worked under a name, and `--preset name` replays it.

Presets set defaults, so anything given on the command line still wins:

    python scripts/export_wang.py data/raw/bigsur.ply --preset bigsur
    python scripts/export_wang.py data/raw/bigsur.ply --preset bigsur --size 2.0
    python scripts/export_wang.py data/raw/x.ply --clean --flip --save-preset x

The file is plain JSON at presets.json, meant to be read and edited by hand.
"""

import json
from pathlib import Path

PRESETS = Path("presets.json")


def load_all(path=PRESETS):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"{path} is not valid JSON: {e}")


def add_preset_args(parser, path=PRESETS):
    known = sorted(load_all(path))
    parser.add_argument(
        "--preset", default=None,
        help="settings saved under this name" +
             (f" (available: {', '.join(known)})" if known else ""))
    parser.add_argument(
        "--save-preset", default=None, metavar="NAME",
        help="save the settings used by this run under NAME")
    parser.add_argument(
        "--list-presets", action="store_true",
        help="print what is saved and exit")
    return parser


def apply(parser, argv=None, path=PRESETS):
    """Parse arguments with a preset supplying the defaults.

    Two passes: the first only reads --preset, the second parses properly
    with that preset's values as defaults. Anything typed explicitly
    overrides the preset, because argparse prefers a given value to a
    default.
    """
    presets = load_all(path)

    # A separate parser for the peek: the real one has required positionals,
    # and --list-presets should work without naming a file.
    import argparse
    peeker = argparse.ArgumentParser(add_help=False)
    peeker.add_argument("--preset", default=None)
    peeker.add_argument("--list-presets", action="store_true")
    peek, _ = peeker.parse_known_args(argv)

    if getattr(peek, "list_presets", False):
        if not presets:
            print(f"no presets yet ({path} does not exist)")
        for name, values in sorted(presets.items()):
            print(f"\n  {name}")
            for k, v in sorted(values.items()):
                print(f"    {k:<18} {v}")
        raise SystemExit(0)

    name = getattr(peek, "preset", None)
    if name:
        if name not in presets:
            raise SystemExit(
                f"no preset called {name!r}. "
                f"Known: {', '.join(sorted(presets)) or 'none'}")
        # One preset per scene should work with every script, so settings
        # this script has no option for are skipped rather than fatal:
        # export_wang knows --cut, render_view does not, and both should be
        # able to say --preset bigsur.
        known = {a.dest for a in parser._actions}
        values = {k: v for k, v in presets[name].items() if k in known}
        skipped = sorted(set(presets[name]) - known)
        parser.set_defaults(**values)
        if skipped:
            print(f"  preset {name!r}: {len(values)} settings applied, "
                  f"{len(skipped)} not used here ({', '.join(skipped[:4])}"
                  f"{'...' if len(skipped) > 4 else ''})")

    args = parser.parse_args(argv)

    if getattr(args, "save_preset", None):
        save(args, args.save_preset, parser, path)
    return args


# Recorded settings describe how to process a scene, not which scene, so
# these are left out.
SKIP = {"preset", "save_preset", "list_presets", "path", "out", "no_cache"}


def save(args, name, parser, path=PRESETS):
    """Write the run's settings under `name`, keeping only what differs."""
    path = Path(path)
    presets = load_all(path)
    values = {}
    for k, v in vars(args).items():
        if k in SKIP or v is None:
            continue
        if isinstance(v, Path):
            continue
        if v == parser.get_default(k) and name in presets and k not in presets[name]:
            continue
        values[k] = list(v) if isinstance(v, tuple) else v

    presets[name] = values
    path.write_text(json.dumps(presets, indent=2, sort_keys=True) + "\n")
    print(f"  saved preset {name!r} to {path} ({len(values)} settings)")