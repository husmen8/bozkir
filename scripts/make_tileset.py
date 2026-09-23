"""From a capture to a tileset in the viewer, in one command.

    python scripts/make_tileset.py my_ground.ply
    python scripts/make_tileset.py my_ground.ply --name meadow --colours 3

For somebody who has a Gaussian splat capture of some ground and has never
seen this project. It runs the exporter with the defaults meant for an
untuned capture, decides for itself whether the capture holds one material
or two, and finishes with a verdict in plain words and the address to open.

The steps it runs are the same `export_wang.py` a tuned workflow uses, with
everything it prints passed straight through - nothing here is hidden from
somebody who later wants the full controls. What it adds is the order, the
retry, and the reading of the result:

  good       the exporter raised no warnings
  marginal   it built a tileset but said something is off, and what
  no         it could not build one, and why - usually that the capture is
             not flat ground, or holds too little clean ground to tile

Anything this script does not know about goes through to the exporter
unchanged, so `--size 2.0 --flip` work here as they do there.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ONE_MATERIAL = "this capture holds one material"
TOO_LITTLE = "still has fewer than"


def run(cmd):
    """Run a command, echo its output as it arrives, and keep a copy."""
    print(f"\n  $ {' '.join(cmd[1:])}\n", flush=True)
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line)
    p.wait()
    return p.returncode, "".join(lines)


def verdict(code, text):
    if code != 0:
        reason = [l.strip() for l in text.splitlines() if l.strip()][-3:]
        return "no", reason
    warnings = [l.strip()[2:] for l in text.splitlines()
                if l.strip().startswith("! ")]
    if "grid will have a grain" in text:
        warnings.append("the two edge sets differ in appearance; the grid "
                        "may show a grain")
    return ("marginal" if warnings else "good"), warnings


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ply", type=Path, help="the capture, a 3DGS .ply")
    ap.add_argument("--name", help="tileset name (default: the file's name)")
    ap.add_argument("--classes", type=int, default=None,
                    help="materials to split into; by default two are tried "
                         "and one is used if the capture holds only one")
    ap.add_argument("--colours", type=int, default=2,
                    help="edge colours per axis (3 needs more clean ground)")
    args, rest = ap.parse_known_args(argv)

    if not args.ply.exists():
        raise SystemExit(f"no such file: {args.ply}")
    name = args.name or args.ply.stem
    out = ROOT / "web" / "data" / f"{name}.splat"
    base = [sys.executable, str(ROOT / "scripts" / "export_wang.py"),
            str(args.ply), "--colours", str(args.colours), "-o", str(out)]

    print(f"making tileset '{name}' from {args.ply.name}")
    tries = [args.classes] if args.classes else [2, 1]
    code, text = 1, ""
    for classes in tries:
        cmd = base + (["--classes", str(classes)] if classes > 1 else []) + rest
        code, text = run(cmd)
        if code == 0:
            break
        if classes > 1 and (ONE_MATERIAL in text or TOO_LITTLE in text):
            print("\n  -> not enough for two materials here; "
                  "building a single-material tileset instead")
            continue
        break

    word, why = verdict(code, text)
    print("\n" + "=" * 68)
    if word == "no":
        print(f"  verdict: no tileset from {args.ply.name}")
        for w in why:
            print(f"    {w}")
        print("\n  scripts/inspect_ply.py says whether the capture is ground at "
              "all;\n  scripts/preview_patches.py shows which filter rejects it.")
        return 1
    print(f"  verdict: {word}")
    for w in why:
        print(f"    - {w}")
    print(f"\n  wrote {out.relative_to(ROOT)} and its .json")
    print("  to see it:   cd web && python -m http.server 8000")
    print(f"  then open:   http://localhost:8000/?scene={name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())