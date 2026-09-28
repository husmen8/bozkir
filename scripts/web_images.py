"""Put screenshots where the viewer's intro and info drawer look for them.

    python scripts/web_images.py intro-1=shot1.png intro-2=shot2.png ...

Each NAME=FILE becomes web/img/NAME.webp, at most 960 px wide. The intro
strip shows pictures only once all four intro-N exist; the info drawer
shows each info-* picture when it exists. See web/img/README.md for what
each one should show.
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pairs', nargs='+', metavar='NAME=FILE')
    argv = ap.parse_args(argv).pairs
    bad = [a for a in argv if '=' not in a]
    if bad:
        ap.error(f'expected NAME=FILE, got {bad[0]}')
    out = ROOT / 'web' / 'img'
    out.mkdir(parents=True, exist_ok=True)
    for arg in argv:
        name, src = arg.split('=', 1)
        img = Image.open(src).convert('RGB')
        if img.width > 960:
            img = img.resize((960, round(img.height * 960 / img.width)), Image.LANCZOS)
        dest = out / f'{name}.webp'
        img.save(dest, 'WEBP', quality=82, method=6)
        print(f'  {dest.relative_to(ROOT)}  {img.width}x{img.height}  '
              f'{dest.stat().st_size / 1024:.0f} KB')
    print('then: python scripts/viewer_guide.py   (the guide shows the info pictures too)')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
