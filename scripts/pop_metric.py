"""Measure popping across a sequence of rendered frames.

Frames are compared in filename order, consecutive pairs, and the score is
the part of each change that camera motion cannot account for. See
bozkir/popping.py for what that means and what it does not.

    python scripts/pop_metric.py frames/topological
    python scripts/pop_metric.py frames/near_side --block 24 --search 16
    python scripts/pop_metric.py frames/a --against frames/b

The intended use is an ablation: render the same camera path twice, once
per ordering, and compare. --against does that in one call and prints the
two side by side, which is the form the numbers want to be read in.

Frames must be the same size in both sequences, and should come from the
same camera path - otherwise the comparison is between two different
journeys and says nothing about ordering.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bozkir.popping import popping, sweep  # noqa: E402

SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def images_in(d):
    """Image files directly inside a folder, in filename order.

    Sorted by name rather than by modification time: a capture writes frames
    faster than the clock resolves, and two frames written in the same
    millisecond would otherwise come back in an arbitrary order.
    """
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in SUFFIXES)


def find_frames(folder):
    """The frames for one run, allowing for one level of nesting.

    Unzipping on Windows puts the contents inside a folder named after the
    archive, so frames land in frames/topological/topological rather than in
    frames/topological. Requiring people to flatten that by hand is a step
    invented by this script for its own convenience, so it looks one level
    down when the folder itself holds no images.

    It does not merge several subfolders. Two runs sitting side by side
    under one parent is exactly the mistake worth catching - the frames
    would interleave into a sequence that never happened, and the metric
    would report the difference between two orderings as if it were popping
    within one.
    """
    d = Path(folder)
    if not d.is_dir():
        raise SystemExit(f'not a folder: {d}')

    here = images_in(d)
    if here:
        return here

    subs = [s for s in sorted(d.iterdir()) if s.is_dir() and images_in(s)]
    if len(subs) == 1:
        print(f'{d}: no frames here, using {subs[0].name}/')
        return images_in(subs[0])
    if len(subs) > 1:
        names = ', '.join(s.name for s in subs)
        raise SystemExit(
            f'{d} holds {len(subs)} folders of frames ({names}).\n'
            f'Each run needs its own folder, so point at one of them:\n'
            f'    python scripts/pop_metric.py {d / subs[0].name} '
            f'--against {d / subs[1].name}')
    raise SystemExit(f'{d}: no images here or one level down')


def load_frames(folder, limit=None):
    """The frames of one run, as arrays."""
    paths = find_frames(folder)
    if len(paths) < 2:
        raise SystemExit(f'{folder}: need at least two frames, '
                         f'found {len(paths)}')
    if limit:
        paths = paths[:limit]
    out = []
    for p in paths:
        with Image.open(p) as im:
            out.append(np.asarray(im.convert('RGB'), dtype=np.uint8))
    return paths, out


def report(label, paths, frames, args):
    s = sweep(frames, block=args.block, search=args.search,
              threshold=args.threshold)
    worst = s['worst_pair']
    print(f'\n{label}')
    print(f'  frames            {len(frames)}  ({paths[0].name} .. {paths[-1].name})')
    print(f'  residual  mean    {s["residual_mean"]:.5f}')
    print(f'  residual  peak    {s["residual_peak"]:.5f}')
    print(f'  popped    mean    {s["fraction_mean"]:.3%}')
    print(f'  popped    peak    {s["fraction_peak"]:.3%}')
    print(f'  worst pair        {paths[worst].name} -> {paths[worst + 1].name}')

    # A number this high is not a finding, it is the instrument at its
    # limit. Block matching undoes a translation; a large turn changes the
    # view enough that near and far parts of the scene move differently,
    # which no per-block translation can remove. The leftover then counts as
    # popping for every ordering alike, and two sequences that look quite
    # different come out within a percent of each other.
    if s['fraction_mean'] > 0.25:
        print(f'  NOTE  {s["fraction_mean"]:.0%} of pixels called popped on '
              f'average. That is too high to')
        print('        be about ordering. The camera is very likely moving too '
              'far between')
        print('        frames - aim for under half a degree a step, by sweeping '
              'a narrow arc')
        print('        rather than a full turn.')
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('frames', help='folder of rendered frames')
    ap.add_argument('--against', help='a second folder, rendered the same '
                                      'camera path with a different ordering')
    ap.add_argument('--block', type=int, default=32,
                    help='motion search block, pixels (default 32)')
    ap.add_argument('--search', type=int, default=12,
                    help='motion search radius, pixels (default 12). Must '
                         'exceed the per-frame movement or motion reads as '
                         'popping')
    ap.add_argument('--threshold', type=float, default=0.02,
                    help='luminance above which a pixel counts as popped '
                         '(default 0.02)')
    ap.add_argument('--limit', type=int, help='use only the first N frames')
    ap.add_argument('--worst', metavar='FILE',
                    help='write the worst pair\'s popped-pixel mask here')
    args = ap.parse_args(argv)

    paths, frames = load_frames(args.frames, args.limit)
    a = report(args.frames, paths, frames, args)

    if args.against:
        bpaths, bframes = load_frames(args.against, args.limit)
        if len(bframes) != len(frames):
            raise SystemExit('the two sequences have different frame counts; '
                             'they cannot be the same camera path')
        if bframes[0].shape != frames[0].shape:
            raise SystemExit(f'frame sizes differ: {frames[0].shape} vs '
                             f'{bframes[0].shape}')
        b = report(args.against, bpaths, bframes, args)

        # Ratios rather than differences: the absolute level depends on the
        # camera path and the scene, and only the comparison is meaningful.
        print('\ncomparison')
        for key, name in (('fraction_mean', 'popped mean'),
                          ('fraction_peak', 'popped peak'),
                          ('residual_mean', 'residual mean')):
            if a[key] == 0:
                print(f'  {name:<14}  first sequence is zero; nothing to divide')
                continue
            print(f'  {name:<14}  {b[key] / a[key]:.3f}x '
                  f'({a[key]:.5f} -> {b[key]:.5f})')

    if args.worst:
        s = a
        i = s['worst_pair']
        r = popping(frames[i], frames[i + 1], block=args.block,
                    search=args.search, threshold=args.threshold)
        Image.fromarray((r['mask'] * 255).astype(np.uint8)).save(args.worst)
        print(f'\nwrote {args.worst}: white is a pixel motion cannot explain')

    return 0


if __name__ == '__main__':
    sys.exit(main())