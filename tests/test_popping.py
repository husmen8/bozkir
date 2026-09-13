"""Tests for bozkir/popping.py.

    python tests/test_popping.py

Numpy only, so this runs without the PLY stack. The frames are synthetic on
purpose: the whole claim of the metric is that it separates motion from
popping, and the only way to check that is to build frames where the answer
is known by construction.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bozkir.popping import to_luma, motion_field, compensate, popping, sweep  # noqa: E402

PASSED = 0
FAILURES = []


def test(name):
    def wrap(fn):
        global PASSED
        try:
            fn()
            PASSED += 1
        except AssertionError as e:
            FAILURES.append((name, str(e)))
        except Exception as e:                       # noqa: BLE001
            FAILURES.append((name, f'{type(e).__name__}: {e}'))
        return fn
    return wrap


def texture(h=192, w=256, seed=0):
    """A frame with detail at every scale. Block matching needs texture to
    lock onto; a flat field has no unique best match and the test would be
    measuring the tie-break rather than the metric."""
    rng = np.random.default_rng(seed)
    img = rng.random((h, w))
    for k in (2, 4, 8):
        coarse = rng.random((h // k + 1, w // k + 1))
        img += np.kron(coarse, np.ones((k, k)))[:h, :w]
    return img / img.max()


def shift(img, dy, dx):
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)


@test('luminance accepts bytes and floats alike')
def _():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 1] = 255
    assert abs(to_luma(rgb).mean() - 0.7152) < 1e-6, 'byte input mis-scaled'
    f = np.ones((4, 4, 3))
    assert abs(to_luma(f).mean() - 1.0) < 1e-9, 'float input mis-scaled'
    g = np.ones((4, 4)) * 0.5
    assert abs(to_luma(g).mean() - 0.5) < 1e-9, 'greyscale should pass through'


@test('a pure translation is found')
def _():
    a = texture()
    b = shift(a, 3, -5)
    field = motion_field(a, b, block=32, search=8)
    dy = np.median(field[..., 0])
    dx = np.median(field[..., 1])
    # The field carries a onto b, so it reports where to read from: the
    # opposite sign of the shift applied.
    assert abs(dy + 3) < 1.5, f'dy found {dy}, expected about -3'
    assert abs(dx - 5) < 1.5, f'dx found {dx}, expected about 5'


@test('compensating a translation removes almost all of the difference')
def _():
    # The central claim. A camera turning moves the whole image, and the
    # metric has to report that as nothing happened.
    a = texture()
    b = shift(a, 2, 4)
    r = popping(a, b, block=32, search=8)
    assert r['raw'] > 0.05, f'the test frames barely differ: raw={r["raw"]}'
    assert r['residual'] < r['raw'] / 10, \
        f'compensation only got {r["raw"]:.4f} down to {r["residual"]:.4f}'
    assert r['fraction'] < 0.02, f'{r["fraction"]:.3%} of pixels called popped'


@test('a small block does not lose the motion to coarse rounding')
def _():
    # Regression. The coarse pass estimates on a quarter-size copy, so its
    # answer is only good to four pixels. With the fine radius set from the
    # block size alone, a 16-pixel block gave a radius of two, and a real
    # motion of three pixels became unreachable - reported as 16% of the
    # frame popping when nothing had popped at all.
    a = texture()
    b = shift(a, 2, 3)
    r = popping(a, b, block=16, search=10)
    assert r['fraction'] < 0.02, \
        f'{r["fraction"]:.2%} of a purely translated frame called popped'


@test('block and search settings agree with each other')
def _():
    # The metric is meant to be robust to how it is configured; a number
    # that changes with the block size is not measuring the scene.
    a = texture()
    b = shift(a, 2, 3)
    for block in (16, 24, 32, 48):
        for search in (6, 12, 20):
            r = popping(a, b, block=block, search=search)
            assert r['fraction'] < 0.03, \
                f'block={block} search={search}: {r["fraction"]:.2%} popped'


@test('a shift far larger than the search radius is still found')
def _():
    # The failure that made a real capture read 55% popped for every
    # ordering. A sweep of 48 frames over a full turn moves the image by
    # more than a hundred pixels between frames; a searched coarse pass
    # with a radius of twelve cannot reach that, so every block failed and
    # the metric saturated. Phase correlation has no radius.
    a = texture(192, 256)
    b = shift(a, 40, -70)
    r = popping(a, b, block=32, search=8)
    assert r['fraction'] < 0.03, \
        f'{r["fraction"]:.1%} popped on a pure 70-pixel translation'


@test('the metric does not saturate as the step grows')
def _():
    # Whatever the step, a translated frame is a translated frame. If the
    # score climbs with step size the metric is measuring the camera.
    a = texture(192, 256)
    for dy, dx in ((1, 2), (8, 14), (25, 45), (60, 90)):
        r = popping(a, shift(a, dy, dx), block=32, search=8)
        assert r['fraction'] < 0.05, \
            f'shift ({dy},{dx}): {r["fraction"]:.1%} popped'


@test('a patch that swaps is caught')
def _():
    # A tile suddenly drawn in front of its neighbour: a rectangle replaced
    # with different content, no motion anywhere.
    a = texture()
    b = a.copy()
    b[60:120, 80:160] = texture(seed=7)[60:120, 80:160]
    r = popping(a, b, block=32, search=8)
    area = (120 - 60) * (160 - 80) / a.size
    assert r['fraction'] > area / 2, \
        f'caught only {r["fraction"]:.3%} of a {area:.1%} patch'
    assert r['p99'] > 0.1, f'p99 {r["p99"]:.3f} is too low for a hard swap'


@test('a patch that swaps while the camera moves is still caught')
def _():
    # The case that matters and the one a raw difference cannot do: both
    # things happening at once.
    a = texture()
    b = shift(a, 2, 3)
    b[60:120, 80:160] = shift(texture(seed=11), 2, 3)[60:120, 80:160]
    moved = popping(a, shift(a, 2, 3), block=32, search=8)
    both = popping(a, b, block=32, search=8)
    assert both['fraction'] > moved['fraction'] * 5, \
        (f'motion alone {moved["fraction"]:.3%} vs motion plus a pop '
         f'{both["fraction"]:.3%} - the two are not separated')


@test('the mask lands on the patch that moved, not elsewhere')
def _():
    a = texture()
    b = a.copy()
    b[64:128, 64:128] = texture(seed=3)[64:128, 64:128]
    m = popping(a, b, block=32, search=8)['mask']
    inside = m[64:128, 64:128].mean()
    outside = (m.sum() - m[64:128, 64:128].sum()) / (m.size - 64 * 64)
    assert inside > 0.5, f'only {inside:.1%} of the patch flagged'
    assert outside < 0.05, f'{outside:.1%} flagged outside the patch'


@test('identical frames score zero')
def _():
    a = texture()
    r = popping(a, a, block=32, search=4)
    assert r['residual'] == 0.0, f'residual {r["residual"]}'
    assert r['fraction'] == 0.0, f'fraction {r["fraction"]}'


@test('a sweep reports the peak, not just the mean')
def _():
    # Forty frames of clean motion with one pop in the middle. The mean is
    # diluted by the clean frames; the peak is what the eye actually sees.
    frames = []
    base = texture()
    for i in range(12):
        f = shift(base, i, 2 * i)
        if i == 6:
            f[60:120, 80:160] = texture(seed=5)[60:120, 80:160]
        frames.append(f)
    s = sweep(frames, block=32, search=8)
    assert s['pairs'] == 11
    assert s['fraction_peak'] > s['fraction_mean'] * 3, \
        'a single popped frame should stand out from the sweep'
    assert s['worst_pair'] in (5, 6), f'blamed pair {s["worst_pair"]}'


@test('mismatched frame sizes are refused')
def _():
    try:
        motion_field(texture(64, 64), texture(64, 96))
    except ValueError as e:
        assert 'size' in str(e)
        return
    raise AssertionError('differing sizes should not be compared')


if __name__ == '__main__':
    for name, msg in FAILURES:
        print(f'FAIL  {name}\n      {msg}')
    print(f'\n{PASSED} passed, {len(FAILURES)} failed')
    sys.exit(1 if FAILURES else 0)