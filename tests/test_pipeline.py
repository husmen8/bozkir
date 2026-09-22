"""Every test for the Python pipeline.

    python tests/test_pipeline.py
    python tests/test_pipeline.py terrain     only sections matching a word

One file rather than one per feature. The split was convenient while each
piece was being written and became five files nobody opens, each with its
own copy of the same six-line harness. Sections below keep the grouping;
the files are gone.

test_all.py is still separate, because it covers the original package and
needs the PLY stack, while everything here runs on numpy and Pillow alone.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

PASSED = 0
FAILURES = []
SECTION = ''
ONLY = [a.lower() for a in sys.argv[1:]]


def test(name):
    """Register a test. Failures are collected, not raised, so one broken
    thing does not hide the state of everything after it."""
    def wrap(fn):
        global PASSED
        if ONLY and not any(o in SECTION.lower() or o in name.lower()
                            for o in ONLY):
            return fn
        try:
            fn()
            PASSED += 1
        except AssertionError as e:
            FAILURES.append((SECTION, name, str(e)))
        except Exception as e:                        # noqa: BLE001
            FAILURES.append((SECTION, name, f'{type(e).__name__}: {e}'))
        return fn
    return wrap


def ok(cond, msg='expected true'):
    if not cond:
        raise AssertionError(msg)


def close(a, b, tol=1e-9, msg='not close'):
    if abs(a - b) > tol:
        raise AssertionError(f'{msg}: {a} vs {b}')


# Everything the sections below reach for. Gathered here rather than beside
# each section so the same name cannot mean two things in one file.
from PIL import Image                                             # noqa: E402
from bozkir.ply import Splats                                     # noqa: E402
from bozkir.popping import (motion_field, popping,                # noqa: E402
                            sweep, to_luma)
from bozkir.scene import SceneConfig                              # noqa: E402
from bozkir.patches import (appearance, apply_settings,           # noqa: E402
                            auto_pick, cached_search,
                            class_separation, choose, describe_trail,
                            pick_patches, search_key, search_kwargs,
                            select_similar, split_classes, SEARCH_KEYS)
from bozkir.terrain import (LANDFORMS, coverage, curvature,       # noqa: E402
                            flow_accumulation, height_from_points,
                            landform, largest_rectangle, read_dsm,
                            read_height_png, slope, tpi,
                            write_height_png)
from heightmap import main as _hm_main                            # noqa: E402
from bozkir.erosion import erode, fbm, generate, ridged             # noqa: E402
import json                                                       # noqa: E402


def main(argv):
    """heightmap.main with its progress report silenced.

    It prints a paragraph per call, which on a clean run buries the one
    line that matters under several screens of things that went right.
    """
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return _hm_main(argv)


# ======================================================================
# popping metric
# ======================================================================

SECTION = 'popping metric'


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


# ======================================================================
# height maps
# ======================================================================

SECTION = 'height maps'


def brute_force(mask):
    """Every rectangle, checked. Correct and far too slow for real sizes,
    which is the point: it is what the fast version has to agree with."""
    rows, cols = mask.shape
    best_area, best = 0, (0, 0, 0, 0)
    for t in range(rows):
        for l in range(cols):
            for b in range(t, rows):
                if not mask[b, l]:
                    break
                for r in range(l, cols):
                    if not mask[t:b + 1, l:r + 1].all():
                        break
                    area = (b - t + 1) * (r - l + 1)
                    if area > best_area:
                        best_area, best = area, (t, l, b - t + 1, r - l + 1)
    return best_area


@test('a solid mask gives the whole thing')
def _():
    m = np.ones((7, 11), dtype=bool)
    t, l, h, w = largest_rectangle(m)
    assert (t, l, h, w) == (0, 0, 7, 11), f'got {(t, l, h, w)}'


@test('an empty mask gives nothing rather than raising')
def _():
    assert largest_rectangle(np.zeros((5, 5), dtype=bool)) == (0, 0, 0, 0)


@test('the rectangle found is actually all valid')
def _():
    rng = np.random.default_rng(0)
    for seed in range(12):
        m = rng.random((14, 18)) > 0.25
        t, l, h, w = largest_rectangle(m)
        if h and w:
            assert m[t:t + h, l:l + w].all(), \
                f'seed {seed}: rectangle at {(t, l, h, w)} includes a hole'


@test('it finds the largest, checked against brute force')
def _():
    rng = np.random.default_rng(7)
    for seed in range(10):
        m = rng.random((9, 11)) > 0.3
        _, _, h, w = largest_rectangle(m)
        assert h * w == brute_force(m), \
            f'seed {seed}: found {h * w}, best is {brute_force(m)}'


@test('a diagonal polygon is handled, which is the real case')
def _():
    # The survey area sits at an angle, so no row or column of the file is
    # entirely empty and a naive bounding-box trim finds nothing to cut.
    n = 60
    yy, xx = np.mgrid[0:n, 0:n]
    m = (xx + yy > n * 0.45) & (xx + yy < n * 1.55) \
        & (xx - yy > -n * 0.55) & (xx - yy < n * 0.55)
    t, l, h, w = largest_rectangle(m)
    assert m[t:t + h, l:l + w].all(), 'rectangle escaped the polygon'
    assert h * w > 0.25 * m.sum(), \
        f'kept only {h * w} of {m.sum()} valid pixels'


@test('a tall thin valid strip is found')
def _():
    m = np.zeros((40, 40), dtype=bool)
    m[:, 17:21] = True
    t, l, h, w = largest_rectangle(m)
    assert (h, w) == (40, 4), f'got {h}x{w}'


@test('nodata is excluded by value when there is no metadata')
def _():
    from PIL import Image
    import tempfile

    a = np.full((20, 20), 5.0, dtype=np.float32)
    a[:4, :] = -9999.0
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'dsm.tif'
        Image.fromarray(a).save(p)
        vals, good, _ = read_dsm(p)
    assert good.shape == a.shape
    assert not good[:4, :].any(), 'sentinel counted as ground'
    assert good[4:, :].all(), 'real ground dropped'


@test('end to end: a DSM becomes a height map and a sidecar')
def _():
    import json
    import tempfile
    from PIL import Image

    n = 80
    yy, xx = np.mgrid[0:n, 0:n]
    # A ridge with a channel, and nodata corners like a real survey.
    a = (3.0 * np.sin(xx / 12.0) + 1.5 * np.cos(yy / 9.0)).astype(np.float32)
    a[(xx + yy) < n * 0.35] = -9999.0
    a[(xx + yy) > n * 1.65] = -9999.0

    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / 'dsm.tif'
        Image.fromarray(a).save(src)
        rc = main([str(src), '--name', 'unit', '--out', d, '--size', '64'])
        assert rc == 0

        png = Path(d) / 'unit.height.png'
        side = Path(d) / 'unit.height.json'
        assert png.exists() and side.exists(), 'outputs missing'

        img = read_height_png(png)
        levels = len(np.unique(np.round(img * 65535)))
        assert levels > 256, \
            f'{levels} distinct heights; the 16-bit encoding did not survive'
        assert abs(float(img.min())) < 1e-6 and abs(float(img.max()) - 1) < 1e-6, \
            'height map does not use its full range'
        assert max(img.shape) <= 64, f'not resampled: {img.shape}'

        meta = json.loads(side.read_text())
        assert meta['metres_range'] > 0
        assert meta['width'] == img.shape[1]
        assert meta['height'] == img.shape[0]
        # The sentinel must not have survived into the range.
        assert meta['metres_low'] > -100, \
            f"nodata leaked into the height range: {meta['metres_low']}"


@test('smoothing lowers the roughness without moving the landforms')
def _():
    import tempfile
    from PIL import Image

    n = 70
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:n, 0:n]
    smoothform = 4.0 * np.sin(xx / 15.0)
    a = (smoothform + rng.normal(0, 0.6, (n, n))).astype(np.float32)

    def run(passes, d):
        main([str(d / 'dsm.tif'), '--name', f's{passes}', '--out', str(d),
              '--size', '64', '--smooth', str(passes)])
        return read_height_png(d / f's{passes}.height.png')

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        Image.fromarray(a).save(d / 'dsm.tif')
        rough = run(0, d)
        smooth = run(2, d)

    def roughness(z):
        return float(np.abs(np.diff(z, axis=1)).mean())

    assert roughness(smooth) < roughness(rough) * 0.8, \
        f'{roughness(rough):.4f} -> {roughness(smooth):.4f}, barely changed'
    # The ridge is still a ridge: column means still rise and fall together.
    c1 = rough.mean(0)
    c2 = smooth.mean(0)
    corr = float(np.corrcoef(c1, c2)[0, 1])
    assert corr > 0.95, f'landform changed shape, correlation {corr:.3f}'


# ======================================================================
# terrain analysis
# ======================================================================

SECTION = 'terrain analysis'







def grid(n=41):
    return np.mgrid[0:n, 0:n]


def plane(n=41, gx=0.3, gy=0.0):
    yy, xx = grid(n)
    return xx * gx + yy * gy


def cone(n=41, height=10.0):
    yy, xx = grid(n)
    c = (n - 1) / 2
    r = np.hypot(xx - c, yy - c)
    return np.maximum(0.0, height * (1 - r / c))


def valley(n=41, slope_x=0.5, fall=0.2):
    """A V running top to bottom, tilted so water flows down it."""
    yy, xx = grid(n)
    return np.abs(xx - n // 2) * slope_x + (n - yy) * fall


@test('slope is the gradient, and a plane has one everywhere')
def _():
    s = slope(plane(gx=0.3))
    assert np.allclose(s, 0.3, atol=1e-9), f'got {s.min()}..{s.max()}'
    assert np.allclose(slope(np.zeros((9, 9))), 0.0)


@test('slope respects the sample spacing')
def _():
    z = plane(gx=1.0)
    assert np.allclose(slope(z, spacing=1.0), 1.0, atol=1e-9)
    assert np.allclose(slope(z, spacing=2.0), 0.5, atol=1e-9), \
        'halving the resolution should halve the measured slope'


@test('TPI is positive on a peak and negative in a pit')
def _():
    z = cone()
    t = tpi(z, radius=6)
    c = z.shape[0] // 2
    assert t[c, c] > 0, f'peak read {t[c, c]:.3f}'
    assert tpi(-z, radius=6)[c, c] < 0, 'an inverted cone should read as a pit'


@test('TPI is near zero on a plane, whatever its tilt')
def _():
    # A planar hillside is not a ridge. If TPI said otherwise, every slope
    # in the scene would collect the ridge material.
    for gx in (0.0, 0.2, 1.0):
        t = tpi(plane(gx=gx), radius=5)
        inner = t[8:-8, 8:-8]
        assert np.abs(inner).max() < 1e-6, \
            f'tilt {gx} gave TPI up to {np.abs(inner).max():.4f}'


@test('curvature separates concave from convex')
def _():
    yy, xx = grid(21)
    bowl = (xx - 10.0) ** 2 + (yy - 10.0) ** 2
    assert curvature(bowl)[10, 10] > 0, 'a bowl should read concave'
    assert curvature(-bowl)[10, 10] < 0, 'a dome should read convex'


@test('flow collects in the valley, not on the flanks')
def _():
    z = valley()
    f = flow_accumulation(z)
    n = z.shape[0]
    channel = f[:, n // 2].sum()
    flank = f[:, 3].sum()
    assert channel > flank * 10, \
        f'channel drained {channel:.0f}, flank {flank:.0f}'


@test('flow accumulates downhill, so totals grow towards the outlet')
def _():
    z = valley()
    f = flow_accumulation(z)
    col = f[:, z.shape[0] // 2]
    # The V falls towards increasing y, so the far end carries the most.
    assert col[-1] > col[0], f'{col[0]:.0f} at the top, {col[-1]:.0f} at the foot'


@test('every cell drains at least itself and no water is invented')
def _():
    rng = np.random.default_rng(2)
    z = rng.random((30, 30))
    f = flow_accumulation(z)
    assert f.min() >= 1.0, f'a cell drained {f.min()}, less than itself'
    assert f.max() <= z.size, \
        f'one cell drained {f.max():.0f} of {z.size} cells'


@test('a flat surface drains nowhere in particular')
def _():
    f = flow_accumulation(np.zeros((20, 20)))
    assert np.allclose(f, 1.0), 'flat ground routed water somewhere'


@test('landform finds a peak on a cone and a valley in a V')
def _():
    names = lambda z: {LANDFORMS[i] for i in np.unique(landform(z))}  # noqa: E731
    c = names(cone())
    assert 'peak' in c or 'ridge' in c, f'cone classified as {sorted(c)}'
    v = names(valley())
    assert 'valley' in v or 'hollow' in v, f'V classified as {sorted(v)}'


@test('a level plane is flat and a tilted one is slope')
def _():
    level = landform(np.zeros((41, 41)))
    assert LANDFORMS[level[20, 20]] == 'flat', \
        f'level ground read as {LANDFORMS[level[20, 20]]}'
    tilted = landform(plane(gx=0.5))
    assert LANDFORMS[tilted[20, 20]] == 'slope', \
        f'a hillside read as {LANDFORMS[tilted[20, 20]]}'


@test('landform thresholds are scale free')
def _():
    # Expressed in standard deviations, so the same numbers work on a gentle
    # survey and on a mountain. A scene scaled up ten times must classify
    # identically or the rule would need retuning per capture.
    z = valley()
    assert (landform(z) == landform(z * 10.0)).all(), \
        'scaling the terrain changed the classification'


@test('coverage sums to one at every cell')
def _():
    c = coverage(valley())
    total = sum(c.values())
    assert np.allclose(total, 1.0), \
        f'coverage sums to {total.min():.3f}..{total.max():.3f}'


@test('collected material goes to the channel, exposed to the ridge')
def _():
    # The claim the whole module exists to support.
    z = valley()
    c = coverage(z)
    n = z.shape[0]
    chan = c['collected'][:, n // 2].mean()
    flank = c['collected'][:, 3].mean()
    assert chan > flank, \
        f'collected material preferred the flank: {chan:.3f} vs {flank:.3f}'
    assert c['exposed'][:, 3].mean() > c['exposed'][:, n // 2].mean(), \
        'exposed material preferred the channel'


@test('sharpness makes the rule more decisive without moving it')
def _():
    z = valley()
    soft = coverage(z, sharpness=1.0)['collected']
    hard = coverage(z, sharpness=6.0)['collected']
    spread = lambda a: float(a.std())  # noqa: E731
    assert spread(hard) > spread(soft), \
        f'sharper rule was not more decisive: {spread(soft):.3f} vs {spread(hard):.3f}'
    # Same places, more strongly: where soft prefers collected, hard must too.
    agree = ((soft > 0.5) == (hard > 0.5)).mean()
    assert agree > 0.9, f'sharpening moved the boundary: {agree:.1%} agreement'


@test('custom rules replace the defaults')
def _():
    z = valley()
    c = coverage(z, rules={
        'steep': lambda m: m['slope'],
        'gentle': lambda m: 1.0 - m['slope'],
    })
    assert set(c) == {'steep', 'gentle'}
    assert np.allclose(sum(c.values()), 1.0)


@test('a featureless surface gives no preference')
def _():
    c = coverage(np.zeros((20, 20)))
    for name, a in c.items():
        assert np.isfinite(a).all(), f'{name} produced non-finite coverage'
    assert np.allclose(sum(c.values()), 1.0)


# ======================================================================
# patch search
# ======================================================================

SECTION = 'patch search'


def scene(xyz, scale=None, opacity=None):
    """A Splats from bare coordinates. Same shape as the helper in
    test_all.py, repeated here so this file runs on its own."""
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    n = len(xyz)
    if scale is None:
        scale = np.full((n, 3), 0.012, np.float32)
    if opacity is None:
        opacity = np.full(n, 0.8, np.float32)
    return Splats(xyz=xyz,
                  opacity=np.asarray(opacity, np.float32).reshape(n),
                  scale=np.asarray(scale, np.float32).reshape(n, 3),
                  rot=np.tile(np.float32([1, 0, 0, 0]), (n, 1)),
                  sh_dc=np.zeros((n, 3), np.float32),
                  sh_rest=np.zeros((n, 0, 3), np.float32), sh_degree=0)


def ground(extent=6.0, density=4000, seed=0, holes=0, hole_r=0.5):
    """A flat slab of splats, optionally with circular bites taken out.

    Holes are what a thin reconstruction looks like from above, and they
    are the thing the coverage filter exists to catch.
    """
    rng = np.random.default_rng(seed)
    n = int(density * extent * extent)
    xy = rng.uniform(-extent / 2, extent / 2, size=(n, 2))

    if holes:
        centres = rng.uniform(-extent / 2, extent / 2, size=(holes, 2))
        keep = np.ones(len(xy), dtype=bool)
        for c in centres:
            keep &= np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1]) > hole_r
        xy = xy[keep]

    z = rng.normal(0.0, 0.01, size=len(xy))
    return scene(np.column_stack([xy[:, 0], xy[:, 1], z]))


@test('clean ground is found at the defaults, with no loosening')
def _():
    s = ground()
    chosen, trail = auto_pick(s, 1.5, 4, up_axis=2, verbose=False,
                              stride=0.5, thickness=0.4)
    assert len(chosen) >= 4, f'only {len(chosen)} patches on clean ground'
    assert len(trail) == 1, \
        f'loosened {len(trail) - 1} setting(s) when the defaults sufficed'


@test('the stats out-parameter reports the tally and the grid')
def _():
    s = ground()
    stats = {}
    pick_patches(s, 1.5, 4, up_axis=2, stride=0.5, thickness=0.4,
                 verbose=False, stats=stats)
    for key in ('sparse', 'holes', 'tilt', 'buried', 'positions', 'viable'):
        assert key in stats, f'{key} missing from stats'
    assert stats['positions'] > 0


@test('progress is called, and ends exactly at the total')
def _():
    s = ground()
    seen = []
    pick_patches(s, 1.5, 4, up_axis=2, stride=0.5, thickness=0.4,
                 verbose=False, progress=lambda d, t, v: seen.append((d, t)))
    assert seen, 'progress was never called'
    done, total = seen[-1]
    assert done == total, f'last progress was {done} of {total}'
    assert all(d <= t for d, t in seen), 'progress went past the total'


@test('holed ground fails the defaults and passes once coverage is loosened')
def _():
    # The case from the desert capture: the defaults return almost nothing,
    # and the tally blames coverage.
    s = ground(holes=60, hole_r=0.42, seed=3)
    stats = {}
    strict = pick_patches(s, 1.5, 4, up_axis=2, stride=0.5, thickness=0.4,
                          verbose=False, stats=stats, min_cover=0.80)
    loose = pick_patches(s, 1.5, 4, up_axis=2, stride=0.5, thickness=0.4,
                         verbose=False, min_cover=0.55)
    assert len(loose) > len(strict), \
        (f'loosening coverage changed nothing: {len(strict)} -> {len(loose)}. '
         f'tally was {stats}')


@test('auto_pick finds what the defaults could not, and records how')
def _():
    s = ground(extent=9.0, holes=90, hole_r=0.42, seed=3)
    chosen, trail = auto_pick(s, 1.5, 4, up_axis=2, verbose=False,
                              stride=0.5, thickness=0.4)
    assert len(chosen) >= 4, f'only {len(chosen)} after loosening'
    assert len(trail) > 1, 'reported no loosening on a scene that needed it'
    # The trail has to be enough to repeat the run by hand.
    _, settings, _, _ = trail[-1]
    assert settings != trail[0][1], 'settings identical to the defaults'


@test('the trail is reported in the order it was tried')
def _():
    s = ground(holes=60, hole_r=0.42, seed=3)
    _, trail = auto_pick(s, 1.5, 4, up_axis=2, verbose=False,
                         stride=0.5, thickness=0.4)
    assert trail[0][0] == 'defaults', f'first step was {trail[0][0]}'
    counts = [got for _, _, _, got in trail]
    assert counts[-1] == max(counts), \
        f'stopped on a worse result than one it had already: {counts}'


@test('a scene that cannot work says so instead of looping')
def _():
    # Far too sparse for any setting to rescue.
    s = ground(density=40, seed=5)
    chosen, trail = auto_pick(s, 1.5, 4, up_axis=2, verbose=False,
                              stride=0.5, thickness=0.4)
    assert len(chosen) < 4, 'expected this scene to fail'
    assert len(trail) < 12, f'tried {len(trail)} times; should give up sooner'
    note = describe_trail(trail, 4, 1.5)
    assert 'not a tiling exemplar' in note, \
        f'failure was not stated plainly:\n{note}'


@test('module state survives a failed search')
def _():
    # auto_pick retires exhausted filters as it goes. Doing that to the
    # module-level table would leave the next call with fewer options than
    # the first, which is the kind of bug that only shows up on the second
    # scene somebody tries.
    from bozkir import patches
    before = list(patches.RELAXATIONS)
    auto_pick(ground(density=40, seed=5), 1.5, 4, up_axis=2, verbose=False,
              stride=0.5, thickness=0.4)
    assert patches.RELAXATIONS == before, 'RELAXATIONS was mutated'


@test('separation is loosened before anything that costs quality')
def _():
    # Plenty of viable candidates, all close together: the fix is to let
    # them sit nearer each other, not to lower the bar for what counts as
    # ground.
    s = ground(extent=3.2)
    chosen, trail = auto_pick(s, 1.2, 4, up_axis=2, verbose=False,
                              stride=0.4, thickness=0.4)
    if len(trail) > 1:
        assert 'separation' in trail[1][0], \
            f'first relaxation was {trail[1][0]}, not separation'


@test('the screening render agrees with the full one on what it returns')
def _():
    # The search screens with a coarse render for speed and measures the
    # survivors properly. If the two disagreed, the reported coverage would
    # not be the coverage that was filtered on.
    s = ground(holes=20, hole_r=0.4, seed=7)
    coarse = pick_patches(s, 1.5, 4, up_axis=2, stride=0.5, thickness=0.4,
                          verbose=False, min_cover=0.5, screen_cap=1_000)
    exact = pick_patches(s, 1.5, 4, up_axis=2, stride=0.5, thickness=0.4,
                         verbose=False, min_cover=0.5, screen_cap=10 ** 9)
    assert coarse and exact, 'no patches to compare'
    for (_, ca, _, ia), (_, cb, _, ib) in zip(coarse, exact):
        assert abs(ia['cover'] - ib['cover']) < 0.05, \
            (f'coverage at {ca} differs between screening and exact: '
             f"{ia['cover']:.3f} vs {ib['cover']:.3f}")


@test('every search script builds the same settings dict')
def _():
    # The three scripts each assembled this by hand, and one had already
    # drifted in whitespace. If they ever drift in content, a preview and an
    # export would search differently and --patches indices would silently
    # point at other patches.
    import argparse
    from bozkir.patches import SEARCH_KEYS, search_kwargs

    ns = argparse.Namespace(**{k: 1 for k in SEARCH_KEYS})
    kw = search_kwargs(ns)
    assert set(kw) == set(SEARCH_KEYS), f'got {sorted(kw)}'
    # thickness overrides when given, since the scripts derive it from size.
    assert search_kwargs(ns, 0.375)['thickness'] == 0.375


@test('resolved settings are written back for the preset to record')
def _():
    import argparse
    from bozkir.patches import apply_settings

    ns = argparse.Namespace(min_cover=0.8, min_separation=1.0, unrelated='x')
    apply_settings(ns, {'min_cover': 0.6, 'min_separation': 0.5,
                        'not_an_arg': 99})
    assert ns.min_cover == 0.6 and ns.min_separation == 0.5
    assert ns.unrelated == 'x', 'clobbered an unrelated field'
    assert not hasattr(ns, 'not_an_arg'), 'invented a field'


@test('the settings that produced a candidate list reproduce it exactly')
def _():
    # The trap this exists to close: indices from one search only mean
    # anything against another search run the same way.
    s = ground(extent=9.0, holes=90, hole_r=0.42, seed=3)
    chosen, trail = auto_pick(s, 1.5, 8, up_axis=2, verbose=False,
                              stride=0.5, thickness=0.4)
    settings = trail[-1][1]
    again = pick_patches(s, 1.5, 8, up_axis=2, verbose=False, **settings)
    assert len(again) == len(chosen), \
        f'{len(chosen)} patches became {len(again)} on replay'
    for (_, a, _, _), (_, b, _, _) in zip(chosen, again):
        assert a == b, f'candidate order changed on replay: {a} vs {b}'


@test('the cache key changes when the source file does')
def _():
    # Retraining a capture and writing it over the same filename used to
    # produce the same key, so the next run silently loaded the previous
    # model. The count printed was the old one, which is the only place it
    # showed.
    import os
    import tempfile
    import time
    from bozkir.scene import SceneConfig

    cfg = SceneConfig()
    fd, name = tempfile.mkstemp(suffix='.ply')
    os.close(fd)
    try:
        with open(name, 'wb') as f:
            f.write(b'x' * 100)
        first = cfg.key(name)
        time.sleep(1.1)                  # mtime has one-second resolution
        with open(name, 'wb') as f:
            f.write(b'y' * 200)
        second = cfg.key(name)
        assert first != second, 'same key for a different file'
        assert cfg.key() == SceneConfig().key(), \
            'the settings-only key stopped being stable'
    finally:
        os.unlink(name)


@test('a recalled search returns the same candidates in the same order')
def _():
    # The whole point. If a recall differed from a search, `--patches 2`
    # would mean one patch in preview and another in export.
    import tempfile
    from bozkir.patches import cached_search, search_kwargs

    s = ground(extent=7.0, seed=11)
    kw = dict(stride=0.5, thickness=0.4, min_cover=0.6)
    with tempfile.TemporaryDirectory() as d:
        first, _, hit1 = cached_search(s, 1.5, 2, kw, cache_dir=d)
        second, _, hit2 = cached_search(s, 1.5, 2, kw, cache_dir=d)
    assert not hit1 and hit2, f'hits were {hit1}, {hit2}'
    assert len(first) == len(second), \
        f'{len(first)} candidates became {len(second)} on recall'
    for (sa, ca, pa, ia), (sb, cb, pb, ib) in zip(first, second):
        assert ca == cb, f'centre moved: {ca} vs {cb}'
        assert abs(sa - sb) < 1e-9, f'score moved at {ca}'
        assert len(pa) == len(pb), f'patch size moved at {ca}'
        assert abs(ia['cover'] - ib['cover']) < 1e-9, f'cover moved at {ca}'


@test('a different scene does not hit another scene\'s cache')
def _():
    # The failure that would be worse than the slowness: patches cut from
    # somewhere else entirely, with the same indices.
    import tempfile
    from bozkir.patches import cached_search

    a = ground(extent=7.0, seed=1)
    b = ground(extent=7.0, seed=2)
    kw = dict(stride=0.5, thickness=0.4, min_cover=0.6)
    with tempfile.TemporaryDirectory() as d:
        ca, _, _ = cached_search(a, 1.5, 2, kw, cache_dir=d)
        cb, _, hit = cached_search(b, 1.5, 2, kw, cache_dir=d)
    assert not hit, 'a different scene hit the cache'
    centres_a = [c for _, c, _, _ in ca]
    centres_b = [c for _, c, _, _ in cb]
    assert centres_a != centres_b or len(ca) != len(cb), \
        'two different scenes produced identical candidates'


@test('changing a setting that moves candidates misses the cache')
def _():
    import tempfile
    from bozkir.patches import cached_search

    s = ground(extent=7.0, holes=20, hole_r=0.4, seed=4)
    with tempfile.TemporaryDirectory() as d:
        cached_search(s, 1.5, 2, dict(stride=0.5, thickness=0.4,
                                      min_cover=0.8), cache_dir=d)
        _, _, hit = cached_search(s, 1.5, 2, dict(stride=0.5, thickness=0.4,
                                                  min_cover=0.6), cache_dir=d)
    assert not hit, 'a looser min_cover reused a stricter search'


@test('separation is not part of the key, since it only re-selects')
def _():
    # It decides which accepted candidates are kept, not which are
    # accepted. Keeping it out of the key is what lets the sweep try
    # several separations without searching again.
    from bozkir.patches import search_key

    s = ground(extent=5.0)
    base = dict(stride=0.5, thickness=0.4, min_cover=0.6)
    k1 = search_key(s, 1.5, 2, dict(base, min_separation=1.0))
    k2 = search_key(s, 1.5, 2, dict(base, min_separation=0.5))
    assert k1 == k2, 'separation changed the key'


@test('choose respects separation and count without searching')
def _():
    from bozkir.patches import choose

    cands = [(10 - i, (float(i), 0.0), None, {}) for i in range(10)]
    wide = choose(cands, 5, size=1.0, min_separation=2.0)
    assert len(wide) <= 5
    for i, (_, (x, _), _, _) in enumerate(wide):
        for _, (ox, _), _, _ in wide[i + 1:]:
            assert abs(x - ox) >= 2.0, f'{x} and {ox} are too close'
    assert len(choose(cands, 3, 1.0, 0.0)) == 3, 'count not respected'


@test('a corrupt cache file is ignored rather than fatal')
def _():
    import tempfile
    from pathlib import Path as P
    from bozkir.patches import cached_search, search_key

    s = ground(extent=6.0)
    kw = dict(stride=0.6, thickness=0.4, min_cover=0.6)
    with tempfile.TemporaryDirectory() as d:
        key = search_key(s, 1.5, 2, kw)
        (P(d) / f'search_{key}.json').write_text('{not json')
        cands, _, hit = cached_search(s, 1.5, 2, kw, cache_dir=d)
    assert not hit, 'claimed a hit on a corrupt file'
    assert cands, 'no candidates after ignoring the corrupt file'


# ======================================================================
# class split
# ======================================================================

SECTION = 'class split'




SH_C0 = 0.28209479177387814




def patch(rgb, n=300, spread=0.02, seed=0):
    """A patch of one colour, with a little variation within it."""
    r = np.random.default_rng(seed)
    dc = ((np.array(rgb) + r.normal(0, spread, (n, 3))) - 0.5) / SH_C0
    return Splats(
        xyz=r.random((n, 3)).astype(np.float32),
        opacity=np.full(n, 0.8, np.float32),
        scale=np.full((n, 3), 0.01, np.float32),
        rot=np.tile(np.float32([1, 0, 0, 0]), (n, 1)),
        sh_dc=dc.astype(np.float32),
        sh_rest=np.zeros((n, 0, 3), np.float32), sh_degree=0)


def candidates(spec):
    """[(colour, how many)] -> a candidate list like pick_patches returns."""
    out, seed = [], 0
    for rgb, count in spec:
        for i in range(count):
            out.append((1.0 - seed * 0.001, (float(seed), 0.0),
                        patch(rgb, seed=seed), {}))
            seed += 1
    return out


SAND = (0.72, 0.66, 0.55)
SCRUB = (0.22, 0.28, 0.18)
GRAVEL = (0.45, 0.44, 0.43)


@test('two materials split into two groups')
def _():
    g = split_classes(candidates([(SAND, 8), (SCRUB, 6)]), classes=2)
    assert len(g) == 2, f'got {len(g)} groups'
    assert all(len(x) == 4 for x in g), f'sizes {[len(x) for x in g]}'


@test('the groups are the materials, not an arbitrary cut')
def _():
    g = split_classes(candidates([(SAND, 8), (SCRUB, 6)]), classes=2)
    means = [np.array([appearance(c[2]) for c in grp])[:, :3].mean(0)
             for grp in g]
    # One group pale, one dark. Which comes first does not matter.
    lum = sorted(float(m.mean()) for m in means)
    assert lum[0] < 0.35 and lum[1] > 0.55, \
        f'group luminances {lum}, expected one dark and one pale'


@test('members of a group look like each other')
def _():
    # Each group still has to tile, so within-group similarity is the
    # property that must survive the split.
    g = split_classes(candidates([(SAND, 8), (SCRUB, 8)]), classes=2)
    for gi, grp in enumerate(g):
        f = np.array([appearance(c[2]) for c in grp])
        spread = float(np.linalg.norm(f - f.mean(0), axis=1).mean())
        assert spread < 0.1, f'group {gi} spread {spread:.3f} is too varied'


@test('separation reports a real split as large and a fake one as small')
def _():
    two = split_classes(candidates([(SAND, 8), (SCRUB, 6)]), classes=2)
    one = split_classes(candidates([(SAND, 14)]), classes=2)
    assert class_separation(two) > 5, \
        f'a genuine two-material split scored {class_separation(two):.2f}'
    assert class_separation(one) < 3, \
        (f'a single material split scored {class_separation(one):.2f}; '
         'noise was mistaken for a second class')


@test('three materials split three ways')
def _():
    g = split_classes(candidates([(SAND, 6), (SCRUB, 6), (GRAVEL, 6)]),
                      classes=3)
    assert len(g) == 3, f'got {len(g)} groups'
    means = sorted(float(np.array([appearance(c[2]) for c in grp])[:, :3]
                         .mean())
                   for grp in g)
    assert means[1] - means[0] > 0.05 and means[2] - means[1] > 0.05, \
        f'three groups but not three materials: {means}'


@test('the split is deterministic')
def _():
    # --patches indices refer to positions in this list, so a run that
    # ordered its groups differently would silently mean other patches.
    c = candidates([(SAND, 8), (SCRUB, 6)])
    a = split_classes(c, classes=2)
    b = split_classes(c, classes=2)
    ca = [[x[1] for x in g] for g in a]
    cb = [[x[1] for x in g] for g in b]
    assert ca == cb, 'two runs produced different groups'


@test('the largest group comes first')
def _():
    g = split_classes(candidates([(SAND, 12), (SCRUB, 5)]), classes=2,
                      per_class=99)
    assert len(g[0]) >= len(g[1]), f'sizes {[len(x) for x in g]}'


@test('a group too small to tile is returned rather than padded')
def _():
    # A capture with one good material and a trace of another should hand
    # back the trace as it is. Padding it from the other class would make
    # a tile set that silently mixes materials.
    g = split_classes(candidates([(SAND, 10), (SCRUB, 2)]), classes=2,
                      per_class=4)
    small = min(g, key=len)
    assert len(small) == 2, f'small group came back with {len(small)}'


@test('asking for one class is the old behaviour')
def _():
    c = candidates([(SAND, 8)])
    assert split_classes(c, classes=1) == [c[:4]]


@test('fewer candidates than classes does not divide by zero')
def _():
    c = candidates([(SAND, 1)])
    g = split_classes(c, classes=4)
    assert len(g) == 1 and len(g[0]) == 1


@test('a single material still selects a usable set')
def _():
    # The common case, and it must not get worse: one material, one group,
    # four patches that look alike.
    c = candidates([(SAND, 12)])
    g = split_classes(c, classes=2, per_class=4)
    best = max(g, key=len)
    assert len(best) == 4
    direct = select_similar(c, 4)
    f1 = np.array([appearance(x[2]) for x in best])
    f2 = np.array([appearance(x[2]) for x in direct])
    s1 = float(np.linalg.norm(f1 - f1.mean(0), axis=1).mean())
    s2 = float(np.linalg.norm(f2 - f2.mean(0), axis=1).mean())
    assert s1 < s2 * 2.5, \
        f'splitting a single material gave a worse set: {s1:.4f} vs {s2:.4f}'


@test('a nearly constant dimension cannot outvote the real difference')
def _():
    # Regression. The signature is three colour means and three colour
    # spreads. The spreads barely differ between patches of one scene, so
    # standardising each dimension by its own deviation multiplied what was
    # left of them - noise - up to the size of the real colour gap. In a
    # scene of ten pale patches and two dark ones, one of the dark ones was
    # clustered with the pale.
    g = split_classes(candidates([(SAND, 10), (SCRUB, 2)]), classes=2,
                      per_class=99)
    small = min(g, key=len)
    assert len(small) == 2, \
        (f'the dark group came back with {len(small)} of 2; '
         'a near-constant feature is being amplified again')
    f = np.array([appearance(c[2]) for c in small])
    assert float(f[:, :3].mean()) < 0.35, 'the small group is not the dark one'


@test('a lopsided split still finds the minority material')
def _():
    for minority in (2, 3, 5):
        g = split_classes(candidates([(SAND, 20), (SCRUB, minority)]),
                          classes=2, per_class=99)
        small = min(g, key=len)
        assert len(small) == minority, \
            f'{minority} dark patches came back as {len(small)}'



# ======================================================================
# class-aware tile sets
# ======================================================================

SECTION = 'class tiles'


def _class_patch(level, n=900, seed=0):
    """A flat square patch of one tone, for building tile sets from."""
    r = np.random.default_rng(seed)
    xy = r.uniform(-0.75, 0.75, (n, 2))
    xyz = np.column_stack([xy[:, 0], xy[:, 1], r.normal(0, 0.005, n)])
    dc = ((np.full((n, 3), level) + r.normal(0, 0.01, (n, 3))) - 0.5) / SH_C0
    return Splats(xyz=xyz.astype(np.float32),
                  opacity=np.full(n, 0.9, np.float32),
                  scale=np.full((n, 3), 0.02, np.float32),
                  rot=np.tile(np.float32([1, 0, 0, 0]), (n, 1)),
                  sh_dc=dc.astype(np.float32),
                  sh_rest=np.zeros((n, 0, 3), np.float32), sh_degree=0)


@test('every class carries the same set of edge codes')
def _():
    # The property the whole scheme rests on. A cell picks a code from its
    # neighbours and a class from the terrain, independently - which only
    # works if every class has a tile for every code. If one class were
    # missing a combination, a cell needing it would have to fall back to
    # another class and the terrain rule would be silently overruled by the
    # matching constraint.
    from bozkir.wang import build_tile_set

    def tile_codes(colour_offset):
        h = [_class_patch(0.6 + colour_offset, seed=i) for i in range(2)]
        v = [_class_patch(0.6 + colour_offset, seed=10 + i) for i in range(2)]
        _, codes = build_tile_set(h, v, 1.0, up_axis=2)
        return sorted(tuple(int(x) for x in c) for c in codes)

    assert tile_codes(0.0) == tile_codes(-0.35), \
        'two classes produced different code sets'


@test('the code set covers every neighbour pair')
def _():
    from bozkir.wang import build_tile_set

    h = [_class_patch(0.6, seed=i) for i in range(2)]
    v = [_class_patch(0.6, seed=10 + i) for i in range(2)]
    _, codes = build_tile_set(h, v, 1.0, up_axis=2)
    got = {tuple(int(x) for x in c) for c in codes}
    assert len(got) == 16, f'{len(got)} distinct codes, expected 16'
    # For any already-placed left and top neighbour there is a tile.
    for w in (0, 1):
        for nn in (0, 1):
            assert any(c[3] == w and c[0] == nn for c in got), \
                f'no tile for west={w}, north={nn}'




# ======================================================================
# generated terrain
# ======================================================================

SECTION = 'erosion'


@test('fractal noise is in range and varies with the seed')
def _():
    a = fbm((32, 32), seed=1)
    b = fbm((32, 32), seed=2)
    assert a.shape == (32, 32)
    assert 0.0 <= a.min() and a.max() <= 1.0, f'{a.min()}..{a.max()}'
    assert not np.allclose(a, b), 'two seeds gave the same field'
    assert np.allclose(a, fbm((32, 32), seed=1)), 'not deterministic'


@test('ridged noise has sharper crests than fractal noise')
def _():
    # Folding each octave about its midpoint turns smooth maxima into
    # creases, so the high end of the distribution is reached by fewer
    # cells: a range rather than a field of mounds.
    def creasing(a):
        # Mean curvature relative to the field's own scale. A crease is a
        # discontinuity in the first derivative, so it shows in the second.
        p = np.pad(a, 1, mode='edge')
        lap = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
               - 4 * a)
        return float(np.abs(lap).mean() / max(a.std(), 1e-9))

    for seed in (4, 7, 11):
        f = creasing(fbm((96, 96), seed=seed))
        r = creasing(ridged((96, 96), seed=seed))
        assert r > f * 1.3, \
            f'seed {seed}: ridged {r:.3f} is not sharper than fractal {f:.3f}'


@test('erosion cuts a drainage network into noise')
def _():
    # The claim the module exists for. Noise has scattered sinks; a
    # landscape has a branching network, so more of it should carry
    # meaningful flow after erosion than before.
    base = fbm((96, 96), seed=2)
    z, _ = erode(base, iterations=40)

    def network(a):
        lf = np.log1p(flow_accumulation(a))
        return float((lf > 2.5).mean())

    assert network(z) > network(base) * 1.5, \
        f'network {network(base):.3f} -> {network(z):.3f}, barely changed'


@test('after erosion the low ground is the draining ground')
def _():
    # A real valley is both low and wet. On noise the two are only weakly
    # related; erosion is what ties them together.
    base = fbm((96, 96), seed=2)
    z, _ = erode(base, iterations=40)

    def tie(a):
        lf = np.log1p(flow_accumulation(a))
        return float(np.corrcoef(a.ravel(), lf.ravel())[0, 1])

    assert tie(z) < tie(base) - 0.05, \
        f'height-flow correlation {tie(base):+.3f} -> {tie(z):+.3f}'


@test('sediment settles where the water arrives')
def _():
    # The map that lets the material rule read what the generator did
    # instead of inferring it from the shape afterwards.
    base = fbm((96, 96), seed=6)
    z, sed = erode(base, iterations=40)
    lf = np.log1p(flow_accumulation(z))
    corr = float(np.corrcoef(lf.ravel(), sed.ravel())[0, 1])
    assert corr > 0.2, f'sediment barely follows drainage: {corr:+.3f}'


@test('erosion conserves the shape of the field, not its detail')
def _():
    # It should carve, not replace: the eroded terrain must still resemble
    # what it started from at coarse scale, or the noise settings would be
    # doing nothing.
    base = fbm((64, 64), seed=8)
    z, _ = erode(base, iterations=40)

    def coarse(a):
        return a.reshape(8, 8, 8, 8).mean(axis=(1, 3)).ravel()

    corr = float(np.corrcoef(coarse(base), coarse(z))[0, 1])
    assert corr > 0.9, f'erosion changed the landscape shape: {corr:.3f}'


@test('more iterations deepen valleys without changing where they are')
def _():
    base = fbm((64, 64), seed=9)
    short, _ = erode(base, iterations=10)
    long, _ = erode(base, iterations=60)
    lf_s = np.log1p(flow_accumulation(short))
    lf_l = np.log1p(flow_accumulation(long))
    agree = float(np.corrcoef(lf_s.ravel(), lf_l.ravel())[0, 1])
    assert agree > 0.5, f'the network moved: {agree:.3f}'


@test('generate returns two normalised fields')
def _():
    z, sed = generate(size=64, seed=1, iterations=20)
    for name, a in (('height', z), ('sediment', sed)):
        assert a.shape == (64, 64), f'{name} is {a.shape}'
        assert abs(a.min()) < 1e-9 and abs(a.max() - 1) < 1e-9, \
            f'{name} is not normalised: {a.min()}..{a.max()}'


@test('a flat input does not divide by zero')
def _():
    z, sed = erode(np.zeros((32, 32)), iterations=5)
    assert np.isfinite(z).all() and np.isfinite(sed).all()


@test('the generator writes what the viewer reads')
def _():
    # The ladder: terrain_gen writes the same pair heightmap.py does, so
    # the renderer needs no change to use generated terrain.
    import tempfile
    from terrain_gen import main as _tg_main
    import contextlib
    import io

    with tempfile.TemporaryDirectory() as d:
        with contextlib.redirect_stdout(io.StringIO()):
            rc = _tg_main(['--name', 'x', '--out', d, '--size', 64,
                           '--iterations', 10] and
                          ['--name', 'x', '--out', d, '--size', '64',
                           '--iterations', '10'])
        assert rc == 0
        png = Path(d) / 'x.height.png'
        side = Path(d) / 'x.height.json'
        assert png.exists() and side.exists()
        img = read_height_png(png)
        assert img.shape == (64, 64), f'shape {img.shape}'
        meta = json.loads(side.read_text())
        assert meta['width'] == 64 and meta['height'] == 64
        assert (Path(d) / meta['sediment']).exists(), 'sediment map missing'




@test('a height grid can be rasterised from a point cloud')
def _():
    # A drone survey has a DSM; a phone capture does not. Both contain the
    # ground, so the surface is read out of the points instead.
    rng = np.random.default_rng(0)
    n = 30000
    x = rng.uniform(-3, 3, n)
    y = rng.uniform(-3, 3, n)
    z = 0.3 * (x * x + y * y) / 9 + rng.normal(0, 0.01, n)
    g = height_from_points(np.column_stack([x, y, z]), resolution=48)
    assert g.shape == (48, 48)
    assert np.isfinite(g).all(), 'gaps left in the grid'
    c = 24
    assert g[c, c] < g[c, 4], \
        f'the bowl came out the wrong way up: {g[c, c]:.3f} vs {g[c, 4]:.3f}'


@test('the ground is read under the clutter, not through it')
def _():
    # A percentile rather than the minimum: the lowest point in a column is
    # as likely to be a floater below the surface as the surface itself,
    # and one drags the whole cell into a spike.
    rng = np.random.default_rng(1)
    n = 20000
    x = rng.uniform(-2, 2, n)
    y = rng.uniform(-2, 2, n)
    z = np.full(n, 1.0) + rng.normal(0, 0.01, n)
    # A handful of floaters far below.
    z[:40] = -5.0
    g = height_from_points(np.column_stack([x, y, z]), resolution=24)
    assert abs(float(np.median(g)) - 1.0) < 0.1, \
        f'floaters pulled the surface to {np.median(g):.2f}'


@test('empty cells are filled rather than left as holes')
def _():
    # A capture is a polygon; the grid is a rectangle. Gaps left as NaN
    # would poison every derivative computed from the field.
    rng = np.random.default_rng(2)
    n = 8000
    a = rng.uniform(0, 2 * np.pi, n)
    r = rng.uniform(0, 1, n) ** 0.5
    x, y = r * np.cos(a), r * np.sin(a)      # a disc, so corners are empty
    z = x * 0.2
    g = height_from_points(np.column_stack([x, y, z]), resolution=32)
    assert np.isfinite(g).all(), 'corners left unfilled'
    raw = height_from_points(np.column_stack([x, y, z]), resolution=32,
                             fill=False)
    assert not np.isfinite(raw).all(), 'fill=False should leave the holes'


@test('the grid is square in world units, not stretched to the cloud')
def _():
    # A grid stretched to the bounding box would make slope and flow depend
    # on which way the capture happened to be walked.
    rng = np.random.default_rng(3)
    n = 20000
    x = rng.uniform(-4, 4, n)                 # twice as wide as tall
    y = rng.uniform(-1, 1, n)
    z = np.hypot(x, y) * 0.1
    g = height_from_points(np.column_stack([x, y, z]), resolution=40)
    # A radial surface sampled on a square grid stays radial: the slope
    # along each axis should be comparable.
    gx = float(np.abs(np.diff(g, axis=1)).mean())
    gy = float(np.abs(np.diff(g, axis=0)).mean())
    assert 0.3 < gx / max(gy, 1e-9) < 3.0, \
        f'axes scaled differently: {gx:.4f} across, {gy:.4f} down'




@test('a height field survives the round trip at 16 bits')
def _():
    # The encoding exists because 16-bit greyscale does not reach a browser
    # at 16 bits. This checks the replacement actually carries the precision
    # it claims, reading the channels the way the browser does.
    import tempfile
    n = 128
    yy, xx = np.mgrid[0:n, 0:n]
    a = np.sin(xx / 20.0) * 0.4 + yy / (n * 3.0)      # a gentle slope
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'h.png'
        norm = write_height_png(a, path)
        back = read_height_png(path)
    err = float(np.abs(back - norm).max())
    assert err < 2.0 / 65535, f'lost precision: max error {err:.2e}'


@test('the encoding carries far more than 256 levels')
def _():
    # Eight bits is what 16-bit greyscale actually delivered to the page.
    # A gentle slope in 256 levels has flat plateaus, and flow routing
    # treats every flat cell as a sink - so this is not only about looks.
    import tempfile
    n = 256
    a = np.linspace(0, 1, n * n).reshape(n, n)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'h.png'
        write_height_png(a, path)
        back = read_height_png(path)
    levels = len(np.unique(np.round(back * 65535)))
    assert levels > 10000, f'only {levels} distinct heights survived'


@test('reading one channel alone would have been eight bits')
def _():
    # The point of the split, stated as a test: the red channel by itself -
    # which is all a browser gave back from a 16-bit greyscale file - holds
    # 256 levels at most. The green channel is what recovers the rest.
    import tempfile
    n = 256
    a = np.linspace(0, 1, n * n).reshape(n, n)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'h.png'
        write_height_png(a, path)
        rgb = np.asarray(Image.open(path).convert('RGB'))
    assert len(np.unique(rgb[..., 0])) <= 256
    assert len(np.unique(rgb[..., 1])) > 1, 'green carries nothing'


@test('older greyscale height files still read')
def _():
    # Files written before the encoding changed carry no marker and were
    # 16-bit greyscale. They must still load rather than be misread as
    # packed bytes.
    import tempfile
    a = (np.linspace(0, 1, 64 * 64).reshape(64, 64) * 65535).astype(np.uint16)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'old.png'
        Image.fromarray(a).save(path)
        back = read_height_png(path, encoding=None)
    assert abs(float(back.max()) - 1.0) < 1e-3 and float(back.min()) < 1e-3


@test('both writers mark the encoding in the sidecar')
def _():
    # The browser reads the encoding from the sidecar. A writer that
    # forgot to mark it would send packed bytes to be read as greyscale.
    import tempfile
    import contextlib
    import io
    from terrain_gen import main as tg

    with tempfile.TemporaryDirectory() as d:
        with contextlib.redirect_stdout(io.StringIO()):
            tg(['--name', 'e', '--out', d, '--size', '48',
                '--iterations', '5'])
        meta = json.loads((Path(d) / 'e.height.json').read_text())
    assert meta.get('encoding') == 'rg16', f"sidecar says {meta.get('encoding')}"



# ======================================================================
# the rule, Python against the viewer
# ======================================================================

SECTION = 'rule parity'

from bozkir import landform as rule                               # noqa: E402


def _rule_cases():
    from bozkir.erosion import fbm
    rng = np.random.default_rng(0)
    cases = []

    def add(z, n, opts, classes=2):
        cases.append({'z': np.asarray(z, float).ravel().tolist(), 'n': n,
                      'classes': classes, 'opts': opts})

    for seed in range(3):
        z = fbm((64, 64), seed=seed)
        add(z, 16, {})
        add(z, 16, {'spacing': 2.5, 'balance': 0.3, 'coherence': 2,
                    'altitude': 0.4, 'sharpness': 3})
        add(z, 16, {'balance': 0.5, 'coherence': 1, 'altitude': 0.0})
        add(z, 16, {'balance': 0.3,
                    'sediment': rng.random(64 * 64).tolist()})
    add(np.zeros((32, 32)), 8, {'balance': 0.3})    # every cell ties
    add(np.zeros((32, 32)), 8, {})
    add(fbm((24, 24), seed=9), 24, {'balance': 0.7})  # no oversampling
    add(fbm((60, 60), seed=4), 20, {}, classes=3)
    return cases


def _run_js(cases):
    import json
    import shutil
    import subprocess
    if not shutil.which('node'):
        return None
    here = Path(__file__).resolve().parent
    out = subprocess.run(['node', str(here / 'landform_oracle.mjs')],
                         input=json.dumps({'cases': cases}),
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@test('the Python rule picks the same class as the viewer in every cell')
def _():
    # The validation script and the figures measure bozkir/landform.py. If
    # it drifted from web/landform.js they would be measuring a rule nobody
    # sees. Skips, rather than fails, on a machine without Node.
    cases = _rule_cases()
    js = _run_js(cases)
    if js is None:
        print('      (skipped: node not found)')
        return
    for k, (c, j) in enumerate(zip(cases, js)):
        o = dict(c['opts'])
        if 'sediment' in o:
            o['sediment'] = np.array(o['sediment'])
        p = rule.classify(np.array(c['z']), c['n'], c['classes'], **o)
        diff = int((p['cls'] != np.array(j['cls'])).sum())
        assert diff == 0, f'case {k}: {diff} cells differ from the viewer'
        err = float(np.abs(p['strength'] - np.array(j['strength'])).max())
        assert err < 1e-9, f'case {k}: strength differs by {err:.2e}'
        assert p['source'] == j['source'], f'case {k}: source differs'


@test('balance gives exactly the share asked for')
def _():
    from bozkir.erosion import fbm
    z = fbm((96, 96), seed=3).ravel()
    for share in (0.2, 0.5, 0.8):
        cls = rule.classify(z, 24, 2, balance=share)['cls']
        got = float((cls == 0).mean())
        assert abs(got - share) < 1.5 / cls.size, \
            f'asked for {share:.0%}, got {got:.1%}'


@test('the Python map functions match their definitions on known surfaces')
def _():
    n = 16
    j, i = np.mgrid[0:n, 0:n].astype(float)
    plane = (0.5 * i).ravel()
    s = rule.slope(plane, n)
    assert np.allclose(s.reshape(n, n)[:, 1:-1], 0.5), 'slope of a plane'
    assert np.allclose(rule.tpi(plane, n, 2).reshape(n, n)[3:-3, 3:-3], 0), \
        'TPI of a plane is not zero in the interior'
    v = np.abs(i - n // 2).ravel()           # V-shaped valley along a column
    acc = rule.flow(v, n).reshape(n, n)
    assert acc[:, n // 2].max() == acc.max(), 'flow did not collect in the valley'


# ======================================================================
# validating the rule against a survey
# ======================================================================

SECTION = 'rule validation'

from bozkir import validate as val                                # noqa: E402


def _geotiff(path, a, dx, x0=500000.0, y0=4000000.0):
    from PIL.TiffImagePlugin import ImageFileDirectory_v2
    ifd = ImageFileDirectory_v2()
    ifd[33550] = (dx, dx, 0.0)
    ifd.tagtype[33550] = 12
    ifd[33922] = (0.0, 0.0, 0.0, x0, y0, 0.0)
    ifd.tagtype[33922] = 12
    Image.fromarray(a).save(path, tiffinfo=ifd)


@test('a GeoTIFF is placed where its tags say')
def _():
    import tempfile
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    a[0, 0] = -9999
    with tempfile.TemporaryDirectory() as d:
        _geotiff(Path(d) / 'a.tif', a, 0.5, x0=10.0, y0=20.0)
        r = val.read_geotiff(Path(d) / 'a.tif')
    assert not r.valid[0, 0] and r.valid[1:].all(), 'nodata not masked'
    close(r.dx, 0.5)
    row, col = r.world_to_pixel(10.25 + 0.5 * 2, 20.0 - 0.25)
    close(float(row), 0.0, msg='row')
    close(float(col), 2.0, msg='col')
    close(float(val.sample_bilinear(r, 11.25, 19.75)), 2.0, msg='sample')


@test('the validation finds a planted relationship and not an absent one')
def _():
    from bozkir.erosion import fbm
    n = 16
    z = fbm((64, 64), seed=5)
    low = z.reshape(-1) < np.quantile(z, 0.3)
    frac = low.reshape(64, 64).reshape(n, 4, n, 4).mean(axis=(1, 3)).ravel()
    planted = val.evaluate(z.ravel(), frac, n, 1.0, altitude=0.8)
    assert planted['rule']['p'] < 0.05, \
        f"missed scrub planted on low ground: p {planted['rule']['p']:.3f}"
    assert planted['rule']['kappa'] > planted['rule, flipped']['kappa']

    unrelated = fbm((n, n), seed=77).ravel()
    frac = (unrelated > np.quantile(unrelated, 0.7)).astype(float)
    none = val.evaluate(z.ravel(), frac, n, 1.0)
    assert none['rule']['p'] > 0.01, \
        f"found a relationship that is not there: p {none['rule']['p']:.3f}"


@test('kappa is 1 for agreement, near 0 for chance, -ish for opposition')
def _():
    a = np.array([1, 1, 0, 0, 1, 0, 1, 0], bool)
    close(val.kappa(a, a), 1.0)
    assert val.kappa(a, ~a) < 0
    close(val.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)


# ====================================================================== run

if __name__ == '__main__':
    for section, name, msg in FAILURES:
        print(f'FAIL  [{section}] {name}\n      {msg}')
    if ONLY and not PASSED and not FAILURES:
        print(f'nothing matched {" ".join(ONLY)}')
    print(f'\n{PASSED} passed, {len(FAILURES)} failed')
    sys.exit(1 if FAILURES else 0)