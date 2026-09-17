"""Tests for the automatic patch search in bozkir/patches.py.

    python tests/test_autopick.py

Numpy only. The scenes are synthetic so the right answer is known by
construction: a slab of uniform ground should yield patches at the default
settings, and a slab with holes punched in it should yield them only after
the coverage filter is loosened.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bozkir.ply import Splats  # noqa: E402
from bozkir.patches import auto_pick, describe_trail, pick_patches  # noqa: E402

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
        except Exception as e:                        # noqa: BLE001
            FAILURES.append((name, f'{type(e).__name__}: {e}'))
        return fn
    return wrap


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


if __name__ == '__main__':
    for name, msg in FAILURES:
        print(f'FAIL  {name}\n      {msg}')
    print(f'\n{PASSED} passed, {len(FAILURES)} failed')
    sys.exit(1 if FAILURES else 0)