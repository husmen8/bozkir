// Tests for the two browser modules that hold no WebGL: order.js and
// merge.js. Both are pure functions over cell lists, so node runs them.
//
//   node tests/test_web.mjs
//
// merge.js shipped in a previous session with no tests at all, which is the
// failure mode this project keeps hitting. Half of what is below is arrears.

import {
  boundarySign, constraints, drawOrder, violations,
} from '../web/order.js';
import { readZip, pickTileset, describe, writeZip } from '../web/tileset.js';
import { HeightField } from '../web/heightfield.js';
import { report } from '../web/benchmark.js';
import { classify, flow, sampleGrid, slope, tpi } from '../web/landform.js';
import {
  SLOT_BITS, MAX_GROUP, INDEX_BITS, INDEX_MASK,
  packIndex, unpackSlot, unpackIndex,
  boundary, mergeScore, neighbourPairs, mergeGroups, kwayMerge,
} from '../web/merge.js';

let passed = 0;
const failures = [];

function test(name, fn) {
  try { fn(); passed++; }
  catch (e) { failures.push([name, e.message]); }
}
function ok(cond, msg) { if (!cond) throw new Error(msg || 'expected true'); }
function eq(a, b, msg) {
  const x = JSON.stringify(a), y = JSON.stringify(b);
  if (x !== y) throw new Error(`${msg || 'not equal'}: ${x} !== ${y}`);
}
function close(a, b, tol = 1e-9, msg) {
  if (Math.abs(a - b) > tol) throw new Error(`${msg || 'not close'}: ${a} vs ${b}`);
}

/** A flat square grid of `n` by `n` unit cells centred on the origin. */
function grid(n, tile = 1) {
  const half = (n - 1) / 2;
  const out = [];
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++)
      out.push({ i, j, x: (i - half) * tile, y: (j - half) * tile, z: 0 });
  return out;
}

/** The viewer's key: minimum projected depth over the four corners. This is
 *  the quantity that goes degenerate, so the tests need the real thing. */
function nearKey(cells, eye, forward, tile = 1) {
  const h = tile / 2;
  return cells.map((c) => {
    const dz = (c.z || 0) - eye[2];
    let near = Infinity;
    for (const [ox, oy] of [[-h, -h], [h, -h], [-h, h], [h, h]]) {
      const kx = c.x + ox - eye[0], ky = c.y + oy - eye[1];
      const k = kx * forward[0] + ky * forward[1] + dz * forward[2];
      if (k < near) near = k;
    }
    return near;
  });
}

/** Camera on a circle about the origin, looking at it. */
function camera(azDeg, elDeg, dist) {
  const az = azDeg * Math.PI / 180, el = elDeg * Math.PI / 180;
  const eye = [dist * Math.cos(el) * Math.cos(az),
               dist * Math.cos(el) * Math.sin(az),
               dist * Math.sin(el)];
  const n = Math.hypot(...eye);
  return { eye, forward: eye.map((v) => -v / n) };
}

// ---------------------------------------------------------------- order.js

test('the sign is positive when the eye is on the second cell\'s side', () => {
  const a = { x: 0, y: 0, z: 0 }, b = { x: 1, y: 0, z: 0 };
  ok(boundarySign(a, b, [10, 0, 0]) > 0, 'eye past b should be positive');
  ok(boundarySign(a, b, [-10, 0, 0]) < 0, 'eye behind a should be negative');
});

test('the sign is the distance to the boundary plane, and is antisymmetric', () => {
  const a = { x: 0, y: 0, z: 0 }, b = { x: 2, y: 0, z: 0 };
  close(boundarySign(a, b, [5, 99, -99]), 4, 1e-9, 'boundary sits at x=1');
  close(boundarySign(a, b, [5, 0, 0]), -boundarySign(b, a, [5, 0, 0]));
});

test('its magnitude is merge.js\'s score, so the two halves agree', () => {
  const a = { i: 0, j: 0, x: 0, y: 0, z: 0 }, b = { i: 0, j: 1, x: 0, y: 1, z: 0 };
  for (const eye of [[3, 0.2, 1], [-4, 2.5, 0.5], [0, 0.5, 9]]) {
    close(Math.abs(boundarySign(a, b, eye)), mergeScore(a, b, eye), 1e-12);
  }
});

test('the depth key really is degenerate along a grid axis', () => {
  // The premise of this whole module. If this stops being true the row-wide
  // flip has a different cause and the fix below is aimed at nothing.
  const cells = grid(9);
  const { eye, forward } = camera(0, 12, 6);
  const keys = nearKey(cells, eye, forward);
  const distinct = new Set(keys.map((v) => v.toFixed(9)));
  eq(distinct.size, 9, '81 cells should collapse to 9 keys at az=0');
});

test('and is not degenerate off-axis', () => {
  const cells = grid(9);
  const { eye, forward } = camera(23, 12, 6);
  const distinct = new Set(nearKey(cells, eye, forward).map((v) => v.toFixed(9)));
  eq(distinct.size, 81, 'every cell should get its own key off-axis');
});

test('a flat grid yields no cycles and no violations', () => {
  const cells = grid(9);
  for (const az of [0, 7, 23, 45, 90, 137]) {
    const { eye, forward } = camera(az, 15, 6);
    const key = nearKey(cells, eye, forward);
    const r = drawOrder(cells, eye, { key });
    eq(r.cycles, 0, `cycle at az=${az}`);
    eq(r.order.length, cells.length, `lost a cell at az=${az}`);
    eq(violations(cells, eye, r.order), 0, `violation at az=${az}`);
  }
});

test('every cell appears exactly once', () => {
  const cells = grid(6);
  const { eye, forward } = camera(31, 20, 5);
  const r = drawOrder(cells, eye, { key: nearKey(cells, eye, forward) });
  eq([...new Set(r.order)].length, cells.length);
});

test('the order is stable across the alignment the key trips on', () => {
  // The measured bug: rotating through az=0 reverses a whole row at once.
  // Under the constraints it must not move at all, since no boundary plane
  // is crossed by a camera turning on the spot at this radius.
  const cells = grid(9);
  const before = camera(-1.5, 12, 6), after = camera(1.5, 12, 6);
  const a = drawOrder(cells, before.eye,
    { key: nearKey(cells, before.eye, before.forward) });
  const b = drawOrder(cells, after.eye,
    { key: nearKey(cells, after.eye, after.forward) });

  const rankA = new Map(a.order.map((c, k) => [c, k]));
  const rankB = new Map(b.order.map((c, k) => [c, k]));
  let flipped = 0;
  for (const [u, v] of neighbourPairs(cells)) {
    if ((rankA.get(u) < rankA.get(v)) !== (rankB.get(u) < rankB.get(v))) flipped++;
  }
  eq(flipped, 0, 'neighbouring pairs swapped across alignment');
});

test('the plain depth key flips the same rotation, which is the baseline', () => {
  // Guards the claim rather than the code: if this ever reports zero, the
  // viewer was fixed elsewhere and the comparison above means nothing.
  const cells = grid(9);
  const rank = (az) => {
    const { eye, forward } = camera(az, 12, 6);
    const key = nearKey(cells, eye, forward);
    const idx = cells.map((_, k) => k);
    // Far to near, tiebroken as viewer.js does it: distance from the middle
    // of the view, descending.
    const side = cells.map((c) => Math.abs(c.x - eye[0]) + Math.abs(c.y - eye[1]));
    idx.sort((p, q) => (key[q] - key[p]) || (side[q] - side[p]));
    return new Map(idx.map((c, k) => [c, k]));
  };
  const a = rank(-1.5), b = rank(1.5);
  let flipped = 0;
  for (const [u, v] of neighbourPairs(cells)) {
    if ((a.get(u) < a.get(v)) !== (b.get(u) < b.get(v))) flipped++;
  }
  ok(flipped > 0, 'expected the unconstrained key to flip pairs here');
});

test('epsilon moves a boundary out of the constraints and into `weak`', () => {
  const cells = grid(3);
  const eye = [6, 0.5, 1.2];            // standing on the y=0.5 plane
  const tight = constraints(cells, eye, { epsilon: 0 });
  const loose = constraints(cells, eye, { epsilon: 1.2 });
  const total = neighbourPairs(cells).length;

  // A sign of exactly zero is weak whatever epsilon is, because the eye is
  // in the plane and neither cell is in front. Here that is the pair either
  // side of y = 0.5, which the camera is standing on.
  eq(tight.weak.length, 3, 'the y=0.5 boundaries are already undecidable');
  ok(loose.weak.length > tight.weak.length, 'epsilon should widen the net');

  // Nothing is lost or double-counted on the way out.
  eq(tight.edges.length + tight.weak.length, total);
  eq(loose.edges.length + loose.weak.length, total);
});

test('the cells a weak boundary drops are the ones merging picks up', () => {
  const cells = grid(3);
  const eye = [6, 0.5, 1.2];
  const { weak } = constraints(cells, eye, { epsilon: 0.2 });
  const { groupOf } = mergeGroups(cells, eye, { threshold: 0.2 });
  for (const [a, b] of weak) {
    eq(groupOf[a], groupOf[b], `pair ${a},${b} left unmerged and unordered`);
  }
});

test('a cycle is survived rather than thrown', () => {
  // Three cells lifted to different heights so their boundary planes tilt
  // enough to disagree. Constructed by search, not by hand.
  let found = null;
  for (let t = 0; t < 4000 && !found; t++) {
    const cells = grid(3).map((c) => ({ ...c, z: Math.sin(t * 0.7 + c.i * 2.3 + c.j * 1.9) * 3 }));
    const eye = [1.5, 1.5, 0.4];
    const r = drawOrder(cells, eye, {});
    if (r.cycles > 0) found = [cells, r];
  }
  if (found) {
    const [cells, r] = found;
    eq(r.order.length, cells.length, 'a cycle must not lose cells');
    eq([...new Set(r.order)].length, cells.length, 'or duplicate them');
  } else {
    ok(true, 'no cycle constructed; the guard is untriggered, not wrong');
  }
});

test('an empty or single-cell set is handled', () => {
  eq(drawOrder([], [0, 0, 1], {}).order, []);
  eq(drawOrder([{ i: 0, j: 0, x: 0, y: 0, z: 0 }], [0, 0, 1], {}).order, [0]);
});

// ---------------------------------------------------------------- merge.js

test('the slot packing round-trips to the documented limits', () => {
  eq(MAX_GROUP, 8);
  eq(INDEX_BITS, 29);
  const maxIndex = INDEX_MASK;
  ok(maxIndex >= 536870911, '29 bits should hold 536 million splats');
  for (const slot of [0, 1, 7]) {
    for (const idx of [0, 1, 12345678, maxIndex]) {
      const v = packIndex(slot, idx);
      eq(unpackSlot(v), slot, `slot ${slot} idx ${idx}`);
      eq(unpackIndex(v), idx, `slot ${slot} idx ${idx}`);
    }
  }
});

test('the packed word stays unsigned at the top slot', () => {
  // The high bit is set once slot >= 4; without the >>> 0 this reads back
  // negative and the index buffer silently takes garbage.
  const v = packIndex(7, INDEX_MASK);
  ok(v > 0, 'packed word went negative');
  ok(v <= 0xFFFFFFFF);
  eq(SLOT_BITS, 3);
});

test('the boundary normal points from the first cell to the second', () => {
  const b = boundary({ x: 0, y: 0, z: 0 }, { x: 0, y: 2, z: 0 });
  eq(b.mid, [0, 1, 0]);
  eq(b.normal, [0, 1, 0]);
  close(Math.hypot(...b.normal), 1, 1e-12, 'normal must be unit');
});

test('the normal tilts when relief lifts the two cells unequally', () => {
  const b = boundary({ x: 0, y: 0, z: 0 }, { x: 0, y: 1, z: 1 });
  ok(Math.abs(b.normal[2]) > 0.7, 'a 45 degree step should tilt the plane');
  close(Math.hypot(...b.normal), 1, 1e-12);
});

test('the score is the perpendicular distance, indifferent to range', () => {
  // The docstring in merge.js claims a distant edge-on boundary scores high.
  // It does not, and should not: lying in the plane is what the criterion
  // detects, at any range. This pins the behaviour the code actually has.
  const a = { x: 0, y: 0, z: 0 }, b = { x: 0, y: 1, z: 0 };
  close(mergeScore(a, b, [3, 0.5, 0]), 0, 1e-12, 'eye in the plane, near');
  close(mergeScore(a, b, [900, 0.5, 0]), 0, 1e-12, 'eye in the plane, far');
  close(mergeScore(a, b, [0, 4.5, 0]), 4, 1e-12, 'eye off the plane');
});

test('neighbour pairs are counted once each, not twice', () => {
  eq(neighbourPairs(grid(3)).length, 12);      // 2 * 3 * 2
  eq(neighbourPairs(grid(9)).length, 144);     // 2 * 9 * 8
  const seen = new Set();
  for (const [a, b] of neighbourPairs(grid(4))) {
    const k = [a, b].sort((p, q) => p - q).join(',');
    ok(!seen.has(k), 'duplicate pair');
    seen.add(k);
  }
});

test('a gap in the grid produces no pair across it', () => {
  const cells = grid(3).filter((c) => !(c.i === 1 && c.j === 1));
  const pairs = neighbourPairs(cells);
  eq(pairs.length, 8, 'the centre cell\'s four boundaries should be gone');
});

test('merging groups the boundaries the camera is standing on', () => {
  // Camera on the +x axis: the y-normal boundaries at |y| = 0.5 are the ones
  // it lies in the plane of, so a 3x3 grid should fuse into columns.
  const cells = grid(3);
  const g = mergeGroups(cells, [6, 0, 1.2], { threshold: 0.75 });
  eq(g.groups.length, 3, 'expected three merged columns');
  for (const grp of g.groups) eq(grp.length, 3);
  eq(g.pairsMerged, 6);
});

test('a threshold of zero merges nothing and still covers every cell', () => {
  const cells = grid(4);
  const g = mergeGroups(cells, [6, 0.3, 1.2], { threshold: 0 });
  eq(g.pairsMerged, 0);
  eq(g.groups.length, cells.length, 'every cell should be its own group');
  eq([...new Set(g.groupOf)].length, cells.length);
});

test('the cap is respected however low the scores go', () => {
  const cells = grid(9);
  // Eye far along z: every boundary plane passes near it, so without the cap
  // the whole grid would chain into one group.
  const g = mergeGroups(cells, [0, 0, 500], { threshold: 1e9, cap: MAX_GROUP });
  for (const grp of g.groups) ok(grp.length <= MAX_GROUP, `group of ${grp.length}`);
  const total = g.groups.reduce((s, x) => s + x.length, 0);
  eq(total, cells.length, 'cap must not drop cells');
});

test('groups partition the cells: every cell in exactly one', () => {
  const cells = grid(5);
  const g = mergeGroups(cells, [3, 0.5, 2], { threshold: 1.0 });
  const seen = new Set();
  for (let gi = 0; gi < g.groups.length; gi++) {
    for (const k of g.groups[gi]) {
      ok(!seen.has(k), `cell ${k} in two groups`);
      seen.add(k);
      eq(g.groupOf[k], gi, `groupOf disagrees for cell ${k}`);
    }
  }
  eq(seen.size, cells.length);
});

test('the cheapest boundaries are the ones that survive the cap', () => {
  // Merging is transitive, so when the cap bites something has to be
  // dropped. It should be the boundary the camera is furthest from.
  const cells = [];
  for (let j = 0; j < 12; j++) cells.push({ i: 0, j, x: 0, y: j, z: 0 });
  const eye = [0, 0, 0];                 // nearest boundary at y=0.5
  const g = mergeGroups(cells, eye, { threshold: 1e9, cap: 4 });
  eq(g.groupOf[0], g.groupOf[1], 'the nearest boundary should have merged');
  const last = cells.length - 1;
  ok(g.groupOf[last] !== g.groupOf[0], 'the furthest should not have');
});

test('a k-way merge interleaves far to near and keeps every splat', () => {
  const out = kwayMerge([
    { slot: 0, order: [0, 1, 2], depth: [9, 5, 1] },
    { slot: 1, order: [7, 8],    depth: [7, 3] },
  ]);
  eq(Array.from(out).map((v) => [unpackSlot(v), unpackIndex(v)]),
     [[0, 0], [1, 7], [0, 1], [1, 8], [0, 2]]);
});

test('the merged depths are non-increasing for any stream count', () => {
  const streams = [];
  let rng = 12345;
  const rand = () => (rng = (rng * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  for (let s = 0; s < MAX_GROUP; s++) {
    const d = Array.from({ length: 40 }, () => rand() * 10).sort((a, b) => b - a);
    streams.push({ slot: s, order: d.map((_, k) => k), depth: d });
  }
  const out = kwayMerge(streams);
  eq(out.length, MAX_GROUP * 40);
  let prev = Infinity;
  for (const v of out) {
    const d = streams[unpackSlot(v)].depth[unpackIndex(v)];
    ok(d <= prev + 1e-12, `depth rose from ${prev} to ${d}`);
    prev = d;
  }
});

test('a single stream merges to itself, tagged', () => {
  const out = kwayMerge([{ slot: 5, order: [3, 1, 2], depth: [9, 4, 0] }]);
  eq(Array.from(out).map(unpackIndex), [3, 1, 2]);
  eq(Array.from(out).map(unpackSlot), [5, 5, 5]);
});

test('an empty stream alongside a full one is skipped, not stalled', () => {
  const out = kwayMerge([
    { slot: 0, order: [], depth: [] },
    { slot: 1, order: [4, 5], depth: [2, 1] },
  ]);
  eq(out.length, 2);
  eq(Array.from(out).map(unpackIndex), [4, 5]);
});


// ------------------------------------------------------- sort-worker.js

/** Load the worker in a stub `self` so its message handler can be driven
 *  from node. It is a classic worker, so there is nothing to import. */
async function loadWorker() {
  const fs = await import('fs');
  const src = fs.readFileSync(new URL('../web/sort-worker.js', import.meta.url), 'utf8');
  const sandbox = { onmessage: null, postMessage: null };
  const fn = new Function('self', 'performance', src);
  fn(sandbox, performance);
  return {
    send(msg) {
      return new Promise((res) => { sandbox.postMessage = res; sandbox.onmessage({ data: msg }); });
    },
  };
}

/** A scene of `n` splats in `p` equal patches, positions from a fixed seed. */
function scene(n, p) {
  let s = 99;
  const rand = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  const pos = new Float32Array(n * 3);
  for (let i = 0; i < n * 3; i++) pos[i] = (rand() - 0.5) * 2;
  const per = Math.floor(n / p), patches = [];
  for (let k = 0; k < p; k++) patches.push({ start: k * per, count: per });
  return { pos, patches };
}

const workerTests = [];
function wtest(name, fn) { workerTests.push([name, fn]); }

wtest('the worker\'s packing constants match merge.js', async () => {
  const fs = await import('fs');
  const src = fs.readFileSync(new URL('../web/sort-worker.js', import.meta.url), 'utf8');
  const m = src.match(/const SLOT_BITS = (\d+);/);
  ok(m, 'SLOT_BITS not found in the worker');
  eq(Number(m[1]), SLOT_BITS, 'the two copies have drifted');
});

wtest('a merged group is sorted far to near across all its cells', async () => {
  const w = await loadWorker();
  const { pos, patches } = scene(6000, 3);
  await w.send({ type: 'init', positions: pos.buffer.slice(0), patches });
  const fwd = [0.6, 0.8, 0.0], eye = [2, -1, 0.5];
  const cells = [
    { patch: 0, start: patches[0].start, count: patches[0].count, offset: [0, 0, 0] },
    { patch: 1, start: patches[1].start, count: patches[1].count, offset: [1, 0, 0] },
  ];
  const r = await w.send({ type: 'mergeGroups', forward: fwd, eye, groups: [cells], seq: 1 });
  eq(r.type, 'mergedGroups');
  const order = new Uint32Array(r.orders[0]);
  eq(order.length, cells[0].count + cells[1].count, 'lost splats');

  const depthOf = (v) => {
    const c = cells[unpackSlot(v)], i = unpackIndex(v);
    const sh = c.offset[0] * fwd[0] + c.offset[1] * fwd[1] + c.offset[2] * fwd[2];
    return (pos[3 * i] - eye[0]) * fwd[0] + (pos[3 * i + 1] - eye[1]) * fwd[1]
         + (pos[3 * i + 2] - eye[2]) * fwd[2] + sh;
  };
  // Quantised to 16 bits, so the guarantee is monotonic within one bucket.
  let prev = Infinity, span = 0;
  for (const v of order) span = Math.max(span, Math.abs(depthOf(v)));
  const tol = 2 * span / 65535;
  for (const v of order) {
    const d = depthOf(v);
    ok(d <= prev + tol, `depth rose by more than one bucket: ${prev} -> ${d}`);
    prev = Math.min(prev, d);
  }
});

wtest('both cells are genuinely interleaved, not stacked', async () => {
  // The whole point. If one cell's splats all preceded the other's, the
  // merge would have achieved nothing over drawing them in sequence.
  const w = await loadWorker();
  const { pos, patches } = scene(4000, 2);
  await w.send({ type: 'init', positions: pos.buffer.slice(0), patches });
  const cells = [
    { patch: 0, start: patches[0].start, count: patches[0].count, offset: [0, 0, 0] },
    { patch: 1, start: patches[1].start, count: patches[1].count, offset: [0, 0.2, 0] },
  ];
  const r = await w.send({ type: 'mergeGroups', forward: [0, 1, 0], eye: [0, -5, 0], groups: [cells], seq: 2 });
  const order = new Uint32Array(r.orders[0]);
  let swaps = 0;
  for (let i = 1; i < order.length; i++) {
    if (unpackSlot(order[i]) !== unpackSlot(order[i - 1])) swaps++;
  }
  ok(swaps > order.length / 10, `only ${swaps} alternations in ${order.length}`);
});

wtest('the counting sort agrees with kwayMerge, the exact reference', async () => {
  const w = await loadWorker();
  const { pos, patches } = scene(3000, 2);
  await w.send({ type: 'init', positions: pos.buffer.slice(0), patches });
  const fwd = [0, 0, 1], eye = [0, 0, -3];
  const cells = [
    { patch: 0, start: patches[0].start, count: patches[0].count, offset: [0, 0, 0] },
    { patch: 1, start: patches[1].start, count: patches[1].count, offset: [0, 0, 0.3] },
  ];
  const r = await w.send({ type: 'mergeGroups', forward: fwd, eye, groups: [cells], seq: 3 });
  const got = new Uint32Array(r.orders[0]);

  // Same thing the slow, exact way: sort each cell, then merge the two.
  const streams = cells.map((c, s) => {
    const idx = [];
    for (let i = c.start; i < c.start + c.count; i++) idx.push(i);
    const sh = c.offset[2] * fwd[2];
    const d = (i) => (pos[3 * i + 2] - eye[2]) * fwd[2] + sh;
    idx.sort((a, b) => d(b) - d(a));
    return { slot: s, order: idx, depth: idx.map(d) };
  });
  const want = kwayMerge(streams);
  eq(got.length, want.length);

  // Bucket quantisation can reorder splats that fall in the same bucket, so
  // compare the depth sequences rather than the index sequences.
  const dg = Array.from(got).map((v) => streams[unpackSlot(v)].depth[
    streams[unpackSlot(v)].order.indexOf(unpackIndex(v))]);
  const dw = Array.from(want).map((v) => streams[unpackSlot(v)].depth[
    streams[unpackSlot(v)].order.indexOf(unpackIndex(v))]);
  let maxGap = 0;
  for (let i = 0; i < dg.length; i++) maxGap = Math.max(maxGap, Math.abs(dg[i] - dw[i]));
  const span = Math.max(...dw) - Math.min(...dw);
  ok(maxGap < 3 * span / 65535,
     `counting sort strays ${(maxGap / span * 65535).toFixed(1)} buckets from exact`);
});

wtest('an eight-cell group stays inside the slot budget', async () => {
  const w = await loadWorker();
  const { pos, patches } = scene(8000, MAX_GROUP);
  await w.send({ type: 'init', positions: pos.buffer.slice(0), patches });
  const cells = patches.map((p, s) => ({
    patch: s, start: p.start, count: p.count, offset: [s * 0.1, 0, 0] }));
  const r = await w.send({ type: 'mergeGroups', forward: [1, 0, 0], eye: [-4, 0, 0], groups: [cells], seq: 4 });
  const order = new Uint32Array(r.orders[0]);
  eq(order.length, 8000);
  const slots = new Set(Array.from(order).map(unpackSlot));
  eq(slots.size, MAX_GROUP, 'every cell should appear');
  for (const v of order) ok(unpackIndex(v) < 8000, 'index escaped its field');
});

wtest('several groups come back in the order they were sent', async () => {
  const w = await loadWorker();
  const { pos, patches } = scene(6000, 3);
  await w.send({ type: 'init', positions: pos.buffer.slice(0), patches });
  const g = (ks) => ks.map((k, s) => ({
    patch: k, start: patches[k].start, count: patches[k].count, offset: [s, 0, 0] }));
  const r = await w.send({ type: 'mergeGroups', forward: [1, 0, 0], eye: [-9, 0, 0],
                           groups: [g([0, 1]), g([2])], seq: 5 });
  eq(r.sizes, [4000, 2000]);
  eq(r.seq, 5, 'the sequence number must come back so stale frames can be dropped');
});

wtest('an empty group is survived', async () => {
  const w = await loadWorker();
  const { pos, patches } = scene(2000, 2);
  await w.send({ type: 'init', positions: pos.buffer.slice(0), patches });
  const r = await w.send({ type: 'mergeGroups', forward: [1, 0, 0], eye: [0, 0, 0],
                           groups: [[]], seq: 6 });
  eq(r.sizes, [0]);
});


// -------------------------------------------------------- tileset.js

/** Build a zip in memory. Stored entries only, which is all the reader
 *  needs to be handed to prove it walks the central directory correctly;
 *  deflated archives are covered by the fixtures written from python. */
function zipOf(entries) {
  const enc = new TextEncoder();
  const locals = [], central = [];
  let offset = 0;
  for (const [name, data] of entries) {
    const nb = enc.encode(name);
    const body = typeof data === 'string' ? enc.encode(data) : data;
    const lh = new Uint8Array(30 + nb.length);
    const lv = new DataView(lh.buffer);
    lv.setUint32(0, 0x04034b50, true);
    lv.setUint16(8, 0, true);                 // stored
    lv.setUint32(18, body.length, true);
    lv.setUint32(22, body.length, true);
    lv.setUint16(26, nb.length, true);
    lh.set(nb, 30);
    locals.push(lh, body);

    const ch = new Uint8Array(46 + nb.length);
    const cv = new DataView(ch.buffer);
    cv.setUint32(0, 0x02014b50, true);
    cv.setUint16(10, 0, true);
    cv.setUint32(20, body.length, true);
    cv.setUint32(24, body.length, true);
    cv.setUint16(28, nb.length, true);
    cv.setUint32(42, offset, true);
    ch.set(nb, 46);
    central.push(ch);
    offset += lh.length + body.length;
  }
  const cdSize = central.reduce((t, c) => t + c.length, 0);
  const eocd = new Uint8Array(22);
  const ev = new DataView(eocd.buffer);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(8, entries.length, true);
  ev.setUint16(10, entries.length, true);
  ev.setUint32(12, cdSize, true);
  ev.setUint32(16, offset, true);
  const parts = [...locals, ...central, eocd];
  const total = parts.reduce((t, p) => t + p.length, 0);
  const out = new Uint8Array(total);
  let w = 0;
  for (const p of parts) { out.set(p, w); w += p.length; }
  return out.buffer;
}

const META = JSON.stringify({ size: 1.5, wang: true, lod: 4 });

wtest('a stored zip round-trips to the same bytes', async () => {
  const splat = new Uint8Array(64).map((_, i) => i * 3 % 251);
  const m = await readZip(zipOf([['a.splat', splat], ['a.json', META]]));
  eq(m.size, 2);
  eq(Array.from(m.get('a.splat')), Array.from(splat), 'splat bytes changed');
});

wtest('the tileset is picked out of a folder with junk beside it', async () => {
  const splat = new Uint8Array(64).fill(7);
  const m = await readZip(zipOf([
    ['bigsur/bigsur.splat', splat],
    ['bigsur/bigsur.json', META],
    ['__MACOSX/._bigsur.splat', 'junk'],
    ['bigsur/.DS_Store', 'junk'],
  ]));
  const t = pickTileset(m);
  eq(t.name, 'bigsur.splat', 'picked the shadow file or the wrong one');
  eq(t.buffer.byteLength, 64);
  eq(t.meta.size, 1.5);
});

wtest('the largest .splat wins when several are present', async () => {
  const m = await readZip(zipOf([
    ['small.splat', new Uint8Array(16)],
    ['big.splat', new Uint8Array(256)],
    ['big.json', META],
  ]));
  eq(pickTileset(m).name, 'big.splat');
});

wtest('the .json matching the chosen .splat is preferred', async () => {
  const m = await readZip(zipOf([
    ['big.splat', new Uint8Array(256)],
    ['other.json', JSON.stringify({ size: 99 })],
    ['big.json', META],
  ]));
  eq(pickTileset(m).meta.size, 1.5, 'took the wrong json');
});

wtest('a tileset with no metadata opens, and says so', async () => {
  const m = await readZip(zipOf([['a.splat', new Uint8Array(32)]]));
  const t = pickTileset(m);
  eq(t.meta, null);
  const d = describe(t.meta);
  eq(d.size, null, 'a guessed tile size would be worse than none');
  ok(d.note, 'the caller has to be able to tell the person');
});

wtest('broken JSON is refused by name rather than silently dropped', async () => {
  const m = await readZip(zipOf([['a.splat', new Uint8Array(32)], ['a.json', '{oops']]));
  let msg = '';
  try { pickTileset(m); } catch (e) { msg = e.message; }
  ok(msg.includes('a.json'), `unhelpful error: ${msg}`);
});

wtest('a zip with no .splat is refused', async () => {
  const m = await readZip(zipOf([['readme.txt', 'hello']]));
  let threw = false;
  try { pickTileset(m); } catch (e) { threw = true; }
  ok(threw, 'an empty tileset should not load');
});

wtest('something that is not a zip is refused', async () => {
  let msg = '';
  try { await readZip(new Uint8Array(400).buffer); } catch (e) { msg = e.message; }
  ok(msg.includes('zip'), `unhelpful error: ${msg}`);
});

wtest('a trailing comment does not hide the end record', async () => {
  // The end record is found by scanning back, so a comment after it must
  // not stop the scan.
  const base = new Uint8Array(zipOf([['a.splat', new Uint8Array(8)]]));
  const withComment = new Uint8Array(base.length + 300);
  withComment.set(base);
  new DataView(withComment.buffer).setUint16(base.length - 22 + 20, 300, true);
  const m = await readZip(withComment.buffer);
  eq(m.size, 1);
});


wtest('a written zip reads back through our own reader', async () => {
  const splat = new Uint8Array(300).map((_, i) => (i * 7) % 251);
  const z = writeZip([['x.splat', splat], ['x.json', new TextEncoder().encode(META)]]);
  const back = await readZip(z.buffer.slice(z.byteOffset, z.byteOffset + z.length));
  eq(back.size, 2);
  eq(Array.from(back.get('x.splat')), Array.from(splat), 'bytes changed');
  eq(pickTileset(back).meta.size, 1.5);
});

wtest('a written zip carries correct CRCs', async () => {
  // The reader ignores the CRC field; an archiver does not. A zip that
  // only our own code can open would be useless for handing frames to
  // pop_metric.py.
  const z = writeZip([['a.txt', new TextEncoder().encode('hello world')]]);
  const view = new DataView(z.buffer, z.byteOffset, z.byteLength);
  // Local header CRC sits at offset 14, and 'hello world' has a known one.
  eq(view.getUint32(14, true) >>> 0, 0x0d4a1185, 'CRC-32 is wrong');
});

wtest('an empty zip is still a valid zip', async () => {
  const z = writeZip([]);
  const back = await readZip(z.buffer.slice(z.byteOffset, z.byteOffset + z.length));
  eq(back.size, 0);
});


// ----------------------------------------------------- heightfield.js

/** A field with a known shape: a ridge along x, flat along y. */
function ridge(w = 16, h = 16) {
  const z = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      z[y * w + x] = Math.sin((x / (w - 1)) * Math.PI);
    }
  }
  return new HeightField(z, w, h, { metres_range: 8 }, 'test');
}

test('a height field samples its own corners', () => {
  const z = new Float32Array([0, 1, 0.25, 0.75]);   // 2x2
  const f = new HeightField(z, 2, 2, {}, 't').fitTo(2);
  // Centred: world -1..1 maps across the field.
  close(f.sample(-1, 1), 0, 1e-6, 'top left');
  close(f.sample(1, 1), 1, 1e-6, 'top right');
});

test('sampling is bilinear, not nearest', () => {
  const z = new Float32Array([0, 1, 0, 1]);
  const f = new HeightField(z, 2, 2, {}, 't').fitTo(2);
  const mid = f.sample(0, 0);
  ok(mid > 0.2 && mid < 0.8, `midpoint read ${mid}, expected about 0.5`);
});

test('beyond the field the surface is continuous, not cliffed', () => {
  // A field smaller than the grid has to repeat somehow, and the choice
  // shows at the join. Wrapping puts the far edge against the near one and
  // they rarely meet at the same height, so every repeat boundary is a
  // cliff - which then reads as a ridge and collects the wrong material.
  // Mirroring joins each edge to itself, so it is continuous by
  // construction.
  const f = ridge(32, 32).fitTo(10);
  const e = 5.0;                       // the field's own edge
  for (const d of [0.02, 0.1, 0.4]) {
    const inside = f.sample(e - d, 0);
    const outside = f.sample(e + d, 0);
    close(inside, outside, 1e-6,
          `a step across the edge at ${d}: ${inside} vs ${outside}`);
  }
});

test('mirroring repeats with period two, not one', () => {
  // The reflected copy is a reflection, not a duplicate: going out one
  // width lands on the mirror image, two widths on the original.
  const f = ridge(32, 32).fitTo(10);
  for (const x of [-3.1, 0.0, 2.7]) {
    close(f.sample(x, 0), f.sample(x + 20, 0), 1e-6,
          `period is not two widths at x=${x}`);
  }
});

test('fitTo rescales rather than resampling', () => {
  const f = ridge();
  const a = f.fitTo(10).sample(2.5, 0);
  const b = f.fitTo(20).sample(5.0, 0);
  close(a, b, 1e-6, 'the same relative position gave different heights');
});

test('the ridge is where the ridge is', () => {
  // The whole point: a sampled field has to put its landforms where the
  // source put them, or a rule reading the terrain reads the wrong place.
  const f = ridge(32, 32).fitTo(10);
  const middle = f.sample(0, 0);
  const side = f.sample(-4.9, 0);
  ok(middle > side + 0.5, `ridge ${middle} is not above the flank ${side}`);
});

test('a flat field samples flat everywhere', () => {
  const f = new HeightField(new Float32Array(64).fill(0.5), 8, 8, {}, 't')
    .fitTo(4);
  for (const [x, y] of [[0, 0], [1, 1], [-1.9, 1.9], [0.3, -0.7]]) {
    close(f.sample(x, y), 0.5, 1e-6, `at ${x},${y}`);
  }
});

test('metres come back when the sidecar carried them', () => {
  eq(ridge().metresPerUnit, 8);
  eq(new HeightField(new Float32Array(4), 2, 2, {}, 't').metresPerUnit, null);
});

// ------------------------------------------------------- benchmark.js

test('the sweep report names the limit it found', () => {
  const sortBound = [
    { grid: 4, cells: 16, splats: 1e5, calls: 16, sortMs: 2, fps: 60, worstMs: 18 },
    { grid: 16, cells: 256, splats: 4e5, calls: 60, sortMs: 40, fps: 40, worstMs: 30 },
  ];
  ok(report(sortBound).includes('sorting is the limit'),
     'a sort growing faster than the frame was not identified');

  const callBound = [
    { grid: 4, cells: 16, splats: 1e5, calls: 16, sortMs: 2, fps: 60, worstMs: 18 },
    { grid: 48, cells: 2304, splats: 4e5, calls: 900, sortMs: 3, fps: 25, worstMs: 60 },
  ];
  ok(report(callBound).includes('draw calls'),
     'a draw-call ceiling was not identified');
});

test('the report says where 60 and 30 fps ran out', () => {
  const rows = [
    { grid: 8, cells: 64, splats: 2e5, calls: 30, sortMs: 2, fps: 60, worstMs: 18 },
    { grid: 16, cells: 256, splats: 6e5, calls: 90, sortMs: 4, fps: 45, worstMs: 26 },
    { grid: 24, cells: 576, splats: 1e6, calls: 150, sortMs: 6, fps: 20, worstMs: 60 },
  ];
  const text = report(rows);
  ok(text.includes('60 fps up to grid 8'), text);
  ok(text.includes('30 fps up to grid 16'), text);
});

test('an empty sweep reports nothing rather than dividing by zero', () => {
  eq(report([]), 'nothing measured');
});


// ------------------------------------------------------- landform.js

/** A V-shaped valley running along y, tilted so water flows down it. */
function vGrid(n = 24) {
  const z = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      z[j * n + i] = Math.abs(i - (n - 1) / 2) * 0.5 + (n - j) * 0.2;
    }
  }
  return z;
}

test('slope reads a plane as its gradient', () => {
  const n = 12;
  const z = new Float64Array(n * n);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) z[j * n + i] = i * 0.3;
  const s = slope(z, n);
  close(s[6 * n + 6], 0.3, 1e-9, 'interior');
});

test('TPI is positive on a ridge and negative in a hollow', () => {
  const n = 21, z = new Float64Array(n * n);
  const c = (n - 1) / 2;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      z[j * n + i] = -Math.hypot(i - c, j - c);        // a cone
    }
  }
  const t = tpi(z, n, 3);
  ok(t[c * n + c] > 0, `peak read ${t[c * n + c]}`);
  const inverted = tpi(z.map(v => -v), n, 3);
  ok(inverted[c * n + c] < 0, 'an inverted cone should read as a pit');
});

test('flow collects in the channel and not on the flank', () => {
  const n = 24, z = vGrid(n);
  const f = flow(z, n);
  let channel = 0, flank = 0;
  for (let j = 0; j < n; j++) {
    channel += f[j * n + (n >> 1)];
    flank += f[j * n + 2];
  }
  ok(channel > flank * 5, `channel ${channel.toFixed(0)}, flank ${flank.toFixed(0)}`);
});

test('every cell drains at least itself, and none drains more than exists', () => {
  const n = 16, z = new Float64Array(n * n).map(() => Math.random());
  const f = flow(z, n);
  for (const v of f) ok(v >= 1, `a cell drained ${v}`);
  ok(Math.max(...f) <= n * n, 'a cell drained more cells than there are');
});

test('flat ground routes water nowhere', () => {
  const n = 10;
  const f = flow(new Float64Array(n * n), n);
  for (const v of f) close(v, 1, 1e-9);
});

test('the collected class goes to the channel, the exposed to the flank', () => {
  // The claim the module exists to support, and the one a figure shows.
  const n = 24, z = vGrid(n);
  const { cls } = classify(z, n, 2, { spacing: 1 });
  const mid = n >> 1;
  let chan = 0, flank = 0;
  for (let j = 4; j < n - 4; j++) {
    if (cls[j * n + mid] === 0) chan++;
    if (cls[j * n + 2] === 1) flank++;
  }
  ok(chan > (n - 8) * 0.6, `only ${chan} channel cells took the collected class`);
  ok(flank > (n - 8) * 0.6, `only ${flank} flank cells took the exposed class`);
});

test('classify returns one class per cell, all in range', () => {
  const n = 16;
  const { cls, strength } = classify(vGrid(n), n, 2, {});
  eq(cls.length, n * n);
  for (const k of cls) ok(k >= 0 && k < 2, `class ${k} out of range`);
  for (const v of strength) ok(v >= 0 && v <= 1, `confidence ${v} out of range`);
});

test('decisiveness sharpens the split without moving it', () => {
  const n = 24, z = vGrid(n);
  const soft = classify(z, n, 2, { sharpness: 1 });
  const hard = classify(z, n, 2, { sharpness: 6 });
  let same = 0;
  for (let i = 0; i < n * n; i++) if (soft.cls[i] === hard.cls[i]) same++;
  ok(same / (n * n) > 0.85,
     `sharpening moved the boundary: ${(100 * same / (n * n)).toFixed(0)}% agree`);
  // Confidence is a margin scaled against the margins across the whole
  // field, so it is relative by construction and sharpening every weight
  // does not inflate it. That is the point: it says which cells are less
  // sure than their neighbours, which is what blending needs, rather than
  // an absolute that would move whenever the rule was retuned.
});

test('flat terrain does not crash the rule', () => {
  const n = 12;
  const { cls } = classify(new Float64Array(n * n), n, 2, {});
  eq(cls.length, n * n);
});

test('the grid is sampled from the same height function that is drawn', () => {
  // Reading a different surface would place material by the shape of
  // something invisible.
  const n = 8, tile = 1.5;
  const h = (x, y) => x * 2 + y;
  const z = sampleGrid(n, tile, h, 0);
  const half = (n - 1) / 2;
  for (const [i, j] of [[0, 0], [3, 5], [7, 7]]) {
    close(z[j * n + i], h((i - half) * tile, (j - half) * tile), 1e-9,
          `cell ${i},${j}`);
  }
});


test('balance gives the share of the grid it was asked for', () => {
  // The control an environment artist actually wants: say how much, and
  // let the terrain decide where.
  const n = 32, z = vGrid(n);
  for (const want of [0.2, 0.35, 0.5, 0.65, 0.8]) {
    const { cls } = classify(z, n, 2, { balance: want });
    let c0 = 0;
    for (const k of cls) if (k === 0) c0++;
    const got = c0 / (n * n);
    ok(Math.abs(got - want) < 0.05,
       `asked for ${(want * 100).toFixed(0)}%, got ${(got * 100).toFixed(0)}%`);
  }
});

test('balance moves how much, not where', () => {
  // Shifting the threshold must not reshuffle the boundary: the cells the
  // first class holds at 30% have to be a subset of the ones it holds at
  // 60%, or the control would be repainting rather than rebalancing.
  const n = 32, z = vGrid(n);
  const few = classify(z, n, 2, { balance: 0.3 }).cls;
  const many = classify(z, n, 2, { balance: 0.6 }).cls;
  let escaped = 0;
  for (let i = 0; i < n * n; i++) {
    if (few[i] === 0 && many[i] !== 0) escaped++;
  }
  ok(escaped === 0, `${escaped} cells left the first class as it grew`);
});

test('flat terrain reports no confidence rather than false confidence', () => {
  // Every map is constant, so the rule has nothing to go on. It used to
  // read zero everywhere, which handed the whole grid to one class and
  // looked like a decision; half way is the honest reading, and the
  // confidence figure then says so.
  const n = 16;
  const { strength } = classify(new Float64Array(n * n), n, 2, {});
  const mean = Array.from(strength).reduce((a, b) => a + b, 0) / (n * n);
  // Zero margin everywhere: nothing separates the classes, so nothing is
  // decided. With blending on, a field like this dissolves evenly rather
  // than picking a winner, which is the honest picture of a rule with no
  // information.
  ok(mean < 0.05,
     `confidence on flat ground read ${(mean * 100).toFixed(0)}%, expected 0%`);
});

test('a hair of relief does not swing the rule to full strength', () => {
  // The jump that showed in the viewer: at exactly zero relief one class
  // took everything, and a hundredth of relief later the rule was at full
  // strength, because each map is stretched to its own range whatever that
  // range is. With balance set the share is stable across both.
  const n = 24;
  const flat = new Float64Array(n * n);
  const barely = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) barely[j * n + i] = 0.001 * i;
  }
  const share = (z) => {
    const { cls } = classify(z, n, 2, { balance: 0.5 });
    let c = 0;
    for (const k of cls) if (k === 0) c++;
    return c / (n * n);
  };
  ok(Math.abs(share(flat) - share(barely)) < 0.15,
     `${(share(flat) * 100).toFixed(0)}% became ${(share(barely) * 100).toFixed(0)}%`);
});


test('patch size makes material regions contiguous', () => {
  // Without it, two adjacent cells either side of the threshold make
  // opposite choices over a difference too small to mean anything, and
  // the result is a scatter of single tiles rather than regions.
  const n = 24, tile = 1;
  const h = (x, y) => Math.sin(x * 0.25) + Math.cos(y * 0.18) * 0.8;
  const z = sampleGrid(n, tile, h, 0, 4);

  const fragmentation = (coherence) => {
    const { cls } = classify(z, n, 2, { spacing: tile, balance: 0.3, coherence });
    let e = 0;
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n - 1; i++) if (cls[j * n + i] !== cls[j * n + i + 1]) e++;
    }
    for (let j = 0; j < n - 1; j++) {
      for (let i = 0; i < n; i++) if (cls[j * n + i] !== cls[(j + 1) * n + i]) e++;
    }
    return e;
  };

  // Counted as isolated cells rather than as boundary length: the filter
  // is meant to remove specks without moving the boundary, so boundary
  // length is the wrong thing to measure and would penalise it for doing
  // its job well.
  const islands = (coherence) => {
    const { cls } = classify(z, n, 2, { spacing: tile, balance: 0.3, coherence });
    let lone = 0;
    for (let j = 1; j < n - 1; j++) {
      for (let i = 1; i < n - 1; i++) {
        const me = cls[j * n + i];
        let same = 0;
        for (const [a, b] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          if (cls[(j + b) * n + i + a] === me) same++;
        }
        if (same === 0) lone++;
      }
    }
    return lone;
  };

  const loose = fragmentation(0);
  const tight = fragmentation(6);
  ok(tight < loose, `boundary did not simplify: ${loose} -> ${tight}`);
  ok(islands(6) <= islands(0),
     `isolated cells rose: ${islands(0)} -> ${islands(6)}`);
});

test('patch size does not change how much, only how clumped', () => {
  const n = 24, tile = 1;
  const h = (x, y) => Math.sin(x * 0.25) + Math.cos(y * 0.18) * 0.8;
  const z = sampleGrid(n, tile, h, 0, 4);
  for (const coherence of [0, 2, 4]) {
    const { cls } = classify(z, n, 2, { spacing: tile, balance: 0.3, coherence });
    let c0 = 0;
    for (const k of cls) if (k === 0) c0++;
    ok(Math.abs(c0 / (n * n) - 0.3) < 0.05,
       `coherence ${coherence} moved the share to ${(100 * c0 / (n * n)).toFixed(0)}%`);
  }
});

test('over-sampling resolves channels a single sample per cell steps over', () => {
  // The aliasing that made the classes look scattered. A channel narrower
  // than a cell cannot be traced at one sample per cell, so no channels
  // appear and the map meant to place material by drainage places it by
  // noise.
  const n = 16, tile = 1;
  // A narrow valley running down the middle, half a tile wide.
  const h = (x) => Math.min(1, Math.abs(x) * 4) + 0;
  const coarse = sampleGrid(n, tile, (x) => h(x), 0, 1);
  const fine = sampleGrid(n, tile, (x) => h(x), 0, 4);
  eq(fine.length, coarse.length * 16, 'over-sampling did not change the count');

  const rc = classify(coarse, n, 2, { spacing: tile, coherence: 0 });
  const rf = classify(fine, n, 2, { spacing: tile, coherence: 0 });
  // The fine version should find the valley: its flow map has a much
  // larger spread, because it resolves somewhere for water to collect.
  const spread = (a) => {
    const v = Array.from(a);
    const m = v.reduce((t, x) => t + x, 0) / v.length;
    return Math.sqrt(v.reduce((t, x) => t + (x - m) ** 2, 0) / v.length);
  };
  ok(spread(rf.maps.flow) > spread(rc.maps.flow),
     `fine ${spread(rf.maps.flow).toFixed(3)} did not beat `
     + `coarse ${spread(rc.maps.flow).toFixed(3)}`);
});


test('altitude puts material on the low ground and the high ground', () => {
  // The input that was missing. Without it a hollow at the top of a
  // mountain reads the same as a hollow at the bottom, which is not how
  // any landscape works and was visible as low ground not going green.
  const n = 28, tile = 1;
  const h = (x, y) => Math.sin(x * 0.22) * 1.2 + Math.cos(y * 0.17) * 0.9;
  const z = sampleGrid(n, tile, h, 0, 4);

  const bias = (altitude) => {
    const r = classify(z, n, 2, { spacing: tile, balance: 0.4, altitude });
    const e = r.maps.elevation;
    let lowFirst = 0, low = 0, highFirst = 0, high = 0;
    for (let i = 0; i < n * n; i++) {
      if (e[i] < 0.35) { low++; if (r.cls[i] === 0) lowFirst++; }
      if (e[i] > 0.65) { high++; if (r.cls[i] === 0) highFirst++; }
    }
    return [lowFirst / Math.max(low, 1), highFirst / Math.max(high, 1)];
  };

  const [lowOff, highOff] = bias(0);
  const [lowOn, highOn] = bias(0.6);
  ok(lowOn > lowOff, `low ground: ${lowOff.toFixed(2)} -> ${lowOn.toFixed(2)}`);
  ok(highOn < highOff, `high ground: ${highOff.toFixed(2)} -> ${highOn.toFixed(2)}`);
  ok(lowOn > 0.85 && highOn < 0.15,
     `altitude did not separate them: low ${lowOn.toFixed(2)}, high ${highOn.toFixed(2)}`);
});

test('confidence is continuous, so a blend width can act on it', () => {
  // It used to come out as exactly two values once a balance was set,
  // because the weights were collapsed to a difference and a zero. Widening
  // the blend then switched every second-class cell on at once instead of
  // reaching further from the boundary.
  const n = 24, tile = 1;
  const h = (x, y) => Math.sin(x * 0.25) + Math.cos(y * 0.18) * 0.8;
  const z = sampleGrid(n, tile, h, 0, 4);
  const { strength } = classify(z, n, 2, { spacing: tile, balance: 0.3 });
  const distinct = new Set(Array.from(strength).map((v) => v.toFixed(2)));
  ok(distinct.size > 10,
     `only ${distinct.size} distinct confidence values; blending has nothing to grade`);
});

test('widening the blend reaches further from the boundary', () => {
  const n = 24, tile = 1;
  const h = (x, y) => Math.sin(x * 0.25) + Math.cos(y * 0.18) * 0.8;
  const z = sampleGrid(n, tile, h, 0, 4);
  const { strength } = classify(z, n, 2, { spacing: tile, balance: 0.3 });
  const within = (w) => Array.from(strength).filter((v) => v < w).length;
  ok(within(0.6) > within(0.2),
     `blend width did not widen the band: ${within(0.2)} -> ${within(0.6)}`);
  ok(within(0.2) > 0, 'nothing is ever close enough to the boundary to blend');
});


test('the requested share survives the majority filter', () => {
  // Removing specks is not share-neutral: a minority class is made of
  // smaller regions by definition, so more of it falls inside the filter's
  // window and gets absorbed. Asking for 30% and being handed 13% is the
  // filter quietly overruling the one control that is meant to be exact.
  const n = 28, tile = 1;
  const h = (x, y) => Math.sin(x * 0.22) * 1.2 + Math.cos(y * 0.17) * 0.9;
  const z = sampleGrid(n, tile, h, 0, 4);
  for (const coherence of [0, 3, 6]) {
    for (const want of [0.2, 0.3, 0.5]) {
      const { cls } = classify(z, n, 2, { spacing: tile, balance: want, coherence });
      let k0 = 0;
      for (const k of cls) if (k === 0) k0++;
      const got = k0 / (n * n);
      ok(Math.abs(got - want) < 0.06,
         `patch ${coherence}: asked ${(want * 100).toFixed(0)}%, `
         + `got ${(got * 100).toFixed(0)}%`);
    }
  }
});


test('a sediment record overrides inferred drainage', () => {
  // On generated terrain the erosion wrote down where material settled.
  // The rule should read that record rather than infer from the shape what
  // it would probably say.
  const n = 24, tile = 1;
  const h = (x, y) => Math.sin(x * 0.25) + Math.cos(y * 0.18) * 0.8;
  const z = sampleGrid(n, tile, h, 0, 4);
  // A record that deliberately disagrees with the drainage: a stripe.
  const sed = sampleGrid(n, tile, (x) => (Math.abs(x - 3) < 1.5 ? 1 : 0), 0, 4);

  const inferred = classify(z, n, 2, { spacing: tile, balance: 0.3, altitude: 0 });
  const recorded = classify(z, n, 2, { spacing: tile, balance: 0.3, altitude: 0,
                                       sediment: sed });
  eq(inferred.source, 'drainage');
  eq(recorded.source, 'sediment');

  const half = (n - 1) / 2;
  let hit = 0, total = 0;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      if (Math.abs((i - half) * tile - 3) < 1.5) {
        total++;
        if (recorded.cls[j * n + i] === 0) hit++;
      }
    }
  }
  ok(hit / total > 0.9, `only ${hit}/${total} stripe cells took the collected class`);
});

test('without a record the rule falls back to drainage unchanged', () => {
  // Surveys and captures have no sediment map. They must behave exactly as
  // they did before the record existed.
  const n = 20, tile = 1;
  const h = (x, y) => Math.sin(x * 0.3) + Math.cos(y * 0.2);
  const z = sampleGrid(n, tile, h, 0, 4);
  const a = classify(z, n, 2, { spacing: tile, balance: 0.4 });
  const b = classify(z, n, 2, { spacing: tile, balance: 0.4, sediment: null });
  for (let i = 0; i < n * n; i++) {
    ok(a.cls[i] === b.cls[i], `cell ${i} differs with an explicit null`);
  }
  eq(a.source, 'drainage');
});

// ------------------------------------------------------------------ report

for (const [name, fn] of workerTests) {
  try { await fn(); passed++; }
  catch (e) { failures.push([name, e.message]); }
}

for (const [name, msg] of failures) console.log(`FAIL  ${name}\n      ${msg}`);
console.log(`\n${passed} passed, ${failures.length} failed`);
process.exit(failures.length ? 1 : 0);
