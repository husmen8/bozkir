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

// ------------------------------------------------------------------ report

for (const [name, fn] of workerTests) {
  try { await fn(); passed++; }
  catch (e) { failures.push([name, e.message]); }
}

for (const [name, msg] of failures) console.log(`FAIL  ${name}\n      ${msg}`);
console.log(`\n${passed} passed, ${failures.length} failed`);
process.exit(failures.length ? 1 : 0);
