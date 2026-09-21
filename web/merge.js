// Selective tile merging. GSWT Section 3.4.
//
// Cells are drawn one at a time, each from its own pre-sorted order, so where
// two neighbours overlap in pixels one simply covers the other. That is wrong
// twice over: the covering cell changes when the two cross in depth, and even
// with the order held fixed the two sets of splats should interleave rather
// than stack.
//
// GSWT merges such a pair into one sorted stream when the camera lies near the
// plane of their shared boundary, detected by the absolute dot product between
// the boundary normal and the *unnormalised* camera-to-edge vector. Leaving it
// unnormalised is the whole trick: the quantity is then the perpendicular
// distance from the camera to the boundary plane, so a boundary the camera is
// standing on scores near zero and a boundary it is well to one side of scores
// high and is left alone. Note that this does not fall off with range - an eye
// lying in a boundary's plane scores zero however far away it is - because
// lying in the plane is the condition being detected, not being near the edge.
//
// Nothing here touches WebGL, so it runs under node and is tested there.

/** How many cells a merged group may hold.
 *
 *  The cap is not arbitrary. A merged draw packs the group-local cell slot
 *  into the high bits of the splat index, and the index has to keep enough
 *  room for the splat count: 3 bits leaves 29, which is 536 million splats.
 *  It also bounds the work, since merging is linear in the splats of a group
 *  and grouping is transitive - without a cap, one chain of low-scoring
 *  boundaries pulls a whole row into a single stream and the frame budget
 *  goes with it. */
import { boundary, boundarySign, neighbourPairs } from './grid.js';
export { boundary, boundarySign, neighbourPairs } from './grid.js';

export const SLOT_BITS = 3;
export const MAX_GROUP = 1 << SLOT_BITS;      // 8
export const INDEX_BITS = 32 - SLOT_BITS;
export const INDEX_MASK = (1 << INDEX_BITS) - 1;

export function packIndex(slot, index) {
  return ((slot << INDEX_BITS) | (index & INDEX_MASK)) >>> 0;
}
export function unpackSlot(v) { return v >>> INDEX_BITS; }
export function unpackIndex(v) { return v & INDEX_MASK; }

/** GSWT's merge criterion, in world units.
 *
 *  Distance from the eye to the plane of the shared boundary. Small means the
 *  camera is near that plane, which is when the two cells' splats genuinely
 *  interpenetrate on screen. */
export function mergeScore(a, b, eye) {
  return Math.abs(boundarySign(a, b, eye));
}

/** Which cells to merge with which, for one camera position.
 *
 *  Pairs below the threshold are unioned, cheapest first, so that when the cap
 *  bites it is the boundaries the camera is furthest from that get dropped -
 *  the ones that needed merging least. Returns one entry per cell holding the
 *  index of its group, and the groups themselves; a cell in no pair gets a
 *  group of its own, so the caller can treat every cell uniformly.
 *
 *  `threshold` is a distance in world units. Tile-relative is the useful way
 *  to set it, since it is the tile that decides how far a splat reaches past
 *  its own boundary. */
export function mergeGroups(cells, eye, { threshold = 1.0, cap = MAX_GROUP } = {}) {
  const parent = new Int32Array(cells.length);
  const size = new Int32Array(cells.length).fill(1);
  for (let k = 0; k < cells.length; k++) parent[k] = k;

  const find = (k) => {
    while (parent[k] !== k) { parent[k] = parent[parent[k]]; k = parent[k]; }
    return k;
  };

  const scored = [];
  for (const [a, b] of neighbourPairs(cells)) {
    const s = mergeScore(cells[a], cells[b], eye);
    if (s < threshold) scored.push([s, a, b]);
  }
  scored.sort((p, q) => p[0] - q[0]);

  let merged = 0;
  for (const [, a, b] of scored) {
    const ra = find(a), rb = find(b);
    if (ra === rb) continue;
    if (size[ra] + size[rb] > cap) continue;
    parent[ra] = rb;
    size[rb] += size[ra];
    merged++;
  }

  const groupOf = new Int32Array(cells.length).fill(-1);
  const groups = [];
  for (let k = 0; k < cells.length; k++) {
    const r = find(k);
    if (groupOf[r] < 0) { groupOf[r] = groups.length; groups.push([]); }
    groupOf[k] = groupOf[r];
    groups[groupOf[r]].push(k);
  }
  return { groupOf, groups, pairsMerged: merged, pairsConsidered: scored.length };
}

/** Interleave several already-sorted streams into one.
 *
 *  Each stream is one cell's pre-sorted order together with the world depth of
 *  each entry, both running far to near. The result is the same splats in one
 *  far-to-near order, with each index tagged by the slot of the cell it came
 *  from so a single draw call can place them all.
 *
 *  A linear scan over the heads beats a heap here: `cap` is eight, and eight
 *  comparisons with no pointer chasing is faster than maintaining the heap. */
export function kwayMerge(streams) {
  let total = 0;
  for (const s of streams) total += s.order.length;
  const out = new Uint32Array(total);
  const head = new Int32Array(streams.length);
  let w = 0;

  while (w < total) {
    let best = -1, bestDepth = -Infinity;
    for (let k = 0; k < streams.length; k++) {
      const h = head[k];
      if (h >= streams[k].order.length) continue;
      const d = streams[k].depth[h];
      if (d > bestDepth) { bestDepth = d; best = k; }
    }
    if (best < 0) break;
    out[w++] = packIndex(streams[best].slot, streams[best].order[head[best]]);
    head[best]++;
  }
  return out;
}