// Selective tile merging, GSWT 3.4.
//
// Each cell is drawn from its own sorted order, so where neighbours overlap
// one simply covers the other - and which one flips as they cross in depth.
// Near the plane of their shared boundary their splats should interleave,
// so the pair is merged into one sorted stream. GSWT detects this with
// |n . (edge - eye)| using the unnormalised vector: the perpendicular
// distance to the plane, which does not fall off with range - being in the
// plane is the condition, not being near the edge.
//
// No WebGL, so it runs under node and is tested there.

import { boundary, boundarySign, neighbourPairs } from './grid.js';
export { boundary, boundarySign, neighbourPairs } from './grid.js';

// A merged draw packs the cell's slot in the group into the top bits of the
// splat index: 3 bits leaves 29 for the index (536M splats). The cap of 8
// also bounds the work, since grouping is transitive and could otherwise
// pull a whole row into one stream.
export const SLOT_BITS = 3;
export const MAX_GROUP = 1 << SLOT_BITS;      // 8
export const INDEX_BITS = 32 - SLOT_BITS;
export const INDEX_MASK = (1 << INDEX_BITS) - 1;

export function packIndex(slot, index) {
  return ((slot << INDEX_BITS) | (index & INDEX_MASK)) >>> 0;
}
export function unpackSlot(v) { return v >>> INDEX_BITS; }
export function unpackIndex(v) { return v & INDEX_MASK; }

/** GSWT's merge criterion: distance from the eye to the shared boundary
 *  plane, in world units. Small = the two cells interpenetrate on screen. */
export function mergeScore(a, b, eye) {
  return Math.abs(boundarySign(a, b, eye));
}

/** Which cells to merge, for one eye position.
 *
 *  Pairs under `threshold` (world units; set it relative to the tile) are
 *  unioned cheapest first, so when the cap bites it is the pairs that
 *  needed it least that are dropped. Every cell gets a group, singletons
 *  included. */
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

/** Interleave already-sorted streams ({order, depth, slot}, far to near)
 *  into one far-to-near stream of packed indices. The exact reference the
 *  worker's counting sort is tested against. A linear scan over the heads
 *  beats a heap at eight streams. */
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
