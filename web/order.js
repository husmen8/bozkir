// Tile-level draw order, GSWT 3.4 "tile-level topological sorting".
//
// Sorting cells by a depth key fails when the view lines up with a grid
// axis: every cell in a row gets the same key, the tie-break decides the
// whole row, and turning through alignment flips it in one frame (the
// row-wide pop). Instead each pair of neighbours is ordered by which side
// of their shared boundary plane the eye is on - a sign test, immune to
// ties - and the pairs are sorted topologically.
//
// A boundary the eye is standing on has no sign; those pairs are dropped
// here and are exactly the ones merge.js merges (same |n . (eye - edge)|).
//
// No WebGL, so it runs under node and is tested there.

import { boundarySign, neighbourPairs } from './grid.js';
export { boundary, boundarySign, neighbourPairs } from './grid.js';

/** Pairwise constraints for one eye position: [before, after], `before`
 *  drawn first. Pairs within `epsilon` of their boundary plane go to `weak`
 *  instead (pinning them only moves the flip). */
export function constraints(cells, eye, { epsilon = 0 } = {}) {
  const edges = [];
  const weak = [];
  for (const [a, b] of neighbourPairs(cells)) {
    const s = boundarySign(cells[a], cells[b], eye);
    if (Math.abs(s) <= epsilon) { weak.push([a, b, s]); continue; }
    edges.push(s > 0 ? [a, b] : [b, a]);
  }
  return { edges, weak };
}

/** Draw order for a set of cells, far to near.
 *
 *  Kahn's algorithm; the depth `key` (larger = further, one per cell) only
 *  chooses among cells that are unblocked at the same time, i.e. cells that
 *  share no boundary and cannot overlap.
 *
 *  On flat ground there are no cycles. Relief can tilt boundary planes so
 *  three cells disagree; the stalled cells are then released by depth key
 *  (the old behaviour) and counted in `cycles`. */
export function drawOrder(cells, eye, { key = null, epsilon = 0 } = {}) {
  const n = cells.length;
  const { edges, weak } = constraints(cells, eye, { epsilon });

  const depth = key || cells.map((c) => Math.hypot(
    c.x - eye[0], c.y - eye[1], (c.z || 0) - eye[2]));

  const after = Array.from({ length: n }, () => []);
  const indeg = new Int32Array(n);
  for (const [u, v] of edges) { after[u].push(v); indeg[v]++; }

  const done = new Uint8Array(n);
  const out = [];
  let cycles = 0;

  while (out.length < n) {
    // Furthest unblocked cell. A linear scan: the grid is a few hundred
    // visible cells at most.
    let best = -1;
    for (let k = 0; k < n; k++) {
      if (done[k] || indeg[k] > 0) continue;
      if (best < 0 || depth[k] > depth[best]) best = k;
    }

    if (best < 0) {
      // Only cycles left: release the furthest and drop its incoming edges.
      cycles++;
      for (let k = 0; k < n; k++) {
        if (done[k]) continue;
        if (best < 0 || depth[k] > depth[best]) best = k;
      }
      indeg[best] = 0;
    }

    done[best] = 1;
    out.push(best);
    // Clamped: breaking a cycle zeroes an in-degree with edges outstanding.
    for (const v of after[best]) if (--indeg[v] < 0) indeg[v] = 0;
  }

  return { order: out, weak, cycles, constrained: edges.length };
}

/** Number of constraints `order` breaks (0 = all satisfied). For tests. */
export function violations(cells, eye, order, { epsilon = 0 } = {}) {
  const rank = new Int32Array(cells.length);
  for (let k = 0; k < order.length; k++) rank[order[k]] = k;
  const { edges } = constraints(cells, eye, { epsilon });
  let bad = 0;
  for (const [u, v] of edges) if (rank[u] > rank[v]) bad++;
  return bad;
}
