// Tile-level draw order. GSWT Section 3.4, "tile-level topological sorting".
//
// Cells are composited far to near, so something has to say which of two
// cells is further. Sorting them by a depth key - distance to the centre, or
// the nearest footprint corner - is only approximately right, and it fails
// in a specific, reproducible way.
//
// The key is a projection onto the view direction. When that direction lines
// up with a grid axis, every cell in a row projects to the *same* value, bit
// for bit: nine cells, one key. Whatever breaks the tie then decides the
// order for the whole row at once, and rotating through alignment reverses
// all nine in a single frame. That is the row-wide repaint the pop meter
// sees. It is not an interleaving problem and no amount of merging fixes it,
// because the order being flipped was never determined in the first place.
//
// GSWT's answer is to stop ranking cells globally and instead constrain them
// pairwise. Two neighbours only ever overlap along the boundary they share,
// so that boundary is what decides them: if the camera is on one cell's side
// of the shared plane, that cell is nearer and renders last. The constraint
// is a sign test, so it is immune to the tie - two cells at equal depth still
// have a well-defined side. Collect one constraint per adjacent pair, then
// topologically sort the graph.
//
// What the sign test cannot do is decide a boundary the camera is standing
// on, where the sign is passing through zero and neither cell is in front.
// Those constraints are dropped here rather than enforced at random, and the
// magnitude that decides whether to drop one is the same |n . (eye - edge)|
// that merge.js thresholds on. The two halves of Section 3.4 meet here: an
// order this module declines to fix is exactly an order that wants merging.
//
// No WebGL, so it runs under node and is tested there.

/** The shared boundary of two adjacent cells: midpoint and unit normal.
 *
 *  Same construction as merge.js - the normal is taken from the line between
 *  the two centres rather than a grid axis, so it survives a rotated layout
 *  and tilts with the terrain when relief lifts the two cells unequally.
 *  Kept as its own function here rather than imported so this module stays
 *  usable on its own. */
export function boundary(a, b) {
    const dx = b.x - a.x, dy = b.y - a.y, dz = (b.z || 0) - (a.z || 0);
    const len = Math.hypot(dx, dy, dz) || 1;
    return {
        mid: [(a.x + b.x) / 2, (a.y + b.y) / 2, ((a.z || 0) + (b.z || 0)) / 2],
        normal: [dx / len, dy / len, dz / len],
    };
}

/** Signed distance from the eye to the plane of the boundary between a and b.
 *
 *  The normal runs a to b, so a positive value puts the camera on b's side:
 *  b is the nearer cell and must be drawn after a in back-to-front order.
 *  The magnitude is how firmly that holds, and it is the quantity GSWT
 *  thresholds for merging. */
export function boundarySign(a, b, eye) {
    const { mid, normal } = boundary(a, b);
    return normal[0] * (eye[0] - mid[0])
        + normal[1] * (eye[1] - mid[1])
        + normal[2] * (eye[2] - mid[2]);
}

/** Adjacent pairs of a grid of cells, each cell carrying integer `i`, `j`. */
export function neighbourPairs(cells) {
    const byIJ = new Map();
    for (let k = 0; k < cells.length; k++) {
        byIJ.set(`${cells[k].i},${cells[k].j}`, k);
    }
    const pairs = [];
    for (let k = 0; k < cells.length; k++) {
        const c = cells[k];
        for (const [di, dj] of [[1, 0], [0, 1]]) {
            const n = byIJ.get(`${c.i + di},${c.j + dj}`);
            if (n !== undefined) pairs.push([k, n]);
        }
    }
    return pairs;
}

/** The pairwise constraints for one camera position.
 *
 *  Each entry is [before, after]: `before` is drawn first. Pairs whose
 *  boundary plane passes within `epsilon` of the eye are left out, since
 *  their sign is the thing about to change and pinning it would only move
 *  the flip rather than remove it. `weak` returns them so a caller can hand
 *  them to merging. */
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
 *  Kahn's algorithm, with the depth key used only to choose among cells that
 *  are simultaneously unblocked. That ordering of the two matters: the
 *  constraints decide wherever they exist, and the key fills in for cells
 *  that share no boundary and therefore cannot overlap - the case where it
 *  is harmless for the key to be degenerate.
 *
 *  `key` is one number per cell, larger meaning further, and is what the
 *  viewer already computes. Passing it keeps the result close to today's
 *  order everywhere the constraints are silent, so the change shows up only
 *  where it is supposed to.
 *
 *  A grid on flat ground cannot produce a cycle: the constraints reduce to
 *  "further along x" and "further along y", each a total order. Relief tilts
 *  the boundary planes and three cells can then disagree. Rather than fail,
 *  the stalled cells are released by depth key - which is what the renderer
 *  did for all of them until now, so a cycle is no worse than the status quo
 *  and is reported in `cycles` so it can be measured rather than guessed at. */
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
        // Furthest cell with nothing left waiting on it. Linear scan: a heap
        // would need reordering on every decrement, and the grid is a few
        // hundred cells at most.
        let best = -1;
        for (let k = 0; k < n; k++) {
            if (done[k] || indeg[k] > 0) continue;
            if (best < 0 || depth[k] > depth[best]) best = k;
        }

        if (best < 0) {
            // Everything left is inside a cycle. Release the furthest of them and
            // drop its incoming constraints so the sort can continue.
            cycles++;
            for (let k = 0; k < n; k++) {
                if (done[k]) continue;
                if (best < 0 || depth[k] > depth[best]) best = k;
            }
            indeg[best] = 0;
        }

        done[best] = 1;
        out.push(best);
        // Clamped, because breaking a cycle zeroes an in-degree while edges
        // into that cell are still outstanding.
        for (const v of after[best]) if (--indeg[v] < 0) indeg[v] = 0;
    }

    return { order: out, weak, cycles, constrained: edges.length };
}

/** True if `order` satisfies every constraint. For tests and for the panel. */
export function violations(cells, eye, order, { epsilon = 0 } = {}) {
    const rank = new Int32Array(cells.length);
    for (let k = 0; k < order.length; k++) rank[order[k]] = k;
    const { edges } = constraints(cells, eye, { epsilon });
    let bad = 0;
    for (const [u, v] of edges) if (rank[u] > rank[v]) bad++;
    return bad;
}