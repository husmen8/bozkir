// Depth sorting, off the main thread.
//
// A comparison sort over ~700k floats takes 100+ ms in JS; a counting sort
// over depth quantised to 16 bits is O(n) and takes a few ms.
//
// Two modes:
//   live    one order per patch per frame along the world view direction.
//           Exact while tiles are flat (copies are pure translations).
//   cached  K orders per patch for K fixed directions, computed once; each
//           cell picks the one nearest the view direction in its own frame,
//           which is what a warped tile sees (GSWT 3.4 uses nine).
// A tile whose frame varies per splat (smooth warp) is exact in neither.

let positions = null;   // Float32Array, 3 per splat
let patches = null;     // [{start, count}, ...]
let order = null;       // Uint32Array, scratch for one pass
let depths = null;      // Int32Array, scratch
let dist = null;        // Float32Array, scratch
// Duplicated from merge.js: a classic worker cannot import. Checked equal
// in tests/test_web.mjs.
const SLOT_BITS = 3;
const INDEX_BITS = 32 - SLOT_BITS;
const INDEX_MASK = (1 << INDEX_BITS) - 1;

let groupDepth = null, groupSlot = null, groupLocal = null;
const counts = new Uint32Array(65536);
const starts = new Uint32Array(65536);

/** k directions over where a camera looking at ground can be: a hemisphere
 *  plus a margin (a tile on a slope can be seen from slightly below). Even
 *  over the whole sphere, half point up out of the ground and the nearest
 *  one was up to 50 degrees off at k = 9. */
function directions(k, margin = 0.25) {
    const out = [];
    const span = 1 + margin;          // 1 is exactly a hemisphere
    for (let i = 0; i < k; i++) {
        const t = (i + 0.5) / k;
        const phi = Math.acos(1 - span * t);      // 0 at straight down
        const theta = Math.PI * (1 + Math.sqrt(5)) * (i + 0.5);
        out.push([Math.cos(theta) * Math.sin(phi),
        Math.sin(theta) * Math.sin(phi),
        -Math.cos(phi)]);               // looking down into the ground
    }
    return out;
}

/** Counting sort of one patch along one direction into `out`, far to near
 *  (the renderer blends with 'over'). */
function sortPatch(p, dir, eye, out) {
    const [fx, fy, fz] = dir;
    const [ex, ey, ez] = eye;
    const a = p.start, b = p.start + p.count;
    if (b <= a) return;

    let min = Infinity, max = -Infinity;
    for (let i = a; i < b; i++) {
        const d = (positions[3 * i] - ex) * fx
            + (positions[3 * i + 1] - ey) * fy
            + (positions[3 * i + 2] - ez) * fz;
        dist[i] = d;
        if (d < min) min = d;
        if (d > max) max = d;
    }

    const scale = max > min ? 65535 / (max - min) : 0;
    counts.fill(0);
    for (let i = a; i < b; i++) {
        const q = (dist[i] - min) * scale | 0;
        depths[i] = q;
        counts[q]++;
    }
    starts[65535] = a;
    for (let q = 65535; q > 0; q--) starts[q - 1] = starts[q] + counts[q];
    for (let i = a; i < b; i++) out[starts[depths[i]]++] = i;
}

/** One sorted stream for a merged group of cells (GSWT 3.4).
 *
 *  GSWT merges the cells' cached orders; here a counting sort over the
 *  union is faster (measured, far to near):
 *
 *      splats     k-way merge     counting sort over the union
 *       300k          9.5 ms                3.8 ms
 *       600k         26.8 ms                7.9 ms
 *      1200k         76.8 ms               25.6 ms
 *
 *  kwayMerge in merge.js stays as the reference the tests compare against.
 *
 *  A cell is a translated copy of its patch, so its depths are the patch's
 *  plus a constant `shift` (exact while flat). The viewer computes `shift`
 *  because it knows the grid rotation; `dir` arrives already in the tile
 *  frame. Tests may pass an `offset` vector instead.
 *
 *  `cells` is [{patch, start, count, shift}], at most MAX_GROUP. Each
 *  cell's slot is packed into the high bits of the index. */
function sortGroup(cells, dir, eye) {
    const [fx, fy, fz] = dir;
    let total = 0;
    for (const c of cells) total += c.count;
    const out = new Uint32Array(total);
    if (!total) return out;

    // Depths first, into a scratch big enough for the union.
    if (!groupDepth || groupDepth.length < total) {
        groupDepth = new Float32Array(total);
        groupSlot = new Uint8Array(total);
        groupLocal = new Uint32Array(total);
    }
    let w = 0, min = Infinity, max = -Infinity;
    for (let s = 0; s < cells.length; s++) {
        const c = cells[s];
        const shift = typeof c.shift === 'number' ? c.shift
            : c.offset[0] * fx + c.offset[1] * fy + c.offset[2] * fz;
        const a = c.start, b = c.start + c.count;
        for (let i = a; i < b; i++) {
            const d = (positions[3 * i] - eye[0]) * fx
                + (positions[3 * i + 1] - eye[1]) * fy
                + (positions[3 * i + 2] - eye[2]) * fz + shift;
            groupDepth[w] = d;
            groupSlot[w] = s;
            groupLocal[w] = i;
            if (d < min) min = d;
            if (d > max) max = d;
            w++;
        }
    }

    const scale = max > min ? 65535 / (max - min) : 0;
    counts.fill(0);
    const q = new Int32Array(total);
    for (let i = 0; i < total; i++) {
        const v = (groupDepth[i] - min) * scale | 0;
        q[i] = v; counts[v]++;
    }
    starts[65535] = 0;
    for (let v = 65535; v > 0; v--) starts[v - 1] = starts[v] + counts[v];
    for (let i = 0; i < total; i++) {
        out[starts[q[i]]++] = ((groupSlot[i] << INDEX_BITS)
            | (groupLocal[i] & INDEX_MASK)) >>> 0;
    }
    return out;
}

self.onmessage = (e) => {
    const msg = e.data;

    if (msg.type === 'init') {
        positions = new Float32Array(msg.positions);
        const n = positions.length / 3;
        patches = msg.patches && msg.patches.length
            ? msg.patches : [{ start: 0, count: n }];
        order = new Uint32Array(n);
        depths = new Int32Array(n);
        dist = new Float32Array(n);
        self.postMessage({ type: 'ready', count: n, patches: patches.length });
        return;
    }

    if (msg.type === 'cache') {
        // K orders per patch. The eye does not matter: patches sit at the
        // origin and only the direction changes the order.
        const dirs = directions(msg.views);
        const n = positions.length / 3;
        const t0 = performance.now();
        const all = new Uint32Array(n * dirs.length);
        const scratch = new Uint32Array(n);
        for (let k = 0; k < dirs.length; k++) {
            for (const p of patches) sortPatch(p, dirs[k], [0, 0, 0], scratch);
            all.set(scratch, k * n);
        }
        self.postMessage({
            type: 'cached', views: dirs.length, dirs, stride: n,
            order: all.buffer, ms: performance.now() - t0,
        }, [all.buffer]);
        return;
    }

    if (msg.type === 'mergeGroups') {
        // One stream per group (single cells go the normal per-cell path).
        const t0 = performance.now();
        const out = [], sizes = [];
        for (const g of msg.groups) {
            const buf = sortGroup(g, msg.forward, msg.eye);
            out.push(buf.buffer); sizes.push(buf.length);
        }
        self.postMessage({ type: 'mergedGroups', orders: out, sizes,
                           seq: msg.seq, ms: performance.now() - t0 }, out);
        return;
    }

    if (msg.type === 'sort') {
        if (!positions) return;
        const t0 = performance.now();
        for (const p of patches) sortPatch(p, msg.forward, msg.eye, order);
        const out = order.slice();
        self.postMessage(
            { type: 'sorted', order: out.buffer, ms: performance.now() - t0 },
            [out.buffer]);
    }
};
