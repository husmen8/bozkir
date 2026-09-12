// Depth sorting, off the main thread.
//
// This is the piece the project is about. Everything else in the renderer is
// standard rasterisation; the order splats are drawn in is what produces
// boundary artifacts, and it lives alone in this file so it can be replaced
// without touching the renderer.
//
// A comparison sort over ~700k floats costs well over 100 ms in JavaScript,
// which would cap the frame rate below 10 fps. A counting sort over depth
// quantised to 16 bits is O(n) and runs in a few milliseconds.
//
// Two ways to use it, and the difference is the experiment.
//
//   live    one order per patch per frame, along the world view direction.
//           Exact while tiles are flat, because every copy of a patch is a
//           pure translation of it and translation preserves depth order.
//
//   cached  K orders per patch, precomputed once for K fixed directions.
//           A tile warped onto a surface is rotated, so the direction that
//           matters is the view direction expressed in that tile's own
//           frame; each cell picks the cached order nearest to its own.
//           This is what GSWT does with nine views (Section 3.4) and what
//           DAV-GSWT varies per tile.
//
// The live order is wrong the moment there is any relief, because it was
// computed along a direction no warped tile actually sees. The cached
// orders fix that for a tile with one frame - and cannot fix it for a tile
// whose frame varies per splat, which is what a smooth warp gives you.

let positions = null;   // Float32Array, 3 per splat
let patches = null;     // [{start, count}, ...]
let order = null;       // Uint32Array, scratch for one pass
let depths = null;      // Int32Array, scratch
let dist = null;        // Float32Array, scratch
// Packing constants, duplicated from merge.js rather than imported: this
// worker is a classic worker, not a module, so it cannot import. They are
// asserted equal in tests/test_web.mjs so the two copies cannot drift.
const SLOT_BITS = 3;
const INDEX_BITS = 32 - SLOT_BITS;
const INDEX_MASK = (1 << INDEX_BITS) - 1;

let groupDepth = null, groupSlot = null, groupLocal = null;
const counts = new Uint32Array(65536);
const starts = new Uint32Array(65536);

/** k directions covering where a camera looking at ground actually is.
 *
 *  Spreading them evenly over the whole sphere wastes half on directions
 *  pointing up out of the terrain, which no camera occupies, and leaves the
 *  nearest cached direction as much as 50 degrees away at k = 9 - worse
 *  than the tile tilt it is meant to correct for. GSWT picks nine by hand
 *  for the same reason: four in the plane, four at 45 degrees, one
 *  top-down.
 *
 *  Here they are spread over a hemisphere plus a margin, since a tile
 *  warped onto a slope can be seen from slightly below level.
 */
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

/** Counting sort of one patch along one direction, written into `out`.
 *
 *  Back-to-front: the prefix sum is walked from the far end, because the
 *  renderer blends with 'over' and needs distant splats first.
 */
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

/** One sorted stream for a merged group of cells. GSWT Section 3.4.
 *
 *  Where two cells straddle a boundary the camera is standing in, neither
 *  can be drawn in front of the other and their splats have to interleave.
 *  That means one order over the union of the group rather than one order
 *  per cell.
 *
 *  GSWT builds that order by merging the cells' cached orders using stored
 *  per-Gaussian projected depths, and falls back to a full sort only when
 *  tiles are very close to the camera. Here the full sort is cheaper than
 *  the merge, because the sort is a counting sort and the merge is not.
 *  Measured on this machine, far to near, total splats in the group:
 *
 *      splats     k-way merge     counting sort over the union
 *       300k          9.5 ms                3.8 ms
 *       600k         26.8 ms                7.9 ms
 *      1200k         76.8 ms               25.6 ms
 *
 *  The merge is O(n k) with a linear scan over k heads; the counting sort
 *  is O(n) and touches memory in one pass. So the union is sorted directly
 *  and `kwayMerge` in merge.js stays as the exact reference the tests check
 *  this against - the same arrangement as the CPU renderer standing behind
 *  the GPU one.
 *
 *  Every cell is a translated copy of its patch, and translation shifts all
 *  of that patch's depths by the same amount, so a cell contributes its
 *  patch's depths plus one constant. That is exact while tiles are flat and
 *  approximate once they are warped, in the same way and for the same
 *  reason as the live order itself.
 *
 *  `cells` is [{patch, start, count, offset:[x,y,z]}], at most MAX_GROUP of
 *  them. The result packs each cell's slot in the group into the high bits
 *  of the index so one draw call can place them all.
 */
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
        const shift = c.offset[0] * fx + c.offset[1] * fy + c.offset[2] * fz;
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
        // K orders per patch, computed once. The eye is irrelevant here: a
        // patch sits at the origin and only the direction changes the order.
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
        // One stream per group. Groups with a single cell are handled by
        // the normal per-cell path and should not be sent here.
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