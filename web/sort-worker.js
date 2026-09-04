// Depth sorting, off the main thread.
//
// This is the piece the project is ultimately about. Everything else in the
// renderer is standard rasterisation; the order splats are drawn in is what
// produces boundary artifacts, and what later experiments will change.
// It lives alone in this file so it can be replaced without touching the
// renderer.
//
// A comparison sort over ~400k floats costs well over 100 ms in JavaScript,
// which would cap the frame rate below 10 fps. A counting sort over depth
// quantised to 16 bits is O(n) and runs in a few milliseconds. The lost
// precision is invisible: 65536 depth buckets across a scene is far finer
// than the depth differences that change a pixel.
//
// Each patch is sorted independently, into its own slice of one shared
// index array. That is what makes tiling affordable: a tile is a translated
// copy of a patch, and translation does not change which splat is in front
// of which along a view direction, so one sort per patch serves every copy
// of it on the grid. Sorting all copies together would be O(splats x tiles)
// and unaffordable, which is exactly why GSWT pre-sorts tiles instead
// (Section 3.4) - and exactly why the boundary artifact exists.

let positions = null;   // Float32Array, 3 per splat
let patches = null;     // [{start, count}, ...]
let order = null;       // Uint32Array, reused
let depths = null;      // Int32Array, reused
let dist = null;        // Float32Array, depth per splat, reused
const counts = new Uint32Array(65536);
const starts = new Uint32Array(65536);

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

    if (msg.type === 'sort') {
        if (!positions) return;
        const t0 = performance.now();
        const [fx, fy, fz] = msg.forward;
        const [ex, ey, ez] = msg.eye;

        for (const p of patches) {
            const a = p.start, b = p.start + p.count;
            if (b <= a) continue;

            // One pass to measure the range, keeping each depth so the second
            // pass does not recompute it. Halves the arithmetic, which matters
            // once a tile set runs into millions of Gaussians.
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

            // Prefix sum walked from the far end, so the output is back-to-front:
            // the renderer blends with 'over', which needs distant splats first.
            // Offsets are relative to this patch's slice of the shared array.
            starts[65535] = a;
            for (let q = 65535; q > 0; q--) starts[q - 1] = starts[q] + counts[q];
            for (let i = a; i < b; i++) order[starts[depths[i]]++] = i;
        }

        const out = order.slice();
        self.postMessage(
            { type: 'sorted', order: out.buffer, ms: performance.now() - t0 },
            [out.buffer]
        );
    }
};