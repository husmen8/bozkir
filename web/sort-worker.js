// Depth sorting, off the main thread.
//
// This is the piece the project is ultimately about. Everything else in the
// renderer is standard rasterisation; the order splats are drawn in is what
// produces boundary artifacts, and what later experiments will change.
// It lives alone in this file so it can be replaced without touching the
// renderer.
//
// A comparison sort over ~700k floats costs well over 100 ms in JavaScript,
// which would cap the frame rate below 10 fps. A counting sort over depth
// quantised to 16 bits is O(n) and runs in a few milliseconds. The lost
// precision is invisible: 65536 depth buckets across a scene is far finer
// than the depth differences that change a pixel.

let positions = null;   // Float32Array, 3 per splat
let count = 0;
let order = null;       // Uint32Array, reused
let depths = null;      // Int32Array, reused
let counts = new Uint32Array(65536);
let starts = new Uint32Array(65536);

self.onmessage = (e) => {
    const msg = e.data;

    if (msg.type === 'init') {
        positions = new Float32Array(msg.positions);
        count = positions.length / 3;
        order = new Uint32Array(count);
        depths = new Int32Array(count);
        self.postMessage({ type: 'ready', count });
        return;
    }

    if (msg.type === 'sort') {
        if (!positions) return;
        const t0 = performance.now();

        // Only the forward row of the view matrix matters: depth is the
        // projection of (position - eye) onto the viewing direction.
        const [fx, fy, fz] = msg.forward;
        const [ex, ey, ez] = msg.eye;

        let min = Infinity, max = -Infinity;
        for (let i = 0; i < count; i++) {
            const d = (positions[3 * i] - ex) * fx
                + (positions[3 * i + 1] - ey) * fy
                + (positions[3 * i + 2] - ez) * fz;
            depths[i] = d;                       // truncated, refined below
            if (d < min) min = d;
            if (d > max) max = d;
        }

        // Quantise to 16 bits across the actual depth range in view.
        const scale = max > min ? 65535 / (max - min) : 0;
        counts.fill(0);
        for (let i = 0; i < count; i++) {
            const d = (positions[3 * i] - ex) * fx
                + (positions[3 * i + 1] - ey) * fy
                + (positions[3 * i + 2] - ez) * fz;
            const b = (d - min) * scale | 0;
            depths[i] = b;
            counts[b]++;
        }

        // Prefix sum, walked from the far end so the output is back-to-front:
        // the renderer blends with 'over', which needs distant splats first.
        starts[65535] = 0;
        for (let b = 65535; b > 0; b--) starts[b - 1] = starts[b] + counts[b];
        for (let i = 0; i < count; i++) order[starts[depths[i]]++] = i;

        const out = order.slice();
        self.postMessage(
            { type: 'sorted', order: out.buffer, ms: performance.now() - t0 },
            [out.buffer]
        );
    }
};