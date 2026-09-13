// Capturing a camera sweep, so popping can be measured instead of watched.
//
// scripts/pop_metric.py compares two sequences of frames and reports the
// part of the change that camera motion cannot account for. That comparison
// is only meaningful if both sequences travel the *same* path, which a hand
// flown camera cannot do twice. So the path is scripted: the sweep starts
// from wherever the camera is, holds elevation and distance, and turns
// through a fixed arc in equal steps.
//
// Anchoring on the current camera rather than on any scene constant is what
// makes this work for a capture nobody has seen before. load() already
// frames the scene - target and distance are chosen from the splat spread -
// so whatever the person dropped in, pointing the camera at what interests
// them and pressing capture gives a path scaled to that scene.
//
// Two things are waited for before each frame is kept, and both matter more
// than they look:
//
//   - the sort worker going idle, because the order arrives a frame or more
//     after the camera moves. Capturing without waiting would measure sort
//     latency, which is a real artifact but not the one being compared, and
//     it would contaminate both orderings equally and hide the difference.
//   - a couple of settle frames afterwards, since LOD cross-fades and the
//     merged-group request both take a frame to catch up.
//
// The result is a zip of PNGs, one file rather than sixty downloads.

import { writeZip } from './tileset.js';

/** Read the canvas back into a 2D canvas, synchronously.
 *
 *  The context is created without preserveDrawingBuffer, so the colour
 *  buffer is undefined once the frame ends. This has to run inside the same
 *  animation frame as the draw, which is why capture is stepped from the
 *  render loop rather than driven by a timer.
 *
 *  Only the pixel read happens here. Turning them into a PNG is much the
 *  slower half and does not need the frame, so it is left to the caller to
 *  start and not wait for.
 */
function readFrame(gl, canvas, maxEdge) {
    const w = canvas.width, h = canvas.height;
    const px = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);

    // GL's origin is bottom left and the 2D canvas's is top left, so the rows
    // come back upside down. An unflipped capture still measures correctly -
    // the metric only ever compares frames with each other - but every figure
    // made from it would be upside down.
    const flipped = new Uint8ClampedArray(w * h * 4);
    const row = w * 4;
    for (let y = 0; y < h; y++) {
        flipped.set(px.subarray((h - 1 - y) * row, (h - y) * row), y * row);
    }

    const full = document.createElement('canvas');
    full.width = w; full.height = h;
    full.getContext('2d').putImageData(new ImageData(flipped, w, h), 0, 0);

    // Captured frames are measured, not looked at, and the cost of measuring
    // them grows with pixel count twice over: the PNG encode here, and the
    // block-matched motion search in pop_metric.py afterwards, which is the
    // slower of the two by a wide margin. Capping the long edge cuts both by
    // the square of the ratio while leaving blocks far larger than the
    // artifacts being counted. Pass 0 for full resolution.
    if (!maxEdge || Math.max(w, h) <= maxEdge) return full;
    const k = maxEdge / Math.max(w, h);
    const small = document.createElement('canvas');
    small.width = Math.round(w * k);
    small.height = Math.round(h * k);
    const ctx = small.getContext('2d');
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(full, 0, 0, small.width, small.height);
    return small;
}

/** PNG bytes from a canvas. Slow, and deliberately not awaited by the
 *  capture loop: encoding one frame overlaps with the next pose's wait. */
function encode(cv) {
    return new Promise((res) => cv.toBlob(
        (b) => b.arrayBuffer().then((a) => res(new Uint8Array(a))), 'image/png'));
}

export class Capture {
    /** `cam` is the live camera, `isBusy` reports whether a sort is in
     *  flight, `onProgress` is called with (done, total) for the UI. */
    constructor({ gl, canvas, cam, isBusy, onProgress, onDone }) {
        Object.assign(this, { gl, canvas, cam, isBusy, onProgress, onDone });
        this.active = false;
    }

    /** Begin a sweep. Returns false if one is already running.
     *
     *  `arc` defaults to a full turn: popping depends on the angle between
     *  the view and the tile grid, so a partial sweep can miss the very
     *  alignments the comparison is about.
     */
    start({ frames = 72, arc = 360, settle = 1, name = 'sweep',
        maxEdge = 1024, target = null } = {}) {
        if (this.active || !frames) return false;
        const c = this.cam;
        this.active = true;
        this.maxEdge = maxEdge;
        this.files = [];
        this.jobs = [];
        this.i = 0;
        this.held = 0;
        this.waited = 0;
        this.warned = false;
        this.plan = {
            frames, settle, name,
            patience: Math.max(settle + 1, 90),
            az0: c.azimuth,
            step: arc / frames,
            elevation: c.elevation,
            distance: c.distance,
            // Whatever the sweep orbits, it orbits for every frame of both runs.
            // Freezing it here is what makes the two sequences comparable; the
            // caller passing a scene centre instead of the live target is what
            // makes them useful, since WASD moves the target and a target that
            // has drifted off the terrain orbits a point in mid air.
            target: (target || c.target).slice(),
            fly: c.fly,
        };
        c.fly = false;                      // a scripted path needs the orbit
        this.pose();
        return true;
    }

    cancel() { this.active = false; this.files = []; }

    /** Put the camera where frame `i` wants it. */
    pose() {
        const p = this.plan, c = this.cam;
        c.azimuth = p.az0 + this.i * p.step;
        this.waited = 0;
        c.elevation = p.elevation;
        c.distance = p.distance;
        c.target = p.target.slice();
        this.held = 0;
    }

    /** Called once per rendered frame, after the draw. Returns nothing; the
     *  sweep advances itself and calls `onDone` with a zip when finished. */
    async step() {
        if (!this.active || this.pending) return;

        this.waited++;
        if (this.isBusy && this.isBusy()) {
            // A sort is still in flight. Waiting is normal for a frame or two;
            // waiting for ever is a bug somewhere upstream, and silently hanging
            // is the worst way to report one. Go on without the wait and say so,
            // so a capture always finishes and always tells you if it is
            // measuring something it should not be.
            if (this.waited < this.plan.patience) return;
            if (!this.warned) {
                this.warned = true;
                console.warn('capture: still busy after ' + this.plan.patience
                    + ' frames; capturing anyway. Frames may show the previous pose\'s'
                    + ' ordering, so treat the numbers with suspicion.');
            }
        }
        if (this.held < this.plan.settle) { this.held++; return; }

        // Read now, encode later. The pixel read has to happen in this frame;
        // the PNG does not, and encoding it here would idle the renderer for
        // longer than the sort it just waited on. Posing the next frame first
        // means every encode overlaps with the next pose's wait, which is dead
        // time otherwise.
        const cv = readFrame(this.gl, this.canvas, this.maxEdge);
        const name = `f${String(this.i).padStart(4, '0')}.png`;
        const slot = this.jobs.length;
        this.jobs.push(encode(cv).then((b) => { this.files[slot] = [name, b]; }));
        if (this.onProgress) this.onProgress(this.jobs.length, this.plan.frames);

        this.i++;
        if (this.i >= this.plan.frames) {
            this.finish();
        } else {
            this.pose();
        }
    }

    async finish() {
        const p = this.plan;
        this.active = false;                 // stop stepping while encodes drain
        await Promise.all(this.jobs);
        const zip = writeZip(this.files);
        const url = URL.createObjectURL(new Blob([zip], { type: 'application/zip' }));
        const a = document.createElement('a');
        a.href = url;
        a.download = `${p.name}.zip`;
        a.click();
        // Revoking immediately can cancel the download on some builds; a tick
        // later is enough and leaks nothing worth worrying about.
        setTimeout(() => URL.revokeObjectURL(url), 10000);

        this.cam.fly = p.fly;
        this.cam.azimuth = p.az0;
        const files = this.files;
        this.files = [];
        this.jobs = [];
        if (this.onDone) this.onDone(files.length, `${p.name}.zip`);
    }
}