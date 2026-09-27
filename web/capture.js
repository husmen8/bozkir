// Capturing a camera sweep, so popping can be measured instead of watched.
//
// scripts/pop_metric.py compares two frame sequences, which only works if
// both travel the same path, so the path is scripted: from the current
// camera, same elevation and distance, a fixed arc in equal steps.
//
// Before each frame is kept it waits for the sort worker to go idle (or it
// would measure sort latency, in both runs, hiding the difference) and for
// a settle frame (LOD fades and merge requests lag a frame). Output is one
// zip of PNGs.

import { writeZip } from './tileset.js';

/** Read the canvas into a 2D canvas. Must run in the frame that drew it (no
 *  preserveDrawingBuffer), which is why capture is stepped from the render
 *  loop. PNG encoding is slower and left to the caller. */
function readFrame(gl, canvas, maxEdge) {
  const w = canvas.width, h = canvas.height;
  const px = new Uint8Array(w * h * 4);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);

  // GL rows come back bottom-up; flip so the frames are usable as figures.
  const flipped = new Uint8ClampedArray(w * h * 4);
  const row = w * 4;
  for (let y = 0; y < h; y++) {
    flipped.set(px.subarray((h - 1 - y) * row, (h - y) * row), y * row);
  }

  const full = document.createElement('canvas');
  full.width = w; full.height = h;
  full.getContext('2d').putImageData(new ImageData(flipped, w, h), 0, 0);

  // Long edge capped (default 1024): encoding and pop_metric's block
  // matching both grow with pixel count. 0 keeps full resolution.
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

/** PNG bytes from a canvas. Not awaited by the loop, so each encode
 *  overlaps the next pose's wait. */
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

  /** Begin a sweep; false if one is running. `arc` defaults to a full
   *  turn, since popping depends on the angle to the grid. */
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
      // Frozen for the whole sweep so both runs orbit the same point. The
      // viewer passes the scene centre (WASD moves the live target).
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

  /** Called once per rendered frame, after the draw; calls `onDone` with a
   *  zip at the end. */
  async step() {
    if (!this.active || this.pending) return;

    this.waited++;
    if (this.isBusy && this.isBusy()) {
      // Waiting a frame or two is normal; for ever is a bug upstream. Carry
      // on after `patience` frames and warn, rather than hang.
      if (this.waited < this.plan.patience) return;
      if (!this.warned) {
        this.warned = true;
        console.warn('capture: still busy after ' + this.plan.patience
          + ' frames; capturing anyway. Frames may show the previous pose\'s'
          + ' ordering, so treat the numbers with suspicion.');
      }
    }
    if (this.held < this.plan.settle) { this.held++; return; }

    // Read now, encode later, pose the next frame meanwhile.
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
    // Revoking at once can cancel the download on some builds.
    setTimeout(() => URL.revokeObjectURL(url), 10000);

    this.cam.fly = p.fly;
    this.cam.azimuth = p.az0;
    const files = this.files;
    this.files = [];
    this.jobs = [];
    if (this.onDone) this.onDone(files.length, `${p.name}.zip`);
  }
}
