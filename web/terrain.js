// Generating terrain in the browser: the same generator as
// bozkir/erosion.py, arithmetic for arithmetic.
//
// Stream power erosion solved the FastScape way (Braun and Willett,
// Geomorphology 2013): every cell drains to its steepest neighbour, the
// incision term is implicit, heights are updated receivers-first. Noise is
// gradient noise from an integer hash, depressions are filled once with
// Priority-Flood (Barnes, Lehman and Mulla 2014) at the start and the end.
// docs/terrain.md has the
// sources; the Python module's docstring has the reasoning.
//
// This file and the Python one are held to the same output by
// tests/test_pipeline.py through tests/bridge.mjs. That only works
// because both do their floating-point work in the same order: the comments
// marking an order are not style, they are the contract. Change one side,
// change the other, and run the tests.
//
// Pure functions, no DOM: runs in a worker (terrain-worker.js), in Node for
// the tests, or on the main thread if it has to.

// ------------------------------------------------------------- profiles

export const PROFILES = {
  plains:    { dome: 0.5, octaves: 5, freq: 2, ridge: 0.0, relief: 0.35,
               iterations: 40, incision: 0.15, diffusion: 0.20 },
  rolling:   { dome: 0.5, octaves: 6, freq: 3, ridge: 0.15, relief: 0.6,
               iterations: 50, incision: 0.25, diffusion: 0.18 },
  hills:     { dome: 0.5, octaves: 6, freq: 4, ridge: 0.3, relief: 0.9,
               iterations: 60, incision: 0.35, diffusion: 0.14 },
  desert:    { dome: 0.25, octaves: 6, freq: 4, ridge: 0.6, relief: 1.0,
               iterations: 60, incision: 0.4, diffusion: 0.10 },
  badlands:  { dome: 0.45, octaves: 7, freq: 6, ridge: 0.5, relief: 1.0,
               iterations: 80, incision: 0.8, diffusion: 0.03 },
  canyon:    { dome: 0.25, octaves: 7, freq: 2, ridge: 0.85, relief: 1.2,
               iterations: 90, incision: 0.9, diffusion: 0.05 },
  mesa:      { dome: 0.4, octaves: 6, freq: 3, ridge: 0.45, relief: 1.0,
               iterations: 55, incision: 0.5, diffusion: 0.06, terraces: 5 },
  plateau:   { dome: 0.35, octaves: 5, freq: 2, ridge: 0.3, relief: 1.0,
               iterations: 50, incision: 0.45, diffusion: 0.08, terraces: 2 },
  foothills: { octaves: 6, freq: 3, ridge: 0.4, relief: 0.3,
               iterations: 120, incision: 0.35, diffusion: 0.10, uplift: 0.012 },
  ridges:    { octaves: 6, freq: 3, ridge: 0.9, relief: 0.3,
               iterations: 150, incision: 0.45, diffusion: 0.06, uplift: 0.02 },
  alpine:    { octaves: 7, freq: 2, ridge: 0.75, relief: 0.4,
               iterations: 180, incision: 0.55, diffusion: 0.04, uplift: 0.03 },
  piedmont:  { octaves: 6, freq: 3, ridge: 0.4, relief: 1.0,
               iterations: 70, incision: 0.45, diffusion: 0.08, tilt: 0.55 },
};

/** The profiles by kind of country, three each, in the order the terrain
 *  window shows them. A list of twelve names asks the reader to know what
 *  "piedmont" means; four kinds of place with three variations each only
 *  asks which kind of place they want. */
export const GROUPS = [
  { name: 'lowlands', profiles: ['plains', 'rolling', 'hills'] },
  { name: 'desert', profiles: ['desert', 'badlands', 'mesa'] },
  { name: 'highlands', profiles: ['plateau', 'canyon', 'piedmont'] },
  { name: 'mountains', profiles: ['foothills', 'ridges', 'alpine'] },
];

/** Fraction of each side simulated but not returned; see generate(). */
export const CROP_MARGIN = 0.15;

const DEFAULTS = {
  octaves: 6, freq: 4, ridge: 0.5, relief: 1.0, iterations: 40,
  incision: 0.3, diffusion: 0.1, uplift: 0.0, terraces: 0, dome: 0.0,
  tilt: 0.0,
};

// ---------------------------------------------------------------- noise

/** 32-bit integer hash of a lattice point. Same bits as _hash_u32. */
function hashU32(x, y, s) {
  let h = Math.imul(x, 0x27D4EB2D) ^ Math.imul(y, 0x165667B1)
        ^ Math.imul(s, 0x9E3779B1);
  h = Math.imul(h ^ (h >>> 15), 0x85EBCA6B);
  h = Math.imul(h ^ (h >>> 13), 0xC2B2AE35);
  return (h ^ (h >>> 16)) >>> 0;
}

// Eight fixed gradient directions: no trigonometry, so no rounding that
// could differ between languages.
const GX = [1, -1, 1, -1, 1, -1, 0, 0];
const GY = [1, 1, -1, -1, 0, 0, 1, -1];

/** One octave of gradient noise into `out` (added with weight `amp`, or
 *  folded into a ridge when `ridge` is set). */
function octave(out, h, w, f, s, amp, ridge) {
  for (let j = 0; j < h; j++) {
    const y = (j * f) / h;
    const y0 = Math.floor(y);
    const fy = y - y0;
    const uy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0);
    for (let i = 0; i < w; i++) {
      const x = (i * f) / w;
      const x0 = Math.floor(x);
      const fx = x - x0;
      const ux = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0);
      const g00 = hashU32(x0, y0, s) & 7;
      const g10 = hashU32(x0 + 1, y0, s) & 7;
      const g01 = hashU32(x0, y0 + 1, s) & 7;
      const g11 = hashU32(x0 + 1, y0 + 1, s) & 7;
      const a = GX[g00] * (fx - 0) + GY[g00] * (fy - 0);
      const b = GX[g10] * (fx - 1) + GY[g10] * (fy - 0);
      const c = GX[g01] * (fx - 0) + GY[g01] * (fy - 1);
      const d = GX[g11] * (fx - 1) + GY[g11] * (fy - 1);
      const top = a + (b - a) * ux;
      const bot = c + (d - c) * ux;
      const v = (top + (bot - top) * uy) * 0.5 + 0.5;
      const k = j * w + i;
      if (ridge) {
        const n = 1.0 - Math.abs(2.0 * v - 1.0);
        out[k] = out[k] + amp * (n * n);
      } else {
        out[k] = out[k] + amp * v;
      }
    }
  }
}

function fractal(n, octaves, freq, seed, ridge, gain = 0.5, lac = 2.0) {
  const out = new Float64Array(n * n);
  let amp = 1.0, f = +freq, norm = 0.0;
  for (let k = 0; k < Math.max(1, octaves | 0); k++) {
    octave(out, n, n, f, (seed | 0) * 1013 + (ridge ? 7919 : 0) + k, amp, ridge);
    norm = norm + amp;
    amp = amp * gain;
    f = f * lac;
  }
  const d = Math.max(norm, 1e-9);
  for (let i = 0; i < out.length; i++) out[i] = out[i] / d;
  return out;
}

/** Fractal (fBm) gradient noise, 0..1 in practice. */
export const fbm = (n, octaves = 6, freq = 4, seed = 0) =>
  fractal(n, octaves, freq, seed, false);

/** Ridged fractal noise: creases where fbm has smooth maxima. */
export const ridged = (n, octaves = 6, freq = 4, seed = 0) =>
  fractal(n, octaves, freq, seed, true);

function unitInPlace(a) {
  let lo = Infinity, hi = -Infinity;
  for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const d = hi - lo;
  for (let i = 0; i < a.length; i++) a[i] = d > 1e-12 ? (a[i] - lo) / d : 0;
  return a;
}

function taperRamp(n, frac) {
  const r = new Float64Array(n);
  const span = Math.max(1.0, frac * (n - 1));
  for (let k = 0; k < n; k++) {
    let d = Math.min(k, (n - 1) - k) / span;
    d = Math.min(1.0, d);
    r[k] = d * d * (3.0 - 2.0 * d);
  }
  return r;
}

/** Pull heights towards `steps` levels: flats with risers between. */
export function terrace(z, steps, sharpness = 0.5) {
  const s = Math.max(1, steps | 0);
  const out = new Float64Array(z.length);
  for (let i = 0; i < z.length; i++) {
    const k = z[i] * s;
    const base = Math.floor(k);
    const frac = k - base;
    const snap = frac >= 0.5 ? 1.0 : 0.0;
    const shaped = base + frac * frac * (3.0 - 2.0 * frac) * (1.0 - sharpness)
                 + sharpness * snap;
    out[i] = Math.min(1.0, Math.max(0.0, shaped / s));
  }
  return out;
}

/** The noise a profile starts from, 0..1, before erosion. */
export function baseSurface(n, seed, p) {
  let z = fbm(n, p.octaves, p.freq, seed);
  if (p.ridge > 0) {
    const r = ridged(n, p.octaves, p.freq, seed);
    for (let i = 0; i < z.length; i++) z[i] = z[i] * (1.0 - p.ridge) + r[i] * p.ridge;
  }
  unitInPlace(z);
  if (p.dome > 0) {
    // The rise sits in the band generate() crops away; see erosion.py.
    const t = taperRamp(n, CROP_MARGIN / (1.0 + 2.0 * CROP_MARGIN));
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const k = j * n + i;
        z[k] = z[k] * (1.0 - p.dome) + (t[j] * t[i]) * p.dome;
      }
    }
    unitInPlace(z);
  }
  if (p.tilt > 0) {
    for (let j = 0; j < n; j++) {
      const ramp = 1.0 - j / (n - 1);
      for (let i = 0; i < n; i++) {
        const k = j * n + i;
        z[k] = z[k] * (1.0 - p.tilt) + ramp * p.tilt;
      }
    }
    unitInPlace(z);
  }
  if (p.terraces) z = terrace(z, p.terraces);
  return z;
}

// ------------------------------------------------------ depressions

const NB = [];
for (let b = -1; b <= 1; b++) {
  for (let a = -1; a <= 1; a++) if (a || b) NB.push([b, a]);
}

/** A binary heap on (value, index), smallest first - the tuple order
 *  Python's heapq uses, so both flood in the same sequence. */
class Heap {
  constructor() { this.v = []; this.k = []; }
  get size() { return this.v.length; }
  less(i, j) {
    return this.v[i] < this.v[j] || (this.v[i] === this.v[j] && this.k[i] < this.k[j]);
  }
  swap(i, j) {
    [this.v[i], this.v[j]] = [this.v[j], this.v[i]];
    [this.k[i], this.k[j]] = [this.k[j], this.k[i]];
  }
  push(v, k) {
    this.v.push(v); this.k.push(k);
    let i = this.v.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!this.less(i, p)) break;
      this.swap(i, p); i = p;
    }
  }
  pop() {
    const v = this.v[0], k = this.k[0];
    const lv = this.v.pop(), lk = this.k.pop();
    if (this.v.length) {
      this.v[0] = lv; this.k[0] = lk;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < this.v.length && this.less(l, m)) m = l;
        if (r < this.v.length && this.less(r, m)) m = r;
        if (m === i) break;
        this.swap(i, m); i = m;
      }
    }
    return [v, k];
  }
}

/** Raise every pit until it drains to the border (Priority-Flood). */
export function fillDepressions(h, n) {
  const g = Float64Array.from(h);
  let lo = Infinity, hi = -Infinity;
  for (const v of g) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const eps = 1e-4 * Math.max(hi - lo, 1e-9) * 256.0 / n;
  const done = new Uint8Array(n * n);
  const heap = new Heap();
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      if (r === 0 || c === 0 || r === n - 1 || c === n - 1) {
        const k = r * n + c;
        heap.push(g[k], k);
        done[k] = 1;
      }
    }
  }
  while (heap.size) {
    const [v, k] = heap.pop();
    const r = Math.floor(k / n), c = k - r * n;
    for (const [b, a] of NB) {
      const rr = r + b, cc = c + a;
      if (rr < 0 || cc < 0 || rr >= n || cc >= n) continue;
      const q = rr * n + cc;
      if (done[q]) continue;
      done[q] = 1;
      if (g[q] <= v) g[q] = v + eps;
      heap.push(g[q], q);
    }
  }
  return g;
}

// -------------------------------------------------------------- erosion

/** Run the stream power law. Mirrors erode() in bozkir/erosion.py.
 *  `progress(done, total)` is called once per step. */
export function erode(z, n, opts = {}, progress = null) {
  const iterations = Math.max(1, (opts.iterations ?? 40) | 0);
  const K = opts.incision ?? 0.3;
  const D = opts.diffusion ?? 0.1;
  const U = opts.uplift ?? 0.0;
  const dt = 1.0;
  const N = n * n;

  const flat = fillDepressions(z, n);
  let umap;
  if (opts.upliftMap) umap = Float64Array.from(opts.upliftMap);
  else umap = unitInPlace(Float64Array.from(flat));
  {
    const t = taperRamp(n, 0.12);
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) umap[j * n + i] = umap[j * n + i] * (t[j] * t[i]);
    }
  }
  const interior = new Uint8Array(N);
  for (let j = 1; j < n - 1; j++) for (let i = 1; i < n - 1; i++) interior[j * n + i] = 1;

  const rec = new Int32Array(N);
  const dist = new Float64Array(N);
  const depth = new Int32Array(N), d2 = new Int32Array(N);
  const ptr = new Int32Array(N), p2 = new Int32Array(N);
  const order = new Int32Array(N);
  const area = new Float64Array(N);
  const F = new Float64Array(N);
  const before = new Float64Array(N);
  const eroded = new Float64Array(N);
  const slope = new Float64Array(N);
  const settle = new Float64Array(N);
  const qs = new Float64Array(N);
  const lapBuf = new Float64Array(N);
  const sediment = new Float64Array(N);

  for (let it = 0; it < iterations; it++) {
    // Receivers: steepest strictly-downhill neighbour, fixed order.
    for (let k = 0; k < N; k++) { rec[k] = k; dist[k] = 1.0; }
    for (let j = 1; j < n - 1; j++) {
      for (let i = 1; i < n - 1; i++) {
        const k = j * n + i;
        const c = flat[k];
        let best = 0.0;
        for (const [b, a] of NB) {
          const d = (a && b) ? Math.SQRT2 : 1.0;
          const q = (j + b) * n + (i + a);
          const s = (c - flat[q]) / d;
          if (s > best) { best = s; rec[k] = q; dist[k] = d; }
        }
      }
    }

    // Distance to outlet by pointer jumping, then a counting sort on it:
    // the same (depth, index) order np.argsort(kind='stable') gives.
    for (let k = 0; k < N; k++) { depth[k] = rec[k] !== k ? 1 : 0; ptr[k] = rec[k]; }
    for (;;) {
      let moved = false;
      for (let k = 0; k < N; k++) {
        d2[k] = depth[k] + depth[ptr[k]];
        p2[k] = ptr[ptr[k]];
        if (p2[k] !== ptr[k]) moved = true;
      }
      depth.set(d2); ptr.set(p2);
      if (!moved) break;
    }
    let maxD = 0;
    for (let k = 0; k < N; k++) if (depth[k] > maxD) maxD = depth[k];
    const starts = new Int32Array(maxD + 2);
    for (let k = 0; k < N; k++) starts[depth[k] + 1]++;
    for (let L = 0; L <= maxD; L++) starts[L + 1] += starts[L];
    {
      const fill = starts.slice(0, maxD + 1);
      for (let k = 0; k < N; k++) order[fill[depth[k]]++] = k;
    }
    const levels = maxD + 1;

    // Drainage area in cells, deepest first.
    area.fill(1.0);
    for (let L = levels - 1; L >= 1; L--) {
      for (let t = starts[L]; t < starts[L + 1]; t++) {
        const k = order[t];
        area[rec[k]] = area[rec[k]] + area[k];
      }
    }

    if (U) {
      for (let k = 0; k < N; k++) if (interior[k]) flat[k] = flat[k] + (U * dt) * umap[k];
    }
    before.set(flat);

    // Implicit incision, receivers first.
    for (let k = 0; k < N; k++) {
      F[k] = ((K * dt) * Math.sqrt(area[k] / N)) / dist[k];
    }
    for (let t = starts[1]; t < N; t++) {
      const k = order[t];
      const f = F[k];
      flat[k] = (flat[k] + f * flat[rec[k]]) / (1.0 + f);
    }

    for (let k = 0; k < N; k++) {
      eroded[k] = Math.max(before[k] - flat[k], 0.0);
      slope[k] = Math.max(flat[k] - flat[rec[k]], 0.0) / dist[k];
    }

    // Hillslope creep, interior only, terms in a fixed order.
    if (D) {
      lapBuf.fill(0.0);
      for (let j = 1; j < n - 1; j++) {
        for (let i = 1; i < n - 1; i++) {
          const k = j * n + i;
          lapBuf[k] = (((flat[k - n] + flat[k + n]) + flat[k - 1]) + flat[k + 1])
                    - 4.0 * flat[k];
        }
      }
      for (let k = 0; k < N; k++) flat[k] = flat[k] + (D * dt) * lapBuf[k];
    }

    // Sediment proxy: material passed down, settling where it flattens.
    qs.set(eroded);
    for (let k = 0; k < N; k++) settle[k] = 0.5 * (0.02 / (0.02 + slope[k]));
    for (let L = levels - 1; L >= 1; L--) {
      for (let t = starts[L]; t < starts[L + 1]; t++) {
        const k = order[t];
        const dep = qs[k] * settle[k];
        sediment[k] = sediment[k] + dep;
        qs[rec[k]] = qs[rec[k]] + (qs[k] - dep);
      }
    }
    for (let t = starts[0]; t < starts[1]; t++) {
      const k = order[t];
      sediment[k] = sediment[k] + qs[k];
    }

    if (progress) progress(it + 1, iterations);
  }
  // Once more at the end: creep can leave shallow closed hollows in
  // channel floors. See erode() in bozkir/erosion.py.
  return { z: fillDepressions(flat, n), sediment };
}

/** Settings for a profile, with any explicit overrides on top. */
export function settingsFor(profile, overrides = {}) {
  if (profile && !PROFILES[profile]) {
    throw new Error(`unknown profile ${profile}; have ${Object.keys(PROFILES).join(', ')}`);
  }
  return { ...DEFAULTS, ...(profile ? PROFILES[profile] : {}), ...overrides };
}

/** A height field and its sediment map, both 0..1, as Float32Arrays.
 *  Mirrors generate() in bozkir/erosion.py. */
export function generate({ size = 256, seed = 0, profile = null, ...overrides } = {},
                         progress = null) {
  const n = size | 0;
  const p = settingsFor(profile, overrides);
  // Simulated larger and cropped to the middle, so the simulation's own
  // border - base level, where everything drains - is not in view. See
  // generate() in bozkir/erosion.py.
  const big = n + 2 * Math.round(n * CROP_MARGIN);
  const k = (big - n) >> 1;
  const z0 = baseSurface(big, seed, p);
  const start = new Float64Array(z0.length);
  for (let i = 0; i < z0.length; i++) start[i] = z0[i] * p.relief;
  const out = erode(start, big, {
    iterations: p.iterations, incision: p.incision, diffusion: p.diffusion,
    uplift: p.uplift * p.relief, upliftMap: z0,
  }, progress);
  const crop = (a) => {
    const c = new Float64Array(n * n);
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) c[j * n + i] = a[(j + k) * big + (i + k)];
    }
    return unitInPlace(c);
  };
  return { z: crop(out.z), sediment: crop(out.sediment), size: n, settings: p };
}