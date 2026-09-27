// Height fields: loaded from a PNG, or generated in the browser.
//
// The built-in surface is two sine waves - enough to test warping, but
// water runs nowhere on it, so a rule that places material by landform has
// nothing to read. scripts/heightmap.py and terrain_gen.py write the pair
// loaded here: a 16-bit PNG of heights in 0..1 and a JSON sidecar. The
// source can be a drone DSM, a capture's height grid or an eroded field.
//
// Sampling is bilinear: nearest-neighbour steps would make the tangent
// frame (finite differences of this) flip about.

/** Load <name>.height.png/.json from `base`. Returns null when there is
 *  none, which is normal: most scenes fall back to the analytic surface. */
export async function loadHeightField(base, name) {
  let meta = null;
  try {
    const r = await fetch(`${base}/${name}.height.json`);
    if (!r.ok) return null;
    meta = await r.json();
  } catch (e) {
    return null;
  }

  const img = new Image();
  img.src = `${base}/${name}.height.png`;
  try {
    await img.decode();
  } catch (e) {
    console.warn(`${name}.height.json exists but the png did not load`);
    return null;
  }

  const split = !!(meta && meta.encoding === 'rg16');
  const { z, w, h, sixteen } = await decode(img, split);
  const out = new HeightField(z, w, h, meta, name);
  out.sixteenBit = sixteen;

  // Generated terrain also records where erosion left sediment; the rule
  // reads that instead of inferring it from drainage.
  if (meta && meta.sediment) {
    try {
      const sed = new Image();
      sed.src = `${base}/${meta.sediment}`;
      await sed.decode();
      const d = await decode(sed, split);
      if (d.w === w && d.h === h) out.sediment = d.z;
    } catch (e) {
      // No sediment map: the rule falls back to drainage.
    }
  }
  return out;
}

/** Heights out of an image, in 0..1.
 *
 *  `split`: 16 bits packed as high byte in red, low byte in green (how the
 *  Python writes them). A canvas decodes a 16-bit grey PNG to 8 bits in every
 *  browser, and 256 levels terraced gentle slopes and made flats that flow
 *  routing read as sinks, so 16-bit greyscale is avoided entirely. */
async function decode(img, split) {
  const cv = document.createElement('canvas');
  cv.width = img.naturalWidth;
  cv.height = img.naturalHeight;
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const px = ctx.getImageData(0, 0, cv.width, cv.height).data;

  const w = cv.width, h = cv.height;
  const z = new Float32Array(w * h);
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < w * h; i++) {
    const v = split ? (px[4 * i] * 256 + px[4 * i + 1]) / 65535
                    : px[4 * i] / 255;
    z[i] = v;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (hi > lo) {
    const k = 1 / (hi - lo);
    for (let i = 0; i < z.length; i++) z[i] = (z[i] - lo) * k;
  }
  return { z, w, h, sixteen: !!split };
}

/** Fold a coordinate into 0..n by reflection (a triangle wave of period
 *  2n), continuous at every join, unlike a modulo. */
function mirror(t, n) {
  if (n <= 0) return 0;
  const p = 2 * n;
  let m = t % p;
  if (m < 0) m += p;
  return m <= n ? m : p - m;
}


/** Sky openness, 0..1: how far a point stands above its neighbourhood (the
 *  same TPI the rule reads), over a window that scales with the field.
 *  Used for ambient darkening, which only removes light and so never
 *  contradicts the light baked into the splats. */
export function openness(z, w, h, radius) {
  const r = Math.max(1, radius || Math.max(2, Math.round(Math.max(w, h) / 48)));
  const mean = boxBlur(z, w, h, r);
  const out = new Float32Array(w * h);
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < out.length; i++) {
    out[i] = z[i] - mean[i];
    if (out[i] < lo) lo = out[i];
    if (out[i] > hi) hi = out[i];
  }
  const d = hi - lo;
  for (let i = 0; i < out.length; i++) {
    out[i] = d > 1e-12 ? (out[i] - lo) / d : 0.5;
  }
  return out;
}

/** Where fine detail belongs, 0..1: steep slopes and convex crests; flats
 *  and hollows stay smooth. A slope ramp as in Frostbite's terrain shader,
 *  plus a crest term from openness. Slope is scaled by its 95th percentile
 *  so the mask means the same on gentle and steep terrain. */
export function roughnessMask(z, w, h, open) {
  const g = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const xl = z[y * w + Math.max(0, x - 1)], xr = z[y * w + Math.min(w - 1, x + 1)];
      const yu = z[Math.max(0, y - 1) * w + x], yd = z[Math.min(h - 1, y + 1) * w + x];
      g[y * w + x] = Math.hypot(xr - xl, yd - yu);
    }
  }
  const sorted = Float32Array.from(g).sort();
  const p95 = Math.max(sorted[Math.floor(0.95 * (sorted.length - 1))], 1e-9);
  const ramp = (a, b, v) => {
    const t = Math.min(1, Math.max(0, (v - a) / (b - a)));
    return t * t * (3 - 2 * t);
  };
  const out = new Float32Array(w * h);
  for (let i = 0; i < out.length; i++) {
    const steep = ramp(0.3, 0.75, g[i] / p95);
    const crest = open ? ramp(0.6, 0.9, open[i]) : 0;
    out[i] = Math.max(steep, crest);
  }
  return out;
}

/** Mean over a (2r+1) square window, clipped at the edges, separable. */
function boxBlur(z, w, h, r) {
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0, n = 0;
      for (let d = -r; d <= r; d++) {
        const xx = x + d;
        if (xx >= 0 && xx < w) { sum += z[y * w + xx]; n++; }
      }
      tmp[y * w + x] = sum / n;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0, n = 0;
      for (let d = -r; d <= r; d++) {
        const yy = y + d;
        if (yy >= 0 && yy < h) { sum += tmp[yy * w + x]; n++; }
      }
      out[y * w + x] = sum / n;
    }
  }
  return out;
}

export class HeightField {
  constructor(z, w, h, meta, name) {
    this.z = z;
    this.w = w;
    this.h = h;
    this.meta = meta || {};
    this.name = name;
    // World units across the long edge, set by fitTo().
    this.extent = 1;
  }

  /** Cover `size` world units across the long edge. */
  fitTo(size) {
    this.extent = Math.max(1e-6, size);
    return this;
  }

  /** Height at a world position, in 0..1. Past the edge it mirrors: a
   *  clamp gives a flat apron, a wrap a cliff at every repeat; a mirror
   *  repeats with no seam. */
  sample(x, y) {
    return this._bilinear(this.z, x, y);
  }

  /** Sediment at a world position, 0..1; null unless the field was generated. */
  sampleSediment(x, y) {
    return this.sediment ? this._bilinear(this.sediment, x, y) : null;
  }

  /** The detail mask (roughnessMask) at a world point, or 1 without one. */
  sampleMask(x, y) {
    return this.mask ? this._bilinear(this.mask, x, y) : 1;
  }

  /** Bilinear read from any grid the size of this field, mirrored past its
   *  edge. Shared so height and sediment line up exactly. */
  _bilinear(grid, x, y) {
    // Texel 0 at one edge, texel w-1 at the other (scaling by w would leave
    // the last texel unreachable).
    const long = Math.max(this.w, this.h) - 1;
    const scale = long / (this.extent || 1);
    // Centred, and v flipped: image rows run down, world y runs up.
    let u = (x * scale) + (this.w - 1) * 0.5;
    let v = (this.h - 1) * 0.5 - (y * scale);

    u = mirror(u, this.w - 1);
    v = mirror(v, this.h - 1);

    const x0 = u | 0, y0 = v | 0;
    const x1 = Math.min(x0 + 1, this.w - 1);
    const y1 = Math.min(y0 + 1, this.h - 1);
    const fx = u - x0, fy = v - y0;
    const a = grid[y0 * this.w + x0];
    const b = grid[y0 * this.w + x1];
    const c = grid[y1 * this.w + x0];
    const d = grid[y1 * this.w + x1];
    return (a * (1 - fx) + b * fx) * (1 - fy)
         + (c * (1 - fx) + d * fx) * fy;
  }

  /** What one unit of `sample` is worth in metres, when that is known. */
  get metresPerUnit() {
    const r = this.meta.metres_range;
    return typeof r === 'number' && r > 0 ? r : null;
  }

  describe() {
    const m = this.metresPerUnit;
    return `${this.w}x${this.h}`
      + (m ? `, ${m.toFixed(1)} m of relief` : '')
      + (this.meta.source ? `, from ${this.meta.source}` : '')
      + (this.sediment ? ', with sediment' : '')
      + (this.sixteenBit === false ? ', 8-bit' : '');
  }
}

// ------------------------------------------------------- generated fields

/** Bump when terrain.js would make different ground from the same profile
 *  and seed, so old cached terrains are not reused. */
export const GENERATOR_VERSION = 2;   // 2: cropped from a larger field

const DB_NAME = 'bozkir', STORE = 'terrains';

function openCache() {
  return new Promise((resolve) => {
    let req;
    try { req = indexedDB.open(DB_NAME, 1); } catch (e) { resolve(null); return; }
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => resolve(null);
  });
}

async function cacheGet(key) {
  const db = await openCache();
  if (!db) return null;
  return new Promise((resolve) => {
    try {
      const r = db.transaction(STORE, 'readonly').objectStore(STORE).get(key);
      r.onsuccess = () => resolve(r.result || null);
      r.onerror = () => resolve(null);
    } catch (e) { resolve(null); }
  });
}

async function cachePut(key, value) {
  const db = await openCache();
  if (!db) return;
  try { db.transaction(STORE, 'readwrite').objectStore(STORE).put(value, key); }
  catch (e) { /* full or private: the terrain just is not remembered */ }
}

let worker = null, nextId = 1;
const pending = new Map();

function terrainWorker() {
  if (worker) return worker;
  worker = new Worker(new URL('./terrain-worker.js', import.meta.url),
                      { type: 'module' });
  worker.onmessage = (e) => {
    const job = pending.get(e.data.id);
    if (!job) return;
    if (e.data.progress != null) { if (job.onProgress) job.onProgress(e.data.progress); return; }
    pending.delete(e.data.id);
    if (e.data.error) job.reject(new Error(e.data.error));
    else job.resolve(e.data);
  };
  return worker;
}

/** A generated terrain as a HeightField: from the browser's cache if made
 *  before, else generated in a worker and cached. No server involved.
 *  `onProgress(fraction, fromCache)` reports. */
export async function generatedField({ profile, seed = 0, size = 256 },
                                     onProgress = null) {
  const key = `v${GENERATOR_VERSION}:${profile}:${seed}:${size}`;
  let r = await cacheGet(key);
  const cached = !!r;
  if (!r) {
    r = await new Promise((resolve, reject) => {
      const id = nextId++;
      pending.set(id, { resolve, reject, onProgress });
      terrainWorker().postMessage({ id, spec: { profile, seed, size } });
    });
    await cachePut(key, { z: r.z, sediment: r.sediment, size: r.size,
                          settings: r.settings });
  }
  if (onProgress) onProgress(1, cached);
  const n = r.size;
  const meta = { source: `generated ${profile} seed ${seed}`,
                 settings: { profile, seed, size: n, ...r.settings },
                 generated: true };
  const f = new HeightField(new Float32Array(r.z), n, n, meta,
                            `${profile}-${seed}`);
  f.sediment = new Float32Array(r.sediment);
  f.sixteenBit = true;
  f.cached = cached;
  return f;
}

/** Terrains made before, newest first: [{ profile, seed, size }]. The
 *  64-pixel previews are left out. */
export async function listCached() {
  const db = await openCache();
  if (!db) return [];
  const keys = await new Promise((resolve) => {
    try {
      const r = db.transaction(STORE, 'readonly').objectStore(STORE).getAllKeys();
      r.onsuccess = () => resolve(r.result || []);
      r.onerror = () => resolve([]);
    } catch (e) { resolve([]); }
  });
  const out = [];
  for (const k of keys) {
    const m = /^v(\d+):([^:]+):(\d+):(\d+)$/.exec(String(k));
    if (!m || +m[1] !== GENERATOR_VERSION || +m[4] <= 64) continue;
    out.push({ profile: m[2], seed: +m[3], size: +m[4] });
  }
  return out.reverse();
}
