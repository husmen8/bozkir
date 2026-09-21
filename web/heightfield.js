// A height field read from an image instead of computed from sine waves.
//
// The surface the tiles are laid on was two sine waves: smooth, periodic,
// and with no drainage anywhere. That is enough to check that warping and
// the tangent frame work, and it is useless for anything that depends on
// terrain being terrain. Water does not run anywhere on a sine wave; there
// are no channels for loose material to collect in and no ridges for it to
// be stripped from. Which is exactly what a rule that places material by
// landform needs to read.
//
// scripts/heightmap.py writes the pair this loads: a 16-bit PNG of heights
// normalised to 0..1, and a sidecar saying what those map to in metres. The
// source can be a drone survey's DSM, a height grid rasterised out of a
// capture, or an eroded procedural field - this side does not care, which
// is the point of going through a file.
//
// Sampling is bilinear and clamped at the edges. Nearest-neighbour would
// put a visible step at every texel, and the tangent frame is computed from
// finite differences of this function, so a stepped height field gives a
// normal that flips about rather than turning.

/** Load a height field written by scripts/heightmap.py.
 *
 *  `name` is the tileset's name, so `desert` looks for desert.height.png
 *  beside desert.splat. Returns null when there is none, which is not an
 *  error: most scenes have no height field and fall back to the analytic
 *  surface.
 */
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

  // The generator also records where sediment settled. The rule can infer
  // where loose material would collect from the shape of the surface, and
  // does when there is nothing better - but on generated terrain there is
  // something better, because the erosion moved the material and wrote
  // down where it stopped. Inference is what you do when you cannot
  // observe.
  if (meta && meta.sediment) {
    try {
      const sed = new Image();
      sed.src = `${base}/${meta.sediment}`;
      await sed.decode();
      const d = await decode(sed, split);
      if (d.w === w && d.h === h) out.sediment = d.z;
    } catch (e) {
      // A missing sediment map is not an error; the rule falls back to
      // reading drainage out of the shape, which is what it always did.
    }
  }
  return out;
}

/** Heights out of an image, in 0..1.
 *
 *  `split` says the file packs 16 bits into two 8-bit channels, high byte
 *  in red and low in green, which is how heightmap.py and terrain_gen.py
 *  write them now. Without it the image is read as plain 8-bit.
 *
 *  The two-channel packing exists because a 16-bit greyscale PNG cannot
 *  reach the page at 16 bits. A canvas decodes to eight bits per channel
 *  whatever the file held and puts the same high byte in all three, so the
 *  low byte is gone before any code sees it - in every browser, not just
 *  some. The field then arrives in 256 levels: 3.5 cm steps on a 9 m range,
 *  which terraces on a gentle slope and, worse, makes flat plateaus that
 *  flow routing reads as sinks. Writing both bytes into ordinary channels
 *  on purpose means no 16-bit image is involved anywhere.
 */
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

/** Fold a coordinate back into 0..n by reflection.
 *
 *  A triangle wave of period 2n: 0..n runs forward, n..2n runs back, and it
 *  repeats. Continuous at every join, which a modulo is not.
 */
function mirror(t, n) {
  if (n <= 0) return 0;
  const p = 2 * n;
  let m = t % p;
  if (m < 0) m += p;
  return m <= n ? m : p - m;
}


export class HeightField {
  constructor(z, w, h, meta, name) {
    this.z = z;
    this.w = w;
    this.h = h;
    this.meta = meta || {};
    this.name = name;
    // World size of the field. The renderer lays tiles out in world units
    // around the origin, so the field has to be told how much ground it
    // covers; without metres in the sidecar it is stretched over whatever
    // the caller asks for.
    this.extent = 1;
  }

  /** Cover `size` world units across the long edge. */
  fitTo(size) {
    this.extent = Math.max(1e-6, size);
    return this;
  }

  /** Height at a world position, in 0..1.
   *
   *  Beyond the field the coordinate is mirrored back, not clamped and not
   *  wrapped. The three differ in what they put at the join:
   *
   *    clamping  holds the edge value, so terrain outside the survey is a
   *              dead flat apron - fine when the field covers everything,
   *              useless when it covers a fraction of the grid.
   *    wrapping  joins the far edge to the near one, and the two rarely
   *              meet at the same height, so there is a cliff on every
   *              repeat boundary.
   *    mirroring joins each edge to itself, so the surface is continuous
   *              by construction. It repeats, but with no seam to see.
   *
   *  Mirroring is what makes the extent worth controlling: a field smaller
   *  than the grid tiles across it, and how much ground it covers becomes
   *  how large the landforms are.
   */
  sample(x, y) {
    return this._bilinear(this.z, x, y);
  }

  /** Where erosion left sediment, at a world position, in 0..1.
   *
   *  Null when the field came with no sediment map - a drone survey, or a
   *  height grid read out of a capture. Only generated terrain has one,
   *  because only there did anything actually move material.
   */
  sampleSediment(x, y) {
    return this.sediment ? this._bilinear(this.sediment, x, y) : null;
  }

  /** Bilinear read from any grid the size of this field, mirrored past its
   *  edge. Shared so the height and the sediment are sampled at exactly the
   *  same place - a sediment map offset by half a texel from the surface it
   *  describes would put material beside its channel rather than in it. */
  _bilinear(grid, x, y) {
    // Texel centres sit on the sampled positions, so the field spans the
    // extent exactly: texel 0 at one edge, texel w-1 at the other. Scaling
    // by w rather than w-1 would leave the last texel unreachable, which
    // reads as the terrain stopping short of its own edge.
    const long = Math.max(this.w, this.h) - 1;
    const scale = long / (this.extent || 1);
    // Centred, and v flipped: image rows run down, world y runs up.
    let u = (x * scale) + (this.w - 1) * 0.5;
    let v = (this.h - 1) * 0.5 - (y * scale);

    u = mirror(u, this.w - 1);
    v = mirror(v, this.h - 1);

    const x0 = u | 0, y0 = v | 0;
    // Clamped rather than wrapped, so the far edge blends against itself
    // instead of against the opposite side of the field.
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