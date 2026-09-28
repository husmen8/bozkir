// A tileset made in the browser, for when there is no capture to show
// (?scene=starter, or when nothing loads): a Wang set in the same format an
// export writes, built in a few milliseconds.
//
// Each tile is four triangles, one per edge, textured by that edge - the
// same construction as the real tiles (Cohen et al. 2003) with procedural
// patches in place of captured ones. Two materials, moss and sand, so the rule,
// blending and class view have something to act on.

const STRIDE = 32;

// Two materials, moss and sand. Each edge colour is its own small texture
// (a base colour and noise at a few scales), indexed in coordinates measured
// from that edge, so the same texture continues across a shared edge - the
// Cohen construction with a procedural patch in place of a captured one.
// Edge colours differ only slightly: enough for the tiles to vary, not
// enough to read as a chequerboard.
const MATERIALS = [
  { base: [0.39, 0.49, 0.28], alt: [0.47, 0.50, 0.30], dark: 0.72,  // moss
    shift: [[1.00, 1.00, 1.00], [0.95, 0.97, 0.93]], clump: 0.30, speck: 0.10 },
  { base: [0.78, 0.69, 0.53], alt: [0.72, 0.64, 0.49], dark: 0.62,  // sand
    shift: [[1.00, 1.00, 1.00], [1.03, 1.01, 0.97]], clump: 0.16, speck: 0.05 },
];

function hash(i) {
  let h = Math.imul(i ^ 0x9E3779B9, 0x85EBCA6B) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xC2B2AE35) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

const smooth = (a, b, v) => {
  const t = Math.min(1, Math.max(0, (v - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/** Value noise on a lattice, seeded. */
function noise(x, y, seed) {
  const ix = Math.floor(x), iy = Math.floor(y), fx = x - ix, fy = y - iy;
  const h = (i, j) => hash((Math.imul(i, 73856093) ^ Math.imul(j, 19349663)) + seed * 83492791);
  const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
  const a = h(ix, iy), b = h(ix + 1, iy), c = h(ix, iy + 1), d = h(ix + 1, iy + 1);
  return (a + (b - a) * sx) * (1 - sy) + (c + (d - c) * sx) * sy;
}

/** The texture of one edge colour at (u, v), measured from that edge, in
 *  tile widths. Clumps, a second tone and fine speckle. */
function edgeTexture(m, code, u, v, seed) {
  const k = seed * 7 + code * 3;
  const clump = noise(u * 5.5, v * 5.5, k + 1);
  const patchy = noise(u * 2.2 + 3.1, v * 2.2 + 1.7, k + 2);
  const fine = noise(u * 19, v * 19, k + 3);
  const tone = smooth(0.35, 0.85, patchy);
  const out = [];
  for (let c = 0; c < 3; c++) {
    let col = m.base[c] * (1 - tone) + m.alt[c] * tone;
    col *= 1 + m.clump * (clump - 0.5) + m.speck * (fine - 0.5);
    out.push(col * m.shift[code][c]);
  }
  // Dark hollows between clumps.
  const hollow = smooth(0.14, 0.0, clump);
  return out.map((v2) => v2 * (1 - (1 - m.dark) * hollow));
}

/** Build the starter tileset. Returns { buffer, manifest } in the same
 *  shape as a .splat file and its .json, ready for the viewer's load(). */
export function starterTileset({ size = 1.5, per = 34 } = {}) {
  const codes = [];
  for (let n = 0; n < 2; n++) for (let e = 0; e < 2; e++)
    for (let s = 0; s < 2; s++) for (let w = 0; w < 2; w++) codes.push([n, e, s, w]);
  const perTile = per * per;
  const total = 2 * codes.length * perTile;
  const buffer = new ArrayBuffer(total * STRIDE);
  const f32 = new Float32Array(buffer);
  const u8 = new Uint8Array(buffer);
  const tiles = [];
  const step = size / per;
  const radius = step * 0.78;
  let k = 0;

  for (let cls = 0; cls < 2; cls++) {
    const mat = MATERIALS[cls];
    for (const [n, e, s, w] of codes) {
      const start = k;
      for (let j = 0; j < per; j++) {
        for (let i = 0; i < per; i++, k++) {
          const r1 = hash(k * 3 + 1), r2 = hash(k * 3 + 2), r3 = hash(k * 3 + 3);
          const x = (i + 0.5 + (r1 - 0.5) * 0.6) * step - size / 2;
          const y = (j + 0.5 + (r2 - 0.5) * 0.6) * step - size / 2;
          const u = x / size, v = y / size;
          // North/south texture measured from the edge the point is nearer
          // (v - 0.5 above the middle, v + 0.5 below); east/west likewise.
          const ns = y >= 0 ? edgeTexture(mat, n, u, v - 0.5, 10 + cls)
                            : edgeTexture(mat, s, u, v + 0.5, 10 + cls);
          const ew = x >= 0 ? edgeTexture(mat, e, u - 0.5, v, 20 + cls)
                            : edgeTexture(mat, w, u + 0.5, v, 20 + cls);
          const toDiag = (Math.abs(y) - Math.abs(x)) / size;
          const t = smooth(-0.06, 0.06, toDiag);        // 1 = north/south side
          const grain = r3 < 0.05 ? 0.8 : 0.97 + 0.06 * r3;
          const o = k * 8;
          f32[o] = x;
          f32[o + 1] = y;
          f32[o + 2] = (hash(k * 7 + 5) - 0.5) * 0.006 * size;
          f32[o + 3] = radius;
          f32[o + 4] = radius;
          f32[o + 5] = 0.004 * size;
          const b = k * STRIDE + 24;
          for (let c = 0; c < 3; c++) {
            const val = (ew[c] * (1 - t) + ns[c] * t) * grain;
            u8[b + c] = Math.max(0, Math.min(255, Math.round(val * 255)));
          }
          u8[b + 3] = 235;
          u8[b + 4] = 255; u8[b + 5] = 128; u8[b + 6] = 128; u8[b + 7] = 128;
        }
      }
      tiles.push({ start, count: perTile, levels: [[start, perTile]],
                   class: cls, n, e, s, w });
    }
  }
  const manifest = { size, wang: true, classes: 2, lod: 1, total, tiles,
                     starter: true, names: ['moss', 'sand'] };
  return { buffer, manifest };
}