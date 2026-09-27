// A tileset made in the browser, for when there is no capture to show
// (?scene=starter, or when nothing loads): a Wang set in the same format an
// export writes, built in a few milliseconds.
//
// Each tile is four triangles, one per edge, coloured by that edge - the
// same construction as the real tiles (Cohen et al. 2003) with colours in
// place of captured patches. Two materials, moss and sand, so the rule,
// blending and class view have something to act on.

const STRIDE = 32;

// Muted, earthy: two hues per axis per material, close enough to read as
// one ground, far enough apart to show the edges.
const PALETTE = [
  { h: [[0.42, 0.52, 0.30], [0.33, 0.44, 0.25]],     // moss: north/south
    v: [[0.50, 0.57, 0.33], [0.38, 0.49, 0.29]] },   //       east/west
  { h: [[0.80, 0.71, 0.54], [0.72, 0.62, 0.46]],     // sand
    v: [[0.85, 0.77, 0.60], [0.69, 0.60, 0.44]] },
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
    const pal = PALETTE[cls];
    for (const [n, e, s, w] of codes) {
      const start = k;
      for (let j = 0; j < per; j++) {
        for (let i = 0; i < per; i++, k++) {
          const r1 = hash(k * 3 + 1), r2 = hash(k * 3 + 2), r3 = hash(k * 3 + 3);
          const x = (i + 0.5 + (r1 - 0.5) * 0.6) * step - size / 2;
          const y = (j + 0.5 + (r2 - 0.5) * 0.6) * step - size / 2;
          // Which edge this point belongs to, and how near a diagonal.
          const ns = y >= 0 ? pal.h[n] : pal.h[s];
          const ew = x >= 0 ? pal.v[e] : pal.v[w];
          const toDiag = (Math.abs(y) - Math.abs(x)) / size;
          const t = smooth(-0.05, 0.05, toDiag);        // 1 = north/south side
          // A weave parallel to the nearest edge: continuous across a
          // shared edge, because both sides measure distance from it.
          const edgeDist = size / 2 - Math.max(Math.abs(x), Math.abs(y));
          const weave = 1 + 0.045 * Math.sin(edgeDist / size * Math.PI * 14);
          const grain = r3 < 0.07 ? 0.72 : 0.95 + 0.1 * r3;
          const o = k * 8;
          f32[o] = x;
          f32[o + 1] = y;
          f32[o + 2] = (hash(k * 7 + 5) - 0.5) * 0.006 * size;
          f32[o + 3] = radius;
          f32[o + 4] = radius;
          f32[o + 5] = 0.004 * size;
          const b = k * STRIDE + 24;
          for (let c = 0; c < 3; c++) {
            const v = (ew[c] * (1 - t) + ns[c] * t) * weave * grain;
            u8[b + c] = Math.max(0, Math.min(255, Math.round(v * 255)));
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
                     starter: true };
  return { buffer, manifest };
}