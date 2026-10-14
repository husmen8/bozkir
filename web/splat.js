// The .splat format, as bozkir/pack.py writes it: 32 bytes a splat,
// position and scale as float32, rgba and a rotation quaternion as bytes.

export const STRIDE = 32;        // bytes per splat in the .splat format

/** Unpack a .splat buffer. */
export function unpack(buffer) {
  const bytes = new Uint8Array(buffer);
  const n = Math.floor(bytes.length / STRIDE);
  const f32 = new Float32Array(buffer);
  const positions = new Float32Array(n * 3);
  const data = new Float32Array(n * 3 * 4);
  const colour = new Uint8Array(n * 4);

  for (let i = 0; i < n; i++) {
    const f = i * 8, b = i * STRIDE, d = i * 12;
    const px = f32[f], py = f32[f + 1], pz = f32[f + 2];
    positions[3 * i] = px; positions[3 * i + 1] = py; positions[3 * i + 2] = pz;
    data[d] = px; data[d + 1] = py; data[d + 2] = pz;
    data[d + 3] = bytes[b + 27] / 255;
    data[d + 4] = f32[f + 3]; data[d + 5] = f32[f + 4]; data[d + 6] = f32[f + 5];
    data[d + 7] = 0;
    let qw = (bytes[b + 28] - 128) / 128, qx = (bytes[b + 29] - 128) / 128;
    let qy = (bytes[b + 30] - 128) / 128, qz = (bytes[b + 31] - 128) / 128;
    const len = Math.hypot(qw, qx, qy, qz) || 1;
    data[d + 8] = qw / len; data[d + 9] = qx / len;
    data[d + 10] = qy / len; data[d + 11] = qz / len;
    colour[4 * i] = bytes[b + 24]; colour[4 * i + 1] = bytes[b + 25];
    colour[4 * i + 2] = bytes[b + 26]; colour[4 * i + 3] = bytes[b + 27];
  }
  return { n, positions, data, colour };
}

/** Mean colour of each class's splats, weighted by opacity. */
export function meanClassColours(colour, tiles, count) {
  const acc = Array.from({ length: count }, () => [0, 0, 0, 0]);
  for (const t of tiles) {
    const a = acc[Math.min(count - 1, t.class || 0)];
    for (let i = t.start; i < t.start + t.count; i++) {
      const w = colour[4 * i + 3] / 255;
      a[0] += colour[4 * i] * w; a[1] += colour[4 * i + 1] * w;
      a[2] += colour[4 * i + 2] * w; a[3] += w;
    }
  }
  return acc.map(([r, g, b, w]) => (w > 0
    ? [r / w / 255, g / w / 255, b / w / 255] : [0.6, 0.6, 0.6]));
}
