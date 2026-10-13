// Where decals go. A tileset made with scripts/decals.py carries copies of
// its plain tiles with a distinct piece of the capture (a shrub and its
// shadow) set in the middle, edges untouched. The layout lays plain tiles
// first; this then swaps a copy in at a few cells.
//
// Too many and the ground looks like random clutter, the same few decals
// seen over and over; so at most one per block of `spacing` x `spacing`
// cells, and never the same decal within two blocks of itself.

/** Copies by the tile they were made from: Map(base -> [{ tile, decal }]). */
export function decalVariants(tiles) {
  const out = new Map();
  tiles.forEach((t, k) => {
    if (t.base == null) return;
    if (!out.has(t.base)) out.set(t.base, []);
    out.get(t.base).push({ tile: k, decal: t.decal ?? 0 });
  });
  return out.size ? out : null;
}

/** Smooth noise in [0, 1) on the cell grid, for which kind grows where. */
function patchNoise(x, y, seed) {
  const h = (i, j) => {
    let v = (i * 374761393 + j * 668265263 + seed * 2246822519) | 0;
    v = Math.imul(v ^ (v >>> 13), 1274126177);
    return ((v ^ (v >>> 16)) >>> 0) / 4294967296;
  };
  const i = Math.floor(x), j = Math.floor(y), fx = x - i, fy = y - j;
  const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
  const a = h(i, j) + (h(i + 1, j) - h(i, j)) * sx;
  const b = h(i, j + 1) + (h(i + 1, j + 1) - h(i, j + 1)) * sx;
  return a + (b - a) * sy;
}

/** Which cells get which copy. `patchOf(cell)` is the plain tile laid at a
 *  cell (row-major, gridN wide). Returns [{ cell, tile, decal }]. Seeded,
 *  and on its own random stream, so the plain layout does not move when
 *  decals are switched on or the spacing changes.
 *
 *  `opts.weight(cell)`, 0..1, lets the terrain lead: each block tries a few
 *  cells and keeps the best placed, and a block whose best is weak usually
 *  stays bare (at weight 0, one in five keeps a decal). `opts.groups` makes
 *  neighbouring decals tend to look alike, from a smooth field over the
 *  map, as one kind of plant spreads across a patch of ground. */
export function placeDecals(patchOf, gridN, variants, spacing, seed, opts = {}) {
  const out = [];
  if (!variants || !(spacing >= 1)) return out;
  const S = Math.max(1, Math.round(spacing));
  const weight = opts.weight || null;
  const tries = weight ? 4 : 1;
  const FLOOR = 0.2;
  let kinds = 0;
  if (opts.groups) {
    const all = new Set();
    for (const list of variants.values()) for (const v of list) all.add(v.decal);
    kinds = all.size;
  }
  let s = (seed ^ 0x5bd1e995) & 0x7fffffff;
  const rand = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  const near = 2 * S;
  for (let bj = 0; bj < gridN; bj += S) {
    for (let bi = 0; bi < gridN; bi += S) {
      // A fixed number of draws per block, so one block's outcome never
      // shifts the next.
      let best = -1, bestW = -1;
      for (let t = 0; t < tries; t++) {
        const i = bi + Math.floor(rand() * S), j = bj + Math.floor(rand() * S);
        if (i >= gridN || j >= gridN) continue;
        const cell = j * gridN + i;
        if (!variants.get(patchOf(cell))) continue;
        const w = weight ? Math.min(1, Math.max(0, weight(cell))) : 1;
        if (w > bestW) { best = cell; bestW = w; }
      }
      const keep = rand(), pickAt = rand();
      if (best < 0) continue;
      if (weight && keep > FLOOR + (1 - FLOOR) * bestW) continue;
      const i = best % gridN, j = Math.floor(best / gridN);
      const opts2 = variants.get(patchOf(best));
      let v = opts2[Math.floor(pickAt * opts2.length) % opts2.length];
      if (kinds > 1) {
        // Decals are numbered in colour order (decals.py), so near numbers
        // look alike: a patch gets shrubs that resemble each other, never
        // the same one twice (below).
        const want = patchNoise(i / (3 * S), j / (3 * S), seed) * (kinds - 1);
        const gap = (o) => Math.abs(o.decal - want);
        v = opts2.reduce((a, b) => (gap(b) < gap(a) ? b : a), v);
      }
      const clash = out.some((o) => o.decal === v.decal
        && Math.abs((o.cell % gridN) - i) < near
        && Math.abs(Math.floor(o.cell / gridN) - j) < near);
      if (!clash) out.push({ cell: best, tile: v.tile, decal: v.decal });
    }
  }
  return out;
}
