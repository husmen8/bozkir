// Boundary geometry shared by ordering and merging (GSWT 3.4).
//
// The eye's signed distance to the plane between two neighbouring cells:
// order.js uses the sign (which cell is in front), merge.js the magnitude
// (how undecidable). One copy, so the two can never disagree about which
// boundaries are undecidable.

/** Shared boundary of two adjacent cells: midpoint and unit normal. The
 *  normal comes from the line between the centres, so it follows a rotated
 *  layout and tilts with relief. */
export function boundary(a, b) {
  const dx = b.x - a.x, dy = b.y - a.y, dz = (b.z || 0) - (a.z || 0);
  const len = Math.hypot(dx, dy, dz) || 1;
  return {
    mid: [(a.x + b.x) / 2, (a.y + b.y) / 2, ((a.z || 0) + (b.z || 0)) / 2],
    normal: [dx / len, dy / len, dz / len],
  };
}

/** Signed distance from the eye to the boundary plane between a and b.
 *  The normal runs a to b, so positive means the eye is on b's side: b is
 *  nearer and is drawn after a. */
export function boundarySign(a, b, eye) {
  const { mid, normal } = boundary(a, b);
  return normal[0] * (eye[0] - mid[0])
       + normal[1] * (eye[1] - mid[1])
       + normal[2] * (eye[2] - mid[2]);
}

// The pairs of the last arrays asked about: the whole grid is asked several
// times a frame (merging at the eye and at turned headings), 16k cells at a
// 128-cell world, and building the lookup took a third of the frame.
const pairCache = new WeakMap();

/** Adjacent pairs of cells (by integer `i`, `j`), each pair once. A missing
 *  cell (culled, or a hole) simply has no pairs. Kept per array: the same
 *  array of the same length gives the same pairs. */
export function neighbourPairs(cells) {
  const hit = pairCache.get(cells);
  if (hit && hit.n === cells.length) return hit.pairs;
  // Integer keys; i and j stay well inside +-2^20.
  const key = (i, j) => (i + 1048576) * 2097152 + (j + 1048576);
  const byIJ = new Map();
  for (let k = 0; k < cells.length; k++) {
    byIJ.set(key(cells[k].i, cells[k].j), k);
  }
  const pairs = [];
  for (let k = 0; k < cells.length; k++) {
    const c = cells[k];
    const e = byIJ.get(key(c.i + 1, c.j));
    if (e !== undefined) pairs.push([k, e]);
    const n = byIJ.get(key(c.i, c.j + 1));
    if (n !== undefined) pairs.push([k, n]);
  }
  pairCache.set(cells, { n: cells.length, pairs });
  return pairs;
}

const geomCache = new WeakMap();

/** neighbourPairs with each pair's boundary (boundary()) in flat arrays:
 *  a, b (cell indices), mid and normal per pair. Kept per array as the
 *  pairs are, so scoring every pair at a new eye allocates nothing. */
export function pairGeometry(cells) {
  const hit = geomCache.get(cells);
  if (hit && hit.cellCount === cells.length) return hit;
  const pairs = neighbourPairs(cells);
  const n = pairs.length;
  const out = { cellCount: cells.length, n, a: new Int32Array(n), b: new Int32Array(n),
                mid: new Float64Array(3 * n), normal: new Float64Array(3 * n) };
  for (let k = 0; k < n; k++) {
    const [ia, ib] = pairs[k];
    const { mid, normal } = boundary(cells[ia], cells[ib]);
    out.a[k] = ia; out.b[k] = ib;
    out.mid.set(mid, 3 * k);
    out.normal.set(normal, 3 * k);
  }
  geomCache.set(cells, out);
  return out;
}
