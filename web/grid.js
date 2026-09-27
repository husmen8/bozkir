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

/** Adjacent pairs of cells (by integer `i`, `j`), each pair once. A missing
 *  cell (culled, or a hole) simply has no pairs. */
export function neighbourPairs(cells) {
  const byIJ = new Map();
  for (let k = 0; k < cells.length; k++) {
    byIJ.set(`${cells[k].i},${cells[k].j}`, k);
  }
  const pairs = [];
  for (let k = 0; k < cells.length; k++) {
    const c = cells[k];
    for (const [di, dj] of [[1, 0], [0, 1]]) {
      const n = byIJ.get(`${c.i + di},${c.j + dj}`);
      if (n !== undefined) pairs.push([k, n]);
    }
  }
  return pairs;
}
