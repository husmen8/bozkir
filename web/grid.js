// The geometry both halves of GSWT Section 3.4 are built on.
//
// Ordering and merging ask the same question about a pair of neighbouring
// cells and use the two halves of one answer. The shared boundary has a
// plane; the eye's signed distance to that plane says which cell is in
// front, and the magnitude of that same distance says how firmly. order.js
// takes the sign, merge.js takes the magnitude.
//
// This file exists because both had their own copy of the geometry -
// identical, and so bound to drift the moment one was corrected. Keeping
// them together also keeps the relationship visible: if the plane is ever
// defined differently for the two, the ordering and the merging would
// disagree about which boundaries are undecidable, and each would look
// correct on its own.

/** The shared boundary of two adjacent cells: midpoint and unit normal.
 *
 *  The normal comes from the line between the two centres rather than from
 *  a grid axis, so it survives a rotated layout and tilts with the terrain
 *  when relief lifts the two cells unequally.
 */
export function boundary(a, b) {
  const dx = b.x - a.x, dy = b.y - a.y, dz = (b.z || 0) - (a.z || 0);
  const len = Math.hypot(dx, dy, dz) || 1;
  return {
    mid: [(a.x + b.x) / 2, (a.y + b.y) / 2, ((a.z || 0) + (b.z || 0)) / 2],
    normal: [dx / len, dy / len, dz / len],
  };
}

/** Signed distance from the eye to the plane of the boundary between a and b.
 *
 *  The normal runs a to b, so a positive value puts the camera on b's side:
 *  b is the nearer cell and must be drawn after a in back-to-front order.
 *
 *  GSWT describes the merging test as the absolute dot product between the
 *  boundary normal and the unnormalised camera-to-edge vector. With a unit
 *  normal that is the perpendicular distance to the plane, which is what
 *  this returns and what `Math.abs` of it gives. Note that it does not fall
 *  off with range: an eye lying in the plane of a boundary scores zero
 *  however far away it is, which is correct - lying in the plane is the
 *  condition being detected, not being near the edge.
 */
export function boundarySign(a, b, eye) {
  const { mid, normal } = boundary(a, b);
  return normal[0] * (eye[0] - mid[0])
       + normal[1] * (eye[1] - mid[1])
       + normal[2] * (eye[2] - mid[2]);
}

/** Adjacent pairs of a grid of cells, each cell carrying integer `i`, `j`.
 *
 *  Each pair appears once, not twice: only the neighbours at +i and +j are
 *  taken, so a pair is owned by its lower cell. A cell missing from the
 *  list - culled, or a hole in the layout - simply produces no pairs, which
 *  is right in both callers, since a cell that is not drawn can neither be
 *  ordered against nor merged with anything.
 */
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