// Frames (export_wang.py --frames, bozkir/frames.py). A tile keeps only its
// frame, the band along its edges that neighbours agree on; each cell gets
// a round centre from its material's pool, turned as far as the material
// allows (any way for scrub; ripples keep their direction), and the two
// meet in a wobbled ring drawn by the splat shader.
//
// Pure functions, no WebGL, so they run under node and are tested there.

/** A turn is stored as a step of 1/256 of a full turn, so cells that share
 *  a centre and a step can share one sorted order. */
export const TURN_STEPS = 256;

/** A fixed hash of a cell, a seed and a salt, in [0, 1). */
export function cellHash(i, j, seed, salt = 0) {
  let v = (i * 374761393 + j * 668265263 + seed * 2246822519 + salt * 3266489917) | 0;
  v = Math.imul(v ^ (v >>> 15), 2246822507);
  v = Math.imul(v ^ (v >>> 13), 3266489909);
  return ((v ^ (v >>> 16)) >>> 0) / 4294967296;
}

/** Per class, the indices of its pool's centres (decal copies, which carry
 *  `base`, are placed by the decal rule, not picked). */
export function poolsByClass(centres, classCount) {
  const out = Array.from({ length: Math.max(1, classCount) }, () => []);
  (centres || []).forEach((c, k) => {
    if (c.base != null) return;
    const cls = Math.min(out.length - 1, c.class || 0);
    out[cls].push(k);
  });
  return out;
}

/** A cell's centre from `pool`: seeded per cell, and not one of `avoid`
 *  (the cells to the left and below) when the pool has others. -1 for an
 *  empty pool. */
export function pickCentre(pool, i, j, seed, avoid = []) {
  if (!pool || !pool.length) return -1;
  const fresh = pool.length > avoid.length ? pool.filter((k) => !avoid.includes(k)) : pool;
  const from = fresh.length ? fresh : pool;
  return from[Math.floor(cellHash(i, j, seed, 1) * from.length) % from.length];
}

/** A cell's turn, in steps of TURN_STEPS, within the material's rule:
 *  {any} turns any way; otherwise up to `spread` degrees either way, and
 *  half turned on top when {half} allows it. `quarter` (a rounded-square
 *  centre, which keeps its shape only so): quarter turns only, of those
 *  the rule allows. */
export function pickTurn(rule, i, j, seed, salt = 2, quarter = false) {
  const u = cellHash(i, j, seed, salt), v = cellHash(i, j, seed, salt + 7);
  if (quarter) {
    if (!rule || rule.any) return (TURN_STEPS / 4) * Math.min(3, Math.floor(u * 4));
    return rule.half && v < 0.5 ? TURN_STEPS / 2 : 0;
  }
  let deg;
  if (!rule || rule.any) {
    deg = u * 360;
  } else {
    deg = (2 * u - 1) * (rule.spread || 0) + (rule.half && v < 0.5 ? 180 : 0);
  }
  const q = Math.round(deg / 360 * TURN_STEPS);
  return ((q % TURN_STEPS) + TURN_STEPS) % TURN_STEPS;
}

/** cos and sin of a turn given in steps. */
export function turnCosSin(q) {
  const a = (q / TURN_STEPS) * 2 * Math.PI;
  return [Math.cos(a), Math.sin(a)];
}

/** Distance from a cell's middle as its centre's shape measures it
 *  (bozkir/frames.py norm): a disc for `shape` 2 (or unset), the p-norm's
 *  rounded square above. Mirrors ringNorm in the shaders. */
export function ringNorm(x, y, shape = 2) {
  if (!(shape > 2)) return Math.hypot(x, y);
  const m = Math.max(Math.abs(x), Math.abs(y));
  if (m <= 0) return 0;
  return m * ((Math.abs(x) / m) ** shape + (Math.abs(y) / m) ** shape) ** (1 / shape);
}

/** The height of a range's ground: a low quantile of its splats' heights
 *  (taller splats than this, and more opaque ones, win the ring). */
export function groundLevel(positions, start, count, q = 0.25, sample = 4096) {
  if (!count) return 0;
  const step = Math.max(1, Math.floor(count / sample));
  const z = [];
  for (let i = start; i < start + count; i += step) z.push(positions[3 * i + 2]);
  z.sort((a, b) => a - b);
  return z[Math.min(z.length - 1, Math.floor(q * z.length))];
}

/** The centre's share of the ground at distance `r` from the cell's
 *  middle, the ring's line at `rho` and `half` its half width; sharpened
 *  (hex-tiling's gamma), so most of the ring is one side or the other. The
 *  frame's share is one minus this. Mirrors the splat shader. */
export function centreShare(r, rho, half, gamma = 3) {
  const s = Math.min(1, Math.max(0, 0.5 - 0.5 * (r - rho) / Math.max(half, 1e-9)));
  const a = Math.pow(s, gamma), b = Math.pow(1 - s, gamma);
  return a + b > 0 ? a / (a + b) : 0;
}

/** A splat's chance to stay in the ring: its side's share, raised for a
 *  prominent splat (taller than the ground and opaque, 0..1) so a tuft is
 *  kept whole rather than cut along the line. Mirrors the splat shader. */
export function keepChance(share, prominence) {
  return Math.pow(share, 1 / (1 + 3 * Math.min(1, Math.max(0, prominence))));
}

/** The class blend by spot: a cell's own-material weight `w` (the mix,
 *  0..1, complementary between the two materials drawn) sharpened about
 *  its middle, so a spot is one material or the other and only a narrow
 *  strip dissolves; a prominent splat (0..1) keeps more of its chance
 *  there. Mirrors the splat shader (uSpots). */
export function spotShare(w, prominence = 0) {
  const s = Math.min(1, Math.max(0, 0.5 + (w - 0.5) * 4));
  const a = s ** 3, b = (1 - s) ** 3;
  return keepChance(a + b > 0 ? a / (a + b) : 0, prominence);
}

// The class blend by spot. A cell next to other materials draws theirs too,
// and a field over the grid says which wins where. On a half-cell grid each
// material has a share, the cells of it among those a point touches (its
// own cell at a middle, two on an edge, four at a corner); a material's
// field is 0.5 plus half its lead over the strongest other there, so 1 in
// the middle of its cells, 0.5 where it ties, under that where another
// leads. Read at a point moved by noise up to SPOT_WARP cells either way,
// each material reaches in and out of its neighbours' ground along every
// boundary, as far one way as the other, three meeting as well as two,
// and a lone cell keeps a rounded patch of its own.

/** How far the boundary wanders from the grid, in cells. Mirrors uMixWarp. */
export const SPOT_WARP = 0.7;

/** The other material a cell blends into: the one most of its eight
 *  neighbours have (the lowest on a tie), or -1 with none. `classAt(i, j)`
 *  gives a cell's material, -1 off the grid. `ring` 2: the cells two away
 *  instead. */
export function partnerClass(classAt, i, j, ring = 1) {
  const own = classAt(i, j), count = new Map();
  for (let dj = -ring; dj <= ring; dj++) {
    for (let di = -ring; di <= ring; di++) {
      if (Math.max(Math.abs(di), Math.abs(dj)) !== ring) continue;
      const k = classAt(i + di, j + dj);
      if (k >= 0 && k !== own) count.set(k, (count.get(k) || 0) + 1);
    }
  }
  let best = -1, most = 0;
  for (const [k, n] of [...count].sort((a, b) => a[0] - b[0])) if (n > most) { best = k; most = n; }
  return best;
}

/** Centres for a cell `step` cells from another material (1 touching, 2
 *  one between): of its material's `pool`, those whose tone (`tone(k)`,
 *  how far the centre's colour lies towards the other material's) is
 *  nearest, half the pool next to it and three quarters one further, so
 *  the two meet in their closest shades and grade back into the whole pool
 *  (the capture's own centres, ordered, none tinted). Any step else. */
export function gradedPool(pool, tone, step) {
  const share = step === 1 ? 0.5 : step === 2 ? 0.75 : 1;
  if (!pool || pool.length < 2 || share >= 1) return pool;
  const keep = Math.max(1, Math.ceil(pool.length * share));
  return [...pool].sort((a, b) => tone(b) - tone(a) || a - b).slice(0, keep);
}

/** Material `cls`'s field around cell (i, j): 7 x 7 values at half-cell
 *  steps from -1 to 2 cells (the cell spans 0..1), row by row from the
 *  south-west; 0.5 plus half its lead in share over the strongest other
 *  material (off the grid does not count). The same point has the same
 *  value from every cell, so their draws agree. */
export function spotField(classAt, i, j, cls) {
  const out = new Float32Array(49);
  for (let b = 0; b < 7; b++) {
    for (let a = 0; a < 7; a++) {
      const x = a / 2 - 1, y = b / 2 - 1;
      const xs = Number.isInteger(x) ? [x - 1, x] : [Math.floor(x)];
      const ys = Number.isInteger(y) ? [y - 1, y] : [Math.floor(y)];
      const share = new Map();
      let n = 0;
      for (const dy of ys) {
        for (const dx of xs) {
          const k = classAt(i + dx, j + dy);
          if (k >= 0) { share.set(k, (share.get(k) || 0) + 1); n++; }
        }
      }
      let mine = 0, best = 0;
      for (const [k, v] of share) if (k === cls) mine = v; else best = Math.max(best, v);
      out[b * 7 + a] = n ? 0.5 + 0.5 * (mine - best) / n : 0.5;
    }
  }
  return out;
}

/** What cell (i, j) draws in the blend: its own material's field first,
 *  then each other material within reach whose field rises over 0.375
 *  somewhere its splats can read (spotShare of it above zero), as
 *  [{ cls, field }]; only its own where nothing reaches. */
export function spotDraws(classAt, i, j, warp = SPOT_WARP) {
  const own = classAt(i, j), near = new Set();
  for (let dj = -2; dj <= 2; dj++) {
    for (let di = -2; di <= 2; di++) {
      const k = classAt(i + di, j + dj);
      if (k >= 0 && k !== own) near.add(k);
    }
  }
  const out = [{ cls: own, field: spotField(classAt, i, j, own) }];
  for (const k of [...near].sort((a, b) => a - b)) {
    const field = spotField(classAt, i, j, k);
    if (fieldReaches(field, warp)) out.push({ cls: k, field });
  }
  return out;
}

/** The field at (x, y) in cell units (0..1 across the cell), bilinear
 *  between the half-cell values. Mirrors the splat shader. */
export function fieldAt(field, x, y) {
  const u = (Math.min(2, Math.max(-1, x)) + 1) * 2, v = (Math.min(2, Math.max(-1, y)) + 1) * 2;
  const i = Math.min(5, Math.floor(u)), j = Math.min(5, Math.floor(v));
  const fx = u - i, fy = v - j;
  const f = (a, b) => field[b * 7 + a];
  return (f(i, j) * (1 - fx) + f(i + 1, j) * fx) * (1 - fy)
    + (f(i, j + 1) * (1 - fx) + f(i + 1, j + 1) * fx) * fy;
}

/** Whether a material's field rises over 0.375 (where spotShare of it
 *  starts) anywhere a cell's splats can read it: its square, moved by up
 *  to `warp`. A bilinear field is highest at the corners of its pieces,
 *  so those points are all it takes. */
export function fieldReaches(field, warp = SPOT_WARP) {
  const at = [-warp, -0.5, 0, 0.5, 1, 1.5, 1 + warp].filter((v) => v >= -warp && v <= 1 + warp);
  for (const y of at) for (const x of at) if (fieldAt(field, x, y) > 0.375) return true;
  return false;
}

/** The ground's colour around the ring, in `sectors` equal angles from
 *  east, counter-clockwise: opacity-weighted, from splats within `half` of
 *  `radius` and no higher than `low` above the ground `z0` (a tuft is not
 *  the ground's tone). Empty sectors take the ring's mean. rgb in 0..1,
 *  `colour` as unpack() gives it. */
export function ringProfile(positions, colour, start, count, radius, half, z0, low,
                            sectors = 16, shape = 2) {
  const acc = new Float64Array(sectors * 4);
  for (let i = start; i < start + count; i++) {
    const x = positions[3 * i], y = positions[3 * i + 1];
    const r = ringNorm(x, y, shape);
    if (Math.abs(r - radius) > half || positions[3 * i + 2] - z0 > low) continue;
    let a = Math.atan2(y, x);
    if (a < 0) a += 2 * Math.PI;
    const k = Math.min(sectors - 1, Math.floor(a / (2 * Math.PI) * sectors));
    const w = colour[4 * i + 3] / 255;
    acc[4 * k] += w * colour[4 * i]; acc[4 * k + 1] += w * colour[4 * i + 1];
    acc[4 * k + 2] += w * colour[4 * i + 2]; acc[4 * k + 3] += w;
  }
  const all = [0, 0, 0, 0];
  for (let k = 0; k < sectors; k++) for (let c = 0; c < 4; c++) all[c] += acc[4 * k + c];
  const out = new Float32Array(sectors * 3);
  for (let k = 0; k < sectors; k++) {
    const w = acc[4 * k + 3] > 1e-6 ? acc[4 * k + 3] : 0;
    for (let c = 0; c < 3; c++) {
      out[3 * k + c] = w ? acc[4 * k + c] / w / 255
        : all[3] > 1e-6 ? all[c] / all[3] / 255 : 0;
    }
  }
  return out;
}

/** How to tint a turned centre so its ground meets its frame's along the
 *  ring: the difference of the two profiles around the ring, the centre's
 *  turned with it, carried inwards smoothly. On a disc that is the
 *  harmonic extension (what a Poisson membrane solves for): each Fourier
 *  term of the difference times (r / radius)^k. Returns 7 rgb triples, a0,
 *  a1, b1, a2, b2, a3, b3, as a Float32Array(21). */
export function toneTerms(frame, centre, turnQ) {
  const n = frame.length / 3;
  const shift = (turnQ / TURN_STEPS) * n;           // sectors the centre turned
  const out = new Float32Array(21);
  for (let k = 0; k < n; k++) {
    // The centre's profile at this tile angle: its own angle minus the turn.
    const at = ((k - shift) % n + n) % n;
    const k0 = Math.floor(at), k1 = (k0 + 1) % n, f = at - k0;
    const phi = (k + 0.5) / n * 2 * Math.PI;
    for (let c = 0; c < 3; c++) {
      const cc = centre[3 * k0 + c] * (1 - f) + centre[3 * k1 + c] * f;
      const d = frame[3 * k + c] - cc;
      out[c] += d / n;
      for (let m = 1; m <= 3; m++) {
        out[3 * (2 * m - 1) + c] += 2 * d * Math.cos(m * phi) / n;
        out[3 * (2 * m) + c] += 2 * d * Math.sin(m * phi) / n;
      }
    }
  }
  return out;
}

/** The tint at `q` = r / radius (capped at 1) and tile angle `phi`, at
 *  most `most` per channel. Mirrors the splat shader. */
export function toneAt(terms, q, phi, most = 0.12) {
  const s = Math.min(1, Math.max(0, q));
  const out = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    let v = terms[c];
    for (let m = 1; m <= 3; m++) {
      v += Math.pow(s, m) * (terms[3 * (2 * m - 1) + c] * Math.cos(m * phi)
                             + terms[3 * (2 * m) + c] * Math.sin(m * phi));
    }
    out[c] = Math.min(most, Math.max(-most, v));
  }
  return out;
}
