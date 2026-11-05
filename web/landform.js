// Choosing a tile's material from the shape of the ground under it.
//
// Hybrid Gaussian Wang Tiles picks a class per position from a coverage
// field alpha_c(x) (their Eq. 1) that a person paints. Here the field is
// computed from the terrain instead - slope, position, drainage, height,
// sediment - the way terrain shaders place materials, and fed to the same
// equation.
//
// The maps mirror bozkir/terrain.py (tested there on surfaces with known
// landforms). The decision is per cell, not per pixel, so even flow
// accumulation costs less than a frame and runs once per layout.

// The rule's weights, shared with Python. fetch + top-level await instead
// of a JSON import attribute, which is a parse error (blank page) before
// Chrome 123 / Safari 17.2 / Firefox 138. Node cannot fetch file: URLs, so
// the tests read it from disk.
import { fillDepressions } from './terrain.js';
const RULE = await loadRule(new URL('./rule.json', import.meta.url));
// The river's settings, for the viewer's channel and water.
export const RIVER = RULE.river;

async function loadRule(url) {
  if (url.protocol === 'file:') {
    const { readFileSync } = await import('node:fs');
    return JSON.parse(readFileSync(url, 'utf8'));
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`could not load ${url}: ${res.status}`);
  return res.json();
}

/** Sample a height function onto the cell grid. Takes the same `height`
 *  the renderer warps with, so the rule reads the terrain that is drawn. */
export function sampleGrid(n, tileSize, height, gridAngle = 0, over = 1) {
  const k = Math.max(1, over | 0);
  const m = n * k;
  const z = new Float64Array(m * m);
  // Step from the span both grids share, so the fine samples cover the same
  // ground as the cells (a step of tileSize/k fell short by most of a tile).
  const span = (n - 1) * tileSize;
  const step = m > 1 ? span / (m - 1) : 0;
  const half = (m - 1) / 2;
  const a = gridAngle * Math.PI / 180;
  const ca = Math.cos(a), sa = Math.sin(a);
  for (let j = 0; j < m; j++) {
    for (let i = 0; i < m; i++) {
      const lx = (i - half) * step, ly = (j - half) * step;
      z[j * m + i] = height(ca * lx - sa * ly, sa * lx + ca * ly);
    }
  }
  return z;
}

/** Bring an over-sampled map back to one value per cell. Slope and
 *  position are averaged; drainage takes the max, because a channel is
 *  narrow and averaging dilutes it out of the cell that contains it. */
export function downsample(a, m, n, how = 'mean') {
  const over = Math.max(1, Math.round(m / n));
  if (over === 1) return a;
  const out = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      let acc = how === 'max' ? -Infinity : 0;
      for (let b = 0; b < over; b++) {
        for (let c = 0; c < over; c++) {
          const v = a[(j * over + b) * m + (i * over + c)];
          acc = how === 'max' ? Math.max(acc, v) : acc + v;
        }
      }
      out[j * n + i] = how === 'max' ? acc : acc / (over * over);
    }
  }
  return out;
}

/** Gradient magnitude: rise over run. */
export function slope(z, n, spacing = 1) {
  const out = new Float64Array(n * n);
  const at = (i, j) => z[Math.min(n - 1, Math.max(0, j)) * n
                        + Math.min(n - 1, Math.max(0, i))];
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const dx = (at(i + 1, j) - at(i - 1, j)) / (2 * spacing);
      const dy = (at(i, j + 1) - at(i, j - 1)) / (2 * spacing);
      out[j * n + i] = Math.hypot(dx, dy);
    }
  }
  return out;
}

/** Topographic position (Weiss 2001): height above the local mean.
 *  Positive on ridges, negative in hollows, ~0 on planar slopes. The radius
 *  decides the scale: small finds bumps, large finds the landscape. */
export function tpi(z, n, radius = 3) {
  const out = new Float64Array(n * n);
  const r = Math.max(1, radius | 0);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      let sum = 0, count = 0;
      for (let b = -r; b <= r; b++) {
        const jj = j + b;
        if (jj < 0 || jj >= n) continue;
        for (let a = -r; a <= r; a++) {
          const ii = i + a;
          if (ii < 0 || ii >= n) continue;
          sum += z[jj * n + ii];
          count++;
        }
      }
      out[j * n + i] = z[j * n + i] - sum / Math.max(count, 1);
    }
  }
  return out;
}

/** Drainage area (D8): each cell sends its water to its lowest neighbour,
 *  accumulated in one pass from the highest cell down. */
export function flow(z, n) {
  return drainage(z, n).acc;
}

/** HAND and the channel mask, as bozkir/landform.py's `hand`: a cell
 *  draining `channelCells` or more is channel; every other cell is measured
 *  against the channel cell its receivers reach first (or its path's end).
 *  Lowest first, so a receiver (strictly lower) is always done. */
export function hand(z, n, to, acc, channelCells) {
  const N = n * n;
  const base = new Int32Array(N);
  for (let k = 0; k < N; k++) base[k] = k;
  const order = Array.from({ length: N }, (_, k) => k).sort((p, q) => z[p] - z[q]);
  for (const k of order) {
    const j = to[k];
    if (!(acc[k] >= channelCells) && j >= 0) base[k] = base[j];
  }
  const h = new Float64Array(N), chan = new Float64Array(N);
  for (let k = 0; k < N; k++) { h[k] = z[k] - z[base[k]]; chan[k] = acc[k] >= channelCells ? 1 : 0; }
  return { hand: h, channel: chan };
}

/** Each sample's drainage base, the height of the channel it drains to:
 *  the filled field minus HAND, with the river rule's receivers, channel
 *  share and lakes (as classify's river option). The viewer's valley mist
 *  pools from there. */
export function drainBase(z, n) {
  const zf = fillDepressions(z, n);
  let lo = Infinity, hi = -Infinity;
  for (const v of z) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const depth = RULE.river.lake_depth * Math.max(hi - lo, 1e-12);
  const rd = drainage(zf, n);
  const racc = rd.acc.map((v, k) => ((zf[k] - z[k]) > depth ? Infinity : v));
  const cells = Math.max(8, Math.round(n * n * RULE.river.channel_share));
  const h = hand(zf, n, rd.to, racc, cells).hand;
  const out = new Float32Array(n * n);
  for (let k = 0; k < n * n; k++) out[k] = zf[k] - h[k];
  return out;
}

/** The rivers of a height field, for waterMap and riverChannel: the
 *  depression-filled field drained as the river rule does, cells draining
 *  more than river.water_share of the field, and only those of the
 *  river.water_rivers largest river systems (cells grouped by the border
 *  outlet they drain to), so a few main rivers run instead of a net over the
 *  whole map. Course: the eight-direction path runs in straight lines and
 *  45 degree steps across filled flats, so each cell's centre is averaged
 *  along its river over river.water_smooth cells either way, then moved
 *  sideways by river.water_meander channel widths on a wave of about 11
 *  widths (meander wavelength, Leopold and Wolman 1960); 0 for either
 *  leaves it out. No standalone lakes. Returns the receivers, the wet cells
 *  downstream first (`order`) and each one's smoothed centre. The last
 *  field's answer is kept: it depends on the heights alone, and the channel
 *  is redone whenever width or relief moves. */
let lastCourse = null;
function riverCourse(z, n) {
  if (lastCourse && lastCourse.z === z && lastCourse.n === n) return lastCourse.course;
  const course = findCourse(z, n);
  lastCourse = { z, n, course };
  return course;
}
function findCourse(z, n) {
  const zf = fillDepressions(z, n);
  const { acc, to } = drainage(zf, n);
  const cells = Math.max(8, Math.round(n * n * RULE.river.water_share));
  // Each cell's outlet: follow the receivers down, marking the path.
  const outlet = new Int32Array(n * n).fill(-1);
  for (let k = 0; k < n * n; k++) {
    const path = [];
    let c = k;
    while (outlet[c] < 0 && to[c] >= 0) { path.push(c); c = to[c]; }
    const root = outlet[c] >= 0 ? outlet[c] : c;
    outlet[c] = root;
    for (const p of path) outlet[p] = root;
  }
  const roots = [...new Set(outlet)].sort((a, b) => acc[b] - acc[a] || a - b)
    .slice(0, RULE.river.water_rivers);
  const wet = new Uint8Array(n * n);
  for (let k = 0; k < n * n; k++) {
    wet[k] = acc[k] >= cells && roots.includes(outlet[k]) ? 1 : 0;
  }
  // Upstream along the river: the wet cell draining into k with most area.
  const up = new Int32Array(n * n).fill(-1);
  for (let k = 0; k < n * n; k++) {
    const t = to[k];
    if (!wet[k] || t < 0 || !wet[t]) continue;
    if (up[t] < 0 || acc[k] > acc[up[t]] || (acc[k] === acc[up[t]] && k < up[t])) up[t] = k;
  }
  const along = (k, steps) => {           // the wet cell `steps` down (+) or up (-)
    let c = k;
    for (let s = 0; s < Math.abs(steps); s++) {
      const next = steps > 0 ? to[c] : up[c];
      if (next < 0 || !wet[next]) break;
      c = next;
    }
    return c;
  };
  const L = Math.max(0, Math.round(RULE.river.water_smooth));
  const meander = RULE.river.water_meander;
  // Downstream first (more area), so a cell's receiver has its phase already.
  const order = [];
  for (let k = 0; k < n * n; k++) if (wet[k]) order.push(k);
  order.sort((p, q) => acc[q] - acc[p] || p - q);
  const phase = new Float64Array(n * n);
  const cx = new Float64Array(n * n), cy = new Float64Array(n * n);
  for (const k of order) {
    const r = 0.5 + Math.sqrt(acc[k] / cells);
    const t = to[k];
    if (t >= 0 && wet[t]) {
      const step = Math.hypot(k % n - t % n, ((k / n) | 0) - ((t / n) | 0));
      phase[k] = phase[t] + 2 * Math.PI * step / (11 * 2 * r);
    }
    let sx = 0, sy = 0, sw = 0;
    for (let d = -L; d <= L; d++) {
      const c = along(k, d), w = Math.exp(-0.5 * (d / Math.max(L / 2, 1)) ** 2);
      sx += w * (c % n); sy += w * ((c / n) | 0); sw += w;
    }
    cx[k] = sx / sw; cy[k] = sy / sw;
    if (meander > 0) {
      const a = along(k, -Math.max(L, 2)), b = along(k, Math.max(L, 2));
      let tx = b % n - a % n, ty = ((b / n) | 0) - ((a / n) | 0);
      const tl = Math.hypot(tx, ty);
      if (tl > 0) {
        const off = meander * 2 * r * Math.sin(phase[k]);
        cx[k] += -ty / tl * off; cy[k] += tx / tl * off;
      }
    }
    // Inside the map: a bend pushed past the border cut a river in two.
    cx[k] = Math.min(n - 1, Math.max(0, cx[k]));
    cy[k] = Math.min(n - 1, Math.max(0, cy[k]));
  }
  return { zf, acc, to, up, wet, order, cells, cx, cy, outlet, roots, along };
}

/** Where rivers run on a height field, 0..1 per cell (1 = channel middle),
 *  in cells: along each river of riverCourse a disc of diameter
 *  1 + 2 sqrt(area / threshold) cells, as width grows with the square root
 *  of drainage area (Leopold and Maddock 1953). The water map's preview. */
export function waterMap(z, n) {
  const { acc, to, wet, order, cells, cx, cy } = riverCourse(z, n);
  // Each cell drawn as a band to its receiver's centre, so a river stays in
  // one piece where a tributary and the main river bend different ways.
  const out = new Float32Array(n * n);
  for (const k of order) {
    const r = 0.5 + Math.sqrt(acc[k] / cells), R = Math.ceil(r);
    const t = to[k];
    const bx = t >= 0 && wet[t] ? cx[t] : cx[k], by = t >= 0 && wet[t] ? cy[t] : cy[k];
    const dx = bx - cx[k], dy = by - cy[k], dd = dx * dx + dy * dy;
    const i0 = Math.floor(Math.min(cx[k], bx)) - R, i1 = Math.ceil(Math.max(cx[k], bx)) + R;
    const j0 = Math.floor(Math.min(cy[k], by)) - R, j1 = Math.ceil(Math.max(cy[k], by)) + R;
    for (let j = Math.max(0, j0); j <= Math.min(n - 1, j1); j++) {
      for (let i = Math.max(0, i0); i <= Math.min(n - 1, i1); i++) {
        const u = dd > 0 ? Math.min(1, Math.max(0, ((i - cx[k]) * dx + (j - cy[k]) * dy) / dd)) : 0;
        const v = 1 - Math.hypot(i - cx[k] - u * dx, j - cy[k] - u * dy) / r;
        if (v > out[j * n + i]) out[j * n + i] = v;
      }
    }
  }
  return out;
}

/** A river channel cut into a height field, in metres: per cell how far
 *  the ground is lowered (carve, 0 or less), the water level (NO_WATER
 *  outside the river) and the direction the water runs (x, y in world
 *  axes, +y up the field). `o`: cellM (metres per cell), metresPerZ (metres
 *  per field unit: the ground is metresPerZ (z - 0.5)) and width (the main
 *  river's width at its outlet, metres).
 *
 *  Courses from riverCourse. Width grows with the square root of drainage
 *  area (Leopold and Maddock 1953): W sqrt(area / its outlet's) within a
 *  system, and W sqrt(its outlet's / the largest outlet's) for the system,
 *  so the second river is narrower than the first. Along each system's main
 *  stem (the larger branch at every junction) the width stays at least
 *  river.stem_floor of the system's and tapers to nothing over two widths
 *  at its head, instead of starting full width in the open; branches
 *  narrower than river.min_width metres are left out.
 *
 *  Cross-section: a parabola river.width_depth times wider than deep (a
 *  gravel bed at bankfull, 15-40; Church 2006) with its rim at the water
 *  level, then banks rising at river.bank_slope until they meet the ground.
 *  In a bend the deepest line moves towards the outer bank (up to
 *  river.bend_shift of the half-width, at a radius of two widths), the
 *  outer bank steepens into a cut bank and the inner one flattens into a
 *  point bar (Leopold and Wolman 1960). Along the river the depth rises and
 *  falls by river.pool_depth every river.pool_spacing widths, pools and
 *  riffles 5-7 widths apart (Keller and Melhorn 1978).
 *  The water level follows the depression-filled field along the river,
 *  smoothed over a width either way, never rising downstream and never
 *  above the filled ground under the course; it is kept only inside the
 *  channel (to its rim), and sits
 *  river.freeboard depths below it: banks stand out of the water and a
 *  filled hollow on the course stays a pool. The carve fades out over the
 *  last half of river.bank_reach metres past the rim, so a valley wall
 *  higher than that is met by a slope, not a cliff, and the whole cut is
 *  softened over river.soften metres. Depth stops at river.max_depth. */
export const NO_WATER = -1e4;
export function riverChannel(z, n, o) {
  const R = RULE.river;
  const { zf, acc, to, up, wet, order, outlet, along, cx, cy } = riverCourse(z, n);
  const cellM = Math.max(o.cellM, 1e-9), mz = o.metresPerZ, W = o.width;
  const N = n * n;
  // Each system's outlet area, and the largest.
  const sysAcc = new Map(), sysOut = new Map();
  for (const k of order) {
    if (!(acc[k] <= (sysAcc.get(outlet[k]) || 0))) { sysAcc.set(outlet[k], acc[k]); sysOut.set(outlet[k], k); }
  }
  const mainAcc = Math.max(1, ...sysAcc.values());
  // Main stems: from each system's outlet up the larger branch, and the
  // distance down the stem from its head, metres.
  const stem = new Uint8Array(N);
  const fromHead = new Float64Array(N).fill(Infinity);
  for (const k0 of sysOut.values()) {
    const path = [];
    for (let c = k0; c >= 0; c = up[c]) { stem[c] = 1; path.push(c); }
    let d = 0;
    for (let q = path.length - 1; q >= 0; q--) {
      fromHead[path[q]] = d;
      if (q > 0) d += Math.hypot(cx[path[q - 1]] - cx[path[q]], cy[path[q - 1]] - cy[path[q]]) * cellM;
    }
  }
  const width = new Float64Array(N);
  for (const k of order) {
    const sa = sysAcc.get(outlet[k]);
    const ws = W * Math.sqrt(sa / mainAcc);
    let w = ws * Math.sqrt(acc[k] / sa);
    if (stem[k]) {
      const t = Math.min(1, fromHead[k] / (2 * ws));
      w = Math.max(w, ws * R.stem_floor) * t * t * (3 - 2 * t);
    }
    width[k] = w >= Math.max(R.min_width, 2 * cellM) ? w : 0;
  }
  // Water level, field units: the filled height averaged over a width either
  // way along the river, then never above the level downstream of it rises.
  const lz = new Float64Array(N);
  for (const k of order) {
    if (!width[k]) continue;
    const L = Math.max(1, Math.round(width[k] / cellM));
    let s = 0, sw = 0;
    for (let d = -L; d <= L; d++) {
      const c = along(k, d), g = Math.exp(-0.5 * (d / Math.max(L / 2, 1)) ** 2);
      s += g * zf[c]; sw += g;
    }
    lz[k] = s / sw;
    const t = to[k];
    if (t >= 0 && wet[t] && width[t]) lz[k] = Math.max(lz[k], lz[t]);
    // Never above the ground it runs on: where the river drops fast, the
    // average over the reach above held the water up in the air. Still
    // never rising downstream, as the filled ground does not.
    lz[k] = Math.min(lz[k], zf[k]);
  }
  // Steps and metres from the outlet; a wide river is drawn from every
  // s-th cell, and pools and riffles follow the distance.
  const step = new Int32Array(N);
  const down = new Float64Array(N);
  for (const k of order) {
    const t = to[k];
    step[k] = t >= 0 && wet[t] ? step[t] + 1 : 0;
    down[k] = t >= 0 && wet[t] ? down[t] + Math.hypot(cx[t] - cx[k], cy[t] - cy[k]) * cellM : 0;
  }
  // How hard each cell's river turns, signed (towards +cross in grid axes),
  // over a width either way: 1 at a radius of two widths.
  const bend = new Float64Array(N);
  for (const k of order) {
    if (!width[k]) continue;
    const m = Math.max(1, Math.round(width[k] / cellM));
    const a = along(k, -m), b = along(k, m);
    const ux = cx[k] - cx[a], uy = cy[k] - cy[a], vx = cx[b] - cx[k], vy = cy[b] - cy[k];
    const lu = Math.hypot(ux, uy), lv = Math.hypot(vx, vy);
    if (lu <= 0 || lv <= 0) continue;
    const turn = Math.atan2(ux * vy - uy * vx, ux * vx + uy * vy);
    const kappa = turn / ((lu + lv) / 2 * cellM);
    bend[k] = Math.max(-1, Math.min(1, kappa * 2 * width[k]));
  }
  // Depth: width / width_depth, at most river.max_depth (past ~0.3 m the
  // water hides the bed anyway, and a deeper cut only bends the tiles).
  const depthOf = (w) => Math.min(w / R.width_depth, R.max_depth);
  const levelAt = (c) => mz * (lz[c] - 0.5) - R.freeboard * depthOf(width[c]);
  const reach = R.bank_reach;
  const best = new Float64Array(N).fill(Infinity);   // lowest ground wanted, m
  const fade = new Float64Array(N);
  const out = new Float32Array(4 * N);
  for (let c = 0; c < N; c++) out[4 * c + 1] = NO_WATER;
  for (const k of order) {
    const wk = width[k];
    if (!wk) continue;
    // A band from k to s cells downstream; the bands overlap, so only
    // every s-th cell starts one (and every head).
    const s = Math.max(1, Math.floor(wk / 4 / cellM));
    if (step[k] % s && up[k] >= 0 && width[up[k]]) continue;
    let e = k;
    for (let q = 0; q < s; q++) {
      const t = to[e];
      if (t < 0 || !wet[t] || !width[t]) break;
      e = t;
    }
    const Lk = levelAt(k), Le = levelAt(e);
    const bk = wk / 2, be = width[e] / 2;
    const pool = (c) => 1 + R.pool_depth * Math.cos(2 * Math.PI * down[c] / (R.pool_spacing * width[c]));
    const Dk = depthOf(wk) * pool(k), De = depthOf(width[e]) * pool(e);
    const Bk = bend[k], Be = bend[e];
    const ax = cx[k], ay = cy[k], dx = cx[e] - ax, dy = cy[e] - ay, dd = dx * dx + dy * dy;
    // Its direction; at the river's end (a band from a cell to itself) the
    // way it came in.
    let gx = dx, gy = dy;
    if (dd <= 0 && up[k] >= 0) { gx = cx[k] - cx[up[k]]; gy = cy[k] - cy[up[k]]; }
    const gl = Math.hypot(gx, gy), fx = gl > 0 ? gx / gl : 0, fy = gl > 0 ? -gy / gl : 0;
    const rC = Math.ceil((Math.max(bk, be) + reach) / cellM) + 1;
    const i0 = Math.max(0, Math.floor(Math.min(ax, cx[e])) - rC), i1 = Math.min(n - 1, Math.ceil(Math.max(ax, cx[e])) + rC);
    const j0 = Math.max(0, Math.floor(Math.min(ay, cy[e])) - rC), j1 = Math.min(n - 1, Math.ceil(Math.max(ay, cy[e])) + rC);
    for (let j = j0; j <= j1; j++) {
      for (let i = i0; i <= i1; i++) {
        const u = dd > 0 ? Math.min(1, Math.max(0, ((i - ax) * dx + (j - ay) * dy) / dd)) : 0;
        const ox = i - ax - u * dx, oy = j - ay - u * dy;
        const x = Math.hypot(ox, oy) * cellM;
        const b = bk + u * (be - bk);
        if (x > b + reach) continue;
        const L = Lk + u * (Le - Lk), D = Dk + u * (De - Dk);
        // Signed across the river: + on the side the bend turns towards
        // (its inside). The deepest line sits towards the outside.
        const bendHere = Bk + u * (Be - Bk);
        const side = (dx * oy - dy * ox) >= 0 ? 1 : -1;
        const xs = side * x, x0 = -R.bend_shift * bendHere * b;
        const q = xs >= x0 ? (xs - x0) / (b - x0) : (x0 - xs) / (b + x0);
        const inner = side * bendHere > 0 ? Math.abs(bendHere) : -Math.abs(bendHere);
        const slope = R.bank_slope * (1 - 0.6 * inner);
        const want = x <= b ? L - D * (1 - Math.min(1, q) ** 2) : L + (x - b) * slope;
        const c = j * n + i;
        if (want >= best[c]) continue;
        best[c] = want;
        const f = Math.min(1, (b + reach - x) / (reach / 2));
        fade[c] = f * f * (3 - 2 * f);
        // Water only in the channel: past its rim the ground beside a
        // river can lie below the level, and the water hung over it.
        out[4 * c + 1] = x <= b ? L : NO_WATER;
        out[4 * c + 2] = fx; out[4 * c + 3] = fy;
      }
    }
  }
  // The cut, softened over river.soften metres: a crease where the bank
  // meets the ground bends every tile across it.
  const cut = new Float32Array(N);
  for (let c = 0; c < N; c++) {
    if (best[c] < Infinity) cut[c] = Math.min(0, best[c] - mz * (z[c] - 0.5)) * fade[c];
  }
  const rs = Math.max(1, Math.round(R.soften / 2 / cellM));
  const soft = boxBlur2(boxBlur2(cut, n, rs), n, rs);
  for (let c = 0; c < N; c++) out[4 * c] = Math.min(0, soft[c]);
  // The current: each band's direction, as fast as the channel is deep
  // there (Manning, v ~ depth^2/3, full at 0.3 m), averaged over half the
  // river's width so it turns smoothly where bands meet; a jump in it is a
  // seam in the moving surface.
  const vx = new Float32Array(N), vy = new Float32Array(N), inWater = new Float32Array(N);
  for (let c = 0; c < N; c++) {
    if (!(out[4 * c + 1] > NO_WATER / 2)) continue;   // the channel only
    const d = Math.max(0, out[4 * c + 1] - best[c]);
    const v = Math.min(1.2, Math.max(0.15, (d / 0.3) ** (2 / 3)));
    vx[c] = out[4 * c + 2] * v; vy[c] = out[4 * c + 3] * v; inWater[c] = 1;
  }
  const r = Math.max(1, Math.round(W / 4 / cellM));
  const [sx, sy, sw] = [vx, vy, inWater].map((g) => boxBlur2(boxBlur2(g, n, r), n, r));
  for (let c = 0; c < N; c++) {
    out[4 * c + 2] = sw[c] > 1e-6 ? sx[c] / sw[c] : 0;
    out[4 * c + 3] = sw[c] > 1e-6 ? sy[c] / sw[c] : 0;
  }
  return out;
}

/** Mean over a (2r+1) square, clipped at the edges, separable. */
function boxBlur2(g, n, r) {
  const tmp = new Float32Array(n * n), out = new Float32Array(n * n);
  for (let j = 0; j < n; j++) {
    let s = 0, k = 0;
    for (let i = -r; i < n + r; i++) {
      if (i + r < n) { s += g[j * n + i + r]; k++; }
      if (i - r - 1 >= 0) { s -= g[j * n + i - r - 1]; k--; }
      if (i >= 0 && i < n) tmp[j * n + i] = s / k;
    }
  }
  for (let i = 0; i < n; i++) {
    let s = 0, k = 0;
    for (let j = -r; j < n + r; j++) {
      if (j + r < n) { s += tmp[(j + r) * n + i]; k++; }
      if (j - r - 1 >= 0) { s -= tmp[(j - r - 1) * n + i]; k--; }
      if (j >= 0 && j < n) out[j * n + i] = s / k;
    }
  }
  return out;
}

/** flow() and each cell's receiver (-1 where nothing is lower). */
export function drainage(z, n) {
  const acc = new Float64Array(n * n).fill(1);
  const to = new Int32Array(n * n).fill(-1);

  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const here = z[j * n + i];
      let best = 0, bestIdx = -1;
      for (let b = -1; b <= 1; b++) {
        for (let a = -1; a <= 1; a++) {
          if (!a && !b) continue;
          const ii = i + a, jj = j + b;
          if (ii < 0 || ii >= n || jj < 0 || jj >= n) continue;
          const drop = (here - z[jj * n + ii]) / ((a && b) ? Math.SQRT2 : 1);
          if (drop > best) { best = drop; bestIdx = jj * n + ii; }
        }
      }
      to[j * n + i] = bestIdx;
    }
  }

  const order = Array.from({ length: n * n }, (_, k) => k)
    .sort((p, q) => z[q] - z[p]);
  for (const k of order) {
    const j = to[k];
    if (j >= 0) acc[j] += acc[k];
  }
  return { acc, to };
}

/** Rescale to 0..1, which makes the rule scale-free. A flat input gives
 *  0.5, not 0: with zeros, one class took the whole grid at zero relief and
 *  the rule jumped to full strength a hundredth of relief later. */
function unit(a) {
  let lo = Infinity, hi = -Infinity;
  for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const span = hi - lo;
  const out = new Float64Array(a.length);
  if (span > 1e-12) {
    for (let i = 0; i < a.length; i++) out[i] = (a[i] - lo) / span;
  } else {
    out.fill(0.5);
  }
  return out;
}

/** Median filter: removes specks without moving edges.
 *
 *  Third attempt at contiguous regions. Blurring the weights enough to
 *  remove the scatter left only a ramp (a dead straight boundary). A
 *  majority vote kept the boundary but broke the balance (30% asked, 13%
 *  given). A median is edge-preserving and runs before the threshold, so
 *  the share asked for is the share given. */
function median(a, n, radius) {
  const r = Math.max(0, radius | 0);
  if (!r) return Float64Array.from(a);
  const out = new Float64Array(n * n);
  const buf = new Float64Array((2 * r + 1) * (2 * r + 1));
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      let count = 0;
      for (let b = -r; b <= r; b++) {
        const jj = j + b;
        if (jj < 0 || jj >= n) continue;
        for (let c = -r; c <= r; c++) {
          const ii = i + c;
          if (ii < 0 || ii >= n) continue;
          buf[count++] = a[jj * n + ii];
        }
      }
      const slice = buf.subarray(0, count).slice().sort();
      out[j * n + i] = slice[count >> 1];
    }
  }
  return out;
}


/** Separable box blur, twice (a cheap Gaussian), so neighbours stop
 *  deciding over differences too small to mean anything. */
function smooth(a, n, radius) {
  const r = Math.max(0, radius | 0);
  if (!r) return a;
  let cur = Float64Array.from(a);
  const tmp = new Float64Array(n * n);
  for (let pass = 0; pass < 2; pass++) {
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        let sum = 0, count = 0;
        for (let c = -r; c <= r; c++) {
          const ii = i + c;
          if (ii < 0 || ii >= n) continue;
          sum += cur[j * n + ii];
          count++;
        }
        tmp[j * n + i] = sum / count;
      }
    }
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        let sum = 0, count = 0;
        for (let b = -r; b <= r; b++) {
          const jj = j + b;
          if (jj < 0 || jj >= n) continue;
          sum += tmp[jj * n + i];
          count++;
        }
        cur[j * n + i] = sum / count;
      }
    }
  }
  return cur;
}


/** Each value's rank among the others, 0 lowest to 1 highest. Ties share a
 *  rank, so a flat field stays flat. */
function rankUnit(a) {
  const n = a.length;
  if (n < 2) return a;
  const order = Array.from({ length: n }, (_, i) => i)
    .sort((p, q) => (a[p] - a[q]) || (p - q));
  const out = new Float64Array(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && a[order[j + 1]] === a[order[i]]) j++;
    const v = ((i + j) / 2) / (n - 1);
    for (let k = i; k <= j; k++) out[order[k]] = v;
    i = j + 1;
  }
  return out;
}

/** Value at quantile q (nearest rank). */
function quantile(values, q) {
  const sorted = Float64Array.from(values).sort();
  if (!sorted.length) return 0;
  const i = Math.min(sorted.length - 1,
                     Math.max(0, Math.round(q * (sorted.length - 1))));
  return sorted[i];
}

/** A class per cell, from the shape of the ground.
 *
 *  Each class scores every cell from the normalised maps (the positions in
 *  rule.json); the highest wins - Hybrid GSWT's Eq. 1 argmax with alpha_c
 *  from geometry. The default mapping (material collects where water runs
 *  and the ground is concave; steep convex ground is exposed) is a choice
 *  about the landscape, so it can be replaced with `opts.rules`.
 *
 *  opts: spacing, radius, sediment, sharpness (width of the undecided
 *  band), altitude (height against shape, 0..1), coherence (median radius),
 *  balance (share of class 0 with two classes, or an array of shares). */
/** Per cell, how far the height line between materials moves there:
 *  smooth noise from `large` cells across (the hills: the line climbs and
 *  dips) down to two cells (ragged edges, stray patches), each scale half
 *  as strong as the one above, as natural boundaries are (fractal; a few
 *  smooth scales gave rounded blobs), within +-amount/2 of the map's
 *  height range. Zero everywhere for amount 0. Seeded, so a layout is the
 *  same each time. */
export function wanderField(n, amount, seed = 0, large = 8) {
  const out = new Float64Array(n * n);
  if (!(amount > 0)) return out;
  const hash = (i, j, salt) => {
    let v = (i * 374761393 + j * 668265263 + seed * 2246822519 + salt * 3266489917) | 0;
    v = Math.imul(v ^ (v >>> 15), 2246822507);
    v = Math.imul(v ^ (v >>> 13), 3266489909);
    return ((v ^ (v >>> 16)) >>> 0) / 4294967296;
  };
  const noise = (x, y, s, salt) => {
    const u = x / s, v = y / s, i = Math.floor(u), j = Math.floor(v);
    const fx = u - i, fy = v - j, sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
    const a = hash(i, j, salt), b = hash(i + 1, j, salt);
    const c = hash(i, j + 1, salt), d = hash(i + 1, j + 1, salt);
    return (a + (b - a) * sx) * (1 - sy) + (c + (d - c) * sx) * sy;
  };
  const scales = [];
  for (let s = large, w = 1; s >= 2 && scales.length < 6; s /= 2.2, w /= 2) scales.push([s, w]);
  if (!scales.length) scales.push([2, 1]);
  const total = scales.reduce((t, [, w]) => t + w, 0);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      let v = 0;
      scales.forEach(([s, w], o) => { v += w * noise(i, j, s, o + 1); });
      out[j * n + i] = amount * (v / total - 0.5);
    }
  }
  return out;
}

export function classify(z, n, classes, opts = {}) {
  // Analysis at the finer sampling, then brought down to cells: at one
  // sample per tile the maps alias, and channels narrower than a cell vanish.
  const m = Math.round(Math.sqrt(z.length));
  const spacing = (opts.spacing || 1) * n / m;
  const s = unit(downsample(slope(z, m, spacing), m, n));
  const t = unit(downsample(tpi(z, m, opts.radius || Math.max(2, m >> 4)),
                            m, n));
  // Drainage spans orders of magnitude, so it is read as a log.
  const dr = drainage(z, m);
  const raw = dr.acc;
  const f = unit(downsample(raw.map(Math.log1p), m, n, 'max'));
  // River rule (opts.river): HAND and channels, from the same receivers.
  let near = null, chan = null;
  if (opts.river) {
    // On the filled field: a pit would end its paths and count its whole
    // basin as next to a channel. What filling raises is lake.
    const zf = fillDepressions(z, m);
    let lo = Infinity, hi = -Infinity;
    for (const v of z) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const depth = RULE.river.lake_depth * Math.max(hi - lo, 1e-12);
    const rd = drainage(zf, m);
    const racc = rd.acc.map((v, k) => ((zf[k] - z[k]) > depth ? Infinity : v));
    const cells = Math.max(8, Math.round(m * m * RULE.river.channel_share));
    const hd = hand(zf, m, rd.to, racc, cells);
    near = unit(downsample(hd.hand, m, n)).map((v) => 1 - v);
    chan = downsample(hd.channel, m, n, 'max');
    // A river drawn as water (opts.barZone, per sample 0..1, 1 in it): what
    // lies along it is the water's edge and bar, whatever the drainage says.
    if (opts.barZone) {
      const w = downsample(opts.barZone, m, n, 'max');
      for (let i = 0; i < near.length; i++) {
        near[i] = Math.max(near[i], w[i]);
        chan[i] = Math.max(chan[i], w[i] > 0.97 ? 1 : 0);
      }
    }
  }
  // Altitude: without it a hollow on a summit read the same as one in a valley.
  // `heightShift` (per cell, optional): where the height at which one
  // material gives way to the next is moved (wanderField), so the line
  // between them is not one contour across the world. None by default.
  const e0 = unit(downsample(z, m, n));
  const e = opts.heightShift ? e0.map((v, i) => v + opts.heightShift[i]) : e0;

  // Sediment, when the generator recorded it: where erosion actually left
  // material, rather than where drainage suggests it would. Max-pooled like
  // drainage. It replaces drainage rather than adding to it (same water twice).
  const sed = opts.sediment
    ? unit(downsample(opts.sediment, m, n, 'max')) : null;
  const drive = sed || f;

  const sharp = opts.sharpness == null ? 2 : opts.sharpness;
  // 0 = shape only, 1 = height only (the crudest terrain shader).
  const alt = opts.altitude == null ? 0.4 : opts.altitude;
  // Two classes are a valley and its walls; more are positions down one
  // hillside (floor, bench, face, crest). Each position in rule.json is
  // weighted terms plus a height term; a middle position scores nearness
  // to the middle of a map.
  const mid = (v) => 1 - 2 * Math.abs(v - 0.5);
  const TERM = {
    drive: (i) => drive[i], hollow: (i) => 1 - t[i], ridge: (i) => t[i],
    slope: (i) => s[i], flat: (i) => 1 - s[i], midpos: (i) => mid(t[i]),
    near: (i) => near[i], channel: (i) => chan[i],
  };
  const HEIGHT = {
    low: (i) => 1 - e[i], high: (i) => e[i], middle: (i) => mid(e[i]),
  };
  const position = (name) => {
    const p = RULE.positions[name];
    const terms = Object.entries(p.terms);
    const h = HEIGHT[p.height];
    return (i) => {
      // Summed in the file's order, from zero, exactly as the Python does.
      let shape = 0;
      for (const [k, wgt] of terms) shape = shape + wgt * TERM[k](i);
      return (1 - alt) * shape + alt * h(i);
    };
  };
  const table = opts.river ? RULE.river_classes : RULE.classes;
  const named = table[String(classes)] || table['4'].slice(0, Math.max(2, classes));
  const rules = opts.rules || named.map(position);

  const out = new Int32Array(n * n);
  const strength = new Float64Array(n * n);

  // Every class's weight at every cell, with one light blur against
  // fourth-decimal disagreements. Contiguity itself comes from the median
  // on the lead field below.
  const w = [];
  for (let k = 0; k < classes; k++) {
    const fn = rules[k % rules.length];
    const col = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) col[i] = Math.max(0, fn(i));
    // No exponent here: with a quantile threshold, powers keep ranks and
    // change nothing (this is why decisiveness used to be inert). It is
    // applied to the margin at the end instead.
    w.push(smooth(col, n, 1));
  }

  // With 3+ classes, rank each field against itself first: raw 'bench'
  // scores high almost anywhere and took 94% of the desert.
  if (classes > 2) for (let k = 0; k < classes; k++) w[k] = rankUnit(w[k]);

  const coherence = opts.coherence == null ? 2 : opts.coherence;

  let bias = 0, picked = null;
  // Two classes along a river drawn as water: the bar is where the river
  // puts it (opts.barZone over 0.5: low beside the water and within reach
  // of it, built by the viewer from the channel), so its share follows the
  // river and the valley instead of the capture's.
  // opts.barPos: which of the two is the bar (0 unless said).
  if (opts.river && opts.barZone && classes === 2) {
    const bar = opts.barPos === 1 ? 1 : 0;
    let lead = downsample(opts.barZone, m, n).map((v) => v - 0.5);
    lead = median(lead, n, coherence);
    picked = new Int32Array(n * n);
    for (let i = 0; i < n * n; i++) {
      picked[i] = lead[i] > 0 ? bar : 1 - bar;
      w[bar][i] = lead[i]; w[1 - bar][i] = 0;
    }
  } else
  // 3+ classes with shares: one threshold cannot do it, so cells are handed
  // out strongest preference first, each to the class it wants most that
  // still has room. A cell placed against its preference reads as undecided.
  if (Array.isArray(opts.balance) && classes > 2) {
    const coh = opts.coherence == null ? 2 : opts.coherence;
    for (let k = 0; k < classes; k++) w[k] = median(w[k], n, coh);

    let sum = 0;
    for (let k = 0; k < classes; k++) sum += Math.max(0, opts.balance[k] || 0);
    const quota = new Int32Array(classes);
    let left = n * n;
    for (let k = 0; k < classes; k++) {
      quota[k] = k === classes - 1 ? left
        : Math.min(left, Math.round(n * n * Math.max(0, opts.balance[k] || 0)
                                    / (sum || 1)));
      left -= quota[k];
    }

    const want = new Int32Array(n * n);
    const conf = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) {
      let best = -Infinity, second = -Infinity, bestK = 0;
      for (let k = 0; k < classes; k++) {
        const v = w[k][i];
        if (v > best) { second = best; best = v; bestK = k; }
        else if (v > second) { second = v; }
      }
      want[i] = bestK;
      conf[i] = best - second;
    }
    const order = Array.from({ length: n * n }, (_, i) => i)
      .sort((p, q) => (conf[q] - conf[p]) || (p - q));
    picked = new Int32Array(n * n).fill(-1);
    for (const i of order) {
      if (quota[want[i]] > 0) { picked[i] = want[i]; quota[want[i]]--; }
    }
    // Whoever is left takes whatever room is left, best first.
    for (const i of order) {
      if (picked[i] >= 0) continue;
      let bestK = -1, bestV = -Infinity;
      for (let k = 0; k < classes; k++) {
        if (quota[k] > 0 && w[k][i] > bestV) { bestV = w[k][i]; bestK = k; }
      }
      picked[i] = bestK < 0 ? want[i] : bestK;
      if (bestK >= 0) quota[bestK]--;
    }
  } else if (opts.balance != null && classes === 2) {
    // Two classes: class 0 takes the cells where it leads by the most, so
    // the share is exact and the terrain decides where. (Left alone, the
    // rule gave ~27% on every scene.)
    let lead = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) lead[i] = w[0][i] - w[1][i];
    // Median before the threshold, so the share asked is the share given.
    lead = median(lead, n, coherence);

    // Flat terrain: every cell ties and one class would take the grid.
    // A per-cell hash keeps the share and looks as arbitrary as it is.
    let lo = Infinity, hi = -Infinity;
    for (const v of lead) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (hi - lo < 1e-9) {
      for (let i = 0; i < n * n; i++) {
        let h = (i * 747796405 + 2891336453) >>> 0;
        h = (((h >>> ((h >>> 28) + 4)) ^ h) * 277803737) >>> 0;
        lead[i] = (((h >>> 22) ^ h) >>> 0) / 4294967296;
      }
    }
    // Rank with the index as tie-break: a plain quantile let every cell on
    // the threshold fall the same way (50% came back as 50.9%).
    const order = Array.from({ length: n * n }, (_, i) => i)
      .sort((p, q) => (lead[q] - lead[p]) || (p - q));
    const take = Math.round(opts.balance * n * n);
    picked = new Int32Array(n * n).fill(1);
    for (let i = 0; i < take; i++) picked[order[i]] = 0;
    // Boundary halfway between the last cell taken and the first left out,
    // so the margin below is a distance from the boundary.
    const hiV = take > 0 ? lead[order[take - 1]] : Infinity;
    const loV = take < n * n ? lead[order[take]] : -Infinity;
    bias = -(take === 0 ? hiV : take === n * n ? loV : (hiV + loV) / 2);
    for (let i = 0; i < n * n; i++) w[0][i] = lead[i];
    for (let i = 0; i < n * n; i++) w[1][i] = 0;
  }

  // The margin each cell decided by, which blending reads. (Computed from
  // the collapsed weights it used to be only 0.5 or 1, so the blend control
  // switched everything at once.) runnerUp is the class to blend towards.
  const margin = new Float64Array(n * n);
  const runnerUp = new Int32Array(n * n);
  for (let i = 0; i < n * n; i++) {
    let best = -Infinity, second = -Infinity, bestK = 0, secondK = 0;
    for (let k = 0; k < classes; k++) {
      const v = w[k][i] + (k === 0 ? bias : 0);
      if (v > best) { second = best; secondK = bestK; best = v; bestK = k; }
      else if (v > second) { second = v; secondK = k; }
    }
    const got = picked ? picked[i] : bestK;
    out[i] = got;
    if (got === bestK) {
      margin[i] = best - second;
      runnerUp[i] = secondK;
    } else {
      // Placed against its preference: undecided, blend towards what it wanted.
      margin[i] = 0;
      runnerUp[i] = bestK;
    }
  }

  // Scaled by the 90th percentile, not the max, so one odd cell cannot
  // flatten the rest. Sharpness acts here and only here: a power cannot
  // move a quantile boundary (it changed 2 cells in 576), but on the margin
  // it sets how wide the undecided, blended band is.
  const scale = Math.max(quantile(margin, 0.9), 1e-9);
  const power = 1 / Math.max(sharp, 1e-6);
  for (let i = 0; i < n * n; i++) {
    strength[i] = Math.min(1, Math.pow(margin[i] / scale, power));
  }

  return { cls: out, strength, runnerUp, weights: cueWeights(alt),
           maps: { slope: s, tpi: t, flow: f, elevation: e, height: e0, sediment: sed, near },
           source: sed ? 'sediment' : 'drainage' };
}

/** What each cue is worth at this altitude setting, as shares of one, for
 *  the collecting class. The panel prints them so "follows height" can be
 *  read rather than guessed. */
export function cueWeights(altitude = 0.4) {
  const a = Math.min(1, Math.max(0, altitude));
  const NAMES = { drive: 'drainage', hollow: 'hollows', ridge: 'ridges',
                  slope: 'slope', flat: 'flat ground', midpos: 'mid-slope' };
  const terms = RULE.positions.collected.terms;
  return [
    ...Object.entries(terms).map(([k, w]) => ({ cue: NAMES[k] || k, w: (1 - a) * w })),
    { cue: 'height', w: a },
  ];
}
