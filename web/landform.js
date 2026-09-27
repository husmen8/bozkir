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
const RULE = await loadRule(new URL('./rule.json', import.meta.url));

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
  return acc;
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
export function classify(z, n, classes, opts = {}) {
  // Analysis at the finer sampling, then brought down to cells: at one
  // sample per tile the maps alias, and channels narrower than a cell vanish.
  const m = Math.round(Math.sqrt(z.length));
  const spacing = (opts.spacing || 1) * n / m;
  const s = unit(downsample(slope(z, m, spacing), m, n));
  const t = unit(downsample(tpi(z, m, opts.radius || Math.max(2, m >> 4)),
                            m, n));
  // Drainage spans orders of magnitude, so it is read as a log.
  const raw = flow(z, m);
  const f = unit(downsample(raw.map(Math.log1p), m, n, 'max'));
  // Altitude: without it a hollow on a summit read the same as one in a valley.
  const e = unit(downsample(z, m, n));

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
  const named = RULE.classes[String(classes)]
    || RULE.classes['4'].slice(0, Math.max(2, classes));
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
           maps: { slope: s, tpi: t, flow: f, elevation: e, sediment: sed },
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
