// Choosing a tile's material from the shape of the ground under it.
//
// Hybrid Gaussian Wang Tiles selects a class at a world position from a
// scalar field per class - the target coverage alpha_c(x) of its Equation
// 1. Their formulation takes that field as input, and their authoring tool
// has a person paint it.
//
// Nobody paints a ten-kilometre map. In procedural terrain the rule has to
// come from the terrain, and it always has: loose material collects where
// water runs, bare ground shows on ridges where it is stripped, fines
// settle in hollows. Every terrain shader ever written places materials by
// slope and height. This does the same thing, with captured Gaussians
// instead of textures, and feeds the result to their equation.
//
// The maps below mirror bozkir/terrain.py, which is tested there against
// surfaces whose landforms are known by construction - a cone has a peak, a
// V has a valley, a tilted plane is a slope and not a ridge.
//
// One thing makes this cheap: the decision is per *cell*, not per pixel. A
// 64 by 64 grid is 4096 samples, so flow accumulation over the whole
// terrain costs less than one frame and is computed once per layout rather
// than per frame.

/** Sample a height function onto the cell grid.
 *
 *  Takes the same `height(x, y)` the renderer warps with, so the terrain
 *  the rule reads is the terrain that is drawn. Reading a different surface
 *  would place material by the shape of something invisible.
 */
export function sampleGrid(n, tileSize, height, gridAngle = 0, over = 1) {
  const k = Math.max(1, over | 0);
  const m = n * k;
  const z = new Float64Array(m * m);
  // The two grids have to cover the same ground. Spacing the fine samples
  // at tileSize/k puts them across (m-1)*tileSize/k, which is short of the
  // cells' own (n-1)*tileSize by most of a tile - so the analysis would
  // read terrain slightly offset from where the cells actually sit, and a
  // small grid would read it noticeably offset. Deriving the step from the
  // span both must share keeps them registered.
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

/** Bring an over-sampled map back down to one value per cell.
 *
 *  `how` matters and differs by map. Slope and position average sensibly:
 *  a cell's slope is the average of the slopes across it. Drainage does
 *  not. A channel is narrow by nature, so a cell holding one is mostly
 *  hillside, and averaging says hillside - the channel is diluted out of
 *  existence exactly where it matters. The largest value in the cell is
 *  the honest summary there: this cell contains a channel.
 */
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

/** Topographic position: height above the local mean.
 *
 *  Positive on ridges and spurs, negative in hollows and valleys, near zero
 *  on planar slopes. The radius decides what counts as local, and that is
 *  the whole character of the result - small finds every bump, large finds
 *  the shape of the landscape.
 */
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

/** How much terrain drains through each cell.
 *
 *  Every cell sends its water to its lowest neighbour; totals accumulate
 *  from the highest cell downwards, so one pass in descending height order
 *  finishes the job with no iteration.
 *
 *  This is the map that produces branching channels, and it produces them
 *  for the right reason: it is tracing where water goes on this surface,
 *  not drawing something channel-shaped.
 */
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

/** Rescale to 0..1.
 *
 *  Returns the midpoint for a flat input rather than zeros. That sounds
 *  like a detail and is not: stretching every map to its own full range
 *  means a terrain with one centimetre of relief and one with a hundred
 *  metres produce identical maps, which is what makes the thresholds
 *  scale-free. But it also means there is no gradual onset. At exactly
 *  zero relief the maps were all-zero, the rule read the same value
 *  everywhere, and one class took the whole grid; a hundredth of relief
 *  later the maps were stretched to full range and the rule was at full
 *  strength. Half way is the honest reading of "no information", and it
 *  leaves `confidence` to say that the rule is not deciding anything.
 */
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

/** The threshold that gives a class the share of the grid asked for.
 *
 *  Scale-free thresholds decide *where* the boundary falls but not how
 *  much land ends up either side - that comes out of the rule's arithmetic
 *  and was 27% in every scene tried, whatever the terrain. An artist wants
 *  the opposite: to say "a third of this should be scrub" and have the
 *  boundary land wherever the terrain says it should for that amount.
 *
 *  So the weights are compared against a quantile of themselves rather
 *  than against each other. Asking for 30% puts the threshold at the 70th
 *  percentile of the difference, which is by construction the value that
 *  gives 30% - on any terrain, at any relief.
 */
/** Median filter: remove specks without moving edges.
 *
 *  The third attempt at making regions contiguous, and the reasons the
 *  first two failed are worth keeping.
 *
 *  Blurring the weights worked but destroyed the terrain: a kernel wide
 *  enough to remove the scatter spanned a third of the grid, and all that
 *  survives that is the field's lowest-frequency component - a ramp. The
 *  boundary ran dead straight across the map.
 *
 *  A majority vote on the decided classes kept the boundary, but silently
 *  broke the balance. Scattered minority cells lose every vote they are
 *  in, so a field set to 30% came out at 13%: the control said one thing
 *  and the picture showed another, which is worse than either problem.
 *
 *  A median does both jobs. It is edge-preserving by construction - the
 *  median of a neighbourhood straddling a boundary is whichever side has
 *  more of it, so the edge stays put rather than smearing - and it runs
 *  before the threshold, so the quantile that sets the balance is taken
 *  on the filtered field and the share asked for is the share given.
 */
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


/** Blur a map, so neighbouring cells stop deciding independently.
 *
 *  Without this, two adjacent cells either side of the threshold make
 *  opposite choices over a difference too small to mean anything, and the
 *  result is a scatter of single tiles rather than regions. Material in a
 *  landscape is contiguous - a patch of scrub is a patch, not a dusting of
 *  isolated bushes on a chequerboard - and the rule has to be told that,
 *  because the terrain maps alone do not say it.
 *
 *  A box blur repeated is a cheap Gaussian, and the radius is the one
 *  number that says how large a patch of one material should be. That is a
 *  property of the landscape being depicted rather than of the terrain, so
 *  it is a control and not a constant.
 */
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


function quantile(values, q) {
  const sorted = Float64Array.from(values).sort();
  if (!sorted.length) return 0;
  const i = Math.min(sorted.length - 1,
                     Math.max(0, Math.round(q * (sorted.length - 1))));
  return sorted[i];
}

/** A class per cell, from the shape of the ground.
 *
 *  `weights` is one function per class of the normalised maps, returning how
 *  much that class wants each cell. The largest wins, which is the argmax of
 *  Hybrid GSWT's Equation 1 with alpha_c supplied by geometry rather than by
 *  hand.
 *
 *  The default rule is deliberately the simplest defensible one rather than
 *  a tuned one: material collects where water runs and the ground is
 *  concave, and is stripped where it is steep and convex. Whether that is
 *  the right rule for a particular pair of materials is a question about
 *  the landscape, not about the code - moss on shaded rock would want the
 *  opposite - so it is a parameter and not a constant.
 */
export function classify(z, n, classes, opts = {}) {
  // The height field may have been sampled finer than one point per cell,
  // and the analysis has to happen at that finer resolution before being
  // averaged down. Taking one sample per tile and computing slope from it
  // means the rule reads whatever the terrain happens to be doing at the
  // tile centres, which at two samples per landform is not the terrain at
  // all but an alias of it - a scatter that changes completely if the
  // grid moves half a tile. Flow accumulation suffers worst: a channel
  // narrower than a cell cannot be traced at all, so no channels appear
  // and the map that was supposed to place material by drainage places it
  // by noise.
  const m = Math.round(Math.sqrt(z.length));
  const spacing = (opts.spacing || 1) * n / m;
  const s = unit(downsample(slope(z, m, spacing), m, n));
  const t = unit(downsample(tpi(z, m, opts.radius || Math.max(2, m >> 4)),
                            m, n));
  // Flow spans orders of magnitude - a main channel drains thousands of
  // cells, a hillside drains one - so it is read as a log, which is also
  // how it is always mapped.
  const raw = flow(z, m);
  const f = unit(downsample(raw.map(Math.log1p), m, n, 'max'));
  // Altitude. Every terrain shader ever written places material by height
  // first - snow above a line, grass below one, bare rock between - and it
  // was the one obvious input missing here. Without it a hollow at the top
  // of a mountain reads the same as a hollow at the bottom, which is not
  // how any landscape works and was visible as high ground and low ground
  // getting the same material.
  const e = unit(downsample(z, m, n));

  // Where erosion actually left material, when the terrain was generated
  // and the generator wrote it down. Drainage above is an inference: water
  // would run here, so loose material probably collects here. The sediment
  // map is the record of where the simulation put it, and where there is a
  // record the rule should read it rather than guess at what it says.
  //
  // Max-pooled like drainage, for the same reason: a deposit is narrow, and
  // a cell containing one should read as containing one.
  const sed = opts.sediment
    ? unit(downsample(opts.sediment, m, n, 'max')) : null;
  // Drainage steps aside for the record rather than being added to it.
  // Adding them would count the same water twice: the sediment is where
  // the drainage put things.
  const drive = sed || f;

  const sharp = opts.sharpness == null ? 2 : opts.sharpness;
  // `altitude` weights height against the rest. At 0 the rule is purely
  // about shape, which is what it did before; at 1 it is almost purely
  // about elevation, which is the crudest terrain shader there is and
  // still looks like something. The interesting settings are in between.
  const alt = opts.altitude == null ? 0.4 : opts.altitude;
  const rules = opts.rules || [
    // 0: collected. Low ground, where water runs, where it is concave.
    (i) => (1 - alt) * (0.6 * drive[i] + 0.4 * (1 - t[i])) + alt * (1 - e[i]),
    // 1: exposed. High ground, steep, convex, nothing draining through.
    (i) => (1 - alt) * (0.5 * s[i] + 0.5 * t[i]) + alt * e[i],
  ];

  const out = new Int32Array(n * n);
  const strength = new Float64Array(n * n);

  // Every class's weight at every cell, so a balance can be struck against
  // the distribution rather than against the other classes pointwise.
  // One light blur of the weights, to stop two neighbours disagreeing over
  // a difference in the fourth decimal. Fixed and small: the work of
  // making regions contiguous is done afterwards by the majority filter,
  // which does it without moving the boundary.
  const w = [];
  for (let k = 0; k < classes; k++) {
    const fn = rules[k % rules.length];
    const col = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) col[i] = Math.max(0, fn(i));
    const blurred = smooth(col, n, 1);
    for (let i = 0; i < n * n; i++) blurred[i] = Math.pow(blurred[i], sharp);
    w.push(blurred);
  }

  // `balance` shifts how much land the first class takes, without moving
  // where the boundary would naturally fall: the first class gets the
  // cells where it leads by the most. Null leaves the rule's own answer
  // alone, which is what it always did.
  const coherence = opts.coherence == null ? 2 : opts.coherence;

  let bias = 0;
  if (opts.balance != null && classes === 2) {
    let lead = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) lead[i] = w[0][i] - w[1][i];
    // Cleaned before the threshold, so the quantile below is taken on the
    // field that will actually be thresholded and the share asked for is
    // the share given.
    lead = median(lead, n, coherence);

    // On terrain that carries no information - dead flat, or relief so
    // small that every map comes out constant - every cell ties, and a
    // threshold has nothing to sort. Ties then all fall the same way and
    // one class takes the whole grid, which reads as a decision when it is
    // the absence of one. A per-cell hash breaks them, so the share asked
    // for is the share given and the arrangement is visibly arbitrary,
    // which is what "the terrain has nothing to say" should look like.
    let lo = Infinity, hi = -Infinity;
    for (const v of lead) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (hi - lo < 1e-9) {
      for (let i = 0; i < n * n; i++) {
        let h = (i * 747796405 + 2891336453) >>> 0;
        h = (((h >>> ((h >>> 28) + 4)) ^ h) * 277803737) >>> 0;
        lead[i] = (((h >>> 22) ^ h) >>> 0) / 4294967296;
      }
    }
    bias = -quantile(lead, 1 - opts.balance);
    for (let i = 0; i < n * n; i++) w[0][i] = lead[i];
    for (let i = 0; i < n * n; i++) w[1][i] = 0;
  }

  // The margin each cell decided by, scaled against how much the margins
  // vary across the whole field. This is what blending reads, and getting
  // it wrong made the blend control do nothing: with a balance set the
  // weights collapse to a difference and a zero, so confidence came out as
  // exactly 1 or exactly 0.5 - two values, no middle. A cell was either
  // fully decided or fully undecided, and widening the blend simply
  // switched every second-class cell on at once.
  const margin = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) {
    let best = -Infinity, second = -Infinity, bestK = 0;
    for (let k = 0; k < classes; k++) {
      const v = w[k][i] + (k === 0 ? bias : 0);
      if (v > best) { second = best; best = v; bestK = k; }
      else if (v > second) { second = v; }
    }
    out[i] = bestK;
    margin[i] = best - second;
  }

  // Scaled by a high percentile rather than the maximum, so one unusual
  // cell cannot flatten everything else to nothing.
  const scale = Math.max(quantile(margin, 0.9), 1e-9);
  for (let i = 0; i < n * n; i++) {
    strength[i] = Math.min(1, margin[i] / scale);
  }

  return { cls: out, strength,
           maps: { slope: s, tpi: t, flow: f, elevation: e, sediment: sed },
           source: sed ? 'sediment' : 'drainage' };
}