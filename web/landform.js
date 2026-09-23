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
// The rule's weights, shared with Python. Import attributes need a current
// browser (Chrome/Edge 123+, Safari 17.2+, Firefox 138+) and Node 22.
import RULE from './rule.json' with { type: 'json' };

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


/** Each value replaced by where it stands among the others, 0 for the
 *  lowest and 1 for the highest. Ties take the same rank, so a flat field
 *  stays flat rather than becoming a gradient of nothing. */
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
  // Two classes are a valley and its walls. More classes are a sequence
  // down the same hillside - crest, face, bench, floor - which is how a
  // slope actually reads, and each rule below names the position it
  // stands for. Hybrid GSWT is a multi-class method, so stopping at two
  // would leave the comparison short.
  //
  // The rule itself lives in rule.json, shared with bozkir/landform.py.
  // Each class is a position on a hillside with weighted terms and a
  // height term; a middle class wants a band, not an extreme, so it scores
  // on nearness to the middle of a map rather than on the map itself.
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

  // Every class's weight at every cell, so a balance can be struck against
  // the distribution rather than against the other classes pointwise.
  // One light blur of the weights, to stop two neighbours disagreeing over
  // a difference in the fourth decimal. Fixed and small: the work of
  // making regions contiguous is done by the median filter on the lead
  // field below, before the threshold, which does it without moving the
  // boundary or the share.
  const w = [];
  for (let k = 0; k < classes; k++) {
    const fn = rules[k % rules.length];
    const col = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) col[i] = Math.max(0, fn(i));
    // No exponent here. Raising every class's field to a power is the
    // shape of Hybrid GSWT's Eq. 1, but with a quantile threshold it
    // changes nothing a viewer can see - ranks survive powers - so it
    // used to make the decisiveness control inert. The exponent is
    // applied to the margin below instead, where it sets the width of
    // the blended band.
    w.push(smooth(col, n, 1));
  }

  // With three or more classes the fields have to be made comparable
  // before they compete. They are not: 'bench' scores high on almost any
  // terrain while 'collected' only scores high in a hollow, so raw values
  // handed the desert 94% bench, and forcing quotas on top of that
  // displaced two cells in three. Ranking each class's field against
  // itself asks the question that actually matters - is this cell more of
  // a bench than most, more of a hollow than most - and leaves two
  // classes, where a difference of two fields is already fair, alone.
  if (classes > 2) for (let k = 0; k < classes; k++) w[k] = rankUnit(w[k]);

  // `balance` shifts how much land the first class takes, without moving
  // where the boundary would naturally fall: the first class gets the
  // cells where it leads by the most. Null leaves the rule's own answer
  // alone, which is what it always did.
  const coherence = opts.coherence == null ? 2 : opts.coherence;

  let bias = 0, picked = null;
  // Several classes, each with a share of the grid. Two classes can be
  // settled by one threshold on the difference; three or more cannot, so
  // the cells are handed out instead: strongest preference first, each
  // cell to the class it wants most that still has room. A cell that ends
  // up somewhere it did not want reads as undecided, which is exactly
  // where blending should be drawing both materials.
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

    // Strongest preference first: a cell that only just prefers its class
    // should give way to one that prefers it by a mile.
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
    // Exactly the cells asked for, taken in order of how strongly they
    // lead. A quantile alone is one rank short of exact: the median
    // filter leaves many cells sharing a value, and every cell sitting
    // on the threshold falls the same way, so a 50% request came back as
    // 50.9%. Ranking with the index as the tie-break gives the count
    // asked for and is still deterministic.
    const order = Array.from({ length: n * n }, (_, i) => i)
      .sort((p, q) => (lead[q] - lead[p]) || (p - q));
    const take = Math.round(opts.balance * n * n);
    picked = new Int32Array(n * n).fill(1);
    for (let i = 0; i < take; i++) picked[order[i]] = 0;
    // Halfway between the last cell taken and the first left out, so the
    // margin below is a distance from the boundary rather than from one
    // cell's value.
    const hiV = take > 0 ? lead[order[take - 1]] : Infinity;
    const loV = take < n * n ? lead[order[take]] : -Infinity;
    bias = -(take === 0 ? hiV : take === n * n ? loV : (hiV + loV) / 2);
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
  // The class each cell would take if it could not have the one it got.
  // Blending needs it: with two classes the other one is obvious, with
  // more it is whichever the terrain ranks next, and drawing an arbitrary
  // third material there would be worse than not blending at all.
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
      // Placed somewhere it did not prefer, so it is undecided by
      // definition, and the class it wanted is the one to blend towards.
      margin[i] = 0;
      runnerUp[i] = bestK;
    }
  }

  // Scaled by a high percentile rather than the maximum, so one unusual
  // cell cannot flatten everything else to nothing.
  //
  // Sharpness lands here, and only here. Raising every class's field to a
  // power cannot move the boundary at all once a quantile sets the
  // threshold - a quantile is a rank, and a power leaves ranks alone - so
  // for a long time the control changed 2 cells in 576 across its whole
  // range. What it can honestly control is how *wide* the undecided band
  // is, which is the band drawn as a per-splat dissolve between the two
  // materials. Exponent 1/sharp: a decisive setting pushes middling
  // margins up towards 1 and narrows the band to the cells that really
  // are a tie; a soft setting pulls them down and widens it. That is
  // Hybrid GSWT's exponent doing its work where per-Gaussian mixing
  // happens, which is where theirs does it too.
  const scale = Math.max(quantile(margin, 0.9), 1e-9);
  const power = 1 / Math.max(sharp, 1e-6);
  for (let i = 0; i < n * n; i++) {
    strength[i] = Math.min(1, Math.pow(margin[i] / scale, power));
  }

  return { cls: out, strength, runnerUp, weights: cueWeights(alt),
           maps: { slope: s, tpi: t, flow: f, elevation: e, sediment: sed },
           source: sed ? 'sediment' : 'drainage' };
}

/** What each cue is actually worth at this altitude setting, as shares of
 *  one, for the class that collects material - the other class mirrors
 *  them with slope and ridge. The rule's weights are written inside the
 *  default rules, where a
 *  slider called "follows height" gives no clue how much of the decision
 *  is left for the shape of the ground. The panel prints these, so the
 *  control can be read rather than guessed at. */
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