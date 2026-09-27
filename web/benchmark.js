// Finding where the renderer stops keeping up.
//
// For each grid size: hold the camera still (a turn mixes in re-sorts and
// LOD fades), wait for sort and merge to settle, average over a fixed
// number of frames. Reading down the table shows which gives out first -
// frame time, the sort or the draw calls - which says what to fix.

const SETTLE_FRAMES = 8;
const MEASURE_FRAMES = 40;

export class Benchmark {
  /** `apply(n)` sets the grid size. `read()` returns the current frame's
   *  numbers. `isBusy()` reports whether a sort or merge is in flight. */
  constructor({ sizes, apply, read, isBusy, onRow, onDone }) {
    Object.assign(this, { sizes, apply, read, isBusy, onRow, onDone });
    this.active = false;
  }

  start() {
    if (this.active || !this.sizes.length) return false;
    this.active = true;
    this.rows = [];
    this.i = -1;
    this.next();
    return true;
  }

  cancel() {
    this.active = false;
    this.rows = [];
  }

  next() {
    this.i++;
    if (this.i >= this.sizes.length) {
      this.active = false;
      if (this.onDone) this.onDone(this.rows);
      return;
    }
    this.apply(this.sizes[this.i]);
    this.settled = 0;
    this.samples = [];
  }

  /** Called once per rendered frame. */
  step(dt) {
    if (!this.active) return;
    if (this.isBusy && this.isBusy()) return;
    if (this.settled < SETTLE_FRAMES) { this.settled++; return; }

    const r = this.read();
    // dt is 0 on the first frame after the tab regains focus.
    if (dt > 0) this.samples.push({ dt, ...r });

    if (this.samples.length >= MEASURE_FRAMES) {
      const mean = (k) => this.samples.reduce((t, s) => t + s[k], 0)
                          / this.samples.length;
      // The worst frame is what people notice; kept beside the mean.
      const worst = Math.max(...this.samples.map((s) => s.dt));
      const row = {
        grid: this.sizes[this.i],
        cells: Math.round(mean('cells')),
        splats: Math.round(mean('splats')),
        calls: Math.round(mean('calls')),
        sortMs: mean('sortMs'),
        fps: 1000 / mean('dt'),
        worstMs: worst,
      };
      this.rows.push(row);
      if (this.onRow) this.onRow(row);
      this.next();
    }
  }
}

/** The measured rows as a table, plus which limit they point at (the same
 *  fps means different fixes if the sort or the fill is the bottleneck). */
export function report(rows) {
  if (!rows.length) return 'nothing measured';
  const lines = [];
  lines.push('grid   cells    splats    calls   sort ms   fps   worst ms');
  for (const r of rows) {
    lines.push(
      `${String(r.grid).padStart(4)}`
      + `${String(r.cells).padStart(7)}`
      + `${r.splats.toLocaleString().padStart(10)}`
      + `${String(r.calls).padStart(8)}`
      + `${r.sortMs.toFixed(1).padStart(10)}`
      + `${r.fps.toFixed(0).padStart(6)}`
      + `${r.worstMs.toFixed(1).padStart(11)}`);
  }

  const smooth = rows.filter((r) => r.fps >= 55);
  const usable = rows.filter((r) => r.fps >= 28);
  lines.push('');
  lines.push(smooth.length
    ? `  60 fps up to grid ${smooth[smooth.length - 1].grid}`
    : '  never reached 60 fps');
  lines.push(usable.length
    ? `  30 fps up to grid ${usable[usable.length - 1].grid}`
    : '  never reached 30 fps');

  // Which quantity turned over first.
  const last = rows[rows.length - 1];
  const first = rows[0];
  const sortGrew = last.sortMs / Math.max(first.sortMs, 0.01);
  const frameGrew = (1000 / last.fps) / Math.max(1000 / first.fps, 0.01);
  if (sortGrew > frameGrew * 1.3) {
    lines.push('  the sort grew faster than the frame: sorting is the limit');
  } else if (last.calls > 200) {
    lines.push(`  ${last.calls} draw calls at the top: per-cell overhead is `
               + 'likely the limit, not the splats');
  } else {
    lines.push('  frame time grew with splat count: fill rate is the limit, '
               + 'so lower LOD distances before anything else');
  }
  return lines.join('\n');
}